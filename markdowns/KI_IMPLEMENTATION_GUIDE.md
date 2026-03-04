# Knowledge Insulation (KI) Implementation Guide

## Overview

Knowledge Insulation is a training methodology for vision-language-action (VLA) models that enables the backbone VLM to learn action-relevant representations while preserving its language understanding capabilities. This document explains the implementation details for the Pi05-KI model.

## Architecture

### Hybrid Token Prediction in a Single Forward Pass

The KI pipeline uses a dual-head architecture that operates in **one forward pass**:

1. **VLM Backbone (PaliGemma)**: Processes prefix and predicts FAST tokens
   - Input: Vision + Language + FAST-tokenized action sequence
   - Output: `prefix_out` - used for FAST token prediction loss
   - Side Effect: Produces enriched latent representations (key-value matrices)
   - Purpose: Forces latent representations to encode action information

2. **Action Expert (Gemma 300M)**: Processes suffix and predicts continuous actions
   - Input: Noisy actions (from flow-matching) + attends to VLM's KV cache
   - Output: `suffix_out` - used for flow-matching loss
   - Purpose: Decodes action information from VLM's enriched latents into precise continuous values

### Key Insight: Knowledge Insulation via Gradient Stopping

**Everything happens in one forward pass, but gradients are insulated:**
```python
# Single forward pass through dual-head architecture
(prefix_out, suffix_out), kv_cache = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens],  # VLM processes prefix, action expert processes suffix
    mask=attn_mask, 
    positions=positions, 
    adarms_cond=[None, adarms_cond]
)

# Two losses from the same pass:
action_loss = flow_matching_loss(suffix_out)  # Action expert output
fast_loss = cross_entropy_loss(prefix_out)     # VLM backbone output

# CRITICAL: Stop gradients from action_loss to VLM backbone
# This prevents action learning from corrupting language understanding
total_loss = action_loss + λ * fast_loss
# With gradient stopping, only fast_loss updates VLM parameters
```

**Gradient Flow Architecture:**
- **VLM Parameters**: Updated ONLY by `fast_loss` (FAST token prediction)
- **Action Expert Parameters**: Updated ONLY by `action_loss` (flow-matching)
- **Forward Pass**: Action expert can attend to VLM features (read-only)
- **Backward Pass**: Action expert gradients STOP at VLM boundary (insulation)

This creates "knowledge insulation" where:
- The VLM learns action-relevant representations via FAST tokens
- The action expert learns to decode these representations
- **Action learning cannot corrupt the VLM's language understanding** (gradients are blocked)

## Current Implementation Status

### ✅ Already Implemented

1. **Model Configuration** ([pi0_config.py](src/openpi/models/pi0_config.py))
   ```python
   knowledge_insulation: bool = False  # Flag to enable KI
   ```

2. **Model Type** ([model.py](src/openpi/models/model.py))
   ```python
   class ModelType(enum.StrEnum):
       PI05_KI = "pi05_ki"  # New model type for KI
   ```

3. **Training Config** ([config.py](src/openpi/training/config.py))
   - FAST tokenizer integration for input transforms
   - Config entry for `pi05_droid_ki` with KI enabled
   - Knowledge insulation lambda parameter: `knowledge_insulation_lambda`

4. **Model Initialization** ([pi0.py](src/openpi/models/pi0.py))
   ```python
   self.knowledge_insulation = config.knowledge_insulation
   ```

## Required Changes

### 1. Dual-Head Loss Computation

**Location**: [pi0.py](src/openpi/models/pi0.py) - `compute_loss()` method

**Current Implementation**:
```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # ... flow-matching setup ...
    
    # Single forward pass produces both outputs
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], 
        mask=attn_mask, 
        positions=positions, 
        adarms_cond=[None, adarms_cond]
    )
    
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # Only action expert loss
```

**Key Insight**: The forward pass already computes both outputs in a single pass:
- `prefix_out`: VLM backbone output (PaliGemma) - used for FAST token prediction
- `suffix_out`: Action expert output (Gemma-300M) - already attends to VLM's latents via shared KV cache

**Required Changes**:
```python
def compute_loss(self, rng, observation, actions, *, train=False):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # Embed prefix and suffix
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    
    # Single forward pass - VLM processes prefix, action expert attends to VLM latents
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], 
        mask=attn_mask, 
        positions=positions, 
        adarms_cond=[None, adarms_cond]
    )
    
    # Action Expert Loss (flow-matching on continuous actions)
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    action_expert_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    if self.knowledge_insulation and train:
        # VLM Backbone Loss (FAST token prediction)
        # Get FAST token targets from observation (added by TokenizeFASTInputs)
        fast_token_targets = observation.fast_action_tokens  # Shape: (b, action_horizon)
        
        # Get logits from VLM backbone for the action token positions in the prefix
        # FAST tokens are part of the prefix sequence (after images + language)
        vlm_logits = self.get_vlm_logits(prefix_out)  # Shape: (b, seq_len, vocab_size)
        
        # Extract logits for action token positions
        # Action tokens are in the prefix after: image tokens + language tokens
        num_image_tokens = sum([tokens.shape[1] for tokens in 
                               [self.PaliGemma.img(obs.images[name], train=False)[0] 
                                for name in obs.images]])
        num_lang_tokens = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
        action_start_idx = num_image_tokens + num_lang_tokens
        action_end_idx = action_start_idx + self.action_horizon
        action_logits = vlm_logits[:, action_start_idx:action_end_idx, :]
        
        # Compute cross-entropy loss for FAST token prediction
        import optax
        fast_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            action_logits, fast_token_targets
        )
        fast_token_loss = jnp.mean(fast_token_loss)
        
        # CRITICAL: Knowledge Insulation via Gradient Stopping
        # Stop gradients from action_expert_loss to VLM parameters
        # This is achieved by stopping gradients through the cross-attention path
        # The VLM's contribution to suffix_out should not receive gradients from action_expert_loss
        
        # Method 1: Stop gradient on prefix_out when computing action loss
        # (This prevents action gradients from flowing back through shared VLM)
        # NOTE: This is done implicitly by the dual-head architecture where
        # action_expert_loss only backprops through suffix_out path
        
        # Method 2: Explicitly separate gradient flows in training loop
        # Return losses as dict and handle gradient application separately
        
        # Combined loss: Only fast_loss updates VLM, only action_loss updates action expert
        total_loss = action_expert_loss + self.knowledge_insulation_lambda * fast_token_loss
        
        # Optional: Return individual losses for logging
        if hasattr(self, '_log_losses'):
            self._log_losses = {
                'action_expert_loss': action_expert_loss,
                'fast_token_loss': fast_token_loss,
                'total_loss': total_loss,
                'lambda': self.knowledge_insulation_lambda
            }
        
        return total_loss
    
    return action_expert_loss


def get_vlm_logits(self, hidden_states):
    """Project VLM hidden states to vocabulary space for FAST token prediction.
    
    Args:
        hidden_states: Output from VLM backbone, shape (b, seq_len, hidden_dim)
    
    Returns:
        logits: Predictions over vocabulary, shape (b, seq_len, vocab_size)
    """
    # TODO: Implement vocabulary projection
    # Need to access Gemma's unembedding layer to project to vocab space
    # This may require modifications to the Gemma module to expose the embedder
    pass
```

**Critical Implementation Note on Gradient Stopping:**

The architecture naturally provides gradient insulation because:
1. `prefix_out` (VLM output) → only used for `fast_token_loss`
2. `suffix_out` (Action expert output) → only used for `action_expert_loss`
3. The action expert's parameters are separate from the VLM's parameters

However, since both share the same forward pass through `self.PaliGemma.llm`, we need to ensure:
- When computing `action_expert_loss.backward()`, gradients stop at the VLM boundary
- When computing `fast_token_loss.backward()`, gradients flow to VLM

**Implementation Strategy:**

The cleanest approach is to handle this in the training loop with separate gradient applications:
```python
# In training loop (train.py or train_pytorch.py)
if model.knowledge_insulation:
    # Compute losses separately
    action_loss, fast_loss = model.compute_loss_separated(...)
    
    # Apply gradients separately with different parameter filters
    # Action loss: update only action expert parameters
    action_grads = jax.grad(lambda params: action_loss)(action_expert_params)
    
    # FAST loss: update only VLM parameters
    fast_grads = jax.grad(lambda params: fast_loss)(vlm_params)
    
    # Apply both gradient updates
    params = apply_gradients(params, action_grads, fast_grads)
```

Alternatively, use `jax.lax.stop_gradient` on the cross-attention path, but this requires careful architectural modifications.

### 2. Observation Dataclass Extension

**Location**: [model.py](src/openpi/models/model.py) - `Observation` dataclass

**Required Addition**:
```python
@dataclasses.dataclass
class Observation:
    images: dict[str, at.UInt8[at.Array, "b h w 3"]]
    image_masks: dict[str, at.Bool[at.Array, "b"]]
    state: at.Float[at.Array, "b state_dim"]
    tokenized_prompt: at.Int32[at.Array, "b prompt_len"] | None
    tokenized_prompt_mask: at.Bool[at.Array, "b prompt_len"] | None
    
    # NEW: For Knowledge Insulation
    fast_action_tokens: at.Int32[at.Array, "b action_horizon"] | None = None
    fast_action_tokens_mask: at.Bool[at.Array, "b action_horizon"] | None = None
```

### 3. FAST Tokenization in Data Pipeline

**Location**: [transforms.py](src/openpi/transforms.py) - `TokenizeFASTInputs` class

**Current Issue**: The transform needs to:
1. Tokenize actions using FAST tokenizer during training
2. Store FAST tokens in observation for loss computation
3. NOT tokenize during inference (since we're predicting actions, not encoding them)

**Required Changes**:
```python
class TokenizeFASTInputs(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, data: dict, *, train: bool = False) -> dict:
        # ... existing prompt & state tokenization ...
        
        if train and "actions" in data:
            # During training: tokenize actions and store FAST tokens
            actions = data["actions"]
            fast_tokens = self.tokenizer.tokenize_actions(actions)
            data["observation"]["fast_action_tokens"] = fast_tokens
            data["observation"]["fast_action_tokens_mask"] = np.ones_like(fast_tokens, dtype=bool)
        
        return data
```

### 4. VLM Logits Extraction

**Location**: [gemma.py](src/openpi/models/gemma.py) or [pi0.py](src/openpi/models/pi0.py)

**Required Addition**:
The PaliGemma LLM needs a method to extract logits from the backbone (first head) before the action expert projection.

**Option A**: Modify Gemma module to return logits
```python
# In Gemma Module class
def get_backbone_logits(self, hidden_states):
    """Get logits from VLM backbone (PaliGemma) head."""
    # Project to vocabulary space
    logits = self.embedder.decode(hidden_states)  # Assuming embedder is tied
    return logits
```

**Option B**: Add logit computation in Pi0 model
```python
# In Pi0 class
def compute_vlm_logits(self, hidden_states):
    """Compute VLM logits for FAST token prediction."""
    # This assumes the VLM has a language modeling head
    # May need to add a separate projection layer
    return self.PaliGemma.llm.lm_head(hidden_states)
```

### 5. Training Loop Integration

**Location**: [train.py](src/openpi/scripts/train.py) or [train_pytorch.py](src/openpi/scripts/train_pytorch.py)

**Current**: Loss is returned from `model.compute_loss()`

**Required**: Pass `knowledge_insulation_lambda` from config to model
```python
# In training config or model init
model.knowledge_insulation_lambda = config.knowledge_insulation_lambda
```

### 6. Action Dimension Mismatch Fix

**Issue**: Current error shows shape mismatch (8,) vs (32,)
- DROID has 8-dim actions (7 joints + 1 gripper)
- Model is configured for 32-dim actions

**Location**: [config.py](src/openpi/training/config.py) - `pi05_droid_ki` config

**Fix**:
```python
TrainConfig(
    name="pi05_droid_ki",
    model=pi0_config.Pi0Config(
        action_horizon=15, 
        pi05=True, 
        knowledge_insulation=True,
        action_dim=8,  # CHANGE: Match DROID action space
    ),
    # ... rest of config ...
)
```

## Implementation Sequence

### Phase 1: Basic Infrastructure (Current)
- [x] Add `knowledge_insulation` flag to config
- [x] Add `PI05_KI` model type
- [x] Integrate FAST tokenizer for inputs
- [x] Update train config for KI

### Phase 2: Data Pipeline
- [ ] Extend `Observation` dataclass with FAST token fields
- [ ] Modify `TokenizeFASTInputs` to store FAST tokens during training
- [ ] Add FAST action tokenization method to tokenizer
- [ ] Test data pipeline with dummy data

### Phase 3: Model Changes
- [ ] Add VLM logits extraction method
- [ ] Implement dual-head loss computation
- [ ] Add lambda weighting parameter
- [ ] Test forward/backward pass

### Phase 4: Training Integration
- [ ] Pass lambda parameter from config to model
- [ ] Update training loop to handle KI loss
- [ ] Add logging for both loss components
- [ ] Test training on small dataset

### Phase 5: Validation
- [ ] Verify FAST tokens are correctly predicted
- [ ] Verify action expert still produces good actions
- [ ] Compare with baseline Pi05 performance
- [ ] Test inference (should only use action expert output)

## Key Design Decisions

### 1. Gradient Insulation (Critical!)

**The Core Principle**: Gradients from action expert loss must NOT flow to VLM parameters.

**Why?**
- Prevents action learning from corrupting the VLM's language understanding
- VLM maintains its pretrained knowledge while learning action-relevant features
- Action expert learns task-specific behavior without damaging the foundation model

**How to Implement:**

**Option A: Separate Parameter Groups (Recommended)**
```python
# In training loop
if model.knowledge_insulation:
    # Define parameter groups
    vlm_params = model.get_vlm_parameters()
    action_params = model.get_action_expert_parameters()
    
    # Compute losses
    action_loss, fast_loss = model.compute_losses(...)
    
    # Apply gradients separately
    vlm_grads = jax.grad(lambda p: fast_loss, allow_int=True)(vlm_params)
    action_grads = jax.grad(lambda p: action_loss)(action_params)
    
    # Update only the respective parameters
    vlm_params = apply_updates(vlm_params, vlm_grads)
    action_params = apply_updates(action_params, action_grads)
```

**Option B: Stop Gradient on Cross-Attention**
```python
def embed_suffix(self, obs, noisy_actions, timestep):
    # ... existing code ...
    
    if self.knowledge_insulation:
        # Stop gradients from action expert to VLM
        # This prevents action_loss from updating VLM parameters
        # But still allows action expert to read VLM features (forward pass)
        action_tokens = jax.lax.stop_gradient(self.get_vlm_features()) 
    
    # ... rest of suffix embedding ...
```

**Option C: Parameter Filtering in Optimizer**
```python
# When creating optimizer, filter which parameters get which gradients
if model.knowledge_insulation:
    optimizer_vlm = optim.create(vlm_params, lr_schedule)
    optimizer_action = optim.create(action_params, lr_schedule)
    
    # Only vlm_params get fast_loss gradients
    # Only action_params get action_loss gradients
```

### 2. FAST Tokens in Inference

**Decision**: During inference, FAST tokens are NOT used from the model output.

**Rationale**: 
- VLM predicts FAST tokens only to enrich its latent representations
- Action expert produces the final continuous actions
- FAST token predictions are auxiliary and used only during training

### 2. Loss Weighting

**Lambda Parameter**: Controls balance between action expert and VLM losses

**Typical Range**: 0.1 to 1.0
- Higher λ: More emphasis on FAST token prediction (better VLM representations)
- Lower λ: More emphasis on action quality (better final performance)

**Recommendation**: Start with λ = 0.5 and tune based on validation performance

### 3. FAST Tokenization vs. Continuous Actions

**Training Data Flow**:
```
Actions (continuous)
    ↓
FAST Tokenizer → FAST tokens (discrete) → VLM loss
    ↓
Flow-matching noise → Noisy actions → Action expert loss
```

### 4. Attention Masks

**Important**: The action expert must attend to VLM's enriched representations
- VLM processes: images + language + FAST-tokenized actions
- Action expert queries: attend to all VLM key-values
- This cross-attention is where knowledge transfer happens

## Testing Strategy

### Unit Tests

1. **FAST Tokenization**
   ```python
   def test_fast_tokenization():
       actions = np.random.randn(1, 15, 8)
       tokens = tokenizer.tokenize_actions(actions)
       assert tokens.shape == (1, 15)
       assert tokens.dtype == np.int32
   ```

2. **Dual Loss Computation**
   ```python
   def test_ki_loss():
       model = Pi0(config_with_ki)
       obs, actions = create_dummy_data()
       loss = model.compute_loss(rng, obs, actions, train=True)
       assert loss.shape == (batch_size,)
   ```

### Integration Tests

1. **Training Loop**
   - Run 10 steps with KI enabled
   - Verify both losses decrease
   - Check gradients flow to both heads

2. **Inference**
   - Load KI checkpoint
   - Run inference on test data
   - Verify output shape and quality

## Debugging Tips

1. **Loss Explosion**: If FAST token loss dominates, reduce λ
2. **Poor Actions**: If actions are bad, increase action expert loss weight
3. **Shape Mismatches**: Check action_dim matches dataset
4. **Token Alignment**: Verify FAST tokens align with action sequence

## References

- Paper: [Insert paper link/name when available]
- Pi0.5 Paper: For baseline architecture
- FAST Tokenizer: For discrete action representation
- Flow Matching: For continuous action prediction

## Questions to Resolve

1. **VLM Logits**: Where in the Gemma architecture should we extract logits?
   - Before final layer norm?
   - After projection to vocab?
   - Need to check Gemma implementation

2. **FAST Token Targets**: How to handle action horizon?
   - One token per timestep?
   - Chunked encoding?
   - Check FAST tokenizer output format

3. **Gradient Flow**: Should VLM gradients flow through to vision encoder?
   - Probably yes, to learn action-relevant visual features
   - May need to tune learning rates separately

4. **Checkpoint Loading**: How to handle pre-trained weights?
   - VLM from Pi0.5: Yes
   - Action expert from Pi0.5: Yes
   - Need to verify weight compatibility
