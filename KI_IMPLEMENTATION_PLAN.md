# Knowledge Insulation (KI) Implementation Plan for π0.5 DROID

## Current Understanding

Your training data:
- **Input**: Images + text prompt
- **Output**: Action chunk (continuous values, shape `[batch, action_horizon, action_dim]`)
- **Current training**: Flow matching to predict continuous actions

## What KI Needs

Knowledge Insulation requires **dual-objective training**:

### 1. VLM Expert (Expert 0) - Trains on FAST Tokens
- **Input**: Images + prompt
- **Output**: Predicts FAST tokens autoregressively (next-token prediction)
- **Loss**: Cross-entropy loss on FAST token sequence
- **Purpose**: Learn to predict discretized action sequences

### 2. Action Expert (Expert 1) - Trains on Continuous Actions  
- **Input**: Detached VLM embeddings + noisy actions + timestep
- **Output**: Predicts velocity for flow matching
- **Loss**: MSE between predicted and true velocity
- **Purpose**: Learn continuous action refinement
- **Key**: Gradients from this loss do NOT update VLM parameters (gradient stopping)

## Data Pipeline Changes Needed

### Step 1: Convert Actions to FAST Tokens

You need to **tokenize your action chunks** using the FAST tokenizer:

```python
# Pseudocode for data pipeline
from openpi.models.tokenizer import FASTTokenizer

# During data loading:
action_chunk = ... # shape: [action_horizon, action_dim], e.g., [16, 7]

# Tokenize actions to FAST tokens
fast_tokenizer = FASTTokenizer(...)
fast_tokens = fast_tokenizer.encode(action_chunk)  # shape: [action_horizon * num_tokens_per_action]
# Example: [16, 7] → [16 * 3] = [48] tokens (if 3 tokens per action dimension)
```

### Step 2: Concatenate with Prompt Tokens

The VLM needs a **single token sequence**:
```
[image tokens] + [prompt tokens] + [FAST action tokens]
```

Example sequence:
```
Image tokens:    [256 tokens from base_image] + [256 tokens from wrist_image]
Prompt tokens:   [10 tokens from "pick up the cup"]
FAST tokens:     [48 tokens from action chunk]
Total:           512 + 10 + 48 = 570 tokens
```

### Step 3: Create Loss Mask

You need to specify **which tokens to compute loss on**:

```python
token_loss_mask = [
    # False for image tokens (don't predict these)
    [False] * 512,
    # False for prompt tokens (don't predict these)  
    [False] * 10,
    # True for FAST action tokens (predict these autoregressively)
    [True] * 48
]
```

## Required Changes by File

### 1. Data Transform (`DroidInputs` or similar)

**File**: `openpi/policies/droid_policy.py` or custom transform

**Changes needed**:
```python
# Add FAST tokenizer initialization
self.fast_tokenizer = FASTTokenizer(
    action_dim=action_dim,
    vocab_size=...,  # Need to determine
    num_bins=...     # Need to determine
)

# In __call__ method, tokenize actions:
fast_tokens = self.fast_tokenizer.encode(actions)

# Concatenate with prompt tokens
full_token_sequence = jnp.concatenate([
    prompt_tokens,      # From PaliGemmaTokenizer
    fast_tokens         # From FASTTokenizer
], axis=-1)

# Create loss mask (only compute loss on FAST tokens)
token_loss_mask = jnp.concatenate([
    jnp.zeros(len(prompt_tokens), dtype=bool),  # Don't predict prompt
    jnp.ones(len(fast_tokens), dtype=bool)       # Do predict actions
])

# Add to observation
observation.tokenized_prompt = full_token_sequence
observation.token_loss_mask = token_loss_mask
```

### 2. Model Configuration (`pi0_config.py`)

**File**: `openpi/src/openpi/models/pi0_config.py`

**Changes needed**:
```python
@dataclasses.dataclass
class Pi0Config(_model.BaseModelConfig):
    # ... existing fields ...
    
    # NEW: KI parameters
    knowledge_insulation: bool = False
    ki_fast_loss_weight: float = 1.0  # Weight for FAST token prediction loss
```

### 3. Model Implementation (`pi0.py`)

**File**: `openpi/src/openpi/models/pi0.py`

**Changes needed in `__init__`**:
```python
def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
    # ... existing code ...
    self.knowledge_insulation = config.knowledge_insulation
    self.ki_fast_loss_weight = config.ki_fast_loss_weight
```

**Changes in `compute_loss`** (this is the main implementation):
```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # ... existing preprocessing ...
    
    if self.knowledge_insulation and self.pi05 and train:
        # === VLM FAST Token Loss ===
        # 1. Forward pass through VLM only for FAST token prediction
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],  # Only Expert 0 (VLM)
            mask=prefix_attn_mask,
            positions=positions,
        )
        
        # 2. Compute cross-entropy loss on FAST tokens
        vocab_logits = self.PaliGemma.llm(pre_logits=prefix_out[:, :-1])[0]
        target_tokens = observation.tokenized_prompt[:, 1:]  # Next-token prediction
        token_loss_mask = observation.token_loss_mask[:, 1:]
        
        fast_loss = cross_entropy_loss(vocab_logits, target_tokens, token_loss_mask)
        
        # === Action Expert Flow Matching Loss ===
        # 3. DETACH prefix tokens (gradient stopping)
        prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)
        
        # 4. Forward pass with action expert
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        
        full_tokens = [prefix_tokens_detached, suffix_tokens]
        (_, suffix_out), _ = self.PaliGemma.llm(
            full_tokens,
            mask=combined_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond]
        )
        
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        
        # 5. Combine losses
        total_loss = action_loss + self.ki_fast_loss_weight * fast_loss
        return total_loss
    else:
        # Standard training (no KI)
        # ... existing code ...
```

### 4. Training Configuration (`training/config.py`)

**File**: `openpi/src/openpi/training/config.py`

**Changes needed**:
```python
# Add new config for KI training
pi05_droid_ki = TrainConfig(
    model_config=pi0_config.Pi0Config(
        pi05=True,
        knowledge_insulation=True,      # Enable KI
        ki_fast_loss_weight=1.0,        # Balance FAST vs action loss
        # ... other DROID settings ...
    ),
    # ... rest of training config ...
)
```

## Key Questions to Answer

1. **FAST Tokenizer Parameters**:
   - What vocab size? (typically 256-1024)
   - How many bins per action dimension? (typically 256)
   - Should we use the existing `FASTTokenizer` or create a new one?

2. **Token Sequence Structure**:
   - Should FAST tokens come before or after the text prompt?
   - Currently: `[images] + [prompt] + [FAST tokens]`
   - Alternative: `[images] + [FAST tokens]` (no prompt during KI)

3. **Loss Weighting**:
   - What weight for FAST loss vs action loss?
   - Default 1.0 means equal weight

## Testing Strategy

1. **Step 1**: Verify FAST tokenization works
   ```python
   # Test script
   actions = jnp.array([...])  # Sample action chunk
   fast_tokens = tokenizer.encode(actions)
   reconstructed = tokenizer.decode(fast_tokens)
   assert jnp.allclose(actions, reconstructed, atol=0.01)
   ```

2. **Step 2**: Verify data pipeline produces correct fields
   ```python
   # Check observation has required fields
   assert hasattr(obs, 'tokenized_prompt')
   assert hasattr(obs, 'token_loss_mask')
   assert obs.tokenized_prompt.shape[-1] == expected_length
   ```

3. **Step 3**: Verify losses compute correctly
   ```python
   # Run one training step
   loss = model.compute_loss(rng, obs, actions, train=True)
   # Should see both fast_loss and action_loss in logs
   ```

## Implementation Order

1. ✅ **Read and understand** this document
2. ⏹️ **Decide** on FAST tokenizer parameters
3. ⏹️ **Implement** FAST tokenization in data pipeline
4. ⏹️ **Add** KI config parameters
5. ⏹️ **Implement** dual-loss compute_loss function
6. ⏹️ **Test** each component individually
7. ⏹️ **Run** full KI training

## Next Steps

**What I need from you:**
1. Should I proceed with implementing these changes?
2. Do you want me to create the FAST tokenizer first, or modify existing files?
3. Any preferences on token sequence structure or loss weights?

Once you confirm, I can implement everything in the order above.
