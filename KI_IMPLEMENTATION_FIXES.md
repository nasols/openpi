# Knowledge Insulation Implementation Fixes

## Executive Summary

The current KI implementation has a **critical distribution mismatch** between training and inference that breaks the pipeline. During training, the model sees sequences with FAST tokens, but during inference, it doesn't. This document outlines a comprehensive fix.

## Problems Identified

### 1. **Critical: Input Distribution Mismatch Between Training and Inference**

**Current Training Flow:**
```python
# Data pipeline: TokenizeFASTInputs
observation.tokenized_prompt = [BOS, "Task:", ..., "State:", ..., "Action:", <FAST_TOKENS>, "|", EOS]
observation.token_loss_mask = [False, ..., False, True, True, ..., True, False]

# Model forward pass
prefix_tokens = embed_prefix(observation)  # Includes FAST tokens
# VLM processes: [image_tokens] + [text_tokens] + [FAST_tokens]
```

**Current Inference Flow:**
```python
# Data pipeline: TokenizePrompt (NOT TokenizeFASTInputs)
observation.tokenized_prompt = [BOS, "Task:", ..., EOS]  # NO FAST tokens!

# Model forward pass
prefix_tokens = embed_prefix(observation)  # Does NOT include FAST tokens
# VLM processes: [image_tokens] + [text_tokens]  # DIFFERENT from training!
```

**Impact**: The model trains on longer sequences including FAST tokens, but at inference time sees shorter sequences. This causes:
- Position embedding misalignment
- Attention pattern differences
- Degraded performance due to train/test mismatch

### 2. **Confusion About FAST Token Usage**

The current code suggests FAST tokens should be in the prefix during training, but:
- During inference, they're never generated
- The action expert uses VLM embeddings that may have seen different context lengths
- It's unclear if this follows the KI paper's intent

### 3. **No Way to Test FAST Token Prediction**

Can't verify if the VLM is actually learning to predict FAST tokens correctly without manual debugging.

---

## Recommended Solution: Clean Separation Architecture

### Core Principle

**Keep the VLM prefix consistent between training and inference.** Only compute FAST token loss separately without mixing FAST tokens into the main prefix.

### Architecture Changes

```
TRAINING:
┌─────────────────────────────────────────────────────────────┐
│ Prefix (Consistent with Inference)                          │
│ [image_tokens] + [text_prompt_tokens]                       │
└─────────────────────────────────────────────────────────────┘
                    ↓
            VLM Forward Pass
                    ↓
        ┌───────────────────────┐
        │   Prefix Embeddings   │
        └───────────────────────┘
                    ↓
            ┌───────┴───────┐
            ↓               ↓
    ┌─────────────┐   ┌─────────────────┐
    │ FAST Token  │   │ Action Expert   │
    │ Prediction  │   │ (Flow Matching) │
    │ (Separate)  │   │                 │
    └─────────────┘   └─────────────────┘
         ↓                     ↓
    FAST Loss          Action Loss
         ↓                     ↓
         └──────────┬──────────┘
                    ↓
            Combined Loss

INFERENCE:
┌─────────────────────────────────────────────────────────────┐
│ Prefix (Same as Training)                                   │
│ [image_tokens] + [text_prompt_tokens]                       │
└─────────────────────────────────────────────────────────────┘
                    ↓
            VLM Forward Pass
                    ↓
        ┌───────────────────────┐
        │   Prefix Embeddings   │
        └───────────────────────┘
                    ↓
        ┌─────────────────────┐
        │   Action Expert     │
        │   (Flow Matching)   │
        │   ONLY              │
        └─────────────────────┘
                    ↓
            Continuous Actions
```

---

## Implementation Changes

### Change 1: Modify `embed_prefix` to Handle FAST Token Sequence Separately

**File**: `src/openpi/models/pi0.py`

**Current Code** (Lines ~108-142):
```python
def embed_prefix(
    self, obs: _model.Observation
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    input_mask = []
    ar_mask = []
    tokens = []
    # embed images
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(...)
        ar_mask += [False] * image_tokens.shape[1]

    # add language (aka tokenized inputs)
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]
    
    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask
```

**Problem**: This embeds `tokenized_prompt` which contains different content during training (with FAST tokens) vs inference (without).

**New Code**:
```python
def embed_prefix(
    self, obs: _model.Observation
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    """Embed the consistent prefix: images + text prompt (NO FAST tokens)."""
    input_mask = []
    ar_mask = []
    tokens = []
    
    # embed images
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
        )
        ar_mask += [False] * image_tokens.shape[1]

    # add language prompt ONLY (not FAST tokens)
    # For KI, we use a separate field for just the text prompt
    if self.knowledge_insulation and obs.text_prompt is not None:
        # Use text_prompt field which contains ONLY the text instruction
        tokenized_inputs = self.PaliGemma.llm(obs.text_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.text_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]
    elif obs.tokenized_prompt is not None:
        # Standard path for non-KI models
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]
    
    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask
```

### Change 2: Add FAST Token Sequence Processing Method

**File**: `src/openpi/models/pi0.py`

**New Method** (Add after `embed_prefix`):
```python
@at.typecheck
def embed_fast_sequence(
    self, obs: _model.Observation
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    """Embed the FAST token sequence for autoregressive prediction during training."""
    # This embeds the full sequence: [BOS, Task:, ..., State:, ..., Action:, <FAST>, |, EOS]
    # Used ONLY during training to compute FAST token loss
    if obs.tokenized_prompt is None:
        raise ValueError("tokenized_prompt required for FAST token prediction")
    
    tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
    input_mask = obs.tokenized_prompt_mask
    # Causal attention for FAST token sequence
    ar_mask = jnp.ones_like(obs.token_ar_mask) if obs.token_ar_mask is not None else jnp.ones(tokenized_inputs.shape[1])
    
    return tokenized_inputs, input_mask, ar_mask
```

### Change 3: Update `compute_loss` to Use Separate Embeddings

**File**: `src/openpi/models/pi0.py`

**Current Code** (Lines ~221-275):
```python
elif self.knowledge_insulation and self.pi05 and train: 
    # We assume the input observation now contains the ground truth FAST tokens aswell. 
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_position = jnp.cumsum(prefix_mask, axis=1) - 1
    
    # 1. VLM forward pass to predict FAST tokens. 
    (prefix_out_FAST, _), _ = self.PaliGemma.llm(
        [prefix_tokens, None], 
        mask=prefix_attn_mask, 
        positions=prefix_position, 
        adarms_cond=[None, None]
    )
    if observation.tokenized_prompt is not None and observation.token_loss_mask is not None: 
        fast_token_targets = jax.nn.one_hot(...)
        FAST_logits = self.PaliGemma.llm(pre_logits=prefix_out_FAST[:, :-1][0])
        FAST_logp = jax.nn.log_softmax(FAST_logits, axis=-1)
        FAST_loss_mask = observation.token_loss_mask[:, 1:]
        FAST_token_pplx = jnp.sum(fast_token_targets * FAST_logp, axis=-1)
        FAST_loss = -jnp.sum(FAST_token_pplx * FAST_loss_mask, axis=-1) / jnp.clip(jnp.sum(FAST_loss_mask, axis=-1) + 1e-8, 1)
        FAST_loss = jnp.mean(FAST_loss)
    else:
        FAST_loss = 0.0

    # 2. Action Expert flow-matching forward pass.
    prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (_, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens_detached, suffix_tokens], 
        mask=attn_mask, 
        positions=positions,
        adarms_cond=[None, adarms_cond]
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])   
    action_loss = jnp.mean(jnp.square(v_t-u_t), axis=-1)

    total_loss = action_loss + self.ki_fast_token_loss_weight * FAST_loss
    return total_loss
```

**New Code**:
```python
elif self.knowledge_insulation and self.pi05 and train: 
    # Knowledge Insulation: Train VLM and Action Expert separately
    # VLM learns FAST tokens, Action Expert learns continuous actions
    # Gradients from Action Expert DO NOT flow to VLM
    
    # ===== PART 1: VLM Forward Pass (FAST Token Prediction) =====
    # Process images + text prompt (consistent with inference)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    
    # Compute FAST token loss using separate sequence
    # This sequence includes: [images] + [Task: ..., State: ..., Action: <FAST_TOKENS> |]
    if observation.tokenized_prompt is not None and observation.token_loss_mask is not None:
        fast_seq_tokens, fast_seq_mask, fast_seq_ar_mask = self.embed_fast_sequence(observation)
        
        # Concatenate image embeddings with FAST sequence
        # Images use bidirectional attention, FAST sequence uses causal attention
        combined_tokens = jnp.concatenate([prefix_tokens[:, :sum(obs.image_masks.values())], fast_seq_tokens], axis=1)
        combined_mask = jnp.concatenate([prefix_mask[:, :sum(obs.image_masks.values())], fast_seq_mask], axis=1)
        # AR mask: bidirectional for images (False), causal for FAST sequence (True)
        num_image_tokens = sum([img.shape[1] for img in self.PaliGemma.img(obs.images.values())])
        combined_ar_mask = jnp.concatenate([
            jnp.zeros(num_image_tokens, dtype=jnp.int32),
            fast_seq_ar_mask
        ])
        
        combined_attn_mask = make_attn_mask(combined_mask, combined_ar_mask)
        combined_positions = jnp.cumsum(combined_mask, axis=1) - 1
        
        # Forward pass for FAST token prediction
        (fast_out, _), _ = self.PaliGemma.llm(
            [combined_tokens, None],
            mask=combined_attn_mask,
            positions=combined_positions,
            adarms_cond=[None, None]
        )
        
        # Compute cross-entropy loss on FAST tokens
        fast_token_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:], 
            self.PaliGemma.llm.module.vocab_size,
        )
        FAST_logits = self.PaliGemma.llm(pre_logits=fast_out[:, :-1])
        FAST_logp = jax.nn.log_softmax(FAST_logits, axis=-1)
        
        FAST_loss_mask = observation.token_loss_mask[:, 1:]
        FAST_token_pplx = jnp.sum(fast_token_targets * FAST_logp, axis=-1)
        FAST_loss = -jnp.sum(FAST_token_pplx * FAST_loss_mask, axis=-1) / jnp.clip(
            jnp.sum(FAST_loss_mask, axis=-1) + 1e-8, 1
        )
        FAST_loss = jnp.mean(FAST_loss)
    else:
        FAST_loss = 0.0

    # ===== PART 2: Action Expert Forward Pass (Flow Matching) =====
    # Use the SAME prefix as inference (images + text only, no FAST tokens)
    # This ensures consistency between training and inference
    prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)
    
    # Prepare suffix (noisy actions)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, time
    )
    
    # Concatenate prefix and suffix
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    # Forward pass for action prediction (gradients blocked from prefix)
    (_, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens_detached, suffix_tokens], 
        mask=attn_mask, 
        positions=positions,
        adarms_cond=[None, adarms_cond]
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])   
    action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # ===== PART 3: Combine Losses =====
    total_loss = action_loss + self.ki_fast_token_loss_weight * FAST_loss
    return total_loss
```

**Key Changes:**
1. Prefix tokens now consistent between training/inference (images + text only)
2. FAST tokens processed separately via `embed_fast_sequence`
3. VLM sees images during FAST token prediction (more context)
4. Action expert uses same prefix length as inference

### Change 4: Update Data Model to Include Separate Text Prompt Field

**File**: `src/openpi/models/model.py`

**Add to Observation dataclass**:
```python
@dataclasses.dataclass
class Observation:
    images: dict[str, at.UInt8[at.Array, "b h w c"]]
    image_masks: dict[str, at.Bool[at.Array, " b"]]
    state: at.Float[at.Array, "b state_dim"]
    
    # Standard tokenized prompt (may include FAST tokens for KI training)
    tokenized_prompt: at.Int[at.Array, "b token_len"] | None = None
    tokenized_prompt_mask: at.Bool[at.Array, "b token_len"] | None = None
    
    # NEW: Separate text-only prompt for KI (consistent between train/inference)
    text_prompt: at.Int[at.Array, "b text_len"] | None = None
    text_prompt_mask: at.Bool[at.Array, "b text_len"] | None = None
    
    token_ar_mask: at.Int[at.Array, "token_len"] | None = None
    token_loss_mask: at.Bool[at.Array, "b token_len"] | None = None
```

### Change 5: Update Tokenizer to Generate Both Fields

**File**: `src/openpi/models/tokenizer.py`

**Modify `FASTTokenizer.tokenize`**:
```python
def tokenize(
    self, prompt: str, state: np.ndarray, actions: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        - full_tokens: Complete sequence with FAST tokens (for FAST loss)
        - full_token_mask: Mask for full sequence
        - full_ar_mask: AR mask for full sequence
        - loss_mask: Which tokens to compute loss on
        - text_only_tokens: Just the text prompt (for consistent prefix)
        - text_only_mask: Mask for text-only sequence
    """
    cleaned_text = prompt.lower().strip().replace("_", " ")

    # Discretize state
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    
    # Text-only prefix (consistent between training and inference)
    text_prefix = f"Task: {cleaned_text}, State: {state_str};\n"
    text_only_tokens = self._paligemma_tokenizer.encode(text_prefix, add_bos=True)

    if actions is not None:
        # Full sequence with FAST tokens (for training FAST prediction)
        action_tokens = self._fast_tokenizer(actions[None])[0]
        action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

        postfix_tokens = (
            self._paligemma_tokenizer.encode("Action: ")
            + action_tokens_in_pg.tolist()
            + self._paligemma_tokenizer.encode("|", add_eos=True)
        )
        
        full_tokens = text_only_tokens + postfix_tokens
        full_token_mask = [True] * len(full_tokens)
        full_ar_mask = [0] * len(text_only_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(text_only_tokens) + [True] * len(postfix_tokens)
    else:
        # Inference: only text prefix
        full_tokens = text_only_tokens
        full_token_mask = [True] * len(full_tokens)
        full_ar_mask = [0] * len(full_tokens)
        loss_mask = [False] * len(full_tokens)

    # Pad sequences
    def pad_sequence(seq, pad_value):
        if len(seq) < self._max_len:
            return seq + [pad_value] * (self._max_len - len(seq))
        else:
            if len(seq) > self._max_len:
                logging.warning(f"Token length ({len(seq)}) exceeds max length ({self._max_len}), truncating.")
            return seq[:self._max_len]
    
    full_tokens = pad_sequence(full_tokens, False)
    full_token_mask = pad_sequence(full_token_mask, False)
    full_ar_mask = pad_sequence(full_ar_mask, False)
    loss_mask = pad_sequence(loss_mask, False)
    
    text_only_tokens = pad_sequence(text_only_tokens, False)
    text_only_mask = [True] * len(text_only_tokens) + [False] * (self._max_len - len(text_only_tokens))
    text_only_mask = text_only_mask[:self._max_len]

    return (
        np.asarray(full_tokens),      # For FAST loss computation
        np.asarray(full_token_mask),
        np.asarray(full_ar_mask),
        np.asarray(loss_mask),
        np.asarray(text_only_tokens),  # For consistent prefix
        np.asarray(text_only_mask)
    )
```

### Change 6: Update Transform to Populate Both Fields

**File**: `src/openpi/transforms.py`

**Modify `TokenizeFASTInputs`**:
```python
@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        (full_tokens, full_token_mask, ar_mask, loss_mask, 
         text_tokens, text_mask) = self.tokenizer.tokenize(prompt, state, actions)
        
        return {
            **data,
            # Full sequence with FAST tokens (for FAST loss during training)
            "tokenized_prompt": full_tokens,
            "tokenized_prompt_mask": full_token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            # Text-only sequence (for consistent prefix)
            "text_prompt": text_tokens,
            "text_prompt_mask": text_mask,
        }
```

### Change 7: Add FAST Token Inference Mode (For Testing)

**File**: `src/openpi/models/pi0.py`

**New Method** (Add after `sample_actions`):
```python
@override
def sample_fast_tokens(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    max_decoding_steps: int = 256,
) -> at.Int[at.Array, "b max_len"]:
    """
    Generate FAST tokens autoregressively (for debugging/testing).
    This allows you to verify the VLM is learning to predict FAST tokens.
    """
    if not self.knowledge_insulation:
        raise ValueError("FAST token sampling only available with knowledge_insulation=True")
    
    observation = _model.preprocess_observation(None, observation, train=False)
    
    # Use consistent prefix (images + text only)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    
    # Fill KV cache with prefix
    prefix_logits, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], 
        mask=prefix_attn_mask, 
        positions=positions
    )
    
    # Start autoregressive sampling
    # First token to sample: "Action:"
    batch_size = observation.state.shape[0]
    output_tokens = jnp.zeros((batch_size, max_decoding_steps), dtype=jnp.int32)
    
    # Sample token by token
    last_logit = prefix_logits[:, -1:]
    prefill_len = jnp.sum(prefix_mask, axis=-1)
    
    def step(carry):
        rng, last_logit, output_tokens, cache, step = carry
        
        # Sample next token
        rng, rng_step = jax.random.split(rng)
        token = jnp.argmax(last_logit, axis=-1)  # Greedy sampling
        output_tokens = output_tokens.at[:, step].set(token[:, 0])
        
        # Check for EOS
        EOS_TOKEN = 1  # PaliGemma EOS token
        has_eos = jnp.any(token == EOS_TOKEN, axis=-1)
        all_eos = jnp.all(has_eos)
        
        # Embed next token
        token_embedding = self.PaliGemma.llm(token, embed_only=True)
        positions = prefill_len[:, None] + step + 1
        
        # Generate next logit
        last_logit, kv_cache = self.PaliGemma.llm(
            embedded_prefix=token_embedding,
            positions=positions,
            decode=True,
            kv_cache=cache
        )
        
        return rng, last_logit, output_tokens, kv_cache, step + 1
    
    def cond(carry):
        _, _, _, _, step = carry
        return step < max_decoding_steps
    
    _, _, output_tokens, _, _ = jax.lax.while_loop(
        cond, step, (rng, last_logit, output_tokens, kv_cache, 0)
    )
    
    return output_tokens
```

---

## Testing Plan

### Test 1: Verify Prefix Consistency

**File**: `src/openpi/tests/ki_prefix_consistency_test.py`

```python
def test_prefix_consistency():
    """Verify that prefix embeddings have same shape in training and inference."""
    config = pi0_config.Pi0Config(
        action_horizon=16,
        pi05=True,
        knowledge_insulation=True,
    )
    model = config.create(jax.random.key(0))
    
    # Create observation for training (with FAST tokens)
    train_obs = create_training_observation()
    train_prefix, train_mask, _ = model.embed_prefix(train_obs)
    
    # Create observation for inference (without FAST tokens)
    infer_obs = create_inference_observation()
    infer_prefix, infer_mask, _ = model.embed_prefix(infer_obs)
    
    # They should have the same sequence length
    assert train_prefix.shape == infer_prefix.shape, \
        f"Prefix shape mismatch: train={train_prefix.shape}, infer={infer_prefix.shape}"
    
    print("✓ Prefix consistency verified")
```

### Test 2: Verify FAST Token Prediction

**File**: `src/openpi/tests/ki_fast_prediction_test.py`

```python
def test_fast_token_prediction():
    """Test that FAST tokens can be generated during inference."""
    config = pi0_config.Pi0Config(
        action_horizon=16,
        pi05=True,
        knowledge_insulation=True,
    )
    model = config.create(jax.random.key(0))
    
    # Create observation
    obs = create_inference_observation()
    
    # Generate FAST tokens
    fast_tokens = model.sample_fast_tokens(jax.random.key(0), obs)
    
    assert fast_tokens.shape[0] == obs.state.shape[0]
    assert fast_tokens.shape[1] > 0
    
    # Decode to actions (optional)
    from openpi.models.tokenizer import FASTTokenizer
    tokenizer = FASTTokenizer()
    actions = tokenizer.extract_actions(
        fast_tokens[0], 
        action_horizon=16, 
        action_dim=8
    )
    
    print(f"✓ FAST tokens generated: {fast_tokens.shape}")
    print(f"✓ Decoded actions: {actions.shape}")
```

### Test 3: End-to-End Training Test

**File**: `src/openpi/tests/ki_e2e_test.py`

```python
def test_ki_training_end_to_end():
    """Test full training loop with KI."""
    config = pi0_config.Pi0Config(
        action_horizon=4,
        pi05=True,
        knowledge_insulation=True,
        ki_fast_token_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    
    # Create training batch
    batch_obs, batch_actions = create_training_batch()
    
    # Compute loss
    loss = model.compute_loss(
        jax.random.key(0),
        batch_obs,
        batch_actions,
        train=True
    )
    
    assert loss.shape == (batch_size, action_horizon)
    assert jnp.all(jnp.isfinite(loss))
    
    # Test inference with same model
    infer_obs = create_inference_observation()
    actions = model.sample_actions(jax.random.key(0), infer_obs)
    
    assert actions.shape == (1, action_horizon, action_dim)
    assert jnp.all(jnp.isfinite(actions))
    
    print("✓ End-to-end KI training test passed")
```

---

## Migration Guide

### Step 1: Update Model Code
1. Apply changes to `src/openpi/models/pi0.py`:
   - Modify `embed_prefix` to check for `text_prompt` field
   - Add `embed_fast_sequence` method
   - Rewrite KI training branch in `compute_loss`
   - Add `sample_fast_tokens` method (optional, for testing)

### Step 2: Update Data Model
1. Add `text_prompt` and `text_prompt_mask` fields to `Observation` in `src/openpi/models/model.py`

### Step 3: Update Tokenizer
1. Modify `FASTTokenizer.tokenize` to return 6 values instead of 4
2. Update return statement to include text-only tokens

### Step 4: Update Transforms
1. Modify `TokenizeFASTInputs` to populate both `tokenized_prompt` and `text_prompt` fields

### Step 5: Test
1. Run `src/openpi/tests/ki_prefix_consistency_test.py`
2. Run `src/openpi/tests/ki_fast_prediction_test.py`
3. Run `src/openpi/tests/ki_e2e_test.py`
4. Run existing inference test: `python src/openpi/tests/inference_test.py`

### Step 6: Retrain (Optional)
If you have existing KI checkpoints, they may need retraining with the new architecture for optimal performance.

---

## Benefits of This Approach

1. **✅ Eliminates Distribution Mismatch**: Training and inference see the same prefix structure
2. **✅ Cleaner Separation**: FAST token prediction is clearly separate from action expert
3. **✅ Testable**: Can verify FAST token learning independently
4. **✅ Follows KI Paper**: VLM and action expert are truly separated
5. **✅ Backward Compatible**: Non-KI models work unchanged
6. **✅ Maintainable**: Clear code structure and intent

---

## FAQ

**Q: Do we lose information by not including FAST tokens in the action expert's prefix?**

A: No! The whole point of knowledge insulation is that the action expert should NOT see the FAST tokens. It learns to predict actions using only the VLM embeddings of images + text, which is exactly what it will see during inference.

**Q: Should we predict FAST tokens during inference?**

A: Generally no, unless you're debugging. The action expert (flow matching) is faster and produces continuous actions directly. However, you CAN predict FAST tokens using `sample_fast_tokens()` to verify the VLM is learning correctly.

**Q: Will this require retraining existing checkpoints?**

A: Yes, if you have KI checkpoints trained with the old approach. However, standard (non-KI) checkpoints are unaffected.

**Q: What if I want to use FAST tokens during inference?**

A: You can! Use `sample_fast_tokens()` to generate them, then decode to actions. This effectively turns your KI model into a pure FAST model for that inference run. This could be useful for comparing the two prediction methods.

---

## Conclusion

These changes fix the critical distribution mismatch in the KI implementation while making the code cleaner and more maintainable. The key insight is to **keep the prefix consistent** between training and inference, and process FAST tokens separately for their specific training objective.
