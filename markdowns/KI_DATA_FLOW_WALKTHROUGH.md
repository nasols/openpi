# Knowledge Insulation (KI) Data Flow: Comprehensive Walkthrough

This document provides a detailed trace through the π0.5 Knowledge Insulation pipeline during both **inference** and **training**. We'll follow the actual data as it flows through the system, examining shapes, transformations, and key decisions at each step.

---

## Table of Contents
1. [Inference Flow](#inference-flow)
2. [Training Flow](#training-flow)
3. [Data Structure Comparison](#data-structure-comparison)
4. [Critical Issues Identified](#critical-issues-identified)

---

## Inference Flow

### Overview
During inference, we want to predict continuous actions given images, robot state, and a text prompt.

```
Raw Input (dict) 
  ↓
Data Transforms (DroidInputs)
  ↓
Model Transforms (TokenizeFASTInputs)  ← ISSUE: Uses FAST tokenizer!
  ↓
Model.sample_actions()
  ↓
Continuous Actions
```

### Step-by-Step Walkthrough

#### **Step 1: Raw Input Data**

**File**: User/Environment provides this
**Code**: `src/openpi/tests/inference_test.py`

```python
data = make_droid_example()
# Returns:
{
    "observation/exterior_image_1_left": np.ndarray,  # (224, 224, 3) uint8
    "observation/wrist_image_left": np.ndarray,       # (224, 224, 3) uint8
    "observation/joint_position": np.ndarray,         # (7,) float
    "observation/gripper_position": np.ndarray,       # (1,) float
    "prompt": "do something",                         # str
}
```

**Data shapes at this point:**
- `exterior_image_1_left`: `(224, 224, 3)` uint8
- `wrist_image_left`: `(224, 224, 3)` uint8
- `joint_position`: `(7,)` float32
- `gripper_position`: `(1,)` float32
- `prompt`: str

---

#### **Step 2: Data Transform (DroidInputs)**

**File**: `src/openpi/policies/droid_policy.py`
**Class**: `DroidInputs.__call__()`

```python
def __call__(self, data: dict) -> dict:
    # Concatenate joint and gripper positions
    state = np.concatenate([
        data["observation/joint_position"],  # (7,)
        data["observation/gripper_position"]  # (1,)
    ])  # Result: (8,)
    
    # Parse images
    base_image = _parse_image(data["observation/exterior_image_1_left"])
    wrist_image = _parse_image(data["observation/wrist_image_left"])
    
    # For PI05_KI model type:
    match self.model_type:
        case ModelType.PI05_KI:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, np.zeros_like(base_image))
            image_masks = (np.True_, np.True_, np.False_)
    
    return {
        "state": state,                    # (8,)
        "image": {
            "base_0_rgb": base_image,      # (224, 224, 3)
            "left_wrist_0_rgb": wrist_image,  # (224, 224, 3)
            "right_wrist_0_rgb": np.zeros_like(base_image),  # (224, 224, 3)
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
        "prompt": "do something",
    }
```

**Data after DroidInputs:**
- `state`: `(8,)` float32 - concatenated joint + gripper positions
- `image`: Dict with 3 images, each `(224, 224, 3)` uint8
- `image_mask`: Dict with 3 boolean masks
- `prompt`: str

---

#### **Step 3: Model Transforms - InjectDefaultPrompt**

**File**: `src/openpi/transforms.py`
**Class**: `InjectDefaultPrompt.__call__()`

```python
def __call__(self, data: DataDict) -> DataDict:
    if "prompt" not in data and self.default_prompt is not None:
        data["prompt"] = self.default_prompt
    return data
```

**Data after InjectDefaultPrompt:**
- Same as before (prompt already exists)

---

#### **Step 4: Model Transforms - ResizeImages**

**File**: `src/openpi/transforms.py`
**Class**: `ResizeImages.__call__()`

```python
def __call__(self, data: DataDict) -> DataDict:
    images = data.get("image")
    for name in images:
        img = images[name]
        # Resize to (224, 224) if needed
        images[name] = resize(img, (self.height, self.width))
    return {**data, "image": images}
```

**Data after ResizeImages:**
- Images still `(224, 224, 3)` (already correct size)

---

#### **Step 5: Model Transforms - TokenizeFASTInputs** ⚠️ **CRITICAL STEP**

**File**: `src/openpi/transforms.py`
**Class**: `TokenizeFASTInputs.__call__()`

```python
def __call__(self, data: DataDict) -> DataDict:
    if (prompt := data.pop("prompt", None)) is None:
        raise ValueError("Prompt is required")
    
    if not isinstance(prompt, str):
        prompt = prompt.item()
    
    state, actions = data["state"], data.get("actions")  # actions=None for inference
    
    # Call FASTTokenizer
    tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(
        prompt="do something", 
        state=state,           # (8,)
        actions=None           # No actions during inference!
    )
    
    return {
        **data,
        "tokenized_prompt": tokens,        # (256,) int32
        "tokenized_prompt_mask": token_mask,  # (256,) bool
        "token_ar_mask": ar_mask,          # (256,) int32
        "token_loss_mask": loss_mask,      # (256,) bool
    }
```

Now let's trace into the tokenizer:

**File**: `src/openpi/models/tokenizer.py`
**Class**: `FASTTokenizer.tokenize()`

```python
def tokenize(
    self, prompt: str, state: np.ndarray, actions: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    cleaned_text = "do something"  # After cleaning
    
    # Discretize state to 256 bins
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    # Result: array of 8 integers in range [0, 255]
    
    state_str = "128 145 92 201 67 189 234 12"  # Example
    
    # Build prefix
    prefix = "Task: do something, State: 128 145 92 201 67 189 234 12;\n"
    prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
    # Result: [2, 8061, 27, ..., 27866, 932]  (example, ~30 tokens)
    
    # INFERENCE: actions is None!
    if actions is not None:
        # This branch is SKIPPED during inference
        ...
    else:
        postfix_tokens = []  # Empty!
    
    # Combine
    tokens = prefix_tokens + postfix_tokens  # Just prefix_tokens
    token_mask = [True] * len(tokens)
    ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
    # ar_mask = [0, 0, 0, ..., 0]  (all bidirectional)
    loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
    # loss_mask = [False, False, ..., False]  (no loss computed)
    
    # Pad to max_len=256
    tokens_len = 30  # Example
    padding = [False] * (256 - 30)
    tokens = tokens + padding  # [token, token, ..., token, False, False, ...]
    token_mask = token_mask + padding
    ar_mask = ar_mask + padding
    loss_mask = loss_mask + padding
    
    return (
        np.asarray(tokens),      # (256,) - mostly False padding
        np.asarray(token_mask),  # (256,) - True for first ~30, False after
        np.asarray(ar_mask),     # (256,) - all 0 (bidirectional)
        np.asarray(loss_mask)    # (256,) - all False
    )
```

**Data after TokenizeFASTInputs:**
- `state`: `(8,)`
- `image`: Dict with 3 images
- `image_mask`: Dict with 3 bools
- `tokenized_prompt`: `(256,)` int32 - tokens for "Task: do something, State: ..."
- `tokenized_prompt_mask`: `(256,)` bool - True for real tokens, False for padding
- `token_ar_mask`: `(256,)` int32 - all 0 (bidirectional attention)
- `token_loss_mask`: `(256,)` bool - all False (no loss during inference)

**Key Point**: No FAST action tokens are included during inference!

---

#### **Step 6: Model Transforms - PadStatesAndActions**

**File**: `src/openpi/transforms.py`
**Class**: `PadStatesAndActions.__call__()`

```python
def __call__(self, data: DataDict) -> DataDict:
    state = data["state"]  # (8,)
    # Pad to action_dim if needed
    if state.shape[-1] < self.action_dim:
        padding = np.zeros(self.action_dim - state.shape[-1])
        state = np.concatenate([state, padding])
    
    return {**data, "state": state}
```

**Data after PadStatesAndActions:**
- `state`: `(8,)` or padded to match `action_dim` if configured differently

---

#### **Step 7: Policy Wrapping - Add Batch Dimension**

**File**: `src/openpi/policies/policy.py`
**Method**: `Policy.infer()`

```python
def infer(self, data: dict) -> dict:
    # Apply all input transforms
    for transform in self.config.data_transforms.inputs:
        data = transform(data)
    
    for transform in self.config.model_transforms.inputs:
        data = transform(data)
    
    # Add batch dimension
    data = jax.tree.map(lambda x: x[None] if isinstance(x, np.ndarray) else x, data)
    
    # Now all arrays have shape (1, ...)
    # Create Observation object
    observation = Observation(
        images={
            name: jnp.asarray(data["image"][name])  # (1, 224, 224, 3)
            for name in data["image"]
        },
        image_masks={
            name: jnp.asarray(data["image_mask"][name])  # (1,)
            for name in data["image_mask"]
        },
        state=jnp.asarray(data["state"]),  # (1, 8)
        tokenized_prompt=jnp.asarray(data["tokenized_prompt"]),  # (1, 256)
        tokenized_prompt_mask=jnp.asarray(data["tokenized_prompt_mask"]),  # (1, 256)
        token_ar_mask=jnp.asarray(data["token_ar_mask"]),  # (256,) - no batch dim!
        token_loss_mask=jnp.asarray(data.get("token_loss_mask")),  # (1, 256)
    )
    
    # Call model
    actions = self.model.sample_actions(self.rng, observation, num_steps=10)
    
    return {"actions": np.asarray(actions[0])}  # Remove batch dim
```

**Data passed to model:**
- `observation.images`: 3 images, each `(1, 224, 224, 3)` uint8
- `observation.image_masks`: 3 bools, each `(1,)`
- `observation.state`: `(1, 8)` float32
- `observation.tokenized_prompt`: `(1, 256)` int32
- `observation.tokenized_prompt_mask`: `(1, 256)` bool
- `observation.token_ar_mask`: `(256,)` int32 (no batch dimension!)
- `observation.token_loss_mask`: `(1, 256)` bool

---

#### **Step 8: Model Inference - sample_actions()**

**File**: `src/openpi/models/pi0.py`
**Method**: `Pi0.sample_actions()`

```python
def sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> _model.Actions:
    # Preprocess observation
    observation = _model.preprocess_observation(None, observation, train=False)
    # This normalizes images, applies norm stats to state, etc.
    
    batch_size = observation.state.shape[0]  # 1
    action_horizon = self.action_horizon  # 50
    action_dim = self.action_dim  # 8
    
    # Initialize noise
    if noise is None:
        noise = jax.random.normal(rng, (1, 50, 8))  # (1, 50, 8)
    
    # ===== EMBED PREFIX =====
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
```

Let's trace `embed_prefix`:

**File**: `src/openpi/models/pi0.py`
**Method**: `Pi0.embed_prefix()`

```python
def embed_prefix(
    self, obs: _model.Observation
) -> tuple[..., ..., ...]:
    input_mask = []
    ar_mask = []
    tokens = []
    
    # Embed images
    for name in obs.images:  # 3 images
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        # Each image → (1, 256) token embeddings
        # SigLIP encoder: (1, 224, 224, 3) → (1, 256, 1152)
        
        tokens.append(image_tokens)  # (1, 256, 1152)
        input_mask.append(
            einops.repeat(obs.image_masks[name], "b -> b s", s=256)
        )  # (1, 256)
        ar_mask += [False] * 256  # Bidirectional for images
    
    # After processing 3 images:
    # tokens: List of 3 tensors, each (1, 256, 1152)
    # input_mask: List of 3 tensors: [(1, 256), (1, 256), (1, 256)]
    #             But mask values: [True×256, True×256, False×256]
    # ar_mask: [False] * 768
    
    # Add language (tokenized prompt)
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        # (1, 256) token IDs → (1, 256, 2048) embeddings (Gemma embedding layer)
        
        tokens.append(tokenized_inputs)  # (1, 256, 2048)
        input_mask.append(obs.tokenized_prompt_mask)  # (1, 256)
        ar_mask += [False] * 256  # Bidirectional for text
    
    # Concatenate
    tokens = jnp.concatenate(tokens, axis=1)
    # (1, 768, 1152) + (1, 256, 2048) → Need to project!
    # Wait, this won't work - different embedding dims!
    
    # Actually, let me check the actual code more carefully...
    # The image tokens go through projection in PaliGemma to match LLM dim
    # So all tokens should be (1, seq_len, 2048)
    
    # After projection:
    # tokens: (1, 1024, 2048) - 768 image tokens + 256 text tokens
    # input_mask: (1, 1024) - [True×512, False×256, True×256]
    # ar_mask: (1024,) - all False (bidirectional)
    
    return tokens, input_mask, ar_mask
```

**Back to sample_actions:**

```python
    # prefix_tokens: (1, 1024, 2048)
    # prefix_mask: (1, 1024)
    # prefix_ar_mask: (1024,)
    
    # Create attention mask
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    # (1, 1024, 1024) - full bidirectional attention
    
    # Calculate positions
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    # (1, 1024) - [0, 1, 2, ..., 511, 0, 0, ..., 0, 512, 513, ...]
    
    # Fill KV cache with prefix
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], 
        mask=prefix_attn_mask, 
        positions=positions
    )
    # kv_cache now contains cached key-value pairs for the prefix
    # This allows efficient inference without recomputing prefix each step
    
    # ===== FLOW MATCHING LOOP =====
    dt = -1.0 / num_steps  # -0.1
    x_t = noise  # (1, 50, 8) - start from pure noise
    time = 1.0  # Start at t=1 (pure noise)
    
    def step(carry):
        x_t, time = carry  # x_t: (1, 50, 8), time: scalar
        
        # Embed suffix (noisy actions + time)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, 1)
        )
        # suffix_tokens: (1, 50, 2048) - action tokens
        # suffix_mask: (1, 50) - all True
        # suffix_ar_mask: (50,) - all True (causal)
        # adarms_cond: (1, 2048) - time conditioning for AdaRMS
        
        # Create attention masks
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # (1, 50, 50) - causal mask
        
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=50)
        # (1, 50, 1024) - suffix can attend to all prefix
        
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        # (1, 50, 1074) - suffix attends to prefix + itself (causally)
        
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        # (1, 50) - [768, 769, 770, ..., 817]
        
        # Forward pass using cached prefix
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # Only process suffix, use cached prefix
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        # suffix_out: (1, 50, 2048)
        
        # Project to velocity prediction
        v_t = self.action_out_proj(suffix_out[:, -50:])
        # (1, 50, 8) - predicted velocity
        
        # Euler step
        x_t = x_t + dt * v_t  # (1, 50, 8)
        time = time + dt
        
        return x_t, time
    
    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2  # Continue until t ≈ 0
    
    # Run flow matching for num_steps iterations
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    # x_0: (1, 50, 8) - denoised actions
    
    return x_0  # (1, 50, 8)
```

**Final output:**
- Actions: `(1, 50, 8)` - continuous action predictions for 50-step horizon

---

#### **Step 9: Return to User**

**File**: `src/openpi/policies/policy.py`

```python
def infer(self, data: dict) -> dict:
    ...
    actions = self.model.sample_actions(self.rng, observation, num_steps=10)
    # actions: (1, 50, 8)
    
    return {"actions": np.asarray(actions[0])}  # (50, 8)
```

**Final output to user:**
- `{"actions": np.ndarray}` - shape `(50, 8)` - 50-step action horizon

---

### Inference Summary

**Key observations during inference:**
1. ✅ No FAST tokens are generated or used
2. ✅ Tokenized prompt only contains: "Task: ..., State: ...\n"
3. ✅ Action expert (flow matching) does all the prediction
4. ❌ **BUT**: VLM was trained with FAST tokens in the prefix!
5. ❌ **Issue**: Sequence lengths differ between training and inference

**Inference data flow:**
```
Raw dict → DroidInputs → ResizeImages → TokenizeFASTInputs (no actions!) 
→ PadStatesAndActions → Observation object → embed_prefix() 
→ [images: 768 tokens] + [text: ~30 tokens] → KV cache 
→ Flow matching loop (10 steps) → Continuous actions (50, 8)
```

---

## Training Flow

### Overview
During training, we want to train both the VLM (to predict FAST tokens) and the action expert (to predict continuous actions) simultaneously without gradient flow between them.

```
Raw Dataset Sample
  ↓
Data Transforms (DroidInputs)
  ↓
Model Transforms (TokenizeFASTInputs)  ← Now includes FAST tokens!
  ↓
Model.compute_loss() with train=True
  ↓
FAST Loss + Action Loss
```

### Step-by-Step Walkthrough

#### **Step 1: Raw Training Data**

**File**: LeRobot dataset or custom dataset
**Code**: `src/openpi/training/train_pytorch.py` or similar

```python
# Dataset returns:
{
    "observation/exterior_image_1_left": np.ndarray,  # (224, 224, 3)
    "observation/wrist_image_left": np.ndarray,       # (224, 224, 3)
    "observation/joint_position": np.ndarray,         # (7,)
    "observation/gripper_position": np.ndarray,       # (1,)
    "actions": np.ndarray,                            # (50, 8) - action chunk!
    "prompt": "pick up the red block",
}
```

**Key difference from inference**: Now we have `actions`!

---

#### **Step 2-4: Same as Inference**

Data transforms (DroidInputs), InjectDefaultPrompt, ResizeImages work identically.

---

#### **Step 5: Model Transforms - TokenizeFASTInputs** ⚠️ **CRITICAL DIFFERENCE**

**File**: `src/openpi/transforms.py`

```python
def __call__(self, data: DataDict) -> DataDict:
    prompt = "pick up the red block"
    state = data["state"]  # (8,)
    actions = data.get("actions")  # (50, 8) - NOW PRESENT!
    
    tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(
        prompt, state, actions  # Actions passed in!
    )
    
    return {
        **data,
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": token_mask,
        "token_ar_mask": ar_mask,
        "token_loss_mask": loss_mask,
    }
```

**File**: `src/openpi/models/tokenizer.py`

```python
def tokenize(
    self, prompt: str, state: np.ndarray, actions: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    cleaned_text = "pick up the red block"
    
    # Discretize state
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = "128 145 92 201 67 189 234 12"
    
    # Build prefix
    prefix = "Task: pick up the red block, State: 128 145 92 201 67 189 234 12;\n"
    prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
    # Length: ~35 tokens
    
    # ===== KEY DIFFERENCE: ACTIONS ARE PRESENT =====
    if actions is not None:  # TRUE during training!
        # Tokenize actions with FAST tokenizer
        action_tokens = self._fast_tokenizer(actions[None])[0]
        # actions: (50, 8) → action_tokens: (150,) int array
        # FAST tokenizes each action dimension separately, ~3 tokens each
        # 50 timesteps × 8 dims × ~0.375 tokens/dim ≈ 150 tokens
        
        # Map FAST tokens to PaliGemma vocab
        action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
        # Remaps to last 128 tokens in PaliGemma vocab (reserved for FAST)
        
        # Build postfix
        postfix_tokens = (
            self._paligemma_tokenizer.encode("Action: ")  # ~2 tokens
            + action_tokens_in_pg.tolist()                # ~150 tokens
            + self._paligemma_tokenizer.encode("|", add_eos=True)  # ~2 tokens
        )
        # postfix_tokens length: ~154 tokens
    else:
        postfix_tokens = []
    
    # Combine sequences
    tokens = prefix_tokens + postfix_tokens
    # Total: ~35 + ~154 = ~189 tokens
    
    token_mask = [True] * len(tokens)  # All valid tokens
    
    # AR mask: prefix uses bidirectional (0), postfix uses causal (1)
    ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
    # [0, 0, ..., 0, 1, 1, ..., 1]
    #  ← 35 zeros →  ← 154 ones →
    
    # Loss mask: only compute loss on FAST action tokens
    loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
    # [False × 35] + [True × 154]
    
    # Pad to 256
    tokens_len = 189
    padding_len = 256 - 189 = 67
    tokens = tokens + [False] * padding_len
    token_mask = token_mask + [False] * padding_len
    ar_mask = ar_mask + [False] * padding_len
    loss_mask = loss_mask + [False] * padding_len
    
    return (
        np.asarray(tokens),      # (256,) - [prefix tokens, postfix tokens, padding]
        np.asarray(token_mask),  # (256,) - [True × 189, False × 67]
        np.asarray(ar_mask),     # (256,) - [0 × 35, 1 × 154, False × 67]
        np.asarray(loss_mask)    # (256,) - [False × 35, True × 154, False × 67]
    )
```

**Data after TokenizeFASTInputs (TRAINING):**
- `tokenized_prompt`: `(256,)` int32 - [Task text, State, "Action:", FAST tokens, "|", EOS, padding]
- `tokenized_prompt_mask`: `(256,)` bool - True for first ~189 tokens
- `token_ar_mask`: `(256,)` int32 - 0 for prefix, 1 for postfix
- `token_loss_mask`: `(256,)` bool - True only for FAST action tokens

**Critical observation**: The sequence is now ~189 tokens vs ~30 during inference!

---

#### **Step 6-7: Same Padding and Batching**

PadStatesAndActions and batch dimension addition work the same.

---

#### **Step 8: Model Training - compute_loss()**

**File**: `src/openpi/models/pi0.py`
**Method**: `Pi0.compute_loss()`

```python
def compute_loss(
    self, 
    rng: at.KeyArrayLike, 
    observation: _model.Observation, 
    actions: _model.Actions,  # (batch, 50, 8)
    *, 
    train: bool = False
) -> at.Float[at.Array, "*b ah"]:
    
    observation = _model.preprocess_observation(rng, observation, train=True)
    
    # Sample random timesteps for flow matching
    batch_size = observation.state.shape[0]
    time = jax.random.uniform(rng, (batch_size,))  # (batch,) in [0, 1]
    
    # Sample noise
    noise = jax.random.normal(rng, actions.shape)  # (batch, 50, 8)
    
    # Flow matching: interpolate between noise and actions
    time_expanded = time[..., None, None]  # (batch, 1, 1)
    x_t = time_expanded * noise + (1 - time_expanded) * actions  # (batch, 50, 8)
    u_t = noise - actions  # True velocity
    
    # Embed prefix and suffix (for action expert)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, time
    )
    
    # ===== KNOWLEDGE INSULATION PATH =====
    if not self.knowledge_insulation:
        # Standard training (PI0, PI05 without KI)
        ...
    
    elif self.knowledge_insulation and self.pi05 and train:
        # ===== PART 1: VLM FAST TOKEN PREDICTION =====
        
        # Embed prefix for FAST token prediction
        # ⚠️ ISSUE: This uses the same embed_prefix as inference!
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_position = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Forward pass to get embeddings
        (prefix_out_FAST, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], 
            mask=prefix_attn_mask, 
            positions=prefix_position, 
            adarms_cond=[None, None]
        )
        # prefix_out_FAST: (batch, seq_len, 2048)
        
        # ⚠️ MAJOR ISSUE HERE!
        # prefix_tokens came from embed_prefix, which embeds:
        #   [images: 768 tokens] + [tokenized_prompt: 256 tokens]
        # 
        # But tokenized_prompt during training contains:
        #   "Task: ..., State: ..., Action: <FAST_TOKENS> | <EOS>"
        # 
        # So prefix_tokens is actually:
        #   [images: 768] + [Task text: 35] + [FAST tokens: 154] + [padding: 67]
        # 
        # This means prefix_out_FAST has embeddings for the FAST tokens!
        
        # Compute FAST token loss
        if observation.tokenized_prompt is not None and observation.token_loss_mask is not None:
            # Create one-hot targets
            fast_token_targets = jax.nn.one_hot(
                observation.tokenized_prompt[:, 1:],  # Shift by 1 for next-token prediction
                self.PaliGemma.llm.module.vocab_size,
            )
            # Shape: (batch, 255, vocab_size)
            
            # Get logits from embeddings
            FAST_logits = self.PaliGemma.llm(pre_logits=prefix_out_FAST[:, :-1][0])
            # ⚠️ This indexing [0] seems wrong! Should be [:, :-1]
            # Assuming it's meant to be prefix_out_FAST[:, :-1]
            # Shape: (batch, 255, vocab_size)
            
            FAST_logp = jax.nn.log_softmax(FAST_logits, axis=-1)
            # Shape: (batch, 255, vocab_size)
            
            # Apply loss mask (only compute loss on FAST tokens)
            FAST_loss_mask = observation.token_loss_mask[:, 1:]
            # Shape: (batch, 255)
            # Values: [False × 35, True × 154, False × 66]
            
            # Compute per-token perplexity
            FAST_token_pplx = jnp.sum(fast_token_targets * FAST_logp, axis=-1)
            # Shape: (batch, 255) - log probability of correct token
            
            # Masked loss
            FAST_loss = -jnp.sum(FAST_token_pplx * FAST_loss_mask, axis=-1) / jnp.clip(
                jnp.sum(FAST_loss_mask, axis=-1) + 1e-8, 1
            )
            # Sum over sequence, normalize by number of FAST tokens (~154)
            # Shape: (batch,)
            
            FAST_loss = jnp.mean(FAST_loss)  # Scalar
        else:
            FAST_loss = 0.0
        
        # ===== PART 2: ACTION EXPERT FLOW MATCHING =====
        
        # Detach prefix to stop gradients
        prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)
        # ⚠️ ISSUE: prefix_tokens still contains FAST tokens!
        # During inference, it won't have them!
        
        # Concatenate prefix and suffix
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        # Forward pass (gradients blocked from prefix)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens_detached, suffix_tokens], 
            mask=attn_mask, 
            positions=positions,
            adarms_cond=[None, adarms_cond]
        )
        # suffix_out: (batch, 50, 2048)
        
        # Predict velocity
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        # (batch, 50, 8)
        
        # Flow matching loss
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        # (batch, 50) - MSE loss per timestep
        
        # ===== COMBINE LOSSES =====
        total_loss = action_loss + self.ki_fast_token_loss_weight * FAST_loss
        # action_loss: (batch, 50)
        # FAST_loss: scalar
        # total_loss: (batch, 50) - broadcasts FAST_loss
        
        return total_loss
```

**Loss computation:**
- `FAST_loss`: Scalar - average cross-entropy loss on FAST tokens
- `action_loss`: `(batch, 50)` - MSE loss for each action timestep
- `total_loss`: `(batch, 50)` - combined loss

---

### Training Summary

**Key observations during training:**
1. ✅ FAST tokens are included in `tokenized_prompt`
2. ✅ VLM learns to predict FAST tokens (supervised by ground truth)
3. ✅ Action expert learns flow matching (supervised by ground truth actions)
4. ✅ Gradients from action expert are stopped before VLM
5. ❌ **BUT**: Prefix includes FAST tokens (~1024 total length)
6. ❌ **Issue**: During inference, prefix only ~798 tokens (no FAST tokens)

**Training data flow:**
```
Dataset sample (with actions) → DroidInputs → TokenizeFASTInputs (encodes FAST tokens!)
→ Observation with tokenized_prompt containing FAST tokens
→ compute_loss():
    ├─ VLM forward → FAST token logits → Cross-entropy loss
    └─ Action expert forward (detached prefix) → Flow matching loss
→ Combined loss → Backprop
```

---

## Data Structure Comparison

### Tokenized Prompt Contents

| Context | Contents | Length | Example |
|---------|----------|--------|---------|
| **Inference** | `[BOS, Task:, ..., State:, ..., \n]` + padding | ~30 real + 226 padding | `[2, 8061, 27, 466, 2673, 518, 1810, ..., 932, 0, 0, ...]` |
| **Training** | `[BOS, Task:, ..., State:, ..., Action:, <FAST>, ..., \|, EOS]` + padding | ~189 real + 67 padding | `[2, 8061, 27, 466, 2673, 518, 1810, ..., 932, 4477, 27, 256872, 256857, ..., 891, 1, 0, ...]` |

### Prefix Embedding Lengths

| Context | Image Tokens | Text Tokens | FAST Tokens | Total Prefix Length |
|---------|-------------|-------------|-------------|-------------------|
| **Inference** | 768 (3×256) | ~30 | 0 | ~798 |
| **Training** | 768 (3×256) | ~35 | ~154 | ~957 |

### Attention Masks

| Context | Prefix Attention | Suffix Attention | Notes |
|---------|-----------------|------------------|-------|
| **Inference** | `(1, 798, 798)` bidirectional | `(1, 50, 50)` causal | Prefix all bidirectional |
| **Training** | `(1, 957, 957)` mixed | `(1, 50, 50)` causal | Prefix: images bidirectional, text bidirectional, FAST tokens causal |

---

## Critical Issues Identified

### Issue 1: Sequence Length Mismatch 🔴 **CRITICAL**

**Problem**: During training, the VLM sees sequences of length ~957 (including FAST tokens). During inference, it sees sequences of length ~798 (no FAST tokens).

**Impact**:
- Position embeddings are different
- Attention patterns are different
- The action expert trains on embeddings from longer sequences
- Distribution shift between training and inference

**Evidence**:
```python
# Training
tokenized_prompt = [...Task...State...Action:<FAST_TOKENS>|<EOS>...]  # 189 tokens
prefix_tokens = [images: 768] + [tokenized_prompt: 189] = 957 tokens

# Inference  
tokenized_prompt = [...Task...State...]  # 30 tokens
prefix_tokens = [images: 768] + [tokenized_prompt: 30] = 798 tokens
```

### Issue 2: Action Expert Sees Different Contexts 🔴 **CRITICAL**

**Problem**: During training, the action expert's prefix includes FAST tokens (even though gradients are stopped). During inference, FAST tokens are not present.

**Impact**:
- The action expert learns to attend to FAST token positions that don't exist at inference
- KV cache structure differs between training and inference
- May hurt action prediction performance

**Evidence**:
```python
# Training
prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)
# prefix_tokens still contains: [images, text, FAST_TOKENS]
# Action expert sees these positions but can't backprop through them

# Inference
prefix_tokens = [images, text]  # No FAST tokens
# Action expert sees different structure
```

### Issue 3: FAST Tokens in Prefix Break Separation 🟡 **DESIGN ISSUE**

**Problem**: The KI paper suggests VLM and action expert should be separate. But currently:
- VLM processes: images + text + FAST tokens
- Action expert processes: (images + text + FAST tokens) + noisy actions

This means the action expert *sees* the FAST token positions (even if gradients are stopped), which violates the separation principle.

**Better design**: VLM and action expert should process the same base prefix (images + text only), with FAST token prediction happening separately.

### Issue 4: Inconsistent AR Masks 🟡 **MINOR**

**Problem**: During training, the prefix has mixed AR masks:
- Images: bidirectional (0)
- Text: bidirectional (0)
- FAST tokens: causal (1)

During inference, all prefix tokens are bidirectional (0).

**Impact**: Attention patterns differ, may cause slight distribution shift.

---

## Recommended Fix

See the companion document `KI_IMPLEMENTATION_FIXES.md` for detailed solutions. The core fix is:

**Keep prefix consistent between training and inference:**
- Prefix should only contain: images + text prompt
- FAST tokens should be processed separately for their training objective
- Action expert should always see the same prefix structure

**Modified training flow:**
```
Training:
  ├─ VLM Path: [images] + [Task/State/Action:<FAST>] → FAST loss
  └─ Action Expert Path: [images] + [Task/State] (detached) + [noisy actions] → Action loss

Inference:
  └─ Action Expert Path: [images] + [Task/State] + [noisy actions] → Actions
     (same prefix as training!)
```

---

## Conclusion

The current KI implementation has a **critical distribution mismatch** between training and inference. The model trains with FAST tokens in the prefix (~957 tokens) but does inference without them (~798 tokens). This violates the train/test consistency principle and likely hurts performance.

The fix requires separating FAST token processing from the main prefix embeddings to ensure the action expert sees consistent input structures during both training and inference.
