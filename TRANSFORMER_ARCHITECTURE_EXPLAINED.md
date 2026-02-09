# π0.5 Transformer Architecture Deep Dive

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Transformer Basics: Attention Mechanism](#transformer-basics-attention-mechanism)
3. [Dual-Expert Architecture](#dual-expert-architecture)
4. [Token Sequences: Prefix vs Suffix](#token-sequences-prefix-vs-suffix)
5. [Attention Masks Explained](#attention-masks-explained)
6. [Complete Forward Pass Walkthrough](#complete-forward-pass-walkthrough)
7. [KV Cache for Inference](#kv-cache-for-inference)
8. [Knowledge Insulation in Detail](#knowledge-insulation-in-detail)

---

## High-Level Overview

The π0.5 model is a **dual-expert transformer** that combines:
- **Expert 0 (VLM)**: A PaliGemma vision-language model that processes images and text
- **Expert 1 (Action Expert)**: A specialized transformer that predicts robotic actions

Think of it like having two brains working together:
- **VLM brain**: "I see a cup and you said 'pick up the cup'"
- **Action brain**: "Given what the VLM sees, here's how to move the robot"

```
Input: [Images] + [Text Prompt] + [Noisy Actions + Time]
         ↓              ↓                    ↓
      [Expert 0]    [Expert 0]         [Expert 1]
         ↓              ↓                    ↓
             [Shared Attention Mechanism]
                        ↓
                [Action Predictions]
```

---

## Transformer Basics: Attention Mechanism

### What is Attention?

Attention allows each token to "look at" other tokens and decide what's important. It's like asking:
- "To predict this action, what parts of the image should I focus on?"
- "Which words in the prompt are most relevant?"

### The Q, K, V Matrices

Every token creates three vectors through learned projection matrices:

```
Token Embedding (e.g., 2048-dim)
    ↓
    ├── Query (Q):  "What am I looking for?"
    ├── Key (K):    "What do I contain?"
    └── Value (V):  "What information can I provide?"
```

**Standard Single-Expert Attention:**
```python
# Each token creates Q, K, V
Q = token @ W_q  # shape: [batch, seq_len, head_dim]
K = token @ W_k  # shape: [batch, seq_len, head_dim]
V = token @ W_v  # shape: [batch, seq_len, head_dim]

# Compute attention scores: how much should each token attend to others?
scores = Q @ K.T / sqrt(head_dim)  # shape: [batch, seq_len, seq_len]

# Apply attention mask (explained later)
masked_scores = scores + mask  # mask is 0 or -inf

# Softmax to get attention weights
attention_weights = softmax(masked_scores)  # shape: [batch, seq_len, seq_len]

# Weighted sum of values
output = attention_weights @ V  # shape: [batch, seq_len, head_dim]
```

**Intuition**: 
- Token A's **Query** asks: "I need information about grasping"
- Token B's **Key** responds: "I contain gripper position info"
- High Q·K score → Token A pays attention to Token B
- Token A gets Token B's **Value** (the actual information)

---

## Dual-Expert Architecture

### Why Two Experts?

The key innovation in π0.5 is that **different tokens use different experts**:
- Image tokens → Processed by VLM expert (Expert 0)
- Text tokens → Processed by VLM expert (Expert 0)
- Action tokens → Processed by Action expert (Expert 1)

### Separate Q, K, V Matrices Per Expert

In `gemma.py`, the `Attention` class has:

```python
class Attention:
    def __init__(self, configs: list):  # Two configs: [VLM config, Action config]
        # Expert 0 (VLM) parameters
        self.q_einsum_0 = [W_q matrices for VLM]
        self.k_einsum_0 = [W_k matrices for VLM]
        self.v_einsum_0 = [W_v matrices for VLM]
        
        # Expert 1 (Action) parameters
        self.q_einsum_1 = [W_q matrices for Action]
        self.k_einsum_1 = [W_k matrices for Action]
        self.v_einsum_1 = [W_v matrices for Action]
```

### How It Works

```python
# Input: two token sequences
x = [prefix_tokens, suffix_tokens]
#     ↑ VLM expert       ↑ Action expert

# Step 1: Each expert creates its own Q, K, V
Q0, K0, V0 = prefix_tokens @ [W_q0, W_k0, W_v0]  # VLM
Q1, K1, V1 = suffix_tokens @ [W_q1, W_k1, W_v1]  # Action

# Step 2: Concatenate into one big sequence
Q_combined = concat([Q0, Q1], axis=seq_dim)  # All queries together
K_combined = concat([K0, K1], axis=seq_dim)  # All keys together
V_combined = concat([V0, V1], axis=seq_dim)  # All values together

# Step 3: Compute attention across ENTIRE sequence
# This is where magic happens: action tokens can attend to image tokens!
scores = Q_combined @ K_combined.T
attention_weights = softmax(scores + mask)
output_combined = attention_weights @ V_combined

# Step 4: Split output back to experts
output_vlm, output_action = split(output_combined)
```

**Key Insight**: 
- Separate parameters (different W matrices) for each expert
- But joint attention computation (tokens from both experts interact)
- This allows action expert to "see" what VLM processed

---

## Token Sequences: Prefix vs Suffix

### Prefix Tokens (VLM Input)

**What**: Images + text prompt

**Created in** `embed_prefix()`:

```python
prefix_tokens = [
    # Image tokens from base camera
    [embedding_0, embedding_1, ..., embedding_255],  # 256 tokens
    
    # Image tokens from wrist camera  
    [embedding_256, ..., embedding_511],             # 256 tokens
    
    # Text prompt tokens
    ["pick", "up", "the", "cup"]                     # 4 tokens
]
# Total: 516 tokens processed by VLM (Expert 0)
```

### Suffix Tokens (Action Input)

**What**: Actions + timestep embedding

**Created in** `embed_suffix()`:

```python
suffix_tokens = [
    # For π0.5: just action tokens (no explicit state token)
    [action_t0, action_t1, ..., action_t15]  # 16 action tokens
]
# Total: 16 tokens processed by Action Expert (Expert 1)
```

### Why This Split?

**During Training**:
```python
# Prefix = what the robot observes
prefix = embed_prefix(observation)

# Suffix = noisy actions at time t
suffix = embed_suffix(observation, noisy_actions, timestep)

# Forward pass with both
output = model([prefix, suffix])  # Predict action velocities
```

**During Inference**:
```python
# Prefix computed once and cached (doesn't change)
prefix, kv_cache = compute_prefix_once(observation)

# Suffix computed iteratively (actions get refined)
for step in range(num_diffusion_steps):
    suffix = embed_suffix(observation, current_actions, current_time)
    output = model([None, suffix], kv_cache=kv_cache)  # Reuse prefix
    current_actions = update(current_actions, output)
```

---

## Attention Masks Explained

There are **three types of masks** that control which tokens can attend to which:

### 1. Input Mask (`input_mask`)

**Purpose**: Distinguish real tokens from padding

```python
input_mask = [True, True, True, False, False]
#            ↑ real tokens    ↑ padding

# Example with batch of 2 sequences:
input_mask = [
    [True, True, True, True],    # Sequence 1: all 4 tokens are real
    [True, True, False, False],  # Sequence 2: only 2 tokens, 2 padding
]
```

**Why needed**: Batches have different lengths, so we pad shorter sequences

### 2. Autoregressive Mask (`ar_mask`)

**Purpose**: Control causal attention (future tokens can't attend to past)

```python
ar_mask = [False, False, False, True, True, True]
#          ↑ prefix (full attention)  ↑ suffix (causal attention)

# False = token shares attention with previous
# True  = token is causally masked from previous
```

**Examples**:

**Pure Causal** (like GPT):
```python
ar_mask = [True, True, True, True]
# Token 0: can only see itself
# Token 1: can see [0, 1]
# Token 2: can see [0, 1, 2]
# Token 3: can see [0, 1, 2, 3]
```

**Prefix-LM** (like π0.5):
```python
ar_mask = [False, False, False, True, True, True]
#          ↑ images + text       ↑ actions

# Image/text tokens: full bidirectional attention among themselves
# Action tokens: causal attention (can't see future actions)
```

### 3. Attention Mask (`attn_mask`)

**Purpose**: The final mask applied to attention scores

**Created by** `make_attn_mask()`:

```python
def make_attn_mask(input_mask, ar_mask):
    # Step 1: Cumulative sum of ar_mask
    cumsum = jnp.cumsum(ar_mask, axis=1)
    # Example: [False, False, True, True] → [0, 0, 1, 2]
    
    # Step 2: Token i can attend to token j if:
    #         cumsum[j] <= cumsum[i]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    
    # Step 3: Also respect padding (input_mask)
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    
    return attn_mask & valid_mask
```

**Example Walkthrough**:

```python
# Setup
input_mask = [True, True, True, True, True, True]
ar_mask = [False, False, False, True, True, True]
#          ↑ prefix (tokens 0-2)  ↑ suffix (tokens 3-5)

# Cumsum
cumsum = [0, 0, 0, 1, 2, 3]

# Attention mask (which tokens can attend to which):
#       Can attend to token →
#       0  1  2  3  4  5
attn = [[1, 1, 1, 0, 0, 0],  # Token 0 (prefix)
        [1, 1, 1, 0, 0, 0],  # Token 1 (prefix)
        [1, 1, 1, 0, 0, 0],  # Token 2 (prefix)
        [1, 1, 1, 1, 0, 0],  # Token 3 (action) - causal
        [1, 1, 1, 1, 1, 0],  # Token 4 (action) - causal
        [1, 1, 1, 1, 1, 1]]  # Token 5 (action) - causal
#    ↑ All actions can attend to all prefix tokens
#                ↑ But only to previous actions (causal)
```

**Visual Representation**:

```
Prefix tokens (images + text): ■ ■ ■
Action tokens: ▲ ▲ ▲

■ can attend to: ■ ■ ■ (full bidirectional)
▲ can attend to: ■ ■ ■ ▲ (prefix + causal actions)

Example:
■₀ ←→ ■₁ ←→ ■₂     (prefix: full attention)
 ↓     ↓     ↓
▲₀ → ▲₁ → ▲₂        (suffix: causal attention)
```

---

## Complete Forward Pass Walkthrough

Let's trace a single training example through the model:

### Input Data

```python
observation = {
    'base_image': [3, 224, 224],      # RGB camera image
    'wrist_image': [3, 224, 224],     # Wrist camera image
    'state': [7],                      # Current joint positions
}
actions = [16, 7]                      # Action chunk (16 timesteps, 7 dims)
prompt = "pick up the cup"
```

### Step 1: Embed Prefix (Images + Text)

```python
# In pi0.py: embed_prefix()

# 1a. Image encoding through SigLIP vision encoder
base_image_tokens = img_encoder(base_image)    # → [256, 2048]
wrist_image_tokens = img_encoder(wrist_image)  # → [256, 2048]

# 1b. Text tokenization and embedding
prompt_ids = tokenizer("pick up the cup")      # → [4] token IDs
prompt_tokens = llm.embed(prompt_ids)          # → [4, 2048]

# 1c. Concatenate
prefix_tokens = concat([
    base_image_tokens,   # [256, 2048]
    wrist_image_tokens,  # [256, 2048]  
    prompt_tokens,       # [4, 2048]
])  # Total: [516, 2048]

# 1d. Create masks
prefix_mask = [True] * 516           # All tokens are valid
prefix_ar_mask = [False] * 516       # Full bidirectional attention
```

### Step 2: Embed Suffix (Actions + Time)

```python
# In pi0.py: embed_suffix()

# 2a. Add noise for flow matching
noise = random.normal([16, 7])
timestep = random.beta(1.5, 1)          # Scalar in [0, 1]
x_t = timestep * noise + (1 - timestep) * actions  # Noisy actions

# 2b. Project actions to embedding space
action_tokens = action_in_proj(x_t)     # [16, 7] → [16, 2048]

# 2c. Embed timestep with sinusoidal encoding
time_emb = posemb_sincos(timestep, 2048)  # [2048]

# 2d. For π0.5: use AdaRMS (adaptive RMS norm)
time_cond = time_mlp(time_emb)          # [2048] - used in attention layers

# 2e. Create masks
suffix_tokens = action_tokens            # [16, 2048]
suffix_mask = [True] * 16                # All tokens are valid
suffix_ar_mask = [True] + [False] * 15   # Causal for actions
```

### Step 3: Combine Tokens

```python
# Concatenate prefix and suffix
combined_tokens = [prefix_tokens, suffix_tokens]  # Expert 0 + Expert 1
combined_mask = concat([prefix_mask, suffix_mask])  # [532]
combined_ar_mask = concat([prefix_ar_mask, suffix_ar_mask])  # [532]

# Create attention mask
attn_mask = make_attn_mask(combined_mask, combined_ar_mask)
# Shape: [532, 532] - which tokens can attend to which

# Compute positions for positional encoding
positions = cumsum(combined_mask) - 1  # [0, 1, 2, ..., 531]
```

### Step 4: Transformer Forward Pass

```python
# In gemma.py: Module.__call__()

# 4a. Create Q, K, V for each expert
Q_vlm, K_vlm, V_vlm = prefix_tokens @ [W_q0, W_k0, W_v0]
Q_act, K_act, V_act = suffix_tokens @ [W_q1, W_k1, W_v1]

# 4b. Concatenate
Q = concat([Q_vlm, Q_act])  # [532, num_heads, head_dim]
K = concat([K_vlm, K_act])  # [532, num_heads, head_dim]
V = concat([V_vlm, V_act])  # [532, num_heads, head_dim]

# 4c. Attention computation
scores = Q @ K.T / sqrt(head_dim)  # [532, 532]
scores = scores + attn_mask        # Apply mask (-inf where not allowed)
weights = softmax(scores)          # Attention weights
output = weights @ V               # Weighted sum of values

# 4d. Add positional encoding (RoPE)
output = apply_rotary_embedding(output, positions)

# 4e. Feed-forward network
output = layer_norm(output)
output = ffn(output)

# 4f. Split back to experts
output_vlm, output_action = split(output)  # [516, 2048], [16, 2048]
```

### Step 5: Predict Actions

```python
# In pi0.py: compute_loss()

# Take only the action tokens output from transformer
action_output = output_action  # [16, 2048]

# Project to action space
v_t = action_out_proj(action_output)  # [16, 2048] → [16, 7]

# Flow matching loss: predict velocity
u_t = noise - actions  # True velocity
loss = mean_squared_error(v_t, u_t)
```

---

## KV Cache for Inference

### Why Cache?

During inference, we do **iterative denoising**:
```python
for step in range(10):  # 10 denoising steps
    # Problem: prefix (images + text) doesn't change!
    # Recomputing it 10 times is wasteful
    actions = denoise_step(images, text, actions)
```

### How KV Cache Works

**Key Insight**: In attention, only **queries** depend on new tokens. **Keys** and **values** from prefix can be reused!

```python
# First step: compute full attention
Q_prefix, K_prefix, V_prefix = attention(prefix_tokens)
Q_suffix, K_suffix, V_suffix = attention(suffix_tokens)

# Attention between suffix queries and all keys
output = softmax(Q_suffix @ concat([K_prefix, K_suffix]).T) @ concat([V_prefix, V_suffix])
#                            ↑ These don't change!

# Cache for reuse
kv_cache = {'K_prefix': K_prefix, 'V_prefix': V_prefix}

# Next steps: only compute suffix
for step in range(1, 10):
    Q_suffix_new, K_suffix_new, V_suffix_new = attention(new_suffix_tokens)
    
    # Reuse cached prefix K, V
    output = softmax(Q_suffix_new @ concat([kv_cache['K_prefix'], K_suffix_new]).T) \
             @ concat([kv_cache['V_prefix'], V_suffix_new])
```

**Savings**: 
- Without cache: 10 full forward passes
- With cache: 1 full + 9 suffix-only passes
- **Speedup**: ~3-5x for typical sequences

### In Code

```python
# In pi0.py: sample_actions()

# Step 1: Compute prefix once
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
_, kv_cache = self.PaliGemma.llm(
    [prefix_tokens, None],  # Only VLM expert
    mask=prefix_attn_mask,
    positions=positions
)

# Step 2: Iterative denoising with cache
for step in range(num_steps):
    suffix_tokens, suffix_mask, _, adarms_cond = self.embed_suffix(
        observation, current_actions, timestep
    )
    
    # Use cache: don't recompute prefix
    (_, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],  # Skip prefix computation
        kv_cache=kv_cache,      # Reuse cached K, V
        ...
    )
    
    # Update actions
    v_t = self.action_out_proj(suffix_out)
    current_actions += dt * v_t
```

---

## Knowledge Insulation in Detail

### The Problem KI Solves

**Without KI**: 
- VLM parameters (millions of parameters) get updated by action prediction loss
- Risk: VLM forgets vision-language understanding to optimize actions
- Like teaching a translator to also cook - they might forget translation

**With KI**:
- VLM learns separate task: predict discretized actions (FAST tokens)
- Action expert learns continuous refinement
- Gradients isolated via `stop_gradient()`

### Architecture Changes

```python
# Standard Training:
[Images + Text] → VLM → Embeddings → Action Expert → Actions
                   ↑←←←←← Gradients flow back from action loss

# Knowledge Insulation:
[Images + Text] → VLM → FAST token predictions
                   ↓ stop_gradient()
                Detached Embeddings → Action Expert → Actions
                                       ↑←←←← Gradients only update Action Expert
```

### Two Forward Passes

In `compute_loss()` with KI enabled:

**Pass 1: VLM FAST Token Prediction**
```python
# Forward through VLM only
prefix_tokens = embed_prefix(observation)
(vlm_output, _), _ = transformer([prefix_tokens, None])

# Decode to vocabulary
logits = vocab_projection(vlm_output)  # [batch, seq_len, vocab_size]

# Cross-entropy loss on FAST tokens
target_tokens = observation.tokenized_prompt[:, 1:]  # Next-token prediction
fast_loss = cross_entropy(logits[:, :-1], target_tokens)
```

**Pass 2: Action Expert Flow Matching**
```python
# Detach prefix (stop gradient)
prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)

# Forward through both experts
suffix_tokens = embed_suffix(observation, noisy_actions, time)
(_, action_output), _ = transformer(
    [prefix_tokens_detached, suffix_tokens]
)

# Flow matching loss
v_t = action_out_proj(action_output)
action_loss = mse(v_t, u_t)
```

**Combined Loss**:
```python
total_loss = action_loss + lambda_fast * fast_loss
```

### Gradient Flow

```
FAST Token Loss:
[VLM Parameters] ←← gradient ←← FAST cross-entropy loss
[Action Parameters] (no gradient)

Action Loss:
[VLM Parameters] (no gradient) ← stop_gradient()
[Action Parameters] ←← gradient ←← Flow matching loss
```

### Why This Works

1. **VLM stays grounded**: Learns vision-language task (FAST prediction)
2. **Action expert specializes**: Only learns continuous control
3. **Information flow preserved**: Action expert still sees VLM output (forward pass)
4. **No catastrophic forgetting**: VLM maintains pre-trained knowledge

---

## Summary

### Key Concepts

| Concept | Purpose | Location |
|---------|---------|----------|
| **Dual Experts** | Separate parameters for VLM and actions | `gemma.py` |
| **Q, K, V matrices** | Enable attention mechanism | `gemma.py: Attention` |
| **Prefix tokens** | Images + text processed by VLM | `pi0.py: embed_prefix()` |
| **Suffix tokens** | Actions processed by Action Expert | `pi0.py: embed_suffix()` |
| **input_mask** | Distinguish real tokens from padding | All forward passes |
| **ar_mask** | Control causal vs bidirectional attention | `pi0.py: embed_*` |
| **attn_mask** | Final mask applied to attention scores | `make_attn_mask()` |
| **KV Cache** | Speed up inference by reusing prefix | `pi0.py: sample_actions()` |
| **stop_gradient()** | Isolate VLM from action gradients | `pi0.py: compute_loss()` |

### Data Flow Summary

```
Training:
  Images + Text → [VLM Encoder] → prefix_tokens [516, 2048]
  Actions + Time → [Action Proj] → suffix_tokens [16, 2048]
                                          ↓
                    [Dual-Expert Transformer with masks]
                                          ↓
                                   output [16, 2048]
                                          ↓
                                   [Action Projection]
                                          ↓
                                   predicted v_t [16, 7]
                                          ↓
                                   MSE(v_t, u_t) → loss

Inference:
  Images + Text → [VLM Encoder] → prefix_tokens → [KV Cache]
                                                        ↓
  For step in denoising:                                ↓
    Current actions → [Action Proj] → suffix_tokens → [Transformer with cache]
                                                              ↓
                                                      Update actions
```

### Next Steps

Now that you understand the architecture, you can:
1. Follow the implementation plan to add KI
2. Debug attention patterns by visualizing `attn_mask`
3. Experiment with different mask patterns
4. Understand how gradients flow through the model

---

**Questions to test your understanding:**
1. Why do we need separate Q, K, V matrices for each expert?
2. What would happen if `ar_mask` was all `True` for prefix tokens?
3. How does KV cache reduce computation?
4. Why does `stop_gradient()` not break the forward pass?

*Answers at bottom of file*

---

<details>
<summary><b>Answers</b></summary>

1. **Separate Q, K, V**: Each expert learns different feature representations. VLM learns visual/language features, Action expert learns control features. Sharing parameters would force a compromise.

2. **All True ar_mask for prefix**: Would make prefix causal (each token only sees previous tokens). This breaks bidirectional understanding - image tokens should see whole image, not just raster order!

3. **KV cache reduction**: Keys and Values from prefix don't change across denoising steps. Cache saves ~50% of transformer computation by skipping prefix attention computation 9 out of 10 steps.

4. **stop_gradient forward pass**: `stop_gradient()` only blocks backward pass (gradients). Forward pass values flow normally, so action expert still "sees" VLM output, just can't modify VLM parameters.

</details>
