# Action Expert Architecture: Deep Dive

## Your Questions Answered

You're asking about a subtle but crucial architectural detail in π0.5. Let me clarify the confusion:

**The action expert IS part of the transformer architecture!** 

It's not a separate flow-matching head outside the transformer. The entire model is one unified transformer, but with **dual experts** that have separate parameters.

---

## The Key Insight: Dual-Expert Transformer

### What "Dual-Expert" Means

Think of it like having **two sets of brains sharing one body**:

```
┌─────────────────────────────────────────────────────────────┐
│                     ONE TRANSFORMER                          │
│                                                              │
│  ┌─────────────────┐           ┌──────────────────┐         │
│  │  Expert 0 (VLM) │           │ Expert 1 (Action)│         │
│  │                 │           │                  │         │
│  │  Own W_q, W_k,  │           │  Own W_q, W_k,   │         │
│  │  W_v, FFN       │           │  W_v, FFN        │         │
│  └────────┬────────┘           └────────┬─────────┘         │
│           │                             │                   │
│           └──────────┬──────────────────┘                   │
│                      ↓                                       │
│              SHARED ATTENTION MECHANISM                      │
│              (All Q, K, V concatenated)                      │
└─────────────────────────────────────────────────────────────┘
```

**Key Point**: Both experts are **transformer layers**. The action expert isn't a separate MLP head - it's a full transformer with its own parameters, but sharing the attention computation with the VLM.

---

## Exact Input to Action Expert

### Step-by-Step Breakdown

#### 1. Raw Action Inputs (from dataset)

```python
# During training: noisy actions from flow matching
actions = [batch, action_horizon, action_dim]  # e.g., [32, 16, 8]
timestep = scalar in [0, 1]  # Flow matching time

# Example values:
actions = [[0.1, 0.2, -0.3, 0.5, 0.1, -0.2, 0.3, 0.0],  # t=0
           [0.15, 0.22, -0.28, 0.52, 0.12, -0.18, 0.32, 0.01], # t=1
           ...  # 14 more timesteps
          ]
timestep = 0.37  # Random time in flow matching process
```

#### 2. Action Embedding (in `pi0.py`)

```python
# File: pi0.py, embed_suffix() method

# Project actions to transformer dimension
action_tokens = self.action_in_proj(actions)
# Input:  [batch, 16, 8]
# Output: [batch, 16, 2048]  # Now in transformer embedding space

# Embed timestep using sinusoidal encoding
time_emb = posemb_sincos(timestep, 2048, min_period=4e-3, max_period=4.0)
# Input:  scalar (e.g., 0.37)
# Output: [batch, 2048]  # Rich encoding of time information

# For π0.5: Use AdaRMS (Adaptive RMS Normalization)
# Time embedding is processed through MLP to create conditioning
time_cond = self.time_mlp_in(time_emb)
time_cond = swish(time_cond)
time_cond = self.time_mlp_out(time_cond)
time_cond = swish(time_cond)
# Output: [batch, 2048]  # Will be used to condition transformer layers
```

**What gets passed to transformer**:
```python
suffix_tokens = action_tokens  # [batch, 16, 2048]
adarms_cond = time_cond        # [batch, 2048]
```

#### 3. Full Input Sequence to Transformer

```python
# Prefix tokens (from VLM - images + text)
prefix_tokens = [batch, 516, 2048]  # 256 + 256 + 4 tokens

# Suffix tokens (from Action Expert - actions)
suffix_tokens = [batch, 16, 2048]

# These are passed to gemma.Module as:
inputs = [prefix_tokens, suffix_tokens]
#         ↑ Expert 0      ↑ Expert 1
```

---

## How the Transformer Processes Dual Experts

### The Architecture in `gemma.py`

Let's trace through one transformer layer:

#### Layer Structure

```python
# File: gemma.py, BlockGated class (lines ~250-320)

class BlockGated:
    """One transformer layer with dual experts"""
    
    def __init__(self, configs: list[Config, Config]):
        # VLM expert (Expert 0) layers
        self.attn_norm_0 = RMSNorm(configs[0].width)
        self.attn_0 = Attention(configs, dual_stream=True)
        self.mlp_norm_0 = RMSNorm(configs[0].width)
        self.mlp_0 = MLP(configs[0])
        
        # Action expert (Expert 1) layers
        self.attn_norm_1 = AdaRMSNorm(configs[1].width)  # Conditioned on time!
        self.attn_1 = Attention(configs, dual_stream=True)  # SAME attention class
        self.mlp_norm_1 = AdaRMSNorm(configs[1].width)   # Conditioned on time!
        self.mlp_1 = MLP(configs[1])
```

**Key observations**:
1. Both experts have full transformer layers (attention + FFN)
2. Action expert uses **AdaRMSNorm** conditioned on timestep
3. **Same Attention class** for both (this is where magic happens!)

#### Forward Pass Through One Layer

```python
def __call__(self, x: list[Array, Array], adarms_cond: list[None, Array]):
    x_vlm, x_action = x
    # x_vlm:    [batch, 516, 2048] - VLM tokens
    # x_action: [batch, 16, 2048]  - Action tokens
    
    # === Pre-normalization ===
    x_vlm_normed = self.attn_norm_0(x_vlm)  # Standard RMSNorm
    x_action_normed = self.attn_norm_1(x_action, cond=adarms_cond[1])  # Time-conditioned!
    
    # === ATTENTION (This is where communication happens!) ===
    attn_out = self.attn_0(
        [x_vlm_normed, x_action_normed],  # Both experts' inputs
        mask=attn_mask,
        positions=positions,
        kv_cache=kv_cache
    )
    # Returns: [attn_vlm, attn_action]
    # Each has attended to ALL tokens (prefix + suffix)
    
    # === Residual connection ===
    x_vlm = x_vlm + attn_out[0]
    x_action = x_action + attn_out[1]
    
    # === Feed-forward network ===
    x_vlm_normed = self.mlp_norm_0(x_vlm)
    x_action_normed = self.mlp_norm_1(x_action, cond=adarms_cond[1])
    
    x_vlm = x_vlm + self.mlp_0(x_vlm_normed)
    x_action = x_action + self.mlp_1(x_action_normed)
    
    return [x_vlm, x_action]
```

---

## The Attention Mechanism: Where Experts Communicate

This is **THE CRITICAL PART** - how action expert attends to VLM.

### Code in `gemma.py` Attention class

```python
# File: gemma.py, Attention class (lines ~164-231)

class Attention:
    def __init__(self, configs: list[Config, Config], dual_stream: bool = True):
        # Expert 0 (VLM) parameters
        self.q_einsum_0 = [create_einsum(config_0) for each head]
        self.k_einsum_0 = [create_einsum(config_0) for each head]
        self.v_einsum_0 = [create_einsum(config_0) for each head]
        
        # Expert 1 (Action) parameters
        self.q_einsum_1 = [create_einsum(config_1) for each head]
        self.k_einsum_1 = [create_einsum(config_1) for each head]
        self.v_einsum_1 = [create_einsum(config_1) for each head]
    
    def __call__(self, x: list[Array, Array], mask, positions):
        x_vlm, x_action = x
        
        # === Step 1: Each expert creates its own Q, K, V ===
        # VLM expert (Expert 0)
        q_vlm = apply_rope(q_einsum_0(x_vlm), positions_vlm)  # [batch, 516, n_heads, head_dim]
        k_vlm = apply_rope(k_einsum_0(x_vlm), positions_vlm)
        v_vlm = v_einsum_0(x_vlm)
        
        # Action expert (Expert 1)
        q_action = apply_rope(q_einsum_1(x_action), positions_action)  # [batch, 16, n_heads, head_dim]
        k_action = apply_rope(k_einsum_1(x_action), positions_action)
        v_action = v_einsum_1(x_action)
        
        # === Step 2: CONCATENATE Q, K, V from both experts ===
        # This is the KEY - all queries, keys, values are combined!
        Q = jnp.concatenate([q_vlm, q_action], axis=1)      # [batch, 532, n_heads, head_dim]
        K = jnp.concatenate([k_vlm, k_action], axis=1)      # [batch, 532, n_heads, head_dim]
        V = jnp.concatenate([v_vlm, v_action], axis=1)      # [batch, 532, n_heads, head_dim]
        
        # === Step 3: Compute attention scores across ENTIRE sequence ===
        logits = jnp.einsum('bqhd,bkhd->bhqk', Q, K) / sqrt(head_dim)
        # Shape: [batch, n_heads, 532, 532]
        # This matrix contains ALL possible attention connections!
        
        # Apply attention mask
        logits = logits + mask  # mask controls which tokens can attend to which
        
        # === Step 4: Softmax to get attention weights ===
        attention_weights = jax.nn.softmax(logits, axis=-1)
        # Shape: [batch, n_heads, 532, 532]
        
        # === Step 5: Apply attention to values ===
        output = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, V)
        # Shape: [batch, 532, n_heads, head_dim]
        
        # === Step 6: Split output back to experts ===
        output_vlm = output[:, :516, ...]     # VLM output
        output_action = output[:, 516:, ...]  # Action output
        
        return [output_vlm, output_action]
```

---

## Visualizing the Attention Flow

### Attention Matrix Structure

When we compute `Q @ K^T`, we get a `[532, 532]` attention matrix:

```
                      Keys (What can be attended to)
                   ┌─────────────┬────────────┐
                   │   Prefix    │   Suffix   │
                   │  (516 VLM)  │ (16 Action)│
              ┌────┼─────────────┼────────────┤
              │    │             │            │
    Queries   │ P  │   P → P     │   P → S    │  VLM tokens can attend
    (Who is   │ r  │   (516x516) │  (516x16)  │  to themselves & actions
    attending)│ e  │             │            │  (but mask prevents P→S)
              │ f  ├─────────────┼────────────┤
              │ i  │             │            │
              │ x  │   S → P     │   S → S    │  Action tokens attend
              │    │   (16x516)  │  (16x16)   │  to VLM (S→P) and
              ├────┤   ✓ ALLOWED │  ✓ CAUSAL  │  themselves (S→S causal)
              │ S  │             │            │
              │ u  │             │            │
              │ f  │             │            │
              │ f  │             │            │
              │ i  │             │            │
              │ x  │             │            │
              └────┴─────────────┴────────────┘
```

### Attention Mask Controls Communication

```python
# From make_attn_mask() in pi0.py
prefix_ar_mask = [False] * 516  # Bidirectional
suffix_ar_mask = [True] + [False] * 15  # Causal

# Result:
# [False, False, ..., False (516x), True, False, False, ..., False (16x)]
#  ↑ Prefix (VLM)                   ↑ Suffix (Action)

# This creates:
attn_mask = [
    # Prefix rows: VLM tokens can attend to each other (bidirectional)
    # but NOT to suffix (action tokens)
    [1, 1, ..., 1, 0, 0, ..., 0],  # VLM token 0
    [1, 1, ..., 1, 0, 0, ..., 0],  # VLM token 1
    ...
    [1, 1, ..., 1, 0, 0, ..., 0],  # VLM token 515
    
    # Suffix rows: Action tokens CAN attend to ALL prefix (VLM)
    # and causally to previous actions
    [1, 1, ..., 1, 1, 0, ..., 0],  # Action token 0 - sees all VLM
    [1, 1, ..., 1, 1, 1, 0, ..., 0],  # Action token 1 - sees VLM + action 0
    [1, 1, ..., 1, 1, 1, 1, 0, ..., 0],  # Action token 2 - sees VLM + actions 0-1
    ...
]
```

**This is the one-way communication!**
- VLM cannot attend to action tokens (0s in top-right)
- Action tokens CAN attend to ALL VLM tokens (1s in bottom-left)

---

## Complete Data Flow: Training Step

### Input Preparation

```python
# 1. Sample from dataset
observation = {
    'images': {'base': [...], 'wrist': [...]},
    'state': [0.5, -0.3, 0.8, ...],
    'prompt': "pick up the cup"
}
actions = [[0.1, 0.2, ...], [0.15, 0.22, ...], ...]  # [16, 8]

# 2. Flow matching: add noise
timestep = random.beta(1.5, 1) * 0.999 + 0.001  # e.g., 0.37
noise = random.normal([16, 8])
x_t = timestep * noise + (1 - timestep) * actions
u_t = noise - actions  # Target velocity
```

### Embedding Phase

```python
# 3. Embed prefix (VLM input)
# Images through SigLIP
image_tokens = SigLIP(images)  # [batch, 512, 2048]

# Text through PaliGemma tokenizer + embedding
text_tokens = LLM.embed(tokenized_prompt)  # [batch, 4, 2048]

prefix_tokens = concat([image_tokens, text_tokens])  # [batch, 516, 2048]

# 4. Embed suffix (Action Expert input)
# Project noisy actions
action_tokens = action_in_proj(x_t)  # [16, 8] → [16, 2048]

# Embed timestep and create conditioning
time_emb = posemb_sincos(timestep, 2048)  # [2048]
time_cond = time_mlp(time_emb)  # [2048] for AdaRMS conditioning

suffix_tokens = action_tokens  # [batch, 16, 2048]
```

### Transformer Processing

```python
# 5. Pass through transformer
inputs = [prefix_tokens, suffix_tokens]
#         [batch, 516, 2048], [batch, 16, 2048]

# For each of 18 layers (in Gemma-2B):
for layer in layers:
    # Pre-norm (action expert uses time-conditioned norm)
    x_vlm_norm = RMSNorm(x_vlm)
    x_action_norm = AdaRMSNorm(x_action, cond=time_cond)  # ← Time injected here!
    
    # Attention (both experts communicate)
    x_vlm, x_action = Attention(
        [x_vlm_norm, x_action_norm],
        mask=attn_mask  # Controls one-way communication
    )
    # Action expert attends to VLM tokens here! ↑
    
    # FFN (separate for each expert)
    x_vlm = x_vlm + FFN_0(RMSNorm(x_vlm))
    x_action = x_action + FFN_1(AdaRMSNorm(x_action, cond=time_cond))

# Output
prefix_out = x_vlm      # [batch, 516, 2048]
suffix_out = x_action   # [batch, 16, 2048]
```

### Loss Computation

```python
# 6. Project action expert output to action space
v_t = action_out_proj(suffix_out[:, -16:])  # [16, 2048] → [16, 8]

# 7. Flow matching loss
loss = mean_squared_error(v_t, u_t)
```

---

## Key Insights: Answering Your Questions

### Q: "Is the action expert just a flow-matching head?"

**A**: No! The action expert is a **full transformer** with:
- Its own Q, K, V projection matrices
- Its own FFN (feed-forward network)
- AdaRMSNorm layers conditioned on timestep
- But shares the attention computation with VLM

The flow-matching aspect is in the **loss function**, not the architecture!

### Q: "How do they communicate via attention?"

**A**: Through **concatenated Q, K, V**:
1. VLM creates: `q_vlm, k_vlm, v_vlm` using its parameters
2. Action expert creates: `q_action, k_action, v_action` using its parameters
3. These are concatenated: `Q = [q_vlm; q_action]`
4. Attention is computed: `softmax(Q @ K^T) @ V`
5. Output is split back: `[output_vlm, output_action]`

Action tokens' queries attend to VLM tokens' keys/values!

### Q: "What is one-way communication?"

**A**: Controlled by **attention mask**:
- `attn_mask[vlm_token, action_token] = -inf` → VLM can't see actions
- `attn_mask[action_token, vlm_token] = 0` → Actions CAN see VLM

### Q: "What is the exact input to action expert?"

**A**: 
1. **Noisy actions** `x_t` (linear projected to 2048-dim)
2. **Timestep** `t` (sinusoidal encoding, processed through MLP)
3. **VLM embeddings** (accessed during attention via K, V)

---

## Why This Architecture?

### Advantages of Shared Transformer

1. **Efficient**: VLM and action expert share attention computation
2. **Information flow**: Action expert sees rich VLM features via attention
3. **Modular**: Separate parameters allow independent learning
4. **Time conditioning**: AdaRMSNorm injects flow-matching time into action expert

### Flow Matching in Transformer Context

```
Flow matching is the TRAINING OBJECTIVE, not a separate architecture!

Training:
- Add noise to actions at random time t
- Action expert (transformer) predicts velocity v_t
- Loss: MSE(v_t, u_t)

Inference:
- Start with pure noise (t=1)
- Iteratively denoise using predicted velocities
- Each step: x_{t-dt} = x_t + dt * v_t
- End with clean actions (t=0)
```

The action expert transformer learns to denoise actions by attending to VLM embeddings!

---

## Visual Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                                │
├──────────────────────────────────────────────────────────────────┤
│ Images [224,224,3]  →  SigLIP  →  [512, 2048]                   │
│ Prompt "pick cup"   →  Embed   →  [4, 2048]                     │
│ Noisy actions [16,8] → Linear  →  [16, 2048]                    │
│ Timestep 0.37       →  MLP     →  [2048] conditioning           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER LAYER                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VLM tokens [516, 2048]         Action tokens [16, 2048]        │
│       ↓                                  ↓                       │
│  RMSNorm                            AdaRMSNorm(time_cond) ← Time│
│       ↓                                  ↓                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │         SHARED ATTENTION MECHANISM                 │         │
│  │                                                     │         │
│  │  q_vlm = W_q0 @ x_vlm     q_action = W_q1 @ x_action        │
│  │  k_vlm = W_k0 @ x_vlm     k_action = W_k1 @ x_action        │
│  │  v_vlm = W_v0 @ x_vlm     v_action = W_v1 @ x_action        │
│  │                                                     │         │
│  │  Q = concat([q_vlm, q_action])  ← All queries      │         │
│  │  K = concat([k_vlm, k_action])  ← All keys         │         │
│  │  V = concat([v_vlm, v_action])  ← All values       │         │
│  │                                                     │         │
│  │  Attention = softmax(Q @ K^T + mask) @ V           │         │
│  │                           ↑                         │         │
│  │                    Mask controls one-way flow      │         │
│  │                                                     │         │
│  │  Split: [out_vlm, out_action]                      │         │
│  └────────────────────────────────────────────────────┘         │
│       ↓                                  ↓                       │
│  FFN with W_0                       FFN with W_1                │
│       ↓                                  ↓                       │
│  output_vlm                         output_action               │
│                                                                  │
│  (Repeat for 18 layers)                                         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
├──────────────────────────────────────────────────────────────────┤
│ output_action [16, 2048] → Linear → v_t [16, 8]                │
│                                                                  │
│ Loss: MSE(v_t, noise - actions)                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Code Location Reference

| Component | File | Function/Class |
|-----------|------|----------------|
| Action embedding | `pi0.py` | `embed_suffix()` |
| Timestep encoding | `pi0.py` | `posemb_sincos()` |
| Dual-expert layer | `gemma.py` | `BlockGated` |
| Shared attention | `gemma.py` | `Attention.__call__()` |
| AdaRMSNorm | `gemma.py` | `AdaRMSNorm` |
| Attention mask | `pi0.py` | `make_attn_mask()` |
| Loss computation | `pi0.py` | `compute_loss()` |

---

## Summary

The action expert is NOT a separate flow-matching head - it's a **full dual-expert transformer** where:

1. ✅ **Action expert IS transformer layers** (attention + FFN)
2. ✅ **Uses separate parameters** (W_q1, W_k1, W_v1, FFN1)
3. ✅ **Shares attention computation** via concatenated Q, K, V
4. ✅ **One-way communication** via attention mask
5. ✅ **Time-conditioned** via AdaRMSNorm
6. ✅ **Flow matching** is the training objective, not architecture

The input to action expert is:
- **Primary**: Noisy actions (linearly projected)
- **Conditioning**: Timestep (via AdaRMSNorm)
- **Context**: VLM embeddings (via attention mechanism)

This design allows the action expert to leverage rich visual-language understanding from the VLM while maintaining its own specialized parameters for action prediction!
