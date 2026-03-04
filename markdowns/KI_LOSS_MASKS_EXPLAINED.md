# Knowledge Insulation: Loss Masks Explained

## Your Question

**Q**: "The `loss_mask` from FASTTokenizer is for FAST tokens, but we also compute loss on continuous actions from the action expert. Is there another mask for the action expert loss?"

**A**: **No, there's no mask for the action expert loss!** Here's why:

---

## Two Fundamentally Different Losses

In KI training, there are **two separate losses** operating on **different representations**:

### 1. VLM FAST Token Loss (Uses `loss_mask`)

```python
# What it operates on:
tokenized_prompt = [
    # Prefix: Don't compute loss here
    [BOS, "Task:", "pick", "up", "cup", "State:", "191", "89", ...],
    
    # Postfix: COMPUTE LOSS HERE
    ["Action:", 256872, 256857, 256669, ..., "|", EOS]
]

# Why we need a mask:
# We only want loss on FAST action tokens, NOT on prompt/state tokens
loss_mask = [False, False, False, ..., True, True, True, ..., False]
#            ↑ Don't compute loss      ↑ Compute loss on FAST   ↑ Don't compute on EOS

# Loss computation:
fast_loss = cross_entropy_loss(
    logits=vlm_output,
    targets=tokenized_prompt[:, 1:],  # Next-token prediction
    mask=loss_mask[:, 1:]              # ← USES MASK
)
```

**Why mask is needed**: The token sequence contains mixed content (prompt + state + FAST tokens), and we only want to train the VLM on predicting FAST tokens.

### 2. Action Expert Flow Matching Loss (No mask needed!)

```python
# What it operates on:
noisy_actions = [batch, 16, 8]  # Pure action sequence, nothing else
predicted_velocity = [batch, 16, 8]  # Model's prediction
true_velocity = [batch, 16, 8]  # Ground truth (noise - actions)

# Loss computation:
action_loss = jnp.mean(jnp.square(predicted_velocity - true_velocity), axis=-1)
# NO MASK NEEDED! ↑
```

**Why no mask is needed**: 
1. The action expert operates on a **pure action sequence** (no prompts, no state tokens mixed in)
2. We want loss on **every action token** in the horizon
3. There's nothing to mask out!

---

## Detailed Comparison

### VLM Loss Input Structure

```python
# Sequence passed to VLM
[Image tokens] + [Text tokens] + [FAST tokens]
[    512     ] + [    4       ] + [   48      ] = 564 tokens total
     ↓               ↓               ↓
  Don't compute   Don't compute   COMPUTE LOSS
  loss here       loss here       HERE ONLY!

# This is why we need loss_mask:
loss_mask = [False] * 516 + [True] * 48
```

### Action Expert Loss Input Structure

```python
# Sequence passed to Action Expert
[Action token 0] [Action token 1] ... [Action token 15]
       ↓                ↓                      ↓
   Compute loss    Compute loss         Compute loss
   
# All tokens are actions, compute loss on all!
# No mask needed!
```

---

## Code Walkthrough in `pi0.py`

Let me show you exactly where each loss is computed:

### VLM FAST Token Loss (With Mask)

```python
# File: pi0.py, compute_loss() with KI enabled

if self.knowledge_insulation and self.pi05 and train:
    # === VLM FAST Token Prediction ===
    
    # Forward through VLM only
    (prefix_out_for_fast, _), _ = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions,
    )
    
    # Get FAST token targets from observation
    if observation.tokenized_prompt is not None and observation.token_loss_mask is not None:
        # Create one-hot targets
        fast_token_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],  # Next-token prediction
            self.PaliGemma.llm.module.vocab_size
        )
        
        # Decode VLM output to vocabulary logits
        fast_logits = self.PaliGemma.llm(pre_logits=prefix_out_for_fast[:, :-1])[0]
        fast_logp = jax.nn.log_softmax(fast_logits, axis=-1)
        
        # ===== THIS IS WHERE loss_mask IS USED =====
        fast_loss_mask = observation.token_loss_mask[:, 1:]  # ← From FASTTokenizer
        fast_token_pplx = jnp.sum(fast_token_targets * fast_logp, axis=-1)
        
        # Compute loss ONLY on masked tokens (FAST action tokens)
        fast_loss = -jnp.sum(fast_token_pplx * fast_loss_mask, axis=-1) / jnp.clip(jnp.sum(fast_loss_mask, -1), 1)
        #                                      ↑ MASK APPLIED HERE
        fast_loss = jnp.mean(fast_loss)
    else:
        fast_loss = 0.0
```

### Action Expert Flow Matching Loss (No Mask)

```python
    # === Action Expert Flow Matching Loss ===
    
    # Detach prefix (gradient stopping)
    prefix_tokens_detached = jax.lax.stop_gradient(prefix_tokens)
    
    # Prepare full input (detached prefix + suffix)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    
    # Forward pass
    (_, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens_detached, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond]
    )
    
    # Project to action space
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    
    # ===== ACTION LOSS: NO MASK USED =====
    action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    #             ↑ Simple MSE, no masking!
    
    # Combine losses
    total_loss = action_loss + self.ki_fast_loss_weight * fast_loss
    return total_loss
```

---

## Why This Design Makes Sense

### Different Input Structures

| Loss Type | Input Contains | Need Mask? |
|-----------|---------------|------------|
| **VLM FAST** | Prompt + State + FAST tokens | ✅ Yes - mask out prompt/state |
| **Action Expert** | Only action tokens | ❌ No - all tokens are valid |

### Different Objectives

| Loss Type | What It Learns | Loss Function |
|-----------|----------------|---------------|
| **VLM FAST** | Next-token prediction on actions | Cross-entropy (categorical) |
| **Action Expert** | Velocity prediction for denoising | MSE (regression) |

### Different Masking Needs

**VLM FAST Loss**:
```python
# Problem: Mixed content in sequence
Sequence: [prompt_tok, prompt_tok, FAST_tok, FAST_tok, FAST_tok]
Loss on:  [   False  ,    False  ,   True   ,   True   ,   True   ]
#         Don't train on prompt    Train on actions!

# Solution: Use loss_mask to select which tokens to compute loss on
```

**Action Expert Loss**:
```python
# No problem: Pure action sequence
Sequence: [action, action, action, action, action]
Loss on:  [ True ,  True ,  True ,  True ,  True ]
#         Compute loss on all actions!

# Solution: No mask needed - loss on entire sequence
```

---

## Visual Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE INSULATION                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PATH 1: VLM FAST Token Loss                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Input:  tokenized_prompt [256]                                │
│         ├─ Prompt tokens [0:15]     ← Don't compute loss      │
│         ├─ State tokens [15:23]     ← Don't compute loss      │
│         └─ FAST tokens [23:71]      ← COMPUTE LOSS HERE       │
│                                                                │
│ Mask:   loss_mask [256]                                       │
│         [False, False, ..., True, True, ..., False]           │
│          ↑ 0:23 = False    ↑ 23:71 = True  ↑ 71:256 = False  │
│                                                                │
│ Loss:   cross_entropy(logits, targets, mask=loss_mask)        │
│         Only tokens where loss_mask=True contribute           │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PATH 2: Action Expert Flow Matching Loss                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Input:  noisy_actions [16, 8]                                 │
│         ├─ Action t=0  ← Compute loss                         │
│         ├─ Action t=1  ← Compute loss                         │
│         ├─ ...                                                 │
│         └─ Action t=15 ← Compute loss                         │
│                                                                │
│ Mask:   NONE!                                                 │
│         All actions are valid targets                         │
│                                                                │
│ Loss:   mse(predicted_velocity, true_velocity)                │
│         All action timesteps contribute equally               │
│                                                                │
└────────────────────────────────────────────────────────────────┘

                              ↓
                              
┌────────────────────────────────────────────────────────────────┐
│ Combined Loss                                                  │
│                                                                │
│ total_loss = action_loss + λ * fast_loss                      │
│              ↑ No mask      ↑ Uses loss_mask                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Common Misconceptions

### ❌ Misconception 1: "Both losses need the same mask"

**Reality**: They operate on different data structures:
- VLM loss: Operates on token sequence with mixed content
- Action loss: Operates on pure action array

### ❌ Misconception 2: "loss_mask controls both losses"

**Reality**: `loss_mask` is ONLY for VLM FAST token loss. Action expert loss doesn't use it.

### ❌ Misconception 3: "We need a mask to ignore bad actions"

**Reality**: In flow matching, we compute loss on the entire action sequence. The model learns to denoise all timesteps.

---

## Where Are Masks Used?

| Mask Type | Source | Used For | Used In |
|-----------|--------|----------|---------|
| `loss_mask` | FASTTokenizer | VLM FAST token loss | `compute_loss()` VLM path |
| `token_mask` | FASTTokenizer | Padding (valid tokens) | Attention computation |
| `ar_mask` | FASTTokenizer | Autoregressive attention | `make_attn_mask()` |
| `attn_mask` | `make_attn_mask()` | Attention masking | Transformer attention |
| _(none)_ | - | Action expert loss | No mask needed! |

---

## Key Takeaways

1. ✅ **`loss_mask` is ONLY for VLM FAST token loss**
   - Masks out prompt/state tokens
   - Only computes loss on FAST action tokens

2. ✅ **Action expert loss uses NO mask**
   - Operates on pure action sequence
   - Computes loss on all action timesteps
   - No need to mask anything out

3. ✅ **Two losses, different structures**
   - VLM: Token sequence with mixed content → needs mask
   - Action Expert: Pure actions → no mask needed

4. ✅ **The masks are independent**
   - `loss_mask` doesn't affect action expert
   - Action expert loss doesn't need any mask
   - They're separate loss pathways

---

## Analogy

Think of it like training two students:

**Student 1 (VLM)**: Given a test with multiple sections
```
Test: [Instructions] [Example] [Questions] [Answers]
Grade: [  No Grade  ] [  No  ] [ No     ] [ Yes!  ]
                                           ↑ Use loss_mask to grade only this part
```

**Student 2 (Action Expert)**: Given a pure math worksheet
```
Worksheet: [Problem 1] [Problem 2] [Problem 3] [Problem 4]
Grade:     [  Grade  ] [  Grade  ] [  Grade  ] [  Grade  ]
           ↑ Grade everything! No mask needed.
```

Both students are evaluated, but Student 1 needs a mask to specify which part to grade, while Student 2's entire worksheet is valid for grading.

---

## Implementation Checklist

When implementing KI, remember:

- [ ] FASTTokenizer creates `loss_mask` for VLM FAST loss
- [ ] VLM FAST loss uses `loss_mask` to compute loss only on FAST tokens
- [ ] Action expert loss computes MSE on entire action sequence
- [ ] Action expert loss does NOT use any mask
- [ ] Two losses are combined: `total_loss = action_loss + λ * fast_loss`
- [ ] No additional masking needed for action expert!

---

## Summary

**Your original question**: "Is there another mask for the action expert loss?"

**Answer**: No! The action expert loss doesn't need a mask because:
1. It operates on a pure action sequence (no mixed content)
2. We want loss on every action timestep
3. It's a regression loss (MSE), not a token prediction loss

The `loss_mask` from FASTTokenizer is exclusively for the VLM FAST token loss, which needs to distinguish between prompt tokens (don't compute loss) and FAST action tokens (do compute loss).

The two losses are fundamentally different in structure and purpose, so they have different masking requirements!
