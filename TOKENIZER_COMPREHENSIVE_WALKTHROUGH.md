# Comprehensive Tokenizer Walkthrough for π0/π0.5 Models

## Overview: Three Tokenization Strategies

The OpenPI codebase has **three different tokenization strategies** depending on the model architecture:

| Strategy | Model Type | Text Tokenizer | Action Representation | Use Case |
|----------|------------|----------------|----------------------|----------|
| **1. Standard** | PI0, PI05 | PaliGemmaTokenizer | **Continuous** (flow matching) | Standard training |
| **2. FAST-only** | PI0_FAST | FASTTokenizer | **Discrete** (FAST tokens) | Fully autoregressive |
| **3. Knowledge Insulation** | PI05_KI | FASTTokenizer | **Both discrete AND continuous** | Dual training |

---

## Tokenizer #1: PaliGemmaTokenizer

**Location**: `openpi/src/openpi/models/tokenizer.py` (lines 14-49)

### What It Does

- Tokenizes **text prompts only** (optionally with discretized state for π0.5)
- Actions remain **continuous** (not tokenized)
- Used for standard π0 and π0.5 training

### Code Structure

```python
class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        # Downloads SentencePiece tokenizer from big_vision
        self._tokenizer = sentencepiece.SentencePieceProcessor(...)
    
    def tokenize(self, prompt: str, state: np.ndarray | None = None):
        # Returns: (tokens, mask)
        # Does NOT tokenize actions
```

### Two Modes

#### Mode 1: π0 (state is continuous)
```python
# Input
prompt = "pick up the cup"
state = None  # State stays in continuous form

# Tokenization
tokens = tokenizer.encode("pick up the cup\n", add_bos=True)
# Output: [1, 1234, 567, 89, 123, 456, 108]  # token IDs
#         ↑ BOS token

# Result: Just text tokens, state and actions stay continuous
```

#### Mode 2: π0.5 (state is discretized into tokens)
```python
# Input
prompt = "pick up the cup"
state = [0.5, -0.3, 0.8, -0.1, 0.2, -0.5, 0.1]  # 7D continuous state

# Discretization (256 bins, range [-1, 1])
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
# Result: [191, 89, 230, 115, 153, 64, 140]  # bin indices

# Tokenization
state_str = "191 89 230 115 153 64 140"
full_prompt = f"Task: pick up the cup, State: {state_str};\nAction: "
tokens = tokenizer.encode(full_prompt, add_bos=True)

# Output: [1, 4567, ..., 890, 108]  # Many more tokens
#         ↑ BOS                 ↑ "Action: " tokens
```

---

## Tokenizer #2: FASTTokenizer

**Location**: `openpi/src/openpi/models/tokenizer.py` (lines 51-145)

### What It Does

- Tokenizes **both text AND actions**
- Actions converted to **discrete FAST tokens** using learned VQ-VAE
- Returns 4 outputs: `tokens, token_mask, ar_mask, loss_mask`

### Code Structure

```python
class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "..."):
        # PaliGemma tokenizer for text
        self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(...)
        
        # FAST tokenizer for actions (learned VQ-VAE from HuggingFace)
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, ...)
        
    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None):
        # Returns: (tokens, token_mask, ar_mask, loss_mask)
        # Tokenizes BOTH text and actions
```

### How It Works

#### Step 1: Tokenize Prefix (Text + State)

```python
# Input
prompt = "pick up the cup"
state = [0.5, -0.3, 0.8, -0.1, 0.2, -0.5, 0.1, 0.0]  # 8D state
actions = [[0.1, 0.2, ...], [0.15, 0.22, ...], ...]  # [16, 8] action chunk

# Discretize state (same as PaliGemma π0.5)
discretized_state = [191, 89, 230, 115, 153, 64, 140, 128]
state_str = "191 89 230 115 153 64 140 128"

# Create prefix
prefix = f"Task: pick up the cup, State: {state_str};\n"
prefix_tokens = paligemma_tokenizer.encode(prefix, add_bos=True)
# Result: [1, 4567, 890, 123, ..., 456]  # ~15-20 tokens
```

#### Step 2: Tokenize Actions with FAST

```python
# FAST tokenizer converts continuous actions to discrete tokens
# Uses learned VQ-VAE (Vector Quantized Variational AutoEncoder)
action_tokens = fast_tokenizer(actions[None])[0]
# Input:  [16, 8] continuous actions
# Output: [48] discrete token IDs (3 tokens per action dimension)
# Example: [0, 15, 203, 45, 67, 89, ...]

# Map FAST tokens to PaliGemma vocabulary space
# PaliGemma has ~257k tokens, last 128 are reserved for FAST
action_tokens_in_pg = 257000 - 128 - action_tokens
# Result: [256872, 256857, 256669, ...]  # High token IDs
```

#### Step 3: Create Postfix (Action Tokens)

```python
# Add "Action: " prefix and "|" suffix
postfix_tokens = (
    paligemma_tokenizer.encode("Action: ")  # [45678, 12345]
    + action_tokens_in_pg.tolist()           # [256872, 256857, ...]
    + paligemma_tokenizer.encode("|", add_eos=True)  # [789, 2]
)
# Result: [45678, 12345, 256872, 256857, ..., 789, 2]
#         ↑ "Action: "  ↑ FAST tokens           ↑ "|" + EOS
```

#### Step 4: Combine and Create Masks

```python
# Combine prefix + postfix
tokens = prefix_tokens + postfix_tokens
# [1, 4567, ..., 456, 45678, 12345, 256872, ..., 789, 2]

# Create masks
token_mask = [True] * len(tokens)  # All valid (not padding)
ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
# 0 = bidirectional attention (prefix)
# 1 = causal attention (postfix actions)

loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
# False = don't compute loss (prompt + state)
# True = compute loss (FAST action tokens)

# Pad to max_len (e.g., 256)
# ... padding logic ...

return tokens, token_mask, ar_mask, loss_mask
```

---

## Data Flow Comparison

### Flow 1: Standard π0/π0.5 (PaliGemmaTokenizer)

```
┌─────────────────────────────────────────────────────────────┐
│ Input Data from Dataset                                     │
│ ├─ images: [224, 224, 3]                                    │
│ ├─ prompt: "pick up the cup"                                │
│ ├─ state: [8] continuous                                    │
│ └─ actions: [16, 8] continuous                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ DroidInputs Transform                                       │
│ (Repackages to standard format)                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PaliGemmaTokenizer Transform                                │
│ Input:  prompt, state (optional)                            │
│ Output: tokenized_prompt, tokenized_prompt_mask             │
│                                                              │
│ Actions stay CONTINUOUS! Not tokenized.                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Passed to Model                                        │
│ {                                                            │
│   'image': {...},                                            │
│   'image_mask': {...},                                       │
│   'state': [8],              ← Continuous                    │
│   'actions': [16, 8],        ← Continuous                    │
│   'tokenized_prompt': [48],  ← Discrete (text only)          │
│   'tokenized_prompt_mask': [48]                              │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Forward Pass                                          │
│                                                              │
│ Prefix (VLM processes):                                     │
│   ├─ Images → SigLIP encoder → image tokens                 │
│   └─ tokenized_prompt → LLM embedding → text tokens         │
│                                                              │
│ Suffix (Action Expert processes):                           │
│   ├─ Noisy actions → linear projection → action tokens      │
│   └─ Timestep → sinusoidal encoding → time embedding        │
│                                                              │
│ Combined attention across prefix + suffix                   │
│                                                              │
│ Output: Predicted velocity v_t [16, 8]                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Loss: MSE(v_t, u_t)                                         │
│ ← Flow matching loss on CONTINUOUS actions                  │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- Text tokenized, actions stay continuous
- Single loss: flow matching on velocity prediction
- All parameters updated by same loss

---

### Flow 2: FAST-only Model (PI0_FAST)

```
┌─────────────────────────────────────────────────────────────┐
│ Input Data from Dataset                                     │
│ (Same as above)                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FASTTokenizer Transform                                     │
│ Input:  prompt, state, actions                              │
│ Output: tokenized_prompt [256]                              │
│         tokenized_prompt_mask [256]                         │
│         token_ar_mask [256]                                 │
│         token_loss_mask [256]                               │
│                                                              │
│ Actions are TOKENIZED into FAST tokens!                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Passed to Model                                        │
│ {                                                            │
│   'image': {...},                                            │
│   'image_mask': {...},                                       │
│   'state': [8],                                              │
│   'actions': [16, 8],        ← Still in data (for reference) │
│   'tokenized_prompt': [256], ← Text + State + FAST tokens    │
│   'tokenized_prompt_mask': [256],                            │
│   'token_ar_mask': [256],    ← Attention mask                │
│   'token_loss_mask': [256]   ← Loss mask (only on FAST)     │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Forward Pass (Fully Autoregressive)                  │
│                                                              │
│ All inputs processed as tokens:                             │
│   ├─ Images → SigLIP encoder → image tokens                 │
│   ├─ Text tokens → LLM embedding                             │
│   └─ FAST action tokens → LLM embedding                      │
│                                                              │
│ Single transformer processes everything                     │
│                                                              │
│ Output: Next token logits [256, vocab_size]                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Loss: Cross-Entropy(logits, next_token)                     │
│ ← Autoregressive language modeling loss                     │
│ ← Only on FAST action tokens (token_loss_mask)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Inference: Autoregressively generate FAST tokens            │
│ Then: FASTTokenizer.extract_actions() → continuous actions  │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- Everything is tokenized (text AND actions)
- Single loss: cross-entropy on next-token prediction
- Fully autoregressive generation
- No flow matching, no action expert

---

### Flow 3: Knowledge Insulation (PI05_KI) ⭐

```
┌─────────────────────────────────────────────────────────────┐
│ Input Data from Dataset                                     │
│ (Same as above)                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FASTTokenizer Transform                                     │
│ Input:  prompt, state, actions                              │
│ Output: tokenized_prompt [256]  ← Text + FAST action tokens │
│         tokenized_prompt_mask [256]                         │
│         token_ar_mask [256]                                 │
│         token_loss_mask [256]   ← Loss on FAST tokens only  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Passed to Model                                        │
│ {                                                            │
│   'image': {...},                                            │
│   'image_mask': {...},                                       │
│   'state': [8],              ← Continuous (still needed!)    │
│   'actions': [16, 8],        ← Continuous (still needed!)    │
│   'tokenized_prompt': [256], ← ALSO discrete FAST tokens     │
│   'tokenized_prompt_mask': [256],                            │
│   'token_ar_mask': [256],                                    │
│   'token_loss_mask': [256]                                   │
│ }                                                            │
│                                                              │
│ KEY: Both continuous actions AND FAST tokens present!       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Forward Pass (DUAL PATH)                             │
│                                                              │
│ PATH 1: VLM FAST Token Prediction                           │
│ ├─ Images → SigLIP → image tokens                           │
│ ├─ tokenized_prompt → LLM embed → text + FAST tokens        │
│ ├─ Forward through VLM expert (Expert 0)                    │
│ ├─ Decode to vocabulary: logits [256, vocab_size]           │
│ └─ Loss: Cross-Entropy on FAST tokens (token_loss_mask)     │
│                                                              │
│ PATH 2: Action Expert Flow Matching                         │
│ ├─ Detach prefix: stop_gradient(VLM output)                 │
│ ├─ Noisy continuous actions → action expert                 │
│ ├─ Forward through Action Expert (Expert 1)                 │
│ ├─ Output: Predicted velocity v_t [16, 8]                   │
│ └─ Loss: MSE(v_t, u_t)                                      │
│                                                              │
│ Total Loss = Action Loss + λ * FAST Loss                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Gradient Flow                                               │
│                                                              │
│ VLM Parameters ←─── FAST Cross-Entropy Loss                 │
│                     (learns discrete action prediction)     │
│                                                              │
│ Action Expert ←───── Flow Matching Loss                     │
│ Parameters          (learns continuous action refinement)   │
│                                                              │
│ stop_gradient() blocks action loss from updating VLM        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Inference: Use Action Expert Only                           │
│ ├─ VLM processes images + text → prefix embeddings          │
│ ├─ Action expert iteratively denoises actions (flow)        │
│ └─ Output: Continuous actions [16, 8]                       │
│                                                              │
│ FAST tokens only used during training for VLM!              │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- **BOTH** tokenization strategies used simultaneously!
- FASTTokenizer creates discrete targets for VLM
- Continuous actions still used for action expert
- Dual loss: cross-entropy (VLM) + MSE (action expert)
- Gradients isolated between experts

---

## Answering Your Questions

### Q1: "When using KI, do I need to use both tokenizers?"

**Answer: You use FASTTokenizer, but it creates BOTH representations!**

Here's what happens:

```python
# FASTTokenizer is called once
tokens, token_mask, ar_mask, loss_mask = fast_tokenizer.tokenize(
    prompt="pick up the cup",
    state=[0.5, -0.3, ...],
    actions=[[0.1, 0.2, ...], ...]  # Continuous actions
)

# This creates:
# 1. tokenized_prompt: Text + FAST discrete action tokens
# 2. token_loss_mask: Where to compute VLM loss

# BUT the original continuous actions are ALSO kept in the data!
data = {
    'actions': [[0.1, 0.2, ...], ...],  # ← Still here for action expert!
    'tokenized_prompt': [...],           # ← FAST tokens for VLM training
    'token_loss_mask': [...],
}
```

**So you have TWO representations of the same actions:**
1. **Discrete FAST tokens** (in `tokenized_prompt`) → for VLM training
2. **Continuous actions** (in `actions`) → for action expert training

### Q2: "Which tokenizer for VLM vs Action Expert?"

**Answer: FASTTokenizer provides data for BOTH**

```python
# In model.compute_loss() with KI enabled:

# For VLM (uses FAST tokens):
fast_token_targets = observation.tokenized_prompt[:, 1:]  # Next-token prediction
fast_loss_mask = observation.token_loss_mask[:, 1:]
fast_loss = cross_entropy(vlm_logits, fast_token_targets, fast_loss_mask)

# For Action Expert (uses continuous actions):
u_t = noise - actions  # Flow matching target
v_t = action_expert_output
action_loss = mse(v_t, u_t)

# Both use data from same input, different fields!
```

---

## Visual Summary: Three Strategies Side-by-Side

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                 │   Standard π0.5  │   FAST-only      │   KI π0.5        │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Tokenizer       │ PaliGemma        │ FAST             │ FAST             │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Text Tokens     │ ✓ Discrete       │ ✓ Discrete       │ ✓ Discrete       │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Action Tokens   │ ✗ Continuous     │ ✓ Discrete       │ ✓ Discrete       │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Action Values   │ ✓ Continuous     │ ✗ Via decode     │ ✓ Continuous     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ VLM Loss        │ None             │ Cross-Entropy    │ Cross-Entropy    │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Action Loss     │ Flow Matching    │ None             │ Flow Matching    │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Experts         │ VLM + Action     │ VLM only         │ VLM + Action     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Gradient Flow   │ All params       │ All params       │ Isolated         │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## Code Location Reference

### Where Tokenizers Are Created

**Standard π0.5**:
```python
# File: training/config.py, ModelTransformFactory
case _model.ModelType.PI05:
    return _transforms.Group(
        inputs=[
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
            ),
        ],
    )
```

**FAST-only**:
```python
# File: training/config.py, ModelTransformFactory
case _model.ModelType.PI0_FAST:
    tokenizer = _tokenizer.FASTTokenizer(model_config.max_token_len)
    return _transforms.Group(
        inputs=[
            _transforms.TokenizeFASTInputs(tokenizer),
        ],
        outputs=[
            _transforms.ExtractFASTActions(tokenizer, ...),
        ],
    )
```

**Knowledge Insulation** (to be added):
```python
# File: training/config.py, ModelTransformFactory
case _model.ModelType.PI05_KI:
    tokenizer = _tokenizer.FASTTokenizer(model_config.max_token_len)
    return _transforms.Group(
        inputs=[
            _transforms.TokenizeFASTInputs(tokenizer),
            # Note: NO ExtractFASTActions in outputs!
            # Actions stay continuous for action expert
        ],
    )
```

### Where Tokenizers Are Applied

**Transform Classes**:
```python
# File: transforms.py

class TokenizePrompt(DataTransformFn):
    """Uses PaliGemmaTokenizer"""
    tokenizer: _tokenizer.PaligemmaTokenizer
    
    def __call__(self, data: DataDict):
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, 
                "tokenized_prompt": tokens, 
                "tokenized_prompt_mask": token_masks}

class TokenizeFASTInputs(DataTransformFn):
    """Uses FASTTokenizer"""
    tokenizer: _tokenizer.FASTTokenizer
    
    def __call__(self, data: DataDict):
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(
            prompt, state, actions
        )
        return {**data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
                "token_ar_mask": ar_mask,
                "token_loss_mask": loss_mask}
```

### Where Data Is Used in Model

**Standard Training** (`pi0.py`):
```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # observation.tokenized_prompt → VLM embedding
    # actions (continuous) → Action expert
    
    prefix_tokens = embed_prefix(observation)  # Uses tokenized_prompt
    suffix_tokens = embed_suffix(observation, x_t, time)  # Uses continuous actions
    
    # Flow matching loss
    loss = mse(v_t, u_t)
```

**KI Training** (`pi0.py` with KI):
```python
def compute_loss(self, rng, observation, actions, *, train=False):
    if self.knowledge_insulation:
        # VLM FAST loss (uses tokenized_prompt)
        fast_token_targets = observation.tokenized_prompt[:, 1:]
        fast_loss_mask = observation.token_loss_mask[:, 1:]
        fast_loss = cross_entropy(vlm_logits, fast_token_targets, fast_loss_mask)
        
        # Action expert loss (uses continuous actions)
        prefix_tokens_detached = stop_gradient(prefix_tokens)
        suffix_tokens = embed_suffix(observation, x_t, time)  # Uses continuous
        action_loss = mse(v_t, u_t)
        
        return action_loss + lambda_fast * fast_loss
```

---

## Key Takeaways

1. **Standard π0.5**: PaliGemmaTokenizer for text, actions stay continuous
2. **FAST-only**: FASTTokenizer for everything, fully autoregressive
3. **KI π0.5**: FASTTokenizer creates BOTH discrete (for VLM) and continuous (for action expert) simultaneously
4. **You only need ONE tokenizer for KI**: FASTTokenizer provides both representations
5. **Continuous actions are never removed**: They stay in the data dict even when FAST tokens are added

The magic of KI is that it maintains **dual representations** of actions:
- Discrete FAST tokens teach the VLM about action structure
- Continuous actions allow precise control via flow matching
- `stop_gradient()` keeps these learning objectives separate

---

## Next Steps for Implementation

Now that you understand how tokenizers work, you're ready to:

1. ✅ Understand that FASTTokenizer provides BOTH representations
2. ✅ Implement `PI05_KI` model type (uses FASTTokenizer but keeps continuous actions)
3. ✅ Model code already expects both `tokenized_prompt` and `actions` to exist
4. ✅ No need for two separate tokenizers - FASTTokenizer does everything!

The implementation in `KI_DATA_PIPELINE_IMPLEMENTATION.md` is correct - we just need to add `PI05_KI` model type and use `TokenizeFASTInputs` transform!
