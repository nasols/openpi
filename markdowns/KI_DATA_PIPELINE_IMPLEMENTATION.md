# Knowledge Insulation: FAST Token Data Pipeline Implementation

## Overview

To enable Knowledge Insulation (KI) training for π0.5 DROID, we need to add FAST token generation to the data pipeline. The VLM expert needs to predict FAST tokens autoregressively, which requires:

1. **Tokenizing actions** into discrete FAST tokens
2. **Concatenating** FAST tokens with prompt tokens
3. **Creating loss masks** to only compute loss on FAST tokens (not prompts)
4. **Passing these through** the data pipeline to the model

---

## Current State

### ✅ What's Already Working

1. **Model code** (`pi0.py`):
   - Added `knowledge_insulation` and `ki_fast_token_loss_weight` parameters
   - `compute_loss()` expects `observation.tokenized_prompt` and `observation.token_loss_mask`
   - Implements dual-path training (VLM FAST loss + Action expert flow matching loss)

2. **Config** (`pi0_config.py`):
   - Added `knowledge_insulation: bool` parameter
   - Added `ki_fast_token_loss_weight: float` parameter

3. **Training config** (`config.py`):
   - Added `pi05_droid_ki` configuration with KI enabled

4. **FASTTokenizer** (`tokenizer.py`):
   - Already implemented and functional
   - Has `tokenize(prompt, state, actions)` method that returns:
     - `tokens`: Combined prompt + state + FAST action tokens
     - `token_mask`: Which tokens are valid (not padding)
     - `ar_mask`: Autoregressive mask (0 = bidirectional, 1 = causal)
     - `loss_mask`: Which tokens to compute loss on

### ❌ What's Missing

The **data pipeline** doesn't use FASTTokenizer for π0.5 KI training. Currently:

```python
# Current pi05_droid_ki config uses:
data_transforms=lambda model: _transforms.Group(
    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
    outputs=[droid_policy.DroidOutputs()],
)
```

This uses `ModelType.PI05`, which calls `ModelTransformFactory` that creates `TokenizePrompt` (PaliGemma tokenizer), not `TokenizeFASTInputs` (FAST tokenizer).

---

## Solution Architecture

We have **two options** for implementation:

### Option 1: Create New ModelType.PI05_KI ⭐ (Recommended)

**Pros**: Clean separation, explicit about KI mode  
**Cons**: Requires changes to ModelType enum

### Option 2: Extend ModelTransformFactory to Check KI Config

**Pros**: No new ModelType needed  
**Cons**: Transform factory needs to inspect config internals

**We'll implement Option 1** as it's cleaner and more explicit.

---

## Implementation Plan

### Step 1: Add PI05_KI to ModelType Enum

**File**: `openpi/src/openpi/models/model.py`

**Location**: Find the `ModelType` enum definition

**Change**:
```python
class ModelType(enum.Enum):
    PI0 = "pi0"
    PI05 = "pi05"
    PI0_FAST = "pi0_fast"
    PI05_KI = "pi05_ki"  # NEW: π0.5 with Knowledge Insulation
```

---

### Step 2: Update pi0_config.py to Return Correct ModelType

**File**: `openpi/src/openpi/models/pi0_config.py`

**Location**: Find the `model_type` property (should be around line 40-60)

**Current code** (approximately):
```python
@property
def model_type(self) -> _model.ModelType:
    return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0
```

**Change to**:
```python
@property
def model_type(self) -> _model.ModelType:
    if self.pi05:
        if self.knowledge_insulation:
            return _model.ModelType.PI05_KI
        return _model.ModelType.PI05
    return _model.ModelType.PI0
```

**Explanation**: Now the model will report `PI05_KI` as its type when both `pi05=True` and `knowledge_insulation=True`.

---

### Step 3: Add PI05_KI Case to ModelTransformFactory

**File**: `openpi/src/openpi/training/config.py`

**Location**: In `ModelTransformFactory.__call__()` method (around lines 107-160)

**Add new case**:
```python
case _model.ModelType.PI05_KI:
    # Knowledge Insulation: use FAST tokenizer for joint VLM token prediction
    assert isinstance(model_config, pi0_config.Pi0Config)
    tokenizer = _tokenizer.FASTTokenizer(model_config.max_token_len)
    return _transforms.Group(
        inputs=[
            _transforms.InjectDefaultPrompt(self.default_prompt),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizeFASTInputs(tokenizer),
            _transforms.PadStatesAndActions(model_config.action_dim),
        ],
        outputs=[
            # KI doesn't need ExtractFASTActions since model outputs are continuous actions
            # from the action expert, not FAST tokens
        ],
    )
```

**Where to insert**: After the `case _model.ModelType.PI05:` block, before `case _model.ModelType.PI0_FAST:`

**Full context** (showing where to add):
```python
case _model.ModelType.PI05:
    assert isinstance(model_config, pi0_config.Pi0Config)
    return _transforms.Group(
        inputs=[
            _transforms.InjectDefaultPrompt(self.default_prompt),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(model_config.action_dim),
        ],
    )
case _model.ModelType.PI05_KI:  # <-- ADD THIS CASE
    # ... code from above ...
case _model.ModelType.PI0_FAST:
    # ... existing PI0_FAST code ...
```

---

### Step 4: Update DroidInputs to Handle PI05_KI

**File**: `openpi/src/openpi/policies/droid_policy.py`

**Location**: In `DroidInputs.__call__()` method, the `match self.model_type:` block (around line 34)

**Current code**:
```python
match self.model_type:
    case _model.ModelType.PI0 | _model.ModelType.PI05:
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)
    case _model.ModelType.PI0_FAST:
        # ... PI0_FAST code ...
```

**Change to**:
```python
match self.model_type:
    case _model.ModelType.PI0 | _model.ModelType.PI05 | _model.ModelType.PI05_KI:
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)
    case _model.ModelType.PI0_FAST:
        # ... PI0_FAST code ...
```

**Explanation**: PI05_KI uses the same image configuration as PI05 (base + wrist camera).

---

### Step 5: Update Training Config to Use PI05_KI

**File**: `openpi/src/openpi/training/config.py`

**Location**: The `pi05_droid_ki` training config (around line 643)

**Current code**:
```python
TrainConfig(
    name="pi05_droid_ki",
    model=pi0_config.Pi0Config(
        action_horizon=50, 
        pi05=True,
        knowledge_insulation=True,
        ki_fast_token_loss_weight=1.0,                       
    ),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],  # <-- WRONG
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
),
```

**Change to**:
```python
TrainConfig(
    name="pi05_droid_ki",
    model=pi0_config.Pi0Config(
        action_horizon=50, 
        pi05=True,
        knowledge_insulation=True,
        ki_fast_token_loss_weight=1.0,                       
    ),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05_KI)],  # <-- FIXED
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
),
```

**Explanation**: Explicitly use `PI05_KI` model type in the data transforms.

---

## Data Flow with KI

### Input Data (from LeRobot/DROID dataset)
```python
{
    'observation/exterior_image_1_left': [224, 224, 3],  # Base camera
    'observation/wrist_image_left': [224, 224, 3],       # Wrist camera
    'observation/joint_position': [7],                    # Joint positions
    'observation/gripper_position': [1],                  # Gripper state
    'actions': [action_horizon, 8],                       # Action chunk
    'prompt': "pick up the cup",                          # Text instruction
}
```

### After DroidInputs Transform
```python
{
    'image': {
        'base_0_rgb': [224, 224, 3],
        'left_wrist_0_rgb': [224, 224, 3],
        'right_wrist_0_rgb': [224, 224, 3],  # Padding (all zeros, masked out)
    },
    'image_mask': {
        'base_0_rgb': True,
        'left_wrist_0_rgb': True,
        'right_wrist_0_rgb': False,
    },
    'state': [8],                    # [joint_position (7) + gripper_position (1)]
    'actions': [action_horizon, 8],  # Action chunk
    'prompt': "pick up the cup",
}
```

### After TokenizeFASTInputs Transform (NEW with KI)
```python
{
    'image': {...},  # Same as above
    'image_mask': {...},  # Same as above
    'state': [8],
    'actions': [action_horizon, 8],
    
    # NEW: Tokenized sequence for VLM
    'tokenized_prompt': [256],  # Combined: [prompt tokens] + [state tokens] + [FAST action tokens]
    'tokenized_prompt_mask': [256],  # Which tokens are valid (not padding)
    'token_ar_mask': [256],  # Autoregressive mask: 0 for prefix, 1 for actions
    'token_loss_mask': [256],  # Loss computed only on FAST action tokens (True)
}
```

### Token Sequence Breakdown

Example with `action_horizon=16`, `action_dim=8`:

```python
# Assume:
# - Prompt "pick up the cup" → 4 tokens
# - State [7 dims] → discretized to 7 tokens
# - Actions [16, 8] → FAST encoded to ~48 tokens (3 tokens per action dim)

tokenized_prompt = [
    # Prefix: Prompt + State (bidirectional attention)
    [1234, 5678, 9012, 3456],  # "Task: pick up the cup,"
    [101, 102, 103, 104, 105, 106, 107],  # "State: [discretized bins];"
    
    # Postfix: FAST action tokens (causal attention)
    [234567, 234568, 234569, ...],  # "Action: [FAST tokens]|"
    # ~48 FAST tokens total
    
    # Padding to max_token_len
    [0, 0, 0, ...]  # Padding
]

token_loss_mask = [
    # Don't compute loss on prompt
    [False, False, False, False],
    
    # Don't compute loss on state
    [False, False, False, False, False, False, False],
    
    # Compute loss on FAST action tokens
    [True, True, True, ...],  # 48 True values
    
    # Don't compute loss on padding
    [False, False, False, ...]
]
```

---

## What Happens in Model Training

### Standard π0.5 Training (without KI)

```python
# Forward pass
[Images + Text] → VLM → embeddings → Action Expert → predict v_t
                                                          ↓
                                                    MSE(v_t, u_t)
```

### π0.5 with Knowledge Insulation (NEW)

```python
# Forward Pass 1: VLM predicts FAST tokens
[Images + Text + FAST tokens] → VLM → logits
                                        ↓
                            Cross-entropy(logits, FAST tokens)
                                        ↓
                                   VLM FAST Loss

# Forward Pass 2: Action Expert predicts continuous actions
[Images + Text] → VLM → embeddings
                          ↓ stop_gradient()
                   Detached embeddings → Action Expert → predict v_t
                                                              ↓
                                                         MSE(v_t, u_t)
                                                              ↓
                                                         Action Loss

# Combined: Total Loss = Action Loss + λ * VLM FAST Loss
```

**Key**: 
- VLM trains to predict next FAST token (like language modeling)
- Action expert trains to predict velocity (like standard flow matching)
- Gradients from action loss don't update VLM (via `stop_gradient()`)

---

## Testing Strategy

### 1. Unit Test: FASTTokenizer

```python
# Test that tokenizer works correctly
from openpi.models.tokenizer import FASTTokenizer
import numpy as np

tokenizer = FASTTokenizer(max_len=256)
state = np.random.rand(8)
actions = np.random.rand(16, 8)
prompt = "pick up the cup"

tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize(prompt, state, actions)

print(f"Total tokens: {len(tokens)}")
print(f"Valid tokens: {token_mask.sum()}")
print(f"Loss computed on: {loss_mask.sum()} tokens")
print(f"Causal tokens: {ar_mask.sum()} tokens")

# Expected output:
# Total tokens: 256 (padded to max_len)
# Valid tokens: ~60-70 (prompt + state + FAST actions)
# Loss computed on: ~48 tokens (only FAST action tokens)
# Causal tokens: ~48 tokens (FAST actions are causal)
```

### 2. Integration Test: Data Pipeline

```python
# Test that data transforms produce correct output
from openpi.policies.droid_policy import DroidInputs, make_droid_example
from openpi.training.config import ModelTransformFactory
from openpi.models.pi0_config import Pi0Config
from openpi.models.model import ModelType

# Create fake data
data = make_droid_example()
data['actions'] = np.random.rand(16, 8)

# Apply transforms
model_config = Pi0Config(pi05=True, knowledge_insulation=True, action_horizon=16)
assert model_config.model_type == ModelType.PI05_KI  # Check model type is correct

# DroidInputs transform
droid_transform = DroidInputs(model_type=ModelType.PI05_KI)
data = droid_transform(data)

# Model transforms
model_transforms = ModelTransformFactory()(model_config)
for transform in model_transforms.inputs:
    data = transform(data)

# Check outputs
assert 'tokenized_prompt' in data
assert 'token_loss_mask' in data
assert 'token_ar_mask' in data
print("✓ Data pipeline produces required fields")
```

### 3. Training Test: Run One Step

```bash
# Run training for 1 step to verify everything works
cd /Users/jonasolsen/Documents/IKT/10_semester/PI-repo/openpi

python scripts/train.py \
    --config pi05_droid_ki \
    --num_steps 1 \
    --log_dir /tmp/ki_test

# Check logs for:
# - No errors about missing fields
# - Both fast_loss and action_loss are computed
# - Gradients flow correctly
```

---

## Summary of Changes

| File | Change | Lines |
|------|--------|-------|
| `models/model.py` | Add `PI05_KI` to `ModelType` enum | ~1 line |
| `models/pi0_config.py` | Update `model_type` property to return `PI05_KI` when KI enabled | ~5 lines |
| `training/config.py` | Add `PI05_KI` case to `ModelTransformFactory` | ~15 lines |
| `policies/droid_policy.py` | Add `PI05_KI` to image handling case | ~1 line |
| `training/config.py` | Update `pi05_droid_ki` config to use `PI05_KI` model type | ~1 line |

**Total**: ~23 lines of code changes across 4 files

---

## Expected Behavior After Implementation

1. **Data Loading**: FAST tokenizer converts actions to discrete tokens
2. **Token Sequence**: Combines prompt + state + FAST action tokens
3. **Loss Masks**: Only FAST action tokens contribute to VLM loss
4. **Training**: 
   - VLM learns to predict FAST tokens (next-token prediction)
   - Action expert learns continuous actions (flow matching)
   - Gradients isolated between experts

5. **Inference**: Model still uses action expert for continuous predictions (FAST tokens only used during training for VLM)

---

## Next Steps

After reviewing this plan:

1. **Confirm approach**: Does this strategy make sense?
2. **Implement changes**: I can make all 5 changes simultaneously
3. **Test**: Run unit tests and training step to verify
4. **Debug**: Address any issues that come up

Let me know when you're ready to proceed with implementation!
