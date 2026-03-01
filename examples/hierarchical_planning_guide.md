# Hierarchical Planning Implementation Guide

## Overview

The hierarchical planning (HI-Robot) structure is **already implemented** in your Pi0.5 model! You don't need to create a new class. The implementation uses the existing `Pi0` class with a `generate_subtask()` method that passes through the VLM to decompose tasks.

## Architecture

```
User Prompt: "Pick up the red cup and place it on the table"
     ↓
[Policy Layer - Hierarchical Mode]
     ↓
1. Tokenize decomposition prompt using Policy's tokenizer
2. Pass to VLM via Pi0.generate_subtask()
     ↓
[Pi0 Model - VLM Forward Pass]
     ↓
3. embed_prefix() - Embed images + decomposition prompt
4. PaliGemma.llm() - Autoregressive generation through VLM ONLY
5. Generate token IDs (greedy or sampled)
     ↓
[Policy Layer - Decode]
     ↓
6. Decode token IDs to text using Policy's tokenizer
7. Replace original prompt with subtask
8. sample_actions() - Generate actions conditioned on subtask
```

## Key Components

### 1. **Pi0.generate_subtask()** ([pi0.py](../src/openpi/models/pi0.py#L417-L489))
- **Input**: Observation with images + decomposition prompt
- **Process**: Passes through VLM (PaliGemma) only - NO action expert
- **Output**: List of token IDs representing the subtask
- **VLM Role**: Generates text autoregressively by predicting next tokens

### 2. **Policy.infer()** ([policy.py](../src/openpi/policies/policy.py#L126-L198))
- Handles hierarchical planning logic
- Creates decomposition prompt from template
- Tokenizes prompt using its own tokenizer (no JAX conflict!)
- Calls `model.generate_subtask()` to get token IDs
- Decodes tokens to text
- Uses subtask as new prompt for action generation

### 3. **Tokenizer** (Policy-owned)
- The tokenizer lives in the **Policy** class, not the model
- This avoids JAX instantiation conflicts
- Encodes decomposition prompts
- Decodes generated token IDs to text

## Usage Example

```python
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig

# Load your trained pi0.5 model
checkpoint_dir = "checkpoints/pi05_droid_ki/KI_test"
train_config = TrainConfig.from_checkpoint(checkpoint_dir)

# Create policy with hierarchical mode enabled
policy = create_trained_policy(
    train_config=train_config,
    checkpoint_dir=checkpoint_dir,
    hierarchical_mode=True,
    subtask_template="You are tasked with decomposing the following goal into a sub-task. Task: {prompt}; Sub-task:",
    subtask_refresh_steps=10,  # Generate new subtask every 10 steps
    completion_check_mode="step_count",  # Simple time-based refresh
)

# Use the policy
obs = {
    "image": camera_image,  # Your robot's camera
    "state": robot_state,   # Joint positions, gripper state
    "prompt": "Pick up the red cup and place it on the table"
}

# Step 1-10: Policy generates first subtask (e.g., "approach red cup")
for step in range(10):
    result = policy.infer(obs)
    actions = result["actions"]
    # Execute actions on robot
    obs = get_new_observation()

# Step 11-20: Policy generates second subtask (e.g., "grasp cup")
for step in range(10):
    result = policy.infer(obs)
    actions = result["actions"]
    # Execute actions on robot
    obs = get_new_observation()

# Reset between episodes
policy.reset_hierarchical_state()
```

## How VLM Decomposes Tasks

The VLM (PaliGemma) processes:

**Input to VLM:**
- Images from robot cameras (base, left wrist, right wrist)
- Decomposition prompt: "Task: pick up red cup; Sub-task:"

**VLM Processing:**
- Embeds images using SigLIP vision encoder
- Embeds text tokens
- Runs transformer forward pass (VLM expert only)
- Generates tokens autoregressively:
  - Predicts next token from vocabulary
  - Embeds predicted token
  - Concatenates to sequence
  - Repeats

**Output from VLM:**
- Token IDs: `[1234, 5678, 910, ...]`
- These are decoded to: "approach the red cup with gripper open"

## Customization Options

### 1. **Decomposition Prompt Template**
Change how the VLM decomposes tasks:

```python
# Shorter, action-focused
subtask_template = "Goal: {prompt}. Next action:"

# More detailed, reasoning-focused  
subtask_template = "High-level task: {prompt}. Break this into steps. Current step:"

# Context-aware
subtask_template = "Mission: {prompt}. Given the current scene, what should I do next?"
```

### 2. **Generation Parameters**
Modify in [pi0.py](../src/openpi/models/pi0.py#L417):

```python
# In generate_subtask() method:
max_tokens = 20        # Longer subtasks
temperature = 0.7      # More creative (0 = greedy, 1.0 = random)
```

### 3. **Refresh Strategy**

**Step Count (Simple):**
```python
completion_check_mode = "step_count"
subtask_refresh_steps = 10  # Fixed interval
```

**Visual Similarity (Future - Adaptive):**
```python
completion_check_mode = "visual_similarity"  
completion_threshold = 0.75  # CLIP similarity threshold
min_steps_per_subtask = 3    # Minimum steps before checking
```
*Note: Visual similarity mode requires implementing `compute_visual_text_similarity()` method*

## Benefits of This Approach

✅ **No separate class needed** - Uses existing Pi0 model  
✅ **Single model** - VLM handles both decomposition AND action generation  
✅ **No tokenizer conflicts** - Tokenizer lives in Policy, not Model  
✅ **Memory efficient** - Reuses same PaliGemma weights  
✅ **Flexible** - Easy to customize prompts and strategies  

## Training Considerations

The hierarchical mode is **inference-only**. During training:
- Model learns to predict actions from prompts
- Model learns language understanding through VLM
- At inference time, we leverage these capabilities for decomposition

To train the model to be BETTER at decomposition, you could:
1. Add hierarchical task datasets with subtask annotations
2. Add a text generation loss during training
3. Use the Knowledge Insulation framework to train VLM and action expert separately

## Troubleshooting

### Issue: Tokenizer instantiation error
**Solution**: Make sure tokenizer is only in Policy class, not in Model class. The Pi0 model should NOT have `self.tokenizer`.

### Issue: VLM generates nonsensical subtasks
**Solution**: 
- Adjust decomposition prompt template to be more specific
- Lower temperature for more deterministic generation
- Ensure VLM was trained on language tasks

### Issue: Subtasks change too frequently
**Solution**: Increase `subtask_refresh_steps` or implement visual completion checking

## Next Steps

1. ✅ **Test basic hierarchical mode** - Use the example code above
2. 🔄 **Customize prompts** - Experiment with decomposition templates
3. 🔄 **Evaluate performance** - Compare with non-hierarchical baseline
4. 🔜 **Implement visual completion** - Add CLIP-based subtask completion detection
5. 🔜 **Train for decomposition** - Add hierarchical task datasets

## Files Modified

- [src/openpi/models/pi0.py](../src/openpi/models/pi0.py#L417-L489) - Uncommented and improved `generate_subtask()`
- [src/openpi/policies/policy.py](../src/openpi/policies/policy.py#L1-L357) - Added tokenizer and hierarchical logic to `infer()`
