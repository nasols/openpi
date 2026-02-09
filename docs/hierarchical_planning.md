# Hierarchical Planning (HI-Robot) in OpenPI

OpenPI supports hierarchical planning where the VLA model generates intermediate subtasks on-the-fly during inference. This simpler, more flexible approach enables the model to break down complex tasks dynamically.

## How It Works

1. **User provides high-level goal**: "pick up the red cup and place it on the table"
2. **Model generates subtasks on-demand**: Every N steps (or when visually complete), the model generates the next immediate action
3. **Model executes subtask**: Uses the generated subtask as the prompt for action generation
4. **Repeats**: Generates new subtask, executes, repeats until task complete

## Completion Checking Modes

### 1. Step Count (Simple)
- Generates new subtask every N steps (fixed interval)
- Fast, no extra computation
- Good for testing and simple tasks

### 2. Visual Similarity (Adaptive - Recommended)
- Uses CLIP/vision encoder to check if subtask looks complete
- Compares current observation with subtask description
- Generates new subtask only when current one appears done
- More adaptive and efficient than fixed step counts

## Key Advantages

- **No separate model**: Uses the same PaliGemma model for both subtask generation and action generation - just different prompt templates
- **Adaptive**: Generates subtasks based on current observation, not rigid pre-planning
- **Simple**: Just 3 parameters - a flag, a template, and a refresh interval
- **Dual training**: Can train the text generation and action generation together end-to-end

## Usage

### Step Count Mode (Simple)

```python
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig

# Load policy
train_config = TrainConfig.from_checkpoint("checkpoints/my_model")

policy = create_trained_policy(
    train_config=train_config,
    checkpoint_dir="checkpoints/my_model",
    hierarchical_mode=True,
    subtask_template="Goal: {prompt}\\n\\nNext immediate action:",
    completion_check_mode="step_count",  # Fixed interval
    subtask_refresh_steps=10,  # New subtask every 10 steps
)

# Use policy normally
obs = {"image": image, "state": state, "prompt": "pick up the cup"}
result = policy.infer(obs)
actions = result["actions"]
```

### Visual Similarity Mode (Adaptive - Recommended)

```python
policy = create_trained_policy(
    train_config=train_config,
    checkpoint_dir="checkpoints/my_model",
    hierarchical_mode=True,
    subtask_template="Goal: {prompt}\\n\\nNext immediate action:",
    completion_check_mode="visual_similarity",  # CLIP-based
    completion_threshold=0.75,  # Similarity threshold (0-1)
    min_steps_per_subtask=3,  # Wait at least 3 steps
)

# Policy will automatically check if subtask is complete by:
# 1. Extracting vision features from current observation
# 2. Computing similarity with subtask text description
# 3. If similarity > threshold, generate new subtask
# 4. Otherwise keep executing current subtask
```

## How It Works Internally

### Step Count Mode:

1. **Step 0**: Policy receives high-level prompt, generates first subtask using `model.generate_text()` with subtask_template
2. **Steps 1-9**: Policy uses first subtask as prompt for action generation via `model.sample_actions()`
3. **Step 10**: New subtask generated based on current observation
4. **Steps 11-19**: Execute new subtask
5. **Repeats** until episode ends

### Visual Similarity Mode:

1. **Step 0**: Generate first subtask
2. **Steps 1-2**: Execute subtask (wait for min_steps_per_subtask)
3. **Step 3+**: Check completion every step:
   - Extract vision features from current observation
   - Compute cosine similarity with subtask text
   - If similarity > threshold → subtask complete, generate new one
   - If similarity < threshold → keep executing current subtask
4. **Repeats** adaptively based on visual feedback

## Requirements

The model must implement a `generate_text()` method for subtask generation. If not implemented, the policy falls back to using the original high-level prompt.

## When to Use

Use hierarchical mode when:
- Tasks are complex multi-step procedures
- You want the model to dynamically adapt its plan based on observations
- You want to enable dual training of planning + execution

**Recommended**: Use `completion_check_mode="visual_similarity"` for adaptive behavior that responds to actual task completion rather than arbitrary time limits.

Don't use when:
- Tasks are simple single-step actions
- You want maximum inference speed (subtask generation + CLIP checking adds overhead)

## Example

See [examples/hi_robot_simple_example.py](examples/hi_robot_simple_example.py) for a complete working example.

## Technical Details

- **No extra memory**: Reuses the same model loaded for action generation
- **Minimal overhead**: Only generates text every N steps (configurable)
- **Stateful**: Policy maintains subtask state across infer() calls
- **Reset**: Call `policy.reset_hierarchical_state()` between episodes
