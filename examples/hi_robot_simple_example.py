"""
Simple example of using hierarchical planning (HI-robot) in OpenPI.

This demonstrates the simplified hierarchical mode that uses a single flag
and the model's text generation capability to generate intermediate subtasks.

Two completion modes are available:
1. Step-count based: Refresh subtask every N steps (simple)
2. Visual similarity based: Use CLIP to check if subtask is done (adaptive)
"""

import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig


def example_step_count_mode():
    """Example with simple step-count based subtask refresh."""
    checkpoint_dir = "path/to/your/checkpoint"
    train_config = TrainConfig.from_checkpoint(checkpoint_dir)
    
    # Create policy with time-based subtask refresh
    policy = create_trained_policy(
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        hierarchical_mode=True,
        subtask_template="Goal: {prompt}\n\nNext immediate action:",
        subtask_refresh_steps=10,  # New subtask every 10 steps
        completion_check_mode="step_count",  # Simple time-based
    )
    
    return policy


def example_visual_similarity_mode():
    """Example with CLIP-based visual completion detection (recommended)."""
    checkpoint_dir = "path/to/your/checkpoint"
    train_config = TrainConfig.from_checkpoint(checkpoint_dir)
    
    # Create policy with CLIP-based completion checking
    policy = create_trained_policy(
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        hierarchical_mode=True,
        subtask_template="Goal: {prompt}\n\nNext immediate action:",
        completion_check_mode="visual_similarity",  # Use CLIP/vision encoder
        completion_threshold=0.75,  # Similarity threshold (0-1)
        min_steps_per_subtask=3,  # Wait at least 3 steps before checking
    )
    
    return policy


def main():
    # Choose which mode to use
    policy = example_visual_similarity_mode()  # Adaptive, recommended
    # policy = example_step_count_mode()  # Simple, faster
    
    # High-level task prompt
    high_level_prompt = "pick up the red cup and place it on the table"
    
    # Example observation
    observation = {
        "image": np.random.randn(224, 224, 3),  # Your camera image
        "state": np.array([0.1, 0.2, 0.3, 0.0]),  # Robot state (pos + gripper)
        "prompt": high_level_prompt,
    }
    
    # Run inference loop
    for step in range(50):
        # With visual_similarity mode:
        # - Policy checks if current subtask looks complete using CLIP
        # - If similarity(observation, subtask) > threshold, generate new subtask
        # - Much more adaptive than fixed step counts!
        
        result = policy.infer(observation)
        actions = result["actions"]
        
        print(f"Step {step}: Actions = {actions}")
        
        # Apply actions to your robot/simulation
        # observation = env.step(actions)
    
    # Reset state between episodes
    policy.reset_hierarchical_state()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
