import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

from openpi.models.pi05_config import Pi05Config
from openpi.models.pi05 import Pi05

logger = logging.getLogger("openpi")


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        hierarchical_mode: If True, use hierarchical planning (HI-robot) - model generates intermediate
                         subtasks and conditions actions on them.
        subtask_template: Template for generating subtasks. Use {prompt} for original goal.
                        Default: "Goal: {prompt}\\n\\nNext immediate action:"
        subtask_refresh_steps: Regenerate subtask every N steps (only used if completion_check_mode="step_count").
        completion_check_mode: How to check subtask completion:
            - "step_count": Refresh after fixed number of steps (simple, fast)
            - "visual_similarity": Use CLIP/vision encoder to check if subtask looks complete (adaptive)
        completion_threshold: For visual_similarity mode, similarity score above this triggers new subtask (0-1).
        min_steps_per_subtask: Minimum steps before checking completion (avoid premature subtask changes).

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safetensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    assert isinstance(train_config.model, Pi05Config), "Expect a PI05 Config as we are"

    logger.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model : Pi05  = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    
    if train_config.model.ki_mode:
        model.ki_mode = train_config.model.ki_mode
            
    if train_config.model.hi_mode:
        model.hi_mode = train_config.model.hi_mode

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # Build input transforms
    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    policy = _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        hi_mode=train_config.model.hi_mode,
        ki_mode=train_config.model.ki_mode,
    )
    
    
    return policy
