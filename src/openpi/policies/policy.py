import logging

from openpi.models.pi0_fast import make_attn_mask
logger = logging.getLogger("openpi")

from collections.abc import Sequence
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.models import tokenizer as _tokenizer
from openpi.models.pi05 import Pi05

import copy

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        hierarchical_mode: bool = False,
  
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            hierarchical_mode: If True, use hierarchical planning (HI-robot style).
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        
        # Hierarchical planning state
        self._hierarchical_mode = hierarchical_mode
        self._current_subtask = None
        self._original_prompt = None                                    

        if self._hierarchical_mode: 
            self._tokenizer = _tokenizer.PaligemmaTokenizer(max_len=model.max_token_len) 

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        ### HIERARCHICAL PLANNING ###
        #############################
        # prompt = obs.get("prompt", None)
        # if self._hierarchical_mode and prompt is not None:
        #     _subtask = self._model.generate_subtask(observation, prompt)
        #     logger.log(level=103, msg=f"[HI-Robot] Generated subtask: {_subtask}")

        ############################
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata



    def _check_subtask_completion(self, inputs: dict, subtask: str) -> bool:
        """Check if current subtask is visually complete using CLIP/vision encoder.
        
        Args:
            inputs: Current observation with image
            subtask: Current subtask description
            
        Returns:
            True if subtask appears complete (high similarity), False otherwise
        """
        # Try to use model's vision encoder for similarity
        if hasattr(self._model, 'compute_visual_text_similarity'):
            try:
                similarity = self._model.compute_visual_text_similarity(
                    inputs["image"], 
                    subtask
                )
                is_complete = similarity > self._completion_threshold
                if is_complete:
                    logger.log(level=103, msg=f"[HI-Robot] Subtask complete (similarity={similarity:.3f})")
                return is_complete
            except Exception as e:
                logging.warning(f"[HI-Robot] Visual completion check failed: {e}")
                return False
        
        # Fallback: Try to access vision encoder directly from PaliGemma
        elif hasattr(self._model, 'paligemma_with_expert'):
            try:
                paligemma = self._model.paligemma_with_expert
                if hasattr(paligemma, 'vision_encoder'):
                    # Get image embedding
                    image = inputs["image"]
                    if self._is_pytorch_model:
                        if not isinstance(image, torch.Tensor):
                            image = torch.from_numpy(np.array(image)).to(self._pytorch_device)
                        if image.ndim == 3:  # Add batch dimension
                            image = image.unsqueeze(0)
                        
                        # Get vision features
                        with torch.no_grad():
                            vision_features = paligemma.vision_encoder(image)
                            # Use text encoder if available
                            if hasattr(paligemma, 'text_encoder'):
                                text_features = paligemma.text_encoder([subtask])
                                # Compute cosine similarity
                                similarity = torch.nn.functional.cosine_similarity(
                                    vision_features.mean(dim=1),  # Pool spatial dimensions
                                    text_features,
                                    dim=-1
                                ).item()
                                is_complete = similarity > self._completion_threshold
                                if is_complete:
                                    logger.log(level=103, msg=f"[HI-Robot] Subtask complete (similarity={similarity:.3f})")
                                return is_complete
            except Exception as e:
                logger.warning(f"[HI-Robot] Direct vision encoder access failed: {e}")
                return False
        
        # No vision encoder available, fallback to step count
        logger.debug("[HI-Robot] Visual completion check not available, using step count")
        return False
    
    def reset_hierarchical_state(self):
        """Reset hierarchical planning state (call between episodes)."""
        self._current_subtask = None
        self._original_prompt = None
        self._step_count = 0


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
