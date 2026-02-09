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
        subtask_template: str | None = None,
        subtask_refresh_steps: int = 10,
        completion_check_mode: str = "step_count",
        completion_threshold: float = 0.75,
        min_steps_per_subtask: int = 3,
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
            subtask_template: Template for generating subtasks. Use {prompt} for original goal.
            subtask_refresh_steps: Regenerate subtask every N steps (only used if completion_check_mode="step_count").
            completion_check_mode: How to check subtask completion:
                - "step_count": Refresh after fixed number of steps (simple, fast)
                - "visual_similarity": Use CLIP/vision encoder to check if subtask looks complete (adaptive)
            completion_threshold: For visual_similarity mode, similarity score above this triggers new subtask (0-1).
            min_steps_per_subtask: Minimum steps before checking completion (avoid premature subtask changes).
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
        self._subtask_template = subtask_template or "Goal: {prompt}; \n Sub-task:"
        self._subtask_refresh_steps = subtask_refresh_steps
        self._completion_check_mode = completion_check_mode
        self._completion_threshold = completion_threshold
        self._min_steps_per_subtask = min_steps_per_subtask
        self._current_subtask = None
        self._original_prompt = None
        self._step_count = 0

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
        
        logger.log(level=103, msg=f'[Policy] Original prompt: {obs["prompt"]}')
        # Hierarchical planning: generate subtask if needed
        if self._hierarchical_mode and "prompt" in obs:
            logger.log(level=103, msg=f"[HI-Robot] Running HI-Robot pipeline!")
            if self._original_prompt is None: # Sets this once at start of episode. 
                self._original_prompt = obs["prompt"]
            
            # Check if we should generate a new subtask
            should_regenerate = False
            if self._current_subtask is None:
                should_regenerate = True
            elif self._step_count >= self._min_steps_per_subtask:
                if self._completion_check_mode == "step_count":
                    # Simple time-based refresh
                    should_regenerate = (self._step_count % self._subtask_refresh_steps == 0)
                elif self._completion_check_mode == "visual_similarity":
                    # CLIP-based completion check
                    is_complete = self._check_subtask_completion(obs, self._current_subtask)
                    should_regenerate = is_complete
            
            if should_regenerate:
                logger.log(level=103, msg="[HI-Robot] Generating new subtask...")
                inputs = jax.tree.map(lambda x: x, obs)
                passing_obs = self._input_transform(inputs)
                passing_obs = _model.Observation.from_dict(passing_obs)  # Ensure correct format
                self._current_subtask = self._generate_subtask(passing_obs)
                self._step_count = 0  # Reset counter for new subtask
                logger.log(level=103, msg=f"[HI-Robot] New subtask: {self._current_subtask}")
            else: 
                logger.log(level=103, msg=f"[HI-Robot] Continuing with current subtask: {self._current_subtask} (step count: {self._step_count})")
            
            # Replace prompt with current subtask for action generation
            obs["prompt"] = self._current_subtask
            self._step_count += 1

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
    
    def _generate_subtask(self, observation: dict) -> str:
        """Generate next subtask using the model's text generation."""
        # Create subtask generation prompt
        # subtask_prompt = self._subtask_template.format(prompt=self._original_prompt)
        
        # # Check if model has text generation capability
        # if hasattr(self._model, 'generate_text'):
        #     try:
        #         # Create observation for text generation
        #         text_gen_inputs = {**inputs, "prompt": subtask_prompt}
                
        #         if self._is_pytorch_model:
        #             # Convert to PyTorch tensors
        #             text_gen_inputs = jax.tree.map(
        #                 lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device),
        #                 text_gen_inputs
        #             )
        #             subtask = self._model.generate_text(text_gen_inputs, max_tokens=64)
        #         else:
        #             # JAX version
        #             text_gen_inputs = jax.tree.map(lambda x: jnp.asarray(x), text_gen_inputs)
        #             self._rng, gen_rng = jax.random.split(self._rng)
        #             subtask = self._model.generate_text(gen_rng, text_gen_inputs, max_tokens=64)
                
        #         return subtask.strip()
        #     except Exception as e:
        #         logging.warning(f"[HI-Robot] Text generation failed: {e}, using original prompt")
        #         return self._original_prompt
        # else:
        #     # Model doesn't have text generation yet, use original prompt

        #     logging.warning("[HI-Robot] Model lacks generate_text(), using original prompt")
        #     return self._original_prompt

        # Using the VLM and the sub-task prompt template, we generate the desired sub-task. 
        prefix_tokens, prefix_mask, prefix_ar_mask = self._model.embed_prefix(observation)
        # prefix attention mask. Separates the image+prompt+state tokens into a bi-directional block 
        
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask) 
        # Positions makes the blocks for the attention mechanism. 
        # Meaning all tokens of lower index can be attended to. Something like that. 
        position = jnp.cumsum(prefix_mask, axis=1) - 1 
        
        # Runs a forward pass through the VLM only to autoregressively predict FAST tokens. 
        (prefix_out, _), _ = self._model.PaliGemma.llm(
            [prefix_tokens, None], # Only running VLM
            mask=prefix_attn_mask,
            positions=position,
        ) 

        # Decode output tokens into text. 
        subtask = self._model.PaliGemma.tokenizer.decode(prefix_out[0, -1, :], skip_special_tokens=True)
        return self._subtask_template.format(prompt=self._original_prompt) + " " + subtask.strip()




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
