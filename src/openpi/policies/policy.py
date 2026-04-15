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
from openpi.models.pi05 import _gen_sample_action

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
        hi_mode: bool = False,
        ki_mode: bool = False,
        guided_inference: bool = False,
  
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
        self._model:Pi05 = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        
        self._ki_mode = ki_mode

        # Hierarchical planning state
        self._hi_mode = hi_mode
        self._current_subtask = None
        self._original_prompt = None      
        self._min_steps_subcount = 5  # Minimum steps before checking subtask completion
        self._step_count = 6
        self._completion_threshold = 0.8  # Similarity threshold for subtask completion                              

        # Action chunking -- guided inference 
        self._guided_inference = guided_inference
        self._previous_action_chunk = None


        if True: #self._hi_mode: 
            self._tokenizer = _tokenizer.PaligemmaTokenizer(max_len=model.max_token_len) 

        if self._guided_inference: 
            assert isinstance(self._model, Pi05), "Guided inference is currently only supported for Pi05 models."
            self._sample_actions = nnx_utils.module_jit(model.guided_inference)
            self._rng = rng or jax.random.key(0)
        elif self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, s: int|None=None, d:int|None=None, j:int|None=None, reset_prev_chunk=False) -> dict:  # type: ignore[misc]
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

        if self._guided_inference: 
            assert d is not None and s is not None, "Neither d nor s can be none when using guided inference"
            print("[DEBUG] Running guided inference")
            sample_kwargs["d"] = d
            sample_kwargs["s"] = s
            sample_kwargs["j"] = j
            #self._previous_action_chunk = _gen_sample_action(action_horizon=15, action_dim=32)
            if reset_prev_chunk:
                self._previous_action_chunk = None

            sample_kwargs["A_prev"] = self._previous_action_chunk[:, s:, :] if self._previous_action_chunk is not None else None  
        else: 
            print("[DEBUG] Not running guided inference")
        observation = _model.Observation.from_dict(inputs)
        
        ### HIERARCHICAL PLANNING ###
        #############################

        if self._hi_mode: 
            """
            Checks if the recent predicted actions converges to a standstill. 
            """
            should_generate_subtask = self._should_generate_subtask()
            rng = jax.random.PRNGKey(0) # Replace with actual RNG key management
            if self._current_subtask is None or self._original_prompt is None:
                self._original_prompt = obs["prompt"]
                self._current_subtask, self._current_subtask_mask = self._model._generate_subtask(rng, observation, self._original_prompt, temperature=0.5)
                # self.step_count = 0 
            
            elif should_generate_subtask:
                self._current_subtask, self._current_subtask_mask = self._model._generate_subtask(rng, observation, self._original_prompt)
                # self.step_count = 0

            # Build action prompt fully on token level to avoid decode->encode roundtrips.
            action_prompt_tokens, action_prompt_mask = self._tokenizer.build_tokenized_prompt_inference(
                np.asarray(observation.tokenized_prompt[0]),
                np.asarray(observation.tokenized_prompt_mask[0], dtype=bool),
                np.asarray(self._current_subtask[0]),
                np.asarray(self._current_subtask_mask[0], dtype=bool),
            )

            observation_w_subtask = _model.Observation(
                images=observation.images,
                image_masks=observation.image_masks,
                state=observation.state,
                tokenized_prompt=jnp.array(action_prompt_tokens)[None, ...],
                tokenized_prompt_mask=jnp.array(action_prompt_mask)[None, ...],
            )
            
            observation = observation_w_subtask
            
        ############################
        

        
        start_time = time.monotonic()

        # actions, hist = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        actions, hist = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        # hist = None
        
        if self._guided_inference: 
            self._previous_action_chunk = actions   
        
        
        outputs = {
            "state": inputs["state"],
            "actions": actions,
            # "hist": hist,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # logger.log(level=103, msg=f"[DEBUG] Logging data from output transform: {outputs}")
        # print(f"[DEBUG] Logging data from output transform: {outputs}")

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        outputs['hist'] = hist
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _should_generate_subtask(self) -> bool:
        """
        Naive regeneration check. 
        Checks if "enogh time" has passed before triggering subtask generation. 
        """
        return True
        if self._step_count < self._min_steps_subcount: 
            return False 

        return True 

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
