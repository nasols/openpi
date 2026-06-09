from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

class PolicyExperimental:
    def __init__(
        self,
        model: _model.BaseModel,
        norm_stats,
        inference_method: str,
        num_steps: int,
        *,
        rng: at.KeyArrayLike | None = None,
        input_transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):

        self._model = model
        self._input_transform = _transforms.compose(input_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._norm_stats = norm_stats
        self._inference_method = inference_method
        self._num_steps = num_steps

        # JAX model setup
        self._rng = rng or jax.random.key(0)

        self._model.num_steps = num_steps

        if inference_method == 'sample_actions':
            print("Compiling sample_actions")
            inference_callable = model.sample_actions
        elif inference_method == 'sample_inpaint':
            print("Compiling sample_inpaint")
            inference_callable = model.sample_inpaint
        elif inference_method == 'sample_inpaint_abs':
            print("Compiling sample_inpaint_abs")
            inference_callable = model.sample_inpaint_abs
        elif inference_method == 'sample_inpaint_pose':
            print("Compiling sample_inpaint_pose")
            inference_callable = model.sample_inpaint_pose
        elif inference_method == 'sample_RTC_and_CG':
            print("Compiling sample_RTC_and_CG")
            inference_callable = model.sample_RTC_and_CG
        else:
            raise ValueError(f"inference_method: ", inference_method)
        
        self._inference = nnx_utils.module_jit(inference_callable)

    def infer(
            self,
            obs: dict,
            *,
            noise: np.ndarray | None = None,
            inpaint_kwargs: dict | None = None,
        ) -> dict:
        q_obs = obs['observation/joint_position'].copy()

        # Make a copy since transformations may modify the inputs in place. (NOTE: Shallow copy)
        inputs = jax.tree.map(lambda x: x, obs)

        # Apply input transform
        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        # Split rng
        self._rng, sample_rng = jax.random.split(self._rng)

        ## Prepare kwargs for sample_actions
        # - noise
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # - unnorm stats
        unnorm_q01 = self._norm_stats['actions'].q01
        unnorm_q99 = self._norm_stats['actions'].q99
        sample_kwargs["unnorm_q01"] = jnp.asarray(unnorm_q01)
        sample_kwargs["unnorm_q99"] = jnp.asarray(unnorm_q99)

        ## Prepare extra kwargs for inpaint method
        if self._inference_method in ('sample_inpaint', 'sample_inpaint_abs'):
            assert inpaint_kwargs is not None
            sample_kwargs.update(
                self._process_inpaint_kwargs(inpaint_kwargs)
            )
        elif self._inference_method == 'sample_inpaint_pose':
            assert inpaint_kwargs is not None
            sample_kwargs.update(
                self._process_inpaint_pose_kwargs(inpaint_kwargs)
            )
        elif self._inference_method == 'sample_RTC_and_CG':
            assert inpaint_kwargs is not None
            sample_kwargs.update(
                self._process_RTC_and_CG_kwargs(inpaint_kwargs)
            )

        # Format observations
        observation = _model.Observation.from_dict(inputs)
        
        # RUN INFERENCE
        t0 = time.perf_counter()
        actions, info = self._inference(sample_rng, observation, **sample_kwargs)
        t1 = time.perf_counter()
        latency = t1 - t0
        
        actions = jax.device_get(actions)
        info = jax.device_get(info)

        # Gather output data
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }    
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)        # Reomve batch dim and convert jax to numpy
        outputs = self._output_transform(outputs)       # Apply output transforms

        # Format info data
        info['t_hist'] = np.asarray(info['t_hist'])
        info['x_hist'] = np.asarray(info['x_hist'][:,0,:,:8])
        # info['P_pos_hist'] = np.asarray(info['P_pos_hist'][:,0,:,:])
        # info['L_hist'] = np.asarray(info['L_hist'])
        # info['L_hist'] = np.asarray(info['L_hist'])
        info['q_obs'] = q_obs
        info['latency'] = latency

        outputs.update(info)
        
        return outputs

    def _process_inpaint_kwargs(
            self,
            inpaint_kwargs,
        ):
            # Support for gripper inpainting not implemented

            Y = np.zeros(shape=(15,32), dtype=np.float32)
            Y[:, :7] = inpaint_kwargs['Y']
            Y = jnp.asarray(Y).astype(jnp.float32)

            W = np.zeros(shape=(15,32), dtype=np.float32)
            W[:, :7] = inpaint_kwargs['W']
            W = jnp.asarray(W).astype(jnp.float32)
            
            guidance_weights = jnp.asarray(inpaint_kwargs['guidance_weights']).astype(jnp.float32)

            out_kwargs = {
                'Y': Y,
                'W': W,
                'guidance_weights': guidance_weights,
            }

            if 'q_obs' in inpaint_kwargs:
                q_obs = np.zeros(shape=(32,), dtype=np.float32)
                q_obs[:7] = inpaint_kwargs['q_obs'][:7]
                out_kwargs['q_obs'] = jnp.asarray(q_obs).astype(jnp.float32)

            return out_kwargs
    
    def _process_inpaint_pose_kwargs(
            self,
            inpaint_kwargs,
        ):
            # Support for gripper inpainting not implemented

            Y_pos = jnp.asarray(inpaint_kwargs['Y_pos']).astype(jnp.float32)
            assert Y_pos.shape==(15,3), f"Y_pos.shape: {Y_pos.shape}"

            W_pos = jnp.asarray(inpaint_kwargs['W_pos']).astype(jnp.float32)
            assert W_pos.shape==(15,3), f"W_pos.shape: {W_pos.shape}"
            
            guidance_weights = jnp.asarray(inpaint_kwargs['guidance_weights']).astype(jnp.float32)
            assert guidance_weights.shape==(self._num_steps,), f"guidance_weights.shape: {guidance_weights.shape}"

            q_obs = jnp.asarray(inpaint_kwargs['q_obs']).astype(jnp.float32)
            assert q_obs.shape==(7,), f"q_obs.shape: {q_obs.shape}"

            out_kwargs = {
                'Y_pos': Y_pos,
                'W_pos': W_pos,
                'guidance_weights': guidance_weights,
                'q_obs': q_obs
            }
            return out_kwargs

    def _process_RTC_and_CG_kwargs(
            self,
            inpaint_kwargs,
        ):
            out_kwargs = {}

            q_obs = inpaint_kwargs['q_obs']
            assert q_obs.shape==(7,)
            out_kwargs['q_obs'] = jnp.asarray(q_obs).astype(jnp.float32)

            Y_RTC = np.zeros(shape=(15,32), dtype=np.float32)
            Y_RTC[:, :8] = inpaint_kwargs['Y_RTC']
            out_kwargs['Y_RTC'] = jnp.asarray(Y_RTC).astype(jnp.float32)

            w_RTC = inpaint_kwargs['w_RTC']
            assert w_RTC.shape==(15,)
            out_kwargs['w_RTC'] = jnp.asarray(w_RTC).astype(jnp.float32)

            g_RTC = inpaint_kwargs['g_RTC']
            assert g_RTC.shape == (self._num_steps,) 
            out_kwargs['g_RTC'] = jnp.asarray(g_RTC).astype(jnp.float32)

            out_kwargs['w_gripper_RTC'] = jnp.float32(inpaint_kwargs['w_gripper_RTC'])
            
            Y_CG_pos = inpaint_kwargs['Y_CG_pos']
            assert Y_CG_pos.shape == (15, 3)
            out_kwargs['Y_CG_pos'] = jnp.asarray(Y_CG_pos).astype(jnp.float32)

            W_CG_pos = inpaint_kwargs['W_CG_pos']
            assert W_CG_pos.shape == (15, 3)
            out_kwargs['W_CG_pos'] = jnp.asarray(W_CG_pos).astype(jnp.float32)

            g_CG = inpaint_kwargs['g_CG']
            assert g_CG.shape == (self._num_steps,) 
            out_kwargs['g_CG'] = jnp.asarray(g_CG).astype(jnp.float32)

            return out_kwargs

# Unchanged
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
    