import os 
import jax 
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import dataclass
from flax import nnx
import cv2
from openpi.models import pi0_config, model as _model
from openpi.training import config as _config
from openpi.models.pi05 import Pi05
from openpi import transforms as _transforms
from openpi.shared import download
from openpi.policies import policy_config
from openpi.training.data_loader import create_torch_dataset, MixedDataset
from scripts.compute_norm_stats import create_torch_dataloader

import time


def print_dict_shapes(obs:dict) -> None :

    for key, item in obs.items():
        if item is None:
            print(key, " -- None")
        elif key == "prompt" or isinstance(item, str) or isinstance(item, bool): 
            print(key, " -- ", item)
        elif isinstance(item, np.ndarray) or isinstance(item, jnp.ndarray):
            print(key, " -- ", item.shape)
        elif isinstance(item, dict): 
            print_dict_shapes(item)
        else: 
            print(key, " -- ", item.shape)  


@dataclass
class TestingParameters:
    """Class to hold testing parameters for easy modification."""
    batch_size: int = 2
    seq_len: int = 64
    action_dim: int = 8
    action_horizon: int = 15
    num_joints: int = 7
    image_size: tuple = (224, 224, 3)
    action_dims: int = 8
    config_name: str = "pi05_droid_finetune" 
    
# Create a global instance
testing_parameters = TestingParameters()

class TestPI05: 

    def __init__(self):
        self.load_policy()

    def load_policy(self): 
        self.config = _config.get_config("pi05_droid_finetune")  
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)
        self.model : Pi05 = self.policy._model
        self.rng = jax.random.PRNGKey(0)

    def warmup(self): 
        dummy_obs = self.create_dummy_observation_DROID()
        self.policy.infer(dummy_obs)

    def _expand_observation_batch(self, observation: _model.Observation, target_batch: int) -> _model.Observation:
        """Repeat a batch-1 observation to the requested batch size."""

        def _repeat_if_needed(x):
            if x is None:
                return None
            x = jnp.asarray(x)
            if x.ndim == 0:
                return x
            if x.shape[0] == target_batch:
                return x
            if x.shape[0] == 1:
                return jnp.repeat(x, target_batch, axis=0)
            return x

        return _model.Observation(
            images={k: _repeat_if_needed(v) for k, v in observation.images.items()},
            image_masks={k: _repeat_if_needed(v) for k, v in observation.image_masks.items()},
            state=_repeat_if_needed(observation.state),
            tokenized_prompt=_repeat_if_needed(observation.tokenized_prompt),
            tokenized_prompt_mask=_repeat_if_needed(observation.tokenized_prompt_mask),
            token_ar_mask=_repeat_if_needed(observation.token_ar_mask),
            token_loss_mask=_repeat_if_needed(observation.token_loss_mask),
            subtask_loss_mask=_repeat_if_needed(observation.subtask_loss_mask),
            subtask_token_mask=_repeat_if_needed(observation.subtask_token_mask),
            action_loss_mask=_repeat_if_needed(observation.action_loss_mask),
            action_token_mask=_repeat_if_needed(observation.action_token_mask),
            subtask_tokens=_repeat_if_needed(observation.subtask_tokens),
            subtask_mask=_repeat_if_needed(observation.subtask_mask),
            subtask_gt_tokens=_repeat_if_needed(observation.subtask_gt_tokens),
            subtask_gt_mask=_repeat_if_needed(observation.subtask_gt_mask),
        )

    def _patch_posemb_for_token_time(self):
        """Patch posemb_sincos to support [batch, action_horizon] timestep arrays in this test script."""
        if getattr(_pi05_module, "_testing_posemb_patched", False):
            return

        def _posemb_sincos_test(pos, embedding_dim: int, min_period: float, max_period: float):
            if embedding_dim % 2 != 0:
                raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

            pos = jnp.asarray(pos)
            original_shape = pos.shape
            pos_flat = pos.reshape(-1)
            fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
            period = min_period * (max_period / min_period) ** fraction
            sinusoid_input = jnp.einsum(
                "i,j->ij",
                pos_flat,
                1.0 / period * 2 * jnp.pi,
                precision=jax.lax.Precision.HIGHEST,
            )
            emb = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
            return emb.reshape(*original_shape, embedding_dim)

        _pi05_module.posemb_sincos = _posemb_sincos_test
        _pi05_module._testing_posemb_patched = True

    def _compute_loss_batch_safe(self, observation: _model.Observation, actions: jnp.ndarray, *, train: bool = True):
        """Compute the standard action loss with per-sample time for batch testing."""
        self._patch_posemb_for_token_time()
        preprocess_rng, noise_rng, time_rng = jax.random.split(self.rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        time_per_token = jnp.repeat(time[..., None], self.model.action_horizon, axis=1)

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.model.embed_suffix(
            observation, x_t, time_per_token
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.model.action_out_proj(suffix_out[:, -self.model.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)


    def create_dummy_observation_DROID(self) -> dict : 
        """Creates a random input example for the Droid policy."""
        return {
            "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), #cv2.imread("./franka_ext_view.jpg"),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  
            "observation/joint_position": np.random.rand(7),
            "observation/gripper_position": np.random.rand(1),
            "prompt": "pick up the cube then place it in the middle of the table.",
            "subtask": None
        }
    
    def create_dummy_observation_DROID_HI(self) -> dict :
        """Creates a random input example for the Droid HI policy."""
        return {
            "observation/exterior_image_1_left": cv2.imread("./franka_ext_view.jpg"), #np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.rand(7),
            "observation/gripper_position": np.random.rand(1),
            "prompt": "pick up the cube",
            "subtask": "move towards the red cube",
        }
    
    def create_batched_dummy_raw_observation_DROID(self, batch_dim=32) -> dict:
        samples = [self.create_dummy_observation_DROID() for _ in range(batch_dim)]
        return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *samples)
    
    def test_data_transforms(self): 
        self.dummy = self.create_dummy_observation_DROID()
        self.actions = np.random.randn(15, 8).astype(np.float32)        
        
        dummy_w_actions = self.dummy.copy()
        dummy_w_actions["actions"] = self.actions
        
        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        data_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
        data_output_transforms = _transforms.compose(data_config.data_transforms.outputs)
        model_transforms = _transforms.compose(data_config.model_transforms.inputs)

        # Transforming obs-dict according to input transforms 
        print(f"Observation dict before transforms -> \n ")
        print_dict_shapes(dummy_w_actions)
        ## Mapping the observation dict variables into a jax computational tree. 
        inputs = jax.tree.map(lambda x: x, dummy_w_actions) 
        ## Applying data transforms 
        inputs = data_input_transforms(inputs) 
        print(f"Observation after input transforms \n ")
        print_dict_shapes(inputs)
        ## Applying model transforms 
        inputs = model_transforms(inputs)
        print(f"Observation after model transforms \n ")
        print_dict_shapes(inputs)

        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...], 
            inputs
        )
        self.observation = _model.Observation.from_dict(inputs)
        print(f"Final observation passed to model \n")
        print_dict_shapes(self.observation.to_dict())

    def test_compute_fast_loss(self): 

        actions = jnp.array(self.actions)[np.newaxis, ...] # Smacks an new axis to the front, so [15, 8] -> [1, 15, 8]
        actions = jnp.pad(actions, ((0, 0), (0, 0), (0, 24)), mode='constant')  # Pad action dim from 8 to 32, so [1, 15, 8] -> [1, 15, 32] 
        print(f"Action shape input to model -> {actions.shape}")


        self.model.compute_loss_ki(self.rng, self.observation, actions, train=True)

    def test_subtask_generation(self):
            assert self.observation is not None, "Observation must be created before testing subtask generation."
            assert self.model.hierarchical_mode, "Policy should be in hierarchical mode for subtask generation."

            print("\nTesting subtask generation...")
            print(f"Hierarchical mode enabled: {self.model.hierarchical_mode}")

            # the decomposition prompt tokenized, not the original prompt
            original_prompt = self.dummy["prompt"]
            decomposition_template = "Decompose the following task into sub-tasks. Task: {prompt}, Sub-tasks:"
            decomposition_prompt = decomposition_template.format(prompt=original_prompt)
            
            print(f"Original prompt: {original_prompt}")
            print(f"Decomposition prompt: {decomposition_prompt}")
            
            # Tokenize the decomposition prompt
            tokenizer = self.policy._tokenizer
            
            decomp_tokens, decomp_mask = tokenizer.tokenize(decomposition_prompt)
            
            # Create new observation with decomposition prompt
            decomp_observation = _model.Observation(
                state=self.observation.state,
                images=self.observation.images,
                image_masks=self.observation.image_masks,
                tokenized_prompt=jnp.array([decomp_tokens]),  # Use decomposition prompt
                tokenized_prompt_mask=jnp.array([decomp_mask]),
                token_loss_mask=None,  # Not needed for generation
            )
            
            # Generate subtask token IDs
            subtask_token_ids = self.model._generate_subtask(
                rng=self.rng,
                observation=decomp_observation,  # Use observation with decomposition prompt
                original_prompt=original_prompt, 
                max_tokens=10, 
                temperature=0.5
            )

            print(f"\nGenerated subtask token IDs: {subtask_token_ids}")
            
            # Decode token IDs to text
            if subtask_token_ids:
                subtask_text = tokenizer._tokenizer.decode(subtask_token_ids)
                print(f"Decoded subtask text: '{subtask_text}'")
                
                # Full subtask = decomposition prompt + generated text
                full_subtask = decomposition_prompt + " " + subtask_text
                print(f"Full subtask for action generation: '{full_subtask}'")
            else:
                print("No tokens generated (EOS reached immediately)")


    def test_compute_loss(self): 
        assert isinstance(self.model, Pi05), "Model should be an instance of Pi05 for this test."

        self.dummy = self.create_dummy_observation_DROID()
        self.actions = np.random.randn(15, 8).astype(np.float32)        
        
        dummy_w_actions = self.dummy.copy()
        dummy_w_actions["actions"] = self.actions
        
        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        data_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
        data_output_transforms = _transforms.compose(data_config.data_transforms.outputs)
        model_transforms = _transforms.compose(data_config.model_transforms.inputs)
        

        inputs = jax.tree.map(lambda x: x, dummy_w_actions) 
        inputs = data_input_transforms(inputs) 
        inputs = model_transforms(inputs)
        
        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...], 
            inputs
        )

        self.observation = _model.Observation.from_dict(inputs)
        # self.observation = self._expand_observation_batch(self.observation, self.actions.shape[0])

        actions = jnp.array(self.actions)[np.newaxis, ...] # Smacks an new axis to the front, so [15, 8] -> [1, 15, 8]
        actions = jnp.pad(actions, ((0, 0), (0, 0), (0, 24)), mode='constant')  # Pad action dim from 8 to 32, so [1, 15, 8] -> [1, 15, 32] 

        print(f"Observation state shape input to model -> {self.observation.state.shape}")
        loss = self.model.compute_loss(self.rng, self.observation, actions, train=True)
        #loss = self._compute_loss_batch_safe(self.observation, actions, train=True)
        print(f"Loss shape -> {loss.shape}")

    def test_inference(self): 
        assert isinstance(self.model, Pi05), "Model should be an instance of Pi05 for this test."
        # if self.model.hierarchical_mode: 
        #     print("creating HI dummy data")
        #     self.dummy = self.create_dummy_observation_DROID_HI()
        # else: 
        self.dummy = self.create_dummy_observation_DROID()

        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        data_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
        model_transforms = _transforms.compose(data_config.model_transforms.inputs)
        inputs = jax.tree.map(lambda x: x, self.dummy)
        inputs = data_input_transforms(inputs)
        inputs = model_transforms(inputs)
        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...],
            inputs
        )
        self.observation = _model.Observation.from_dict(inputs) 
        result = self.policy.infer(self.dummy)
        print(f"Inference result: {result}")

    def test_inference_HI(self): 
        assert self.model.hierarchical_mode, "Policy should be in hierarchical mode for this test."
        dummy = self.create_dummy_observation_DROID_HI()

        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        data_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
        model_transforms = _transforms.compose(data_config.model_transforms.inputs)

        inputs = jax.tree.map(lambda x: x, dummy)
        inputs = data_input_transforms(inputs)
        inputs = model_transforms(inputs)
        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...],
            inputs
        ) 
        self.observation = _model.Observation.from_dict(inputs)


        result = self.policy.infer(dummy)

        return result

    def test_mixed_training(self): 
        
        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        print(f"TESTING DATASET CREATION")
        mixedset : MixedDataset = create_torch_dataset(data_config, action_horizon=self.config.model.action_horizon, model_config=self.config.model)
        print(f"DATASET CREATED!")
        print(f"TESTING DATALOADER CREATION")
        create_torch_dataloader(
            data_config,
            self.model.action_horizon,
            self.config.batch_size,
            self.config.model,
            self.config.num_workers,
        )
        print(f"DATALOADER CREATED!")

        print("TESTING GETITEM")
        mixedset.__getitem__(0)
        print("GETITEM WORKED!")
        pass     

    def test_batch_32_observation_pipeline(self):
        """Test the training pipeline with a batch of 32 observations."""
        # Step 1: Create a batched observation
        batch_size = 32
        batched_obs = self.create_batched_dummy_raw_observation_DROID(batch_dim=batch_size)

        # Step 2: Split batch into individual samples
        individual_samples = [jax.tree.map(lambda x: x[i], batched_obs) for i in range(batch_size)]

        # Step 3: Apply transformations to each sample
        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        data_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
        transformed_samples = [data_input_transforms(sample) for sample in individual_samples]

        # Step 4: Recombine transformed samples into a batch
        transformed_batched_obs = jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed_samples)

        # Step 5: Convert to Observation object
        observation = _model.Observation.from_dict(transformed_batched_obs)

        # Step 6: Create dummy actions
        actions = jnp.zeros((batch_size, self.model.action_horizon, self.model.action_dim), dtype=jnp.float32)

        # Step 7: Pass through the training pipeline
        loss = self.model.compute_loss(self.rng, observation, actions, train=True)
        print(f"Computed loss: {loss}")


if __name__ == "__main__":
    
    test_pi05 = TestPI05()
    print("Model type:", test_pi05.model.model_type)
    assert isinstance(test_pi05.model, Pi05), "Loaded model should be an instance of Pi05"
    # test_pi05.warmup()

    # test_pi05.test_data_transforms()
    
    # test_pi05.test_compute_fast_loss()
    # test_pi05.test_subtask_generation()
    # prefix_out = test_pi05.test_batch_32_observation_pipeline()
    prefix_out = test_pi05.test_compute_loss()
    # print(prefix_out)
    
    # test_pi05.test_mixed_training()

    # tb = time.time()
    # test_pi05.test_inference()
    # ta = time.time()
    # print(f"Inference time: {ta - tb:.2f} seconds")
    # for i in range(3): 
    #     tb = time.time()
    #     test_pi05.test_inference()
    #     ta = time.time()
    #     print(f"Compute loss time: {ta - tb:.2f} seconds")

    # create_torch_dataset(
    #     test_pi05.config.data.create(test_pi05.config.assets_dirs, test_pi05.config.model),
    #     test_pi05.config.model.action_horizon, 
    #     test_pi05.config.model
    #     )
    # print("MANAGED TO LOAD TORCH SET")
    # for i in range(0, 3): 
    #     inference_out = test_pi05.test_inference_HI()
    #     print(f"Inference output loop {i}:", inference_out)

    
# python decode_tokens.py "255667 255495 573 255649 255649 16616 573 255649 255649 16616 16616 255642 573 16616 573 255649 3124 255495 235248 255616"
# python decode_tokens.py "7071 235292 4788 908 573 28660 235269 3040 235292 235248 235274 235324 235324 235248 235274 235324 235318 235248 235284 235310"
# python decode_tokens.py "255667 1 1 573 1 235292 573 573 8277 235292 1 255495 235248 573 573 573 573 255649 255642 573"
# python decode_tokens.py "18075   908   573 28660   108     0     0     0     0     0     0     0 0     0     0     0     0     0     0     0"
# python decode_tokens.py "255667    908   3868   3868      1   3351"



