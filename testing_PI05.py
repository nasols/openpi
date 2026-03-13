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




def print_dict_shapes(obs:dict) -> None :

    for key, item in obs.items():
        if item is None:
            print(key, " -- None")
        elif key == "prompt" or isinstance(item, str) or isinstance(item, bool): 
            print(key, " -- ", item)
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
    config_name: str = "pi05_ki" 
    
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


    def create_dummy_observation_DROID(self) -> dict : 
        """Creates a random input example for the Droid policy."""
        return {
            "observation/exterior_image_1_left": cv2.imread("./franka_ext_view.jpg"), #np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.rand(7),
            "observation/gripper_position": np.random.rand(1),
            "prompt": "pick up the cube then place it in the middle of the table.",
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
        if self.model.hierarchical_mode: 
            print("creating HI dummy data")
            self.dummy = self.create_dummy_observation_DROID_HI()
        else: 
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

        actions = jnp.array(self.actions)[np.newaxis, ...] # Smacks an new axis to the front, so [15, 8] -> [1, 15, 8]
        actions = jnp.pad(actions, ((0, 0), (0, 0), (0, 24)), mode='constant')  # Pad action dim from 8 to 32, so [1, 15, 8] -> [1, 15, 32] 

        self.model.compute_loss(self.rng, self.observation, actions, train=True)

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

        

        


if __name__ == "__main__":
    
    test_pi05 = TestPI05()
    print("Model type:", test_pi05.model.model_type)
    assert isinstance(test_pi05.model, Pi05), "Loaded model should be an instance of Pi05"
    # test_pi05.test_data_transforms()
    
    # test_pi05.test_compute_fast_loss()
    # test_pi05.test_subtask_generation()
    # prefix_out = test_pi05.test_compute_loss()
    # print(prefix_out)

    # create_torch_dataset(
    #     test_pi05.config.data.create(test_pi05.config.assets_dirs, test_pi05.config.model),
    #     test_pi05.config.model.action_horizon, 
    #     test_pi05.config.model
    #     )
    print("MANAGED TO LOAD TORCH SET")
    for i in range(0, 3): 
        inference_out = test_pi05.test_inference_HI()
        print(f"Inference output loop {i}:", inference_out)

    
# python decode_tokens.py "255667 255495 573 255649 255649 16616 573 255649 255649 16616 16616 255642 573 16616 573 255649 3124 255495 235248 255616"
# python decode_tokens.py "7071 235292 4788 908 573 28660 235269 3040 235292 235248 235274 235324 235324 235248 235274 235324 235318 235248 235284 235310"
# python decode_tokens.py "255667 1 1 573 1 235292 573 573 8277 235292 1 255495 235248 573 573 573 573 255649 255642 573"
# python decode_tokens.py "18075   908   573 28660   108     0     0     0     0     0     0     0 0     0     0     0     0     0     0     0"
# python decode_tokens.py "255667    908   3868   3868      1   3351"
    

     
