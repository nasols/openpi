import os 
import jax 
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import dataclass
from flax import nnx

from openpi.models import pi0_config, model as _model
from openpi.training import config as _config
from openpi.models.pi0 import Pi0 
from openpi import transforms as _transforms
from openpi.shared import download
from openpi.policies import policy_config


def print_dict_shapes(obs:dict) -> None :

    for key, item in obs.items():
        if key == "prompt" or isinstance(item, str) or isinstance(item, bool): 
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



def create_dummy_observation_DROID() -> dict : 
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

if __name__ == "__main__":
    
    config = _config.get_config("pi05_droid_ki")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    model = policy._model

    print(f"    ✓ Knowledge Insulation enabled: {model.knowledge_insulation}")
    
    # Create dummy data
    print("\n[2] Creating dummy data...")
    dummy = create_dummy_observation_DROID()
    gt_action = np.random.randn(15, 8).astype(np.float32)
    dummy_w_actions = dummy.copy()
    dummy_w_actions["actions"] = gt_action

    # Data transform
    data_config = config.data.create(config.assets_dirs, config.model)
    data_input_transforms = _transforms.compose(data_config.data_transforms.inputs)
    data_output_transforms = _transforms.compose(data_config.data_transforms.outputs)
    model_transforms = _transforms.compose(data_config.model_transforms.inputs)

    ##############################
    ### DATA TRANSFORM TESTING ###
    ##############################
    
    # Transforming obs-dict according to input transforms 
    print(f"Observation dict before transforms -> \n ")
    print_dict_shapes(dummy)
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
    observation = _model.Observation.from_dict(inputs)
    print(f"Final observation passed to model \n")
    print_dict_shapes(observation.to_dict())

    ##############################
    ##############################

    # Pi05 model is trained with a constant action dim set to 32, so it expects an array of size 32 with the first 8 items populated
    # Pad actions to match model dimension (32)
    actions = jnp.array(gt_action)[np.newaxis, ...] # Smacks an new axis to the front, so [15, 8] -> [1, 15, 8]
    actions = jnp.pad(actions, ((0, 0), (0, 0), (0, 24)), mode='constant')  # Pad action dim from 8 to 32, so [1, 15, 8] -> [1, 15, 32] 
    print(f"Action shape input to model -> {actions.shape}")




    rng = jax.random.PRNGKey(0)

    model.compute_loss(rng, observation, gt_action)
    

     
