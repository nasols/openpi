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
    config_name: str = "debug_pi05" 
    
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
    
    config = _config.get_config(testing_parameters.config_name)

    print(config)
    

     
