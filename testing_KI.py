import os 
import jax 
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from openpi.models import pi0_config, model as _model
from openpi.models.pi0 import Pi0 

class TestingParameters:
    """Class to hold testing parameters for easy modification."""
    def __init__(self):
        self.batch_size = 2
        self.seq_len = 64
        self.action_dim = 8
        self.action_horizon = 15
        self.num_joints = 7
        self.image_size = (224, 224, 3)
        self.action_dims = 8

@pytest.fixture
def testing_parameters(): 
    return TestingParameters()


def create_dummy_observation_DROID() -> dict : 

    return {"test":testing_parameters.image_size}

if __name__ == "__main__":
    
    print(create_dummy_observation_DROID())

     
