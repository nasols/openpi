import os 
import jax 
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from openpi.models import pi0_config, model as _model
from openpi.policies.policy import Policy
from openpi.models.pi0 import Pi0 
from openpi.policies.droid_policy import DroidInputs, DroidOutputs, make_droid_example
from openpi.models.model import ModelType
from openpi import transforms

class TestingParameters:
    """Class to hold testing parameters for easy modification."""
    def __init__(self):
        self.batch_size = 2
        self.seq_len = 64
        self.action_dim = 32  # Match the model config
        self.action_horizon = 15
        self.num_joints = 7
        self.action_dims = 32  # Match the model config
        self.image_size = (224, 224, 3)

@pytest.fixture
def testing_parameters(): 
    return TestingParameters()


def create_dummy_observation_DROID() -> dict : 
    return make_droid_example()


def create_dummy_policy_with_transforms(model: Pi0, model_type: ModelType) -> Policy:
    """
    Create a dummy policy with the proper transforms for testing.
    
    This mimics what create_trained_policy does, but without needing a checkpoint.
    For testing with raw DROID observations, you need:
    - DroidInputs: converts raw obs dict to model format
    - DroidOutputs: converts model outputs back to raw format
    """
    # Dummy norm stats (identity normalization - no actual normalization)
    dummy_norm_stats = {
        "state": transforms.NormStats(min=np.zeros(8), max=np.ones(8)),
        "actions": transforms.NormStats(min=np.zeros(8), max=np.ones(8)),
    }
    
    input_transforms = [
        DroidInputs(model_type=model_type),
        transforms.Normalize(dummy_norm_stats, use_quantiles=False),
    ]
    
    output_transforms = [
        transforms.Unnormalize(dummy_norm_stats, use_quantiles=False),
        DroidOutputs(),
    ]
    
    return Policy(
        model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        is_pytorch=False,
    )


def create_KI_model() -> Pi0:
    """Create a KI model for testing."""
    config = pi0_config.Pi0Config(
        pi05=True, 
        action_dim=32, 
        action_horizon=15,
        knowledge_insulation=True, 
        paligemma_variant="dummy", 
        action_expert_variant="dummy"
    )
    rngs = nnx.Rngs(0)
    return Pi0(config, rngs)


def create_KI_policy() -> Policy:
    """Create a KI policy with proper transforms for testing."""
    model = create_KI_model()
    return create_dummy_policy_with_transforms(model, ModelType.PI05_KI)

def test_KI_model_forward_pass(testing_parameters): 
    dummy_ki_policy = create_KI_policy()
    dummy_obs = create_dummy_observation_DROID()

    outputs = dummy_ki_policy.infer(dummy_obs)
    print("Output keys:", outputs.keys())
    print("Actions shape:", outputs["actions"].shape)
    
    assert "actions" in outputs, "Model output should contain 'actions'"
    # DroidOutputs slices to first 8 dims, and infer removes batch dimension
    assert outputs["actions"].shape == (testing_parameters.action_horizon, 8), \
        f"Expected shape ({testing_parameters.action_horizon}, 8), got {outputs['actions'].shape}"

if __name__ == "__main__":
    # Run the test function directly
    test_params = TestingParameters()
    test_KI_model_forward_pass(test_params)


    
    
    

     
