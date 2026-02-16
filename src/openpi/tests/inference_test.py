import logging

# Configure logging BEFORE importing other modules
logging.basicConfig(
    level=103,  # Use INFO to reduce verbosity, DEBUG for everything
    format="",
)

import jax
from openpi.training import config as _config
from openpi.models import model as _model
from openpi import transforms as _transforms
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies.droid_policy import make_droid_example
from openpi.models.pi0 import Pi0
import jax.numpy as jnp
import numpy as np
import cv2 as cv


def print_dict_structure(d, prefix="", max_depth=5, _depth=0):
    """Print dictionary keys with shapes for arrays and values for strings."""
    if _depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    
    
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict_structure(value, prefix=prefix + "  ", max_depth=max_depth, _depth=_depth+1)
        elif isinstance(value, str):
            print(f"{prefix}{key}: '{value}'")
        elif hasattr(value, 'shape'):
            print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"{prefix}{key}: {type(value).__name__} len={len(value)}")
        else:
            print(f"{prefix}{key}: {type(value).__name__}")


def print_observation_structure(obs, prefix=""):
    """Print Observation dataclass structure with shapes."""
    print(f"{prefix}Observation:")
    
    # Handle images dict
    if hasattr(obs, 'images') and obs.images:
        print(f"{prefix}  images:")
        for key, value in obs.images.items():
            if hasattr(value, 'shape'):
                print(f"{prefix}    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{prefix}    {key}: {type(value).__name__}")
    
    # Handle image_masks dict
    if hasattr(obs, 'image_masks') and obs.image_masks:
        print(f"{prefix}  image_masks:")
        for key, value in obs.image_masks.items():
            if hasattr(value, 'shape'):
                print(f"{prefix}    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{prefix}    {key}: {type(value).__name__}")
    
    # Handle all other fields
    for field_name in ['state', 'tokenized_prompt', 'tokenized_prompt_mask', 'token_ar_mask', 'token_loss_mask']:
        if hasattr(obs, field_name):
            value = getattr(obs, field_name)
            if value is None:
                print(f"{prefix}  {field_name}: None")
            elif hasattr(value, 'shape'):
                print(f"{prefix}  {field_name}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{prefix}  {field_name}: {type(value).__name__}")



print("Running inference test PI05_KI model")

ext_view = cv.imread("./franka_ext_view.jpg")
prompt = "Pick up the cube and place it in the middle of the table."


config = _config.get_config("pi05_droid_ki")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

print("[DEBUG] Loaded checkpoint from:", checkpoint_dir)

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir, hierarchical_mode=True)
print("[DEBUG] Knowledge Insulation enabled? ", policy._model.knowledge_insulation)
print("[DEBUG] Model Type? ", policy._model.model_type)
# Run inference on a dummy example.
dummy = make_droid_example()
dummy["observation/exterior_image_1_left"] = ext_view
dummy["prompt"] = prompt
gt_action = np.random.rand(15, 8).astype(np.float32)  # action_horizon=15, action_dim=8 for PI05 DROID, gets padded later

print("[DEBUG] Generated dummy example: ")
print_dict_structure(dummy)

print("[DEBUG] Running inference...")
action_chunk = policy.infer(dummy)["actions"]
print("[DEBUG] Inference action chunk: ", action_chunk)
print("[DEBUG] Inference test passed!")

# print("\n" + "="*60)
# print("Testing KI Training Pipeline Setup (no actual training)")
# print("="*60)

# rng = jax.random.key(42)
# data_config = config.data.create(config.assets_dirs, config.model)
# # For training, we need to add actions to the dummy data BEFORE transforms
# # so that TokenizeFASTInputs can tokenize them
# dummy = dummy.copy()
# dummy["actions"] = gt_action  
# # Generate random actions 
# actions = jnp.array(gt_action)[np.newaxis, ...]
# # Pad actions to match model dimension (32)
# actions = jnp.pad(actions, ((0, 0), (0, 0), (0, 24)), mode='constant')  # Pad from 8 to 32


# # Transform to get tokenized prompt
# ## create data config - Specifies things like input- output transforms (data transforms), model-transforms, norm-stats, data repo and asset id.
# data_config = config.data.create(config.assets_dirs, config.model)
# ## We pluck out the data and model transforms - compose creates a single transform from a sequence of transforms.
# data_transforms = _transforms.compose(data_config.data_transforms.inputs)
# model_transforms = _transforms.compose(data_config.model_transforms.inputs)
# ## Just maps the dictionary to be a jax tree 
# inputs = jax.tree.map(lambda x: x, dummy)
# inputs = data_transforms(inputs) # <-- applies data transform
# inputs = model_transforms(inputs) # <-- applies model transform
# inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
# observation = _model.Observation.from_dict(inputs)

# print("[DEBUG] Transformed observation structure:")
# print_observation_structure(observation)

# losses = policy._model.compute_loss(rng, observation, actions)
# print("[DEBUG] Computed losses: ", losses)