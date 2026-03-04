# Video Frame Loading for OpenPI

Quick reference for loading video frames into the model.

## Installation

```bash
# For cv2 backend (recommended)
pip install opencv-python

# For imageio backend (alternative)
pip install imageio imageio-ffmpeg
```

## Quick Start

### Load Single Frame

```python
from openpi.shared import video_utils

# Load frame at timestep 15 (from 100-step episode)
frame = video_utils.extract_frame_at_timestep(
    video_path="episode_003.mp4",
    timestep=15,
    total_timesteps=100,
    target_size=(224, 224),
    normalize=True,  # Convert to [-1, 1] for model
)
# frame shape: (224, 224, 3), dtype: float32, range: [-1, 1]
```

### Load Multiple Frames

```python
# Load specific frames
frames = video_utils.load_video_frames(
    "episode.mp4",
    frame_indices=[0, 10, 20, 30],
    target_size=(224, 224),
)

# Sample 10 frames uniformly
frames = video_utils.load_video_frames(
    "episode.mp4",
    num_frames=10,
    target_size=(224, 224),
)

# Load all frames
frames = video_utils.load_video_frames("episode.mp4")
```

### Use with Model

```python
from openpi.shared import video_utils
from openpi.policies import policy_config
from openpi.training import config as _config

# Load policy
config = _config.get_config("pi05_droid")
policy = policy_config.create_trained_policy(config, "checkpoints/pi05_droid")

# Load frame from video
frame = video_utils.extract_frame_at_timestep(
    "demo.mp4", timestep=25, total_timesteps=100, normalize=True
)

# Create observation
obs = {
    "observation/exterior_image_1_left": frame,
    "observation/wrist_image_left": frame,
    "observation/joint_position": robot_state[:7],
    "observation/gripper_position": robot_state[7:8],
    "prompt": "pick up the cube",
}

# Inference
result = policy.infer(obs)
actions = result["actions"]
```

## Advanced Usage

### VideoFrameLoader Class

```python
with video_utils.VideoFrameLoader("video.mp4", target_size=(224, 224)) as loader:
    print(f"Video: {loader.total_frames} frames @ {loader.fps} fps")
    
    # Load single frame
    frame = loader.load_frame(15)
    
    # Load frames at specific times (seconds)
    frames = loader.load_frames_at_times([0.5, 1.0, 1.5, 2.0])
    
    # Load frame range with step
    frames = loader.load_frames_range(start_idx=10, end_idx=50, step=5)
    
    # Load uniformly sampled frames
    frames = loader.load_frames_uniform(num_frames=20)
```

### Multiple Camera Views

```python
timestep = 15

# Load frames from different camera videos
base_frame = video_utils.extract_frame_at_timestep(
    "base_view.mp4", timestep, 100, normalize=True
)
wrist_frame = video_utils.extract_frame_at_timestep(
    "wrist_view.mp4", timestep, 100, normalize=True
)

obs = {
    "observation/exterior_image_1_left": base_frame,
    "observation/wrist_image_left": wrist_frame,
    ...
}
```

### Convert Frames to Model Format

```python
# frames: (N, H, W, 3), uint8, [0, 255]
model_frames = video_utils.frames_to_model_input(
    frames, 
    normalize=True  # Convert to float32, [-1, 1]
)
# model_frames: (N, H, W, 3), float32, [-1, 1]
```

## API Reference

### `load_video_frames()`

```python
frames = load_video_frames(
    video_path,
    frame_indices=None,      # List of frame indices to load
    num_frames=None,         # Number of frames to uniformly sample
    target_size=(224, 224),  # Resize to (H, W)
    backend="cv2",           # "cv2" or "imageio"
)
# Returns: (N, H, W, 3), uint8, [0, 255]
```

### `extract_frame_at_timestep()`

```python
frame = extract_frame_at_timestep(
    video_path,
    timestep,                # Timestep index (0-based)
    total_timesteps,         # Total timesteps in episode
    target_size=(224, 224),
    normalize=True,          # Convert to [-1, 1]
)
# Returns: (H, W, 3), float32 if normalize else uint8
```

### `VideoFrameLoader`

```python
loader = VideoFrameLoader(
    video_path,
    backend="cv2",           # "cv2" or "imageio"
    target_size=(224, 224),  # Optional resize
)

# Properties
loader.total_frames  # int
loader.fps          # float
loader.duration     # float (seconds)
loader.shape        # (H, W, C)

# Methods
loader.load_frame(idx)                    # Single frame
loader.load_frames([0, 10, 20])          # Specific frames
loader.load_all_frames()                 # All frames
loader.load_frames_uniform(num_frames)   # Uniform sampling
loader.load_frames_range(start, end, step)  # Frame range
loader.load_frames_at_times([0.5, 1.0]) # At time points (seconds)

loader.close()  # Release resources
```

### `frames_to_model_input()`

```python
model_frames = frames_to_model_input(
    frames,          # (N, H, W, 3) or (H, W, 3), uint8
    normalize=True,  # Convert to [-1, 1]
    format="float32", # "uint8" or "float32"
)
```

## Tips

1. **Use context manager**: `with VideoFrameLoader(...) as loader:` to auto-close
2. **Normalize for model**: Always use `normalize=True` when feeding to the model
3. **Target size**: Model expects (224, 224) for most variants
4. **Backend choice**: 
   - `cv2`: Faster, widely used
   - `imageio`: Pure Python, better format support
5. **Memory**: Use `load_frames_uniform()` instead of `load_all_frames()` for long videos

## Examples

See complete examples in: `examples/load_video_frames_example.py`

Run test:
```bash
python -m openpi.shared.video_utils path/to/video.mp4 --num-frames 10
```
