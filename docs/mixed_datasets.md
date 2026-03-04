# Mixed Dataset Training

This guide explains how to train OpenPi models on a mixture of multiple LeRobot datasets.

## Overview

The mixed dataset feature allows you to combine multiple LeRobot datasets during training, with configurable sampling weights for each dataset. This is useful when you want to:

- Combine data from multiple tasks or environments
- Balance datasets of different sizes
- Control the contribution of each dataset to the training process

## Configuration

### Basic Usage

To use mixed datasets, modify your data config to use `repo_ids` instead of a single `repo_id`:

```python
from openpi.training.config import LeRobotDatasetConfig

data=LeRobotDROIDDataConfig(
    repo_ids=[
        LeRobotDatasetConfig(repo_id="your_username/dataset1", weight=1.0),
        LeRobotDatasetConfig(repo_id="your_username/dataset2", weight=2.0),
        LeRobotDatasetConfig(repo_id="your_username/dataset3", weight=1.0),
    ],
    base_config=DataConfig(prompt_from_task=True),
    assets=AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
        asset_id="droid",
    ),
)
```

### Sampling Weights

The `weight` parameter controls how often samples are drawn from each dataset:

- **Equal weights** (e.g., all `weight=1.0`): Each dataset contributes proportionally to its size
- **Higher weight**: More samples will be drawn from this dataset
- **Example**: If dataset A has `weight=2.0` and dataset B has `weight=1.0`, samples from A will be drawn twice as often as samples from B

### Weight Calculation

The actual sampling probability for each sample is calculated as:

```
sample_probability = (dataset_weight / total_weight) / dataset_size
```

This means:
- Larger datasets get more samples overall (because they have more data)
- But weights can be adjusted to balance the contribution of each dataset

## Example Configurations

### Example 1: Combining Three DROID Datasets

```python
TrainConfig(
    name="pi05_droid_mixed",
    model=pi05_config.Pi05Config(
        action_dim=32,
        action_horizon=15,
    ),
    data=LeRobotDROIDDataConfig(
        repo_ids=[
            LeRobotDatasetConfig(repo_id="your_username/droid_pickupcube", weight=1.0),
            LeRobotDatasetConfig(repo_id="your_username/droid_stackblocks", weight=1.5),
            LeRobotDatasetConfig(repo_id="your_username/droid_openbox", weight=1.0),
        ],
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
            asset_id="droid",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
    num_train_steps=2000,
    save_interval=1000,
    batch_size=32,
    fsdp_devices=2,
)
```

### Example 2: Balancing Small and Large Datasets

If you have one large dataset and one small dataset, you can increase the weight of the small dataset to ensure it contributes meaningfully to training:

```python
repo_ids=[
    LeRobotDatasetConfig(repo_id="your_username/large_dataset", weight=1.0),  # 10,000 episodes
    LeRobotDatasetConfig(repo_id="your_username/small_dataset", weight=5.0),  # 1,000 episodes
]
```

This ensures the small dataset gets sampled more frequently per-episode, balancing its contribution.

### Example 3: Single Dataset (Backward Compatible)

The original single-dataset configuration still works:

```python
data=LeRobotDROIDDataConfig(
    repo_id="your_username/single_dataset",  # Single dataset
    base_config=DataConfig(prompt_from_task=True),
    assets=AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
        asset_id="droid",
    ),
)
```

## Training with Mixed Datasets

Training with mixed datasets works the same as single datasets:

```bash
uv run scripts/train.py --config-name=pi05_droid_mixed
```

The training logs will show information about each dataset, including:
- Number of samples in each dataset
- Sampling weight for each dataset
- Percentage of total samples from each dataset

## Important Considerations

### 1. Normalization Stats

All datasets in the mixture should use the same normalization statistics (specified via `AssetsConfig`). This ensures consistent preprocessing across datasets.

### 2. Data Format Compatibility

All datasets should have compatible data formats:
- Same action space dimensions
- Same observation keys
- Compatible image sizes (if using vision)

### 3. Task Prompts

When using `prompt_from_task=True`, the tasks from all datasets are merged. Make sure task indices don't conflict across datasets.

### 4. Dataset Availability

All datasets specified in `repo_ids` must be available (either locally or on HuggingFace Hub) before training starts.

## Implementation Details

### MixedDataset Class

The `MixedDataset` class in `data_loader.py` handles the mixing:

```python
class MixedDataset(Dataset):
    """Dataset that mixes multiple LeRobot datasets with specified sampling weights."""
    
    def __init__(self, datasets: Sequence[Dataset], weights: Sequence[float]):
        # Stores datasets and calculates sample weights
        ...
    
    def __getitem__(self, index: int) -> dict:
        # Maps global index to appropriate dataset and local index
        ...
```

### Weighted Sampling

When creating a DataLoader, you can use the sample weights for weighted random sampling:

```python
from torch.utils.data import WeightedRandomSampler

# If using mixed dataset with PyTorch DataLoader
if isinstance(dataset, MixedDataset):
    sampler = WeightedRandomSampler(
        weights=dataset.get_sample_weights(),
        num_samples=len(dataset),
        replacement=True
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

Note: The OpenPi training pipeline handles this automatically - you don't need to manually create the sampler.

## Troubleshooting

### Error: "Number of datasets must match number of weights"

Make sure each `LeRobotDatasetConfig` has a corresponding weight value. If not specified, the default weight is `1.0`.

### Error: "repo_ids list is empty"

Ensure you have at least one dataset configuration in the `repo_ids` list.

### Error: "Normalization stats not found"

Make sure you've specified valid `AssetsConfig` pointing to normalization stats that are compatible with all your datasets.

## Performance Notes

- **Memory**: Each dataset is loaded independently, so memory usage scales with the number of datasets
- **Loading time**: All datasets are loaded at initialization, which may take longer than loading a single dataset
- **Sampling**: The weighted sampling adds minimal overhead during training

## Further Reading

- [LeRobot Dataset Documentation](https://github.com/huggingface/lerobot)
- [Data Pipeline Implementation](KI_DATA_PIPELINE_IMPLEMENTATION.md)
- [Normalization Stats Guide](norm_stats.md)
