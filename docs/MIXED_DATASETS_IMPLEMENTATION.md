# Mixed Dataset Training - Implementation Summary

This document provides a summary of the changes made to support training on mixed LeRobot datasets.

## Changes Overview

### 1. New Data Structures ([config.py](../src/openpi/training/config.py))

#### `LeRobotDatasetConfig`
A new dataclass to represent individual dataset configurations in a mixed dataset setup:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotDatasetConfig:
    """Configuration for a single LeRobot dataset in a mixed dataset setup."""
    repo_id: str          # The LeRobot repository ID
    weight: float = 1.0   # Sampling weight (default 1.0)
```

#### Updated `DataConfig`
Added support for multiple datasets:

```python
@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None                              # Single dataset (original)
    repo_ids: Sequence[LeRobotDatasetConfig] | None = None  # Multiple datasets (new)
    # ... other fields
```

#### Updated `DataConfigFactory`
Modified to support both single and multiple dataset configurations:

```python
@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str | None = tyro.MISSING                     # Single dataset
    repo_ids: Sequence[LeRobotDatasetConfig] | None = None # Multiple datasets
    # ... other fields
```

### 2. Mixed Dataset Implementation ([data_loader.py](../src/openpi/training/data_loader.py))

#### `MixedDataset` Class
A new dataset class that combines multiple LeRobot datasets with weighted sampling:

**Key Features:**
- Combines multiple datasets into a single unified dataset
- Supports configurable sampling weights per dataset
- Maintains index mapping to route samples to the correct underlying dataset
- Provides sample weights for use with PyTorch's `WeightedRandomSampler`

**API:**
```python
class MixedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], weights: Sequence[float])
    def __getitem__(self, index: int) -> dict
    def __len__(self) -> int
    def get_sample_weights(self) -> list[float]
```

#### Updated `create_torch_dataset`
Enhanced to handle both single and mixed dataset configurations:

1. **Mixed Dataset Mode** (`repo_ids` is specified):
   - Creates multiple LeRobotDataset instances
   - Wraps them in a `MixedDataset`
   - Merges task mappings from all datasets
   - Applies prompt transforms if needed

2. **Single Dataset Mode** (`repo_id` is specified):
   - Original behavior preserved for backward compatibility

#### Updated Helper Functions
- `transform_dataset`: Now checks for both `repo_id` and `repo_ids`
- `transform_iterable_dataset`: Same update for consistency

### 3. Example Configuration

Added `pi05_droid_mixed_example` in [config.py](../src/openpi/training/config.py):

```python
TrainConfig(
    name="pi05_droid_mixed_example",
    model=pi05_config.Pi05Config(action_dim=32, action_horizon=15),
    data=LeRobotDROIDDataConfig(
        repo_ids=[
            LeRobotDatasetConfig(repo_id="your_username/droid_dataset1", weight=1.0),
            LeRobotDatasetConfig(repo_id="your_username/droid_dataset2", weight=2.0),
            LeRobotDatasetConfig(repo_id="your_username/droid_dataset3", weight=1.0),
        ],
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
            asset_id="droid",
        ),
    ),
    # ... other config
)
```

### 4. Documentation

Created comprehensive user guide: [docs/mixed_datasets.md](mixed_datasets.md)

**Contents:**
- Overview and use cases
- Configuration examples
- Sampling weight explanation
- Multiple example configurations
- Important considerations (normalization, compatibility, etc.)
- Implementation details
- Troubleshooting guide
- Performance notes

## How to Use

### Basic Usage

Replace the single `repo_id` with a list of `repo_ids`:

**Before:**
```python
data=LeRobotDROIDDataConfig(
    repo_id="your_username/dataset1",
    # ... other config
)
```

**After:**
```python
data=LeRobotDROIDDataConfig(
    repo_ids=[
        LeRobotDatasetConfig(repo_id="your_username/dataset1", weight=1.0),
        LeRobotDatasetConfig(repo_id="your_username/dataset2", weight=2.0),
    ],
    # ... other config
)
```

### Training

Use the same training command:
```bash
uv run scripts/train.py --config-name=your_mixed_config_name
```

## Backward Compatibility

✅ **All existing configurations continue to work** without modification:
- Single `repo_id` configurations work as before
- No breaking changes to existing code
- `repo_ids=None` defaults to single dataset mode

## Key Design Decisions

1. **Weighted Sampling**: Each sample's probability = (dataset_weight / total_weight) / dataset_size
   - Allows balancing datasets of different sizes
   - Higher weights mean more frequent sampling from that dataset

2. **Index Mapping**: MixedDataset uses cumulative sizes to map global indices to dataset-local indices
   - Efficient O(log n) lookup using binary search
   - No memory overhead for index mapping

3. **Task Merging**: When `prompt_from_task=True`, tasks from all datasets are merged
   - Assumes task indices don't conflict across datasets
   - All datasets contribute their task mappings

4. **Normalization**: All datasets must use the same normalization stats
   - Ensures consistent preprocessing across datasets
   - Specified once in the `AssetsConfig`

## Testing Recommendations

1. **Verify dataset loading**:
   ```python
   # Check that all datasets load correctly
   # Check that weights are applied as expected
   ```

2. **Check sample distribution**:
   ```python
   # Verify sampling proportions match expected weights
   # Monitor which datasets samples come from during training
   ```

3. **Monitor training metrics**:
   - Watch for any issues with task conflicts
   - Ensure normalization works correctly across datasets

## Future Enhancements

Potential improvements for future versions:

1. **Per-dataset normalization**: Support different norm stats per dataset
2. **Dynamic reweighting**: Adjust weights during training based on loss
3. **Stratified sampling**: Ensure each batch contains samples from multiple datasets
4. **Dataset metadata tracking**: Track which dataset each sample came from for analytics

## Files Modified

1. `/home/ril_mtplab/workspace/openpi-repo/openpi/src/openpi/training/config.py`
   - Added `LeRobotDatasetConfig` dataclass
   - Updated `DataConfig` to support `repo_ids`
   - Updated `DataConfigFactory` to handle both modes
   - Added example configuration

2. `/home/ril_mtplab/workspace/openpi-repo/openpi/src/openpi/training/data_loader.py`
   - Added `MixedDataset` class
   - Updated `create_torch_dataset` for mixed datasets
   - Updated `transform_dataset` and `transform_iterable_dataset`
   - Added `WeightedRandomSampler` import

3. `/home/ril_mtplab/workspace/openpi-repo/openpi/docs/mixed_datasets.md`
   - Created comprehensive user documentation

## Questions or Issues?

If you encounter any problems or have questions:

1. Check the [mixed_datasets.md](mixed_datasets.md) documentation
2. Review the example configuration in `config.py`
3. Verify that all datasets are compatible (same action space, observation keys, etc.)
4. Ensure normalization stats are appropriate for all datasets

## Contributors

This feature was implemented to support training on multiple datasets with configurable sampling weights, enabling better multi-task learning and dataset balancing.
