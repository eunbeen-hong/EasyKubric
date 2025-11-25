# DiffSynth-Studio Multi-View Kubric Integration

This package provides seamless integration between the MultiViewKubricDataset and the DiffSynth-Studio training framework. It allows you to use multi-view Kubric rendered data directly in your DiffSynth-Studio training pipelines.

## Features

- **Full DiffSynth-Studio compatibility**: Integrates with the unified dataset framework
- **Multi-view data support**: Handles synchronized multi-view sequences
- **Flexible data loading**: Optional loading of specific render passes
- **Camera pose handling**: Automatic extraction and formatting of camera poses
- **Image processing**: Built-in image resizing and preprocessing
- **Training utilities**: Ready-to-use training scripts and configurations

## Files Overview

- **`diffsynth_multi_view_dataset.py`**: Core adapter classes and dataset implementations
- **`diffsynth_multi_view_config.py`**: Configuration utilities and factory functions
- **`example_diffsynth_training.py`**: Example training script with 4D reconstruction model
- **`test_diffsynth_integration.py`**: Test script to verify integration works correctly

## Quick Start

### 1. Basic Usage

```python
from diffsynth_multi_view_dataset import create_multi_view_kubric_dataset

# Create dataset
dataset = create_multi_view_kubric_dataset(
    data_dir="/path/to/multi_view_output",
    sequence_length=4,
    height=512,
    width=512,
    camera_subset=[0, 1, 2, 3]
)

# Use in training
for sample in dataset:
    images = sample['images']  # List of lists of PIL Images
    camera_poses = sample['camera_poses']  # [num_cameras, num_frames, 4, 4]
    camera_intrinsics = sample['camera_intrinsics']  # [num_cameras, 3, 3]
    # Your training code here...
```

### 2. Using with DiffSynth-Studio DataLoader

```python
from diffsynth_multi_view_config import create_training_dataset

# Create DataLoader
dataloader = create_training_dataset(
    multi_view_data_dir="/path/to/multi_view_output",
    batch_size=2,
    sequence_length=4,
    height=512,
    width=512,
    camera_subset=[0, 1, 2, 3]
)

# Use in training loop
for batch in dataloader:
    images = batch['images']  # List of lists of PIL Images
    camera_poses = batch['camera_poses']  # [batch_size, num_cameras, num_frames, 4, 4]
    # Your training code here...
```

### 3. Unified Dataset (Mixed Data Types)

```python
from diffsynth_multi_view_config import create_unified_training_dataset

# Create unified dataset that can handle both regular and multi-view data
dataloader = create_unified_training_dataset(
    base_path="/path/to/regular_dataset",
    multi_view_data_dir="/path/to/multi_view_output",
    metadata_path="/path/to/metadata.json",
    data_file_keys=("image_path", "multi_view"),
    batch_size=2
)
```

## Data Format

### Sample Structure

Each sample returned by the dataset contains:

```python
{
    'images': List[List[PIL.Image]],           # [num_cameras][num_frames] - PIL Images
    'camera_poses': np.ndarray,               # [num_cameras, num_frames, 4, 4] - 4x4 transformation matrices
    'camera_intrinsics': np.ndarray,          # [num_cameras, 3, 3] - Camera intrinsic matrices
    'depth': List[List[np.ndarray]],          # [num_cameras][num_frames] - Depth maps (optional)
    'normal': List[List[PIL.Image]],          # [num_cameras][num_frames] - Normal maps (optional)
    'segmentation': List[List[np.ndarray]],   # [num_cameras][num_frames] - Segmentation masks (optional)
    'forward_flow': List[List[np.ndarray]],   # [num_cameras][num_frames] - Forward optical flow (optional)
    'backward_flow': List[List[np.ndarray]],  # [num_cameras][num_frames] - Backward optical flow (optional)
    'object_coordinates': List[List[PIL.Image]], # [num_cameras][num_frames] - Object coordinates (optional)
    'metadata': dict,                         # Additional metadata
    'num_cameras': int,                       # Number of cameras
    'num_frames': int                         # Number of frames in sequence
}
```

### Camera Poses

Camera poses are provided as 4x4 transformation matrices in world coordinates:
- `camera_poses[cam_idx, frame_idx]` gives the pose of camera `cam_idx` at frame `frame_idx`
- The pose transforms from camera coordinates to world coordinates

### Camera Intrinsics

Camera intrinsics are 3x3 matrices with:
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (image center)
- Format: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`

## API Reference

### Core Classes

#### MultiViewKubricDataset

```python
MultiViewKubricDataset(
    data_dir: str,                          # Path to dataset directory
    sequence_length: Optional[int] = None,  # Frames per sequence (None = all)
    load_flows: bool = True,                # Load optical flow data
    load_segmentation: bool = True,         # Load segmentation masks
    load_depth: bool = True,                # Load depth maps
    load_normals: bool = True,              # Load normal maps
    load_object_coords: bool = True,        # Load object coordinates
    camera_subset: Optional[List[int]] = None,  # Camera IDs to load (None = all)
    max_pixels: int = 1920*1080,           # Maximum pixels for image resizing
    height: Optional[int] = None,           # Target height for images
    width: Optional[int] = None,            # Target width for images
    height_division_factor: int = 16,       # Height must be divisible by this factor
    width_division_factor: int = 16,        # Width must be divisible by this factor
    repeat: int = 1                        # Number of times to repeat the dataset
)
```

#### MultiViewKubricUnifiedDataset

```python
MultiViewKubricUnifiedDataset(
    base_path: str,                         # Base path for the dataset
    multi_view_data_dir: str,               # Path to multi-view dataset
    metadata_path: Optional[str] = None,    # Path to metadata file
    data_file_keys: tuple = (),             # Keys for data files in metadata
    main_data_operator: Optional[DataProcessingOperator] = None,  # Main data operator
    special_operator_map: Optional[Dict[str, DataProcessingOperator]] = None,  # Special operators
    # ... multi-view specific parameters
)
```

### Factory Functions

#### create_multi_view_kubric_dataset

```python
create_multi_view_kubric_dataset(data_dir: str, **kwargs) -> MultiViewKubricDataset
```

#### create_unified_multi_view_dataset

```python
create_unified_multi_view_dataset(
    base_path: str, 
    multi_view_data_dir: str,
    **kwargs
) -> MultiViewKubricUnifiedDataset
```

### Configuration Functions

#### create_training_dataset

```python
create_training_dataset(
    multi_view_data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    sequence_length: int = 4,
    height: int = 512,
    width: int = 512,
    camera_subset: list = None,
    load_additional_data: bool = True
) -> DataLoader
```

#### create_unified_training_dataset

```python
create_unified_training_dataset(
    base_path: str,
    multi_view_data_dir: str,
    metadata_path: str = None,
    data_file_keys: tuple = ("image_path",),
    batch_size: int = 1,
    num_workers: int = 4,
    sequence_length: int = 4,
    height: int = 512,
    width: int = 512,
    camera_subset: list = None,
    load_additional_data: bool = True
) -> DataLoader
```

## Example Training Scripts

### Basic Training

```bash
# Run the example training script
python example_diffsynth_training.py --data_dir /path/to/multi_view_output --batch_size 2 --num_epochs 100
```

### Test Integration

```bash
# Test the integration
python test_diffsynth_integration.py /path/to/multi_view_output
```

## Configuration Examples

### Multi-View Only

```python
from diffsynth_multi_view_config import create_training_dataset

dataloader = create_training_dataset(
    multi_view_data_dir="/path/to/multi_view_output",
    batch_size=2,
    sequence_length=4,
    height=512,
    width=512,
    camera_subset=[0, 1, 2, 3],
    load_additional_data=True
)
```

### Unified Dataset

```python
from diffsynth_multi_view_config import create_unified_training_dataset

dataloader = create_unified_training_dataset(
    base_path="/path/to/regular_dataset",
    multi_view_data_dir="/path/to/multi_view_output",
    metadata_path="/path/to/metadata.json",
    data_file_keys=("image_path", "multi_view"),
    batch_size=2,
    sequence_length=4
)
```

### Custom Configuration

```python
from diffsynth_multi_view_config import get_example_training_config

config = get_example_training_config()
print(config)
```

## Integration with DiffSynth-Studio

### 1. Data Processing Pipeline

The adapter integrates with DiffSynth-Studio's data processing pipeline:

```python
from diffsynth_multi_view_config import get_multi_view_kubric_operator

# Create operator for multi-view data
operator = get_multi_view_kubric_operator(
    data_dir="/path/to/multi_view_output",
    sequence_length=4,
    height=512,
    width=512
)

# Use in pipeline
pipeline = DataProcessingPipeline([operator])
processed_data = pipeline(data)
```

### 2. Unified Dataset

The unified dataset can handle both regular and multi-view data:

```python
from diffsynth_multi_view_dataset import MultiViewKubricUnifiedDataset

dataset = MultiViewKubricUnifiedDataset(
    base_path="/path/to/regular_dataset",
    multi_view_data_dir="/path/to/multi_view_output",
    metadata_path="/path/to/metadata.json",
    data_file_keys=("image_path", "multi_view")
)
```

### 3. Training Integration

The adapter provides seamless integration with DiffSynth-Studio training:

```python
from diffsynth_multi_view_config import create_training_dataset

# Create training dataloader
dataloader = create_training_dataset(
    multi_view_data_dir="/path/to/multi_view_output",
    batch_size=2,
    sequence_length=4
)

# Use in training loop
for batch in dataloader:
    # Process multi-view data
    images = batch['images']
    camera_poses = batch['camera_poses']
    camera_intrinsics = batch['camera_intrinsics']
    
    # Your model training code here...
```

## Requirements

- Python 3.7+
- PyTorch
- DiffSynth-Studio (optional, with fallback)
- NumPy
- PIL/Pillow
- OpenCV
- Matplotlib (for visualization examples)

## Installation

```bash
# Install required packages
pip install torch torchvision numpy pillow opencv-python matplotlib

# Install DiffSynth-Studio (optional)
pip install diffsynth

# The adapter is ready to use - no additional installation needed
```

## Notes

- The adapter automatically handles image resizing and preprocessing
- Camera poses are automatically extracted and converted to 4x4 transformation matrices
- All data is returned in DiffSynth-Studio compatible format
- The unified dataset can handle mixed data types (regular images + multi-view sequences)
- Memory usage scales with sequence length and number of cameras - adjust batch size accordingly

## Troubleshooting

### Import Errors

If you get import errors for `diffsynth.trainers.unified_dataset`, the adapter will fall back to basic functionality. This is normal if DiffSynth-Studio is not installed.

### Memory Issues

For large sequences or many cameras, reduce batch size or use fewer cameras:

```python
dataloader = create_training_dataset(
    multi_view_data_dir="/path/to/multi_view_output",
    batch_size=1,  # Reduce batch size
    camera_subset=[0, 1]  # Use fewer cameras
)
```

### Data Loading Issues

Make sure your multi-view dataset directory structure is correct:

```
multi_view_output/
├── global_metadata.json
├── camera_00/
│   ├── rgba_00000.png
│   ├── depth_00000.tiff
│   └── ...
├── camera_01/
│   └── ...
└── ...
```

## Support

For issues or questions about the DiffSynth-Studio integration, please check:

1. The test script: `python test_diffsynth_integration.py /path/to/multi_view_output`
2. The example training script: `python example_diffsynth_training.py --help`
3. The configuration examples in `diffsynth_multi_view_config.py`
