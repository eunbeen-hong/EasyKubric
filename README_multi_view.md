# Multi-View Dataset Generator

This tool generates multi-view datasets using the Kubric renderer with MOVi-E format. It creates synchronized renderings from four different camera trajectories for a single scene.

## Features

- **Four synchronized camera trajectories**: Each camera follows a different linear path around the scene
- **MOVi-E format compatibility**: Follows the MOVi-E dataset structure and metadata format
- **Separate directories**: Each camera trajectory gets its own output directory
- **Complete metadata**: Camera parameters, object information, and scene metadata are saved
- **Multiple render passes**: RGB, depth, normal, segmentation, optical flow, and object coordinates

## Usage

### Basic Usage

```bash
python multi_view_dataset_generator.py --output_dir my_dataset --num_frames 24 --num_cameras 4
```

### Command Line Arguments

- `--output_dir`: Output directory for the dataset (default: "multi_view_output")
- `--resolution`: Image resolution as width height (default: 256 256)
- `--num_frames`: Number of frames to render (default: 24)
- `--num_cameras`: Number of camera trajectories (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--samples_per_pixel`: Rendering quality parameter (default: 64)

### Example Usage

```python
from multi_view_dataset_generator import MultiViewDatasetGenerator

# Create generator
generator = MultiViewDatasetGenerator(
    output_dir="my_multi_view_dataset",
    resolution=(512, 512),
    num_frames=48,
    num_cameras=4,
    samples_per_pixel=128
)

# Generate dataset
generator.generate_dataset(seed=42)
```

## Output Structure

The generator creates the following directory structure:

```
output_dir/
├── camera_00/                    # First camera trajectory
│   ├── rgba_00000.png           # RGB images
│   ├── rgba_00001.png
│   ├── ...
│   ├── depth_00000.tiff         # Depth maps
│   ├── normal_00000.png         # Normal maps
│   ├── segmentation_00000.png   # Instance segmentation
│   ├── forward_flow_00000.png   # Optical flow
│   ├── backward_flow_00000.png
│   ├── object_coordinates_00000.png
│   ├── camera_metadata.json     # Camera-specific metadata
│   └── scene_metadata.json      # Scene metadata
├── camera_01/                    # Second camera trajectory
│   └── ... (same structure)
├── camera_02/                    # Third camera trajectory
│   └── ... (same structure)
├── camera_03/                    # Fourth camera trajectory
│   └── ... (same structure)
└── global_metadata.json          # Global dataset metadata
```

## Camera Trajectories

Each camera follows a linear trajectory:

1. **Starting position**: Randomly sampled from a half-sphere shell around the scene
2. **Movement**: Linear movement in a random direction with random speed
3. **Look-at target**: Slight random variation around the scene center
4. **Synchronization**: All cameras render the same timesteps simultaneously

## Metadata Format

### Camera Metadata (`camera_metadata.json`)
```json
{
  "camera_id": 0,
  "focal_length": 35.0,
  "sensor_width": 32.0,
  "field_of_view": 0.85755605,
  "positions": [[x, y, z], ...],
  "quaternions": [[w, x, y, z], ...],
  "lookat_target": [x, y, z],
  "num_frames": 24,
  "resolution": [256, 256],
  "num_instances": 15
}
```

### Global Metadata (`global_metadata.json`)
```json
{
  "dataset_type": "multi_view_movi_e",
  "num_cameras": 4,
  "num_frames": 24,
  "resolution": [256, 256],
  "frame_rate": 12,
  "seed": 42,
  "camera_trajectories": [...]
}
```

## Requirements

- Python 3.7+
- Kubric
- Blender
- PyBullet
- NumPy
- PIL/Pillow

## Installation

Make sure you have Kubric installed and configured:

```bash
# Install Kubric dependencies
pip install kubric

# Make sure Blender is available in your PATH
# The script will use the default Blender installation
```

## Example Script

Run the example script to see the generator in action:

```bash
python example_multi_view.py
```

This will create a sample multi-view dataset in the `example_multi_view_output` directory.

## Notes

- The script uses the same object spawning and simulation logic as the original MOVi-E format
- Each camera trajectory is independent but synchronized in time
- All cameras render the same scene with the same object dynamics
- The output follows the MOVi-E format structure for compatibility with existing tools
