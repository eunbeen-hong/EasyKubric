#!/usr/bin/env python3
"""
Example script demonstrating how to use the multi-view dataset generator.
"""

import sys
from pathlib import Path
from multi_view_dataset_generator import MultiViewDatasetGenerator

def main():
    """Example usage of the multi-view dataset generator."""
    
    # Create generator with custom parameters
    generator = MultiViewDatasetGenerator(
        output_dir="example_multi_view_output",
        resolution=(256, 256),
        num_frames=24,
        frame_rate=12,
        num_cameras=4,
        samples_per_pixel=32  # Lower for faster rendering
    )
    
    print("Generating multi-view dataset...")
    print(f"Output directory: {generator.output_dir}")
    print(f"Resolution: {generator.resolution}")
    print(f"Number of frames: {generator.num_frames}")
    print(f"Number of cameras: {generator.num_cameras}")
    
    # Generate the dataset
    generator.generate_dataset(seed=123)
    
    print("\nDataset generation complete!")
    print("\nDirectory structure:")
    print("example_multi_view_output/")
    print("├── camera_00/")
    print("│   ├── rgba_00000.png")
    print("│   ├── depth_00000.tiff")
    print("│   ├── normal_00000.png")
    print("│   ├── segmentation_00000.png")
    print("│   ├── forward_flow_00000.png")
    print("│   ├── backward_flow_00000.png")
    print("│   ├── object_coordinates_00000.png")
    print("│   ├── camera_metadata.json")
    print("│   └── scene_metadata.json")
    print("├── camera_01/")
    print("│   └── ... (same structure)")
    print("├── camera_02/")
    print("│   └── ... (same structure)")
    print("├── camera_03/")
    print("│   └── ... (same structure)")
    print("└── global_metadata.json")

if __name__ == "__main__":
    main()
