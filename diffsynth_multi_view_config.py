#!/usr/bin/env python3
"""
Configuration file for DiffSynth-Studio multi-view Kubric dataset training.
"""

import torch

# Import from unified_dataset with fallback
try:
    from diffsynth.trainers.unified_dataset import (
        DataProcessingPipeline, RouteByType, ToAbsolutePath, 
        LoadImage, ImageCropAndResize, SequencialProcess, ToList
    )
except ImportError:
    # Fallback for when diffsynth is not available
    print("Warning: diffsynth.trainers.unified_dataset not found. Some features may not work.")
    DataProcessingPipeline = object
    RouteByType = object
    ToAbsolutePath = object
    LoadImage = object
    ImageCropAndResize = object
    SequencialProcess = object
    ToList = object

from diffsynth_multi_view_dataset import (
    MultiViewKubricDataset, MultiViewKubricUnifiedDataset,
    create_multi_view_kubric_dataset, create_unified_multi_view_dataset
)


def get_multi_view_kubric_operator(
    data_dir: str,
    max_pixels: int = 1920*1080,
    height: int = 512,
    width: int = 512,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
    sequence_length: int = 4,
    load_flows: bool = True,
    load_segmentation: bool = True,
    load_depth: bool = True,
    load_normals: bool = True,
    load_object_coords: bool = True,
    camera_subset: list = None
):
    """
    Get a data processing operator for multi-view Kubric data.
    
    Args:
        data_dir: Path to the multi-view dataset directory
        max_pixels: Maximum number of pixels for image resizing
        height: Target height for images
        width: Target width for images
        height_division_factor: Height must be divisible by this factor
        width_division_factor: Width must be divisible by this factor
        sequence_length: Number of frames per sequence
        load_flows: Whether to load optical flow data
        load_segmentation: Whether to load segmentation masks
        load_depth: Whether to load depth maps
        load_normals: Whether to load normal maps
        load_object_coords: Whether to load object coordinates
        camera_subset: List of camera IDs to load (None = all cameras)
    
    Returns:
        DataProcessingOperator for multi-view Kubric data
    """
    from diffsynth_multi_view_dataset import MultiViewKubricOperator
    
    return MultiViewKubricOperator(
        data_dir=data_dir,
        sequence_length=sequence_length,
        load_flows=load_flows,
        load_segmentation=load_segmentation,
        load_depth=load_depth,
        load_normals=load_normals,
        load_object_coords=load_object_coords,
        camera_subset=camera_subset,
        max_pixels=max_pixels,
        height=height,
        width=width,
        height_division_factor=height_division_factor,
        width_division_factor=width_division_factor
    )


def create_training_dataset(
    multi_view_data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    sequence_length: int = 4,
    height: int = 512,
    width: int = 512,
    camera_subset: list = None,
    load_additional_data: bool = True
):
    """
    Create a training dataset for multi-view Kubric data.
    
    Args:
        multi_view_data_dir: Path to the multi-view dataset directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        sequence_length: Number of frames per sequence
        height: Target height for images
        width: Target width for images
        camera_subset: List of camera IDs to load (None = all cameras)
        load_additional_data: Whether to load additional render passes
    
    Returns:
        PyTorch DataLoader
    """
    # Create the dataset
    dataset = create_multi_view_kubric_dataset(
        data_dir=multi_view_data_dir,
        sequence_length=sequence_length,
        load_flows=load_additional_data,
        load_segmentation=load_additional_data,
        load_depth=load_additional_data,
        load_normals=load_additional_data,
        load_object_coords=load_additional_data,
        camera_subset=camera_subset,
        height=height,
        width=width,
        height_division_factor=16,
        width_division_factor=16
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_unified_training_dataset(
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
):
    """
    Create a unified training dataset that can handle both regular and multi-view data.
    
    Args:
        base_path: Base path for the dataset
        multi_view_data_dir: Path to the multi-view dataset directory
        metadata_path: Path to metadata file (optional)
        data_file_keys: Keys for data files in metadata
        batch_size: Batch size for training
        num_workers: Number of worker processes
        sequence_length: Number of frames per sequence
        height: Target height for images
        width: Target width for images
        camera_subset: List of camera IDs to load (None = all cameras)
        load_additional_data: Whether to load additional render passes
    
    Returns:
        PyTorch DataLoader
    """
    # Create main data operator for regular images
    main_data_operator = RouteByType(operator_map=[
        (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(
            height, width, 1920*1080, 16, 16
        )),
        (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(
            height, width, 1920*1080, 16, 16
        ))),
    ])
    
    # Create special operator for multi-view data
    special_operator_map = {
        "multi_view": get_multi_view_kubric_operator(
            data_dir=multi_view_data_dir,
            sequence_length=sequence_length,
            height=height,
            width=width,
            camera_subset=camera_subset,
            load_flows=load_additional_data,
            load_segmentation=load_additional_data,
            load_depth=load_additional_data,
            load_normals=load_additional_data,
            load_object_coords=load_additional_data
        )
    }
    
    # Create unified dataset
    dataset = create_unified_multi_view_dataset(
        base_path=base_path,
        multi_view_data_dir=multi_view_data_dir,
        metadata_path=metadata_path,
        data_file_keys=data_file_keys,
        main_data_operator=main_data_operator,
        special_operator_map=special_operator_map,
        multi_view_sequence_length=sequence_length,
        multi_view_height=height,
        multi_view_width=width,
        multi_view_camera_subset=camera_subset,
        multi_view_load_flows=load_additional_data,
        multi_view_load_segmentation=load_additional_data,
        multi_view_load_depth=load_additional_data,
        multi_view_load_normals=load_additional_data,
        multi_view_load_object_coords=load_additional_data
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


# Example configurations
MULTI_VIEW_CONFIG = {
    "data_dir": "/path/to/multi_view_output",
    "sequence_length": 4,
    "height": 512,
    "width": 512,
    "camera_subset": [0, 1, 2, 3],  # All 4 cameras
    "load_flows": True,
    "load_segmentation": True,
    "load_depth": True,
    "load_normals": True,
    "load_object_coords": True
}

TRAINING_CONFIG = {
    "batch_size": 2,
    "num_workers": 4,
    "sequence_length": 4,
    "height": 512,
    "width": 512,
    "camera_subset": [0, 1, 2, 3],
    "load_additional_data": True
}

UNIFIED_CONFIG = {
    "base_path": "/path/to/base_dataset",
    "multi_view_data_dir": "/path/to/multi_view_output",
    "metadata_path": "/path/to/metadata.json",
    "data_file_keys": ("image_path", "multi_view"),
    "batch_size": 2,
    "num_workers": 4,
    "sequence_length": 4,
    "height": 512,
    "width": 512,
    "camera_subset": [0, 1, 2, 3],
    "load_additional_data": True
}


def get_example_training_config():
    """Get an example training configuration."""
    return {
        "dataset": {
            "type": "multi_view_kubric",
            "data_dir": "/path/to/multi_view_output",
            "sequence_length": 4,
            "height": 512,
            "width": 512,
            "camera_subset": [0, 1, 2, 3],
            "load_flows": True,
            "load_segmentation": True,
            "load_depth": True,
            "load_normals": True,
            "load_object_coords": True
        },
        "training": {
            "batch_size": 2,
            "num_workers": 4,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "save_interval": 10,
            "log_interval": 100
        },
        "model": {
            "type": "4d_reconstruction",
            "num_cameras": 4,
            "sequence_length": 4,
            "input_channels": 3,
            "hidden_dim": 256
        }
    }


if __name__ == "__main__":
    # Example usage
    print("Multi-View Kubric Dataset Configuration")
    print("=" * 50)
    
    # Example 1: Multi-view dataset only
    print("\n1. Multi-view dataset only:")
    dataloader = create_training_dataset(
        multi_view_data_dir="/path/to/multi_view_output",
        batch_size=2,
        sequence_length=4,
        height=512,
        width=512
    )
    print(f"   DataLoader created with {len(dataloader)} batches")
    
    # Example 2: Unified dataset
    print("\n2. Unified dataset:")
    unified_dataloader = create_unified_training_dataset(
        base_path="/path/to/base",
        multi_view_data_dir="/path/to/multi_view_output",
        batch_size=2,
        sequence_length=4
    )
    print(f"   Unified DataLoader created with {len(unified_dataloader)} batches")
    
    # Example 3: Configuration
    print("\n3. Example configuration:")
    config = get_example_training_config()
    print(f"   Dataset type: {config['dataset']['type']}")
    print(f"   Sequence length: {config['dataset']['sequence_length']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Number of cameras: {config['dataset']['camera_subset']}")
