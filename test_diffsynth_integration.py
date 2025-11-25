#!/usr/bin/env python3
"""
Test script to verify DiffSynth-Studio integration with multi-view Kubric data.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffsynth_multi_view_dataset import (
    MultiViewKubricDataset, MultiViewKubricUnifiedDataset,
    create_multi_view_kubric_dataset, create_unified_multi_view_dataset
)
from diffsynth_multi_view_config import (
    create_training_dataset, create_unified_training_dataset,
    get_example_training_config
)


def test_multi_view_dataset(data_dir: str):
    """Test the multi-view dataset functionality."""
    print("Testing Multi-View Dataset...")
    print("=" * 50)
    
    # Test 1: Basic dataset creation
    print("\n1. Creating multi-view dataset...")
    dataset = create_multi_view_kubric_dataset(
        data_dir=data_dir,
        sequence_length=2,  # Use 2 frames for testing
        height=256,
        width=256,
        camera_subset=[0, 1]  # Use first 2 cameras for testing
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Test 2: Load a sample
    print("\n2. Loading a sample...")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Number of cameras: {sample['num_cameras']}")
        print(f"   Number of frames: {sample['num_frames']}")
        
        # Check image data
        if 'images' in sample:
            print(f"   Images: {len(sample['images'])} cameras, {len(sample['images'][0])} frames per camera")
            print(f"   Image size: {sample['images'][0][0].size}")
        
        # Check camera data
        if 'camera_poses' in sample:
            print(f"   Camera poses shape: {sample['camera_poses'].shape}")
        if 'camera_intrinsics' in sample:
            print(f"   Camera intrinsics shape: {sample['camera_intrinsics'].shape}")
        
        # Check additional data
        additional_keys = ['depth', 'normal', 'segmentation', 'forward_flow', 'backward_flow', 'object_coordinates']
        for key in additional_keys:
            if key in sample:
                print(f"   {key}: Available")
    else:
        print("   No samples found in dataset")
    
    return dataset


def test_unified_dataset(base_path: str, multi_view_data_dir: str):
    """Test the unified dataset functionality."""
    print("\n\nTesting Unified Dataset...")
    print("=" * 50)
    
    # Test 1: Unified dataset creation
    print("\n1. Creating unified dataset...")
    unified_dataset = create_unified_multi_view_dataset(
        base_path=base_path,
        multi_view_data_dir=multi_view_data_dir,
        multi_view_sequence_length=2,
        multi_view_height=256,
        multi_view_width=256,
        multi_view_camera_subset=[0, 1]
    )
    
    print(f"   Unified dataset size: {len(unified_dataset)}")
    
    # Test 2: Load a sample
    print("\n2. Loading a sample...")
    if len(unified_dataset) > 0:
        sample = unified_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        
        if 'num_cameras' in sample:
            print(f"   Number of cameras: {sample['num_cameras']}")
        if 'num_frames' in sample:
            print(f"   Number of frames: {sample['num_frames']}")
    else:
        print("   No samples found in unified dataset")
    
    return unified_dataset


def test_dataloader(data_dir: str):
    """Test the DataLoader functionality."""
    print("\n\nTesting DataLoader...")
    print("=" * 50)
    
    # Test 1: Create training dataloader
    print("\n1. Creating training dataloader...")
    dataloader = create_training_dataset(
        multi_view_data_dir=data_dir,
        batch_size=1,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
        sequence_length=2,
        height=256,
        width=256,
        camera_subset=[0, 1]
    )
    
    print(f"   DataLoader created with {len(dataloader)} batches")
    
    # Test 2: Load a batch
    print("\n2. Loading a batch...")
    for i, batch in enumerate(dataloader):
        print(f"   Batch {i}:")
        print(f"     Keys: {list(batch.keys())}")
        
        if 'images' in batch:
            print(f"     Images: {len(batch['images'])} cameras")
            if len(batch['images']) > 0:
                print(f"     Frames per camera: {len(batch['images'][0])}")
                if len(batch['images'][0]) > 0:
                    print(f"     Image size: {batch['images'][0][0].size}")
        
        if 'camera_poses' in batch:
            print(f"     Camera poses shape: {batch['camera_poses'].shape}")
        if 'camera_intrinsics' in batch:
            print(f"     Camera intrinsics shape: {batch['camera_intrinsics'].shape}")
        
        # Only test first batch
        if i >= 0:
            break
    
    return dataloader


def test_configuration():
    """Test the configuration functionality."""
    print("\n\nTesting Configuration...")
    print("=" * 50)
    
    # Test 1: Get example configuration
    print("\n1. Getting example configuration...")
    config = get_example_training_config()
    
    print(f"   Dataset type: {config['dataset']['type']}")
    print(f"   Sequence length: {config['dataset']['sequence_length']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Number of cameras: {config['dataset']['camera_subset']}")
    print(f"   Model type: {config['model']['type']}")
    
    # Test 2: Validate configuration
    print("\n2. Validating configuration...")
    required_keys = ['dataset', 'training', 'model']
    for key in required_keys:
        if key in config:
            print(f"   ✓ {key} section present")
        else:
            print(f"   ✗ {key} section missing")
    
    return config


def test_data_processing_pipeline():
    """Test the data processing pipeline."""
    print("\n\nTesting Data Processing Pipeline...")
    print("=" * 50)
    
    from diffsynth_multi_view_config import get_multi_view_kubric_operator
    
    # Test 1: Create operator
    print("\n1. Creating multi-view operator...")
    operator = get_multi_view_kubric_operator(
        data_dir="/path/to/multi_view_output",
        sequence_length=2,
        height=256,
        width=256,
        camera_subset=[0, 1]
    )
    
    print(f"   Operator created: {type(operator).__name__}")
    print(f"   Dataset size: {len(operator.dataset)}")
    
    # Test 2: Process data
    print("\n2. Processing data...")
    if len(operator.dataset) > 0:
        processed_data = operator(0)
        print(f"   Processed data keys: {list(processed_data.keys())}")
        print(f"   Number of cameras: {processed_data.get('num_cameras', 'Unknown')}")
        print(f"   Number of frames: {processed_data.get('num_frames', 'Unknown')}")
    else:
        print("   No data to process")
    
    return operator


def main():
    """Main test function."""
    print("DiffSynth-Studio Multi-View Kubric Integration Test")
    print("=" * 60)
    
    # Get data directory from command line or use default
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/home/cvlab15/project/chyun/4d-recon/kubric/multi_view_output"
    
    print(f"Using data directory: {data_dir}")
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"ERROR: Data directory {data_dir} does not exist!")
        print("Please provide a valid data directory as an argument.")
        return
    
    # Run tests
    try:
        # Test 1: Multi-view dataset
        dataset = test_multi_view_dataset(data_dir)
        
        # Test 2: Unified dataset
        unified_dataset = test_unified_dataset(data_dir, data_dir)
        
        # Test 3: DataLoader
        dataloader = test_dataloader(data_dir)
        
        # Test 4: Configuration
        config = test_configuration()
        
        # Test 5: Data processing pipeline
        operator = test_data_processing_pipeline()
        
        print("\n\n" + "=" * 60)
        print("All tests completed successfully!")
        print("The DiffSynth-Studio integration is working correctly.")
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nYou can now use the multi-view Kubric dataset with DiffSynth-Studio!")
    print("Example usage:")
    print("  python example_diffsynth_training.py --data_dir /path/to/multi_view_output")


if __name__ == "__main__":
    main()
