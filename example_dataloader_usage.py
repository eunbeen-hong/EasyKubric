#!/usr/bin/env python3
"""
Example usage of the multi-view Kubric dataloader.
Demonstrates how to load and process multi-view sequences.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from multi_view_dataloader import MultiViewKubricDataset, create_dataloader


def visualize_multi_view_sequence(dataset, sequence_idx=0, save_path=None):
    """
    Visualize a multi-view sequence by showing RGB images from all cameras.
    
    Args:
        dataset: MultiViewKubricDataset instance
        sequence_idx: Index of the sequence to visualize
        save_path: Path to save the visualization (optional)
    """
    # Load the sequence
    sample = dataset[sequence_idx]
    
    num_cameras = sample['rgba'].shape[0]
    num_frames = sample['rgba'].shape[1]
    
    # Create subplot grid
    fig, axes = plt.subplots(num_cameras, num_frames, figsize=(num_frames * 3, num_cameras * 3))
    if num_cameras == 1:
        axes = axes.reshape(1, -1)
    if num_frames == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot images
    for cam_idx in range(num_cameras):
        for frame_idx in range(num_frames):
            ax = axes[cam_idx, frame_idx]
            
            # Get RGB image (remove alpha channel if present)
            rgb = sample['rgba'][cam_idx, frame_idx]
            if rgb.shape[2] == 4:  # RGBA
                rgb = rgb[:, :, :3]  # Remove alpha
            
            ax.imshow(rgb)
            ax.set_title(f'Camera {cam_idx}, Frame {frame_idx}')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def analyze_camera_trajectories(dataset):
    """
    Analyze and visualize camera trajectories.
    
    Args:
        dataset: MultiViewKubricDataset instance
    """
    print("=== Camera Trajectory Analysis ===")
    
    # Get global info
    global_info = dataset.get_global_info()
    print(f"Number of cameras: {global_info['num_cameras']}")
    print(f"Number of frames: {global_info['num_frames']}")
    print(f"Resolution: {global_info['resolution']}")
    
    # Analyze each camera
    for cam_idx in range(global_info['num_cameras']):
        cam_info = dataset.get_camera_info(cam_idx)
        print(f"\nCamera {cam_idx}:")
        print(f"  Focal length: {cam_info['focal_length']}")
        print(f"  Sensor width: {cam_info['sensor_width']}")
        print(f"  Field of view: {cam_info['field_of_view']:.4f}")
        print(f"  Look-at target: {cam_info['lookat_target']}")
        
        # Analyze trajectory
        positions = np.array(cam_info['positions'])
        print(f"  Position range:")
        print(f"    X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
        print(f"    Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
        print(f"    Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")


def test_dataloader_functionality(data_dir):
    """
    Test the dataloader functionality with various configurations.
    
    Args:
        data_dir: Path to the multi-view dataset directory
    """
    print("=== Testing Multi-View Dataloader ===")
    
    # Test 1: Load all data
    print("\n1. Loading all data...")
    dataset_full = MultiViewKubricDataset(
        data_dir=data_dir,
        sequence_length=None,  # Load all frames
        load_flows=True,
        load_segmentation=True,
        load_depth=True,
        load_normals=True,
        load_object_coords=True
    )
    
    print(f"   Dataset size: {len(dataset_full)}")
    if len(dataset_full) > 0:
        sample = dataset_full[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   RGBA shape: {sample['rgba'].shape}")
        print(f"   Camera poses shape: {sample['camera_poses'].shape}")
    
    # Test 2: Load subset of cameras
    print("\n2. Loading subset of cameras (first 2)...")
    dataset_subset = MultiViewKubricDataset(
        data_dir=data_dir,
        sequence_length=2,
        camera_subset=[0, 1],
        load_flows=False,  # Skip flows for faster loading
        load_segmentation=True,
        load_depth=True,
        load_normals=False,
        load_object_coords=False
    )
    
    print(f"   Dataset size: {len(dataset_subset)}")
    if len(dataset_subset) > 0:
        sample = dataset_subset[0]
        print(f"   RGBA shape: {sample['rgba'].shape}")
        print(f"   Number of cameras: {sample['metadata']['num_cameras']}")
    
    # Test 3: Create dataloader
    print("\n3. Creating dataloader...")
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=1,
        shuffle=False,
        sequence_length=2,
        camera_subset=[0, 1, 2]  # First 3 cameras
    )
    
    print(f"   Dataloader batches: {len(dataloader)}")
    
    # Test loading batches
    print("\n4. Testing batch loading...")
    for i, batch in enumerate(dataloader):
        print(f"   Batch {i}: RGBA shape = {batch['rgba'].shape}")
        if i >= 2:  # Only test first few batches
            break
    
    return dataset_full, dataset_subset, dataloader


def main():
    """Main function to demonstrate dataloader usage."""
    data_dir = "/home/cvlab15/project/chyun/4d-recon/kubric/multi_view_output"
    
    print("Multi-View Kubric Dataloader Example")
    print("=" * 50)
    
    # Test dataloader functionality
    dataset_full, dataset_subset, dataloader = test_dataloader_functionality(data_dir)
    
    # Analyze camera trajectories
    analyze_camera_trajectories(dataset_full)
    
    # Visualize a sequence (if matplotlib is available)
    try:
        print("\n5. Visualizing multi-view sequence...")
        visualize_multi_view_sequence(
            dataset_full, 
            sequence_idx=0, 
            save_path="multi_view_visualization.png"
        )
    except ImportError:
        print("   Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"   Visualization failed: {e}")
    
    print("\n=== Example Complete ===")
    print("The dataloader is ready for use in your 4D reconstruction pipeline!")


if __name__ == "__main__":
    main()
