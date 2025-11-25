#!/usr/bin/env python3
"""
Quick utility to inspect multi-view Kubric dataset structure and content.
"""

import json
import numpy as np
from pathlib import Path
import argparse


def inspect_dataset_structure(data_dir):
    """Inspect the structure of a multi-view dataset."""
    data_path = Path(data_dir)
    
    print(f"Inspecting dataset at: {data_path}")
    print("=" * 60)
    
    # Check if directory exists
    if not data_path.exists():
        print(f"ERROR: Directory {data_path} does not exist!")
        return
    
    # Load global metadata
    global_metadata_path = data_path / "global_metadata.json"
    if global_metadata_path.exists():
        with open(global_metadata_path, 'r') as f:
            global_metadata = json.load(f)
        
        print("Global Metadata:")
        print(f"  Dataset type: {global_metadata.get('dataset_type', 'Unknown')}")
        print(f"  Number of cameras: {global_metadata.get('num_cameras', 'Unknown')}")
        print(f"  Number of frames: {global_metadata.get('num_frames', 'Unknown')}")
        print(f"  Resolution: {global_metadata.get('resolution', 'Unknown')}")
        print(f"  Frame rate: {global_metadata.get('frame_rate', 'Unknown')}")
        print(f"  Seed: {global_metadata.get('seed', 'Unknown')}")
    else:
        print("WARNING: global_metadata.json not found!")
    
    print("\n" + "=" * 60)
    
    # Find camera directories
    camera_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('camera_')])
    
    if not camera_dirs:
        print("ERROR: No camera directories found!")
        return
    
    print(f"Found {len(camera_dirs)} camera directories:")
    
    for cam_dir in camera_dirs:
        cam_id = cam_dir.name.split('_')[1]
        print(f"\nCamera {cam_id}:")
        print(f"  Path: {cam_dir}")
        
        # Check for metadata files
        camera_metadata_path = cam_dir / "camera_metadata.json"
        scene_metadata_path = cam_dir / "scene_metadata.json"
        
        if camera_metadata_path.exists():
            with open(camera_metadata_path, 'r') as f:
                cam_meta = json.load(f)
            print(f"  Focal length: {cam_meta.get('focal_length', 'Unknown')}")
            print(f"  Sensor width: {cam_meta.get('sensor_width', 'Unknown')}")
            print(f"  Field of view: {cam_meta.get('field_of_view', 'Unknown'):.4f}")
            print(f"  Number of instances: {cam_meta.get('num_instances', 'Unknown')}")
        else:
            print("  WARNING: camera_metadata.json not found!")
        
        if scene_metadata_path.exists():
            print("  Scene metadata: Available")
        else:
            print("  WARNING: scene_metadata.json not found!")
        
        # Check for image files
        image_types = ['rgba', 'depth', 'normal', 'segmentation', 'forward_flow', 'backward_flow', 'object_coordinates']
        
        print("  Image files:")
        for img_type in image_types:
            if img_type == 'depth':
                pattern = f"{img_type}_*.tiff"
            else:
                pattern = f"{img_type}_*.png"
            
            files = list(cam_dir.glob(pattern))
            if files:
                print(f"    {img_type}: {len(files)} files")
                # Show file range
                frame_indices = []
                for f in files:
                    try:
                        idx = int(f.stem.split('_')[1])
                        frame_indices.append(idx)
                    except:
                        pass
                
                if frame_indices:
                    frame_indices.sort()
                    print(f"      Frames: {frame_indices[0]:05d} to {frame_indices[-1]:05d}")
            else:
                print(f"    {img_type}: No files found")


def analyze_camera_trajectories(data_dir):
    """Analyze camera trajectories from the dataset."""
    data_path = Path(data_dir)
    global_metadata_path = data_path / "global_metadata.json"
    
    if not global_metadata_path.exists():
        print("ERROR: global_metadata.json not found!")
        return
    
    with open(global_metadata_path, 'r') as f:
        global_metadata = json.load(f)
    
    print("\n" + "=" * 60)
    print("Camera Trajectory Analysis:")
    print("=" * 60)
    
    if 'camera_trajectories' in global_metadata:
        trajectories = global_metadata['camera_trajectories']
        
        for i, traj in enumerate(trajectories):
            print(f"\nCamera {i}:")
            positions = np.array(traj['positions'])
            quaternions = np.array(traj['quaternions'])
            
            print(f"  Position range:")
            print(f"    X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] (range: {positions[:, 0].max() - positions[:, 0].min():.2f})")
            print(f"    Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] (range: {positions[:, 1].max() - positions[:, 1].min():.2f})")
            print(f"    Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] (range: {positions[:, 2].max() - positions[:, 2].min():.2f})")
            
            # Calculate movement distance
            total_distance = 0
            for j in range(1, len(positions)):
                dist = np.linalg.norm(positions[j] - positions[j-1])
                total_distance += dist
            
            print(f"  Total movement distance: {total_distance:.2f}")
            print(f"  Look-at target: {traj['lookat_target']}")


def check_data_consistency(data_dir):
    """Check for data consistency across cameras."""
    data_path = Path(data_dir)
    camera_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('camera_')])
    
    if not camera_dirs:
        print("ERROR: No camera directories found!")
        return
    
    print("\n" + "=" * 60)
    print("Data Consistency Check:")
    print("=" * 60)
    
    # Check frame consistency
    frame_counts = {}
    for cam_dir in camera_dirs:
        cam_id = cam_dir.name.split('_')[1]
        rgba_files = list(cam_dir.glob("rgba_*.png"))
        frame_counts[cam_id] = len(rgba_files)
    
    print(f"Frame counts per camera: {frame_counts}")
    
    if len(set(frame_counts.values())) == 1:
        print("✓ All cameras have the same number of frames")
    else:
        print("⚠ WARNING: Cameras have different numbers of frames!")
    
    # Check for missing files
    print("\nChecking for missing files...")
    for cam_dir in camera_dirs:
        cam_id = cam_dir.name.split('_')[1]
        print(f"\nCamera {cam_id}:")
        
        # Get frame indices from rgba files
        rgba_files = list(cam_dir.glob("rgba_*.png"))
        frame_indices = []
        for f in rgba_files:
            try:
                idx = int(f.stem.split('_')[1])
                frame_indices.append(idx)
            except:
                pass
        
        frame_indices.sort()
        
        # Check each image type
        image_types = ['rgba', 'depth', 'normal', 'segmentation', 'forward_flow', 'backward_flow', 'object_coordinates']
        
        for img_type in image_types:
            if img_type == 'depth':
                pattern = f"{img_type}_*.tiff"
            else:
                pattern = f"{img_type}_*.png"
            
            files = list(cam_dir.glob(pattern))
            file_indices = []
            for f in files:
                try:
                    idx = int(f.stem.split('_')[1])
                    file_indices.append(idx)
                except:
                    pass
            
            file_indices.sort()
            
            if file_indices == frame_indices:
                print(f"  {img_type}: ✓ Complete")
            else:
                missing = set(frame_indices) - set(file_indices)
                extra = set(file_indices) - set(frame_indices)
                if missing:
                    print(f"  {img_type}: ⚠ Missing frames: {sorted(missing)}")
                if extra:
                    print(f"  {img_type}: ⚠ Extra frames: {sorted(extra)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Inspect multi-view Kubric dataset")
    parser.add_argument("data_dir", help="Path to the multi-view dataset directory")
    parser.add_argument("--analyze-trajectories", action="store_true", 
                       help="Analyze camera trajectories")
    parser.add_argument("--check-consistency", action="store_true",
                       help="Check data consistency across cameras")
    
    args = parser.parse_args()
    
    # Basic structure inspection
    inspect_dataset_structure(args.data_dir)
    
    # Optional analyses
    if args.analyze_trajectories:
        analyze_camera_trajectories(args.data_dir)
    
    if args.check_consistency:
        check_data_consistency(args.data_dir)


if __name__ == "__main__":
    main()
