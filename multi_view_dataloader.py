#!/usr/bin/env python3
"""
Multi-view Kubric dataset dataloader for PyTorch.
Handles loading and preprocessing of multi-view rendered data from Kubric.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2


class MultiViewKubricDataset(Dataset):
    """
    PyTorch Dataset for multi-view Kubric rendered data.
    
    Loads synchronized multi-view sequences with all render passes:
    - RGB/RGBA images
    - Depth maps
    - Normal maps
    - Segmentation masks
    - Optical flow (forward/backward)
    - Object coordinates
    """
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 load_flows: bool = True,
                 load_segmentation: bool = True,
                 load_depth: bool = True,
                 load_normals: bool = True,
                 load_object_coords: bool = True,
                 camera_subset: Optional[List[int]] = None):
        """
        Initialize the multi-view Kubric dataset.
        
        Args:
            data_dir: Path to the multi-view dataset directory
            sequence_length: Number of frames to load per sequence (None = all frames)
            transform: Torchvision transforms to apply to images
            load_flows: Whether to load optical flow data
            load_segmentation: Whether to load segmentation masks
            load_depth: Whether to load depth maps
            load_normals: Whether to load normal maps
            load_object_coords: Whether to load object coordinates
            camera_subset: List of camera IDs to load (None = all cameras)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.load_flows = load_flows
        self.load_segmentation = load_segmentation
        self.load_depth = load_depth
        self.load_normals = load_normals
        self.load_object_coords = load_object_coords
        
        # Load global metadata
        self.global_metadata = self._load_global_metadata()
        
        # Get available cameras
        self.camera_dirs = self._get_camera_dirs()
        if camera_subset is not None:
            self.camera_dirs = [cam for cam in self.camera_dirs if cam['camera_id'] in camera_subset]
        
        # Get frame sequences
        self.sequences = self._get_sequences()
        
        # Load camera metadata
        self.camera_metadata = {}
        for camera_dir in self.camera_dirs:
            cam_id = camera_dir['camera_id']
            self.camera_metadata[cam_id] = self._load_camera_metadata(camera_dir['path'])
    
    def _load_global_metadata(self) -> Dict:
        """Load global dataset metadata."""
        metadata_path = self.data_dir / "global_metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _get_camera_dirs(self) -> List[Dict]:
        """Get list of available camera directories."""
        camera_dirs = []
        for cam_dir in sorted(self.data_dir.glob("camera_*")):
            if cam_dir.is_dir():
                # Extract camera ID from directory name
                cam_id = int(cam_dir.name.split("_")[1])
                camera_dirs.append({
                    'camera_id': cam_id,
                    'path': cam_dir
                })
        return camera_dirs
    
    def _load_camera_metadata(self, camera_path: Path) -> Dict:
        """Load camera-specific metadata."""
        metadata_path = camera_path / "camera_metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _get_sequences(self) -> List[Dict]:
        """Get list of available sequences."""
        sequences = []
        
        # Get frame indices from the first camera
        if not self.camera_dirs:
            return sequences
            
        first_camera = self.camera_dirs[0]
        rgba_files = sorted((first_camera['path'] / "rgba_*.png").glob("rgba_*.png"))
        
        if not rgba_files:
            return sequences
        
        # Extract frame indices
        frame_indices = []
        for file_path in rgba_files:
            frame_idx = int(file_path.stem.split("_")[1])
            frame_indices.append(frame_idx)
        
        frame_indices = sorted(frame_indices)
        
        # Create sequences
        if self.sequence_length is None:
            # Single sequence with all frames
            sequences.append({
                'frame_indices': frame_indices,
                'start_idx': 0,
                'end_idx': len(frame_indices)
            })
        else:
            # Multiple sequences of specified length
            for start_idx in range(0, len(frame_indices) - self.sequence_length + 1, self.sequence_length):
                end_idx = min(start_idx + self.sequence_length, len(frame_indices))
                sequences.append({
                    'frame_indices': frame_indices[start_idx:end_idx],
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        return sequences
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load an image file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        image = Image.open(file_path)
        if image.mode == 'RGBA':
            # Convert RGBA to RGB
            image = image.convert('RGB')
        
        return np.array(image)
    
    def _load_depth(self, file_path: Path) -> np.ndarray:
        """Load a depth map from TIFF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Depth file not found: {file_path}")
        
        # Load depth using OpenCV (supports TIFF)
        depth = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not load depth file: {file_path}")
        
        return depth
    
    def _load_flow(self, file_path: Path) -> np.ndarray:
        """Load optical flow from PNG file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Flow file not found: {file_path}")
        
        # Load flow using OpenCV
        flow = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if flow is None:
            raise ValueError(f"Could not load flow file: {file_path}")
        
        # Convert from PNG format to flow format
        # Flow is typically stored as (u, v) displacement
        # The exact format depends on how Kubric saves it
        return flow
    
    def _load_segmentation(self, file_path: Path) -> np.ndarray:
        """Load segmentation mask."""
        if not file_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {file_path}")
        
        image = Image.open(file_path)
        return np.array(image)
    
    def _load_normal(self, file_path: Path) -> np.ndarray:
        """Load normal map."""
        if not file_path.exists():
            raise FileNotFoundError(f"Normal file not found: {file_path}")
        
        image = Image.open(file_path)
        return np.array(image)
    
    def _load_object_coordinates(self, file_path: Path) -> np.ndarray:
        """Load object coordinates."""
        if not file_path.exists():
            raise FileNotFoundError(f"Object coordinates file not found: {file_path}")
        
        image = Image.open(file_path)
        return np.array(image)
    
    def __len__(self) -> int:
        """Return number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sequence of multi-view data.
        
        Returns:
            Dictionary containing:
            - 'rgba': List of RGB images for each camera [num_cameras, height, width, 3]
            - 'depth': List of depth maps for each camera [num_cameras, height, width]
            - 'normal': List of normal maps for each camera [num_cameras, height, width, 3]
            - 'segmentation': List of segmentation masks for each camera [num_cameras, height, width]
            - 'forward_flow': List of forward flow maps for each camera [num_cameras, height, width, 2]
            - 'backward_flow': List of backward flow maps for each camera [num_cameras, height, width, 2]
            - 'object_coordinates': List of object coordinate maps for each camera [num_cameras, height, width, 3]
            - 'camera_poses': Camera poses for each camera [num_cameras, num_frames, 4, 4] (4x4 transformation matrices)
            - 'camera_intrinsics': Camera intrinsics for each camera [num_cameras, 3, 3]
            - 'metadata': Additional metadata
        """
        sequence = self.sequences[idx]
        frame_indices = sequence['frame_indices']
        
        # Initialize data containers
        data = {
            'rgba': [],
            'depth': [],
            'normal': [],
            'segmentation': [],
            'forward_flow': [],
            'backward_flow': [],
            'object_coordinates': [],
            'camera_poses': [],
            'camera_intrinsics': [],
            'metadata': {
                'frame_indices': frame_indices,
                'sequence_length': len(frame_indices),
                'num_cameras': len(self.camera_dirs)
            }
        }
        
        # Load data for each camera
        for camera_dir in self.camera_dirs:
            cam_id = camera_dir['camera_id']
            camera_path = camera_dir['path']
            
            # Load camera poses and intrinsics
            camera_poses = []
            for frame_idx in frame_indices:
                pose = self._get_camera_pose(cam_id, frame_idx)
                camera_poses.append(pose)
            
            data['camera_poses'].append(camera_poses)
            data['camera_intrinsics'].append(self._get_camera_intrinsics(cam_id))
            
            # Load images for this camera
            camera_rgba = []
            camera_depth = []
            camera_normal = []
            camera_segmentation = []
            camera_forward_flow = []
            camera_backward_flow = []
            camera_object_coords = []
            
            for frame_idx in frame_indices:
                # Load RGBA
                rgba_path = camera_path / f"rgba_{frame_idx:05d}.png"
                rgba = self._load_image(rgba_path)
                camera_rgba.append(rgba)
                
                # Load depth
                if self.load_depth:
                    depth_path = camera_path / f"depth_{frame_idx:05d}.tiff"
                    depth = self._load_depth(depth_path)
                    camera_depth.append(depth)
                
                # Load normal
                if self.load_normals:
                    normal_path = camera_path / f"normal_{frame_idx:05d}.png"
                    normal = self._load_normal(normal_path)
                    camera_normal.append(normal)
                
                # Load segmentation
                if self.load_segmentation:
                    seg_path = camera_path / f"segmentation_{frame_idx:05d}.png"
                    segmentation = self._load_segmentation(seg_path)
                    camera_segmentation.append(segmentation)
                
                # Load optical flow
                if self.load_flows:
                    forward_flow_path = camera_path / f"forward_flow_{frame_idx:05d}.png"
                    backward_flow_path = camera_path / f"backward_flow_{frame_idx:05d}.png"
                    
                    forward_flow = self._load_flow(forward_flow_path)
                    backward_flow = self._load_flow(backward_flow_path)
                    
                    camera_forward_flow.append(forward_flow)
                    camera_backward_flow.append(backward_flow)
                
                # Load object coordinates
                if self.load_object_coords:
                    obj_coords_path = camera_path / f"object_coordinates_{frame_idx:05d}.png"
                    obj_coords = self._load_object_coordinates(obj_coords_path)
                    camera_object_coords.append(obj_coords)
            
            # Add camera data to main data structure
            data['rgba'].append(camera_rgba)
            if self.load_depth:
                data['depth'].append(camera_depth)
            if self.load_normals:
                data['normal'].append(camera_normal)
            if self.load_segmentation:
                data['segmentation'].append(camera_segmentation)
            if self.load_flows:
                data['forward_flow'].append(camera_forward_flow)
                data['backward_flow'].append(camera_backward_flow)
            if self.load_object_coords:
                data['object_coordinates'].append(camera_object_coords)
        
        # Convert to numpy arrays
        data['rgba'] = np.array(data['rgba'])  # [num_cameras, num_frames, height, width, 3]
        data['camera_poses'] = np.array(data['camera_poses'])  # [num_cameras, num_frames, 4, 4]
        data['camera_intrinsics'] = np.array(data['camera_intrinsics'])  # [num_cameras, 3, 3]
        
        if self.load_depth:
            data['depth'] = np.array(data['depth'])  # [num_cameras, num_frames, height, width]
        if self.load_normals:
            data['normal'] = np.array(data['normal'])  # [num_cameras, num_frames, height, width, 3]
        if self.load_segmentation:
            data['segmentation'] = np.array(data['segmentation'])  # [num_cameras, num_frames, height, width]
        if self.load_flows:
            data['forward_flow'] = np.array(data['forward_flow'])  # [num_cameras, num_frames, height, width, 2]
            data['backward_flow'] = np.array(data['backward_flow'])  # [num_cameras, num_frames, height, width, 2]
        if self.load_object_coords:
            data['object_coordinates'] = np.array(data['object_coordinates'])  # [num_cameras, num_frames, height, width, 3]
        
        # Apply transforms if specified
        if self.transform is not None:
            data = self._apply_transforms(data)
        
        return data
    
    def _get_camera_pose(self, cam_id: int, frame_idx: int) -> np.ndarray:
        """Get camera pose (4x4 transformation matrix) for a specific camera and frame."""
        camera_meta = self.camera_metadata[cam_id]
        
        # Find the frame index in the camera metadata
        frame_positions = camera_meta['positions']
        frame_quaternions = camera_meta['quaternions']
        
        # Find the closest frame index
        frame_indices = list(range(len(frame_positions)))
        closest_frame_idx = min(frame_indices, key=lambda x: abs(x - frame_idx))
        
        position = np.array(frame_positions[closest_frame_idx])
        quaternion = np.array(frame_quaternions[closest_frame_idx])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)
        
        # Create 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = position
        
        return pose
    
    def _get_camera_intrinsics(self, cam_id: int) -> np.ndarray:
        """Get camera intrinsics matrix (3x3) for a specific camera."""
        camera_meta = self.camera_metadata[cam_id]
        resolution = camera_meta['resolution']
        focal_length = camera_meta['focal_length']
        sensor_width = camera_meta['sensor_width']
        
        # Calculate focal length in pixels
        fx = fy = focal_length * resolution[0] / sensor_width
        
        # Principal point (assume center of image)
        cx = resolution[0] / 2
        cy = resolution[1] / 2
        
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return intrinsics
    
    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = quaternion
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        rotation_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return rotation_matrix
    
    def _apply_transforms(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to the data."""
        # This is a placeholder - implement specific transforms as needed
        # For now, just convert to tensors
        for key, value in data.items():
            if isinstance(value, np.ndarray) and key != 'metadata':
                data[key] = torch.from_numpy(value).float()
        
        return data
    
    def get_camera_info(self, cam_id: int) -> Dict[str, Any]:
        """Get camera information for a specific camera."""
        return self.camera_metadata[cam_id]
    
    def get_global_info(self) -> Dict[str, Any]:
        """Get global dataset information."""
        return self.global_metadata


def create_dataloader(data_dir: str,
                     batch_size: int = 1,
                     shuffle: bool = False,
                     num_workers: int = 0,
                     sequence_length: Optional[int] = None,
                     camera_subset: Optional[List[int]] = None,
                     **kwargs) -> DataLoader:
    """
    Create a DataLoader for the multi-view Kubric dataset.
    
    Args:
        data_dir: Path to the multi-view dataset directory
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        sequence_length: Number of frames per sequence
        camera_subset: List of camera IDs to load
        **kwargs: Additional arguments for MultiViewKubricDataset
    
    Returns:
        PyTorch DataLoader
    """
    dataset = MultiViewKubricDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        camera_subset=camera_subset,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )


def custom_collate_fn(batch):
    """
    Custom collate function for multi-view data.
    Handles the complex nested structure of multi-view sequences.
    """
    if len(batch) == 1:
        return batch[0]
    
    # For multiple samples, we need to handle the nested structure
    # This is a simplified version - you may need to customize based on your needs
    collated = {}
    
    for key in batch[0].keys():
        if key == 'metadata':
            collated[key] = [sample[key] for sample in batch]
        else:
            # Stack along batch dimension
            collated[key] = torch.stack([torch.from_numpy(sample[key]) if isinstance(sample[key], np.ndarray) else sample[key] for sample in batch])
    
    return collated


# Example usage and testing
if __name__ == "__main__":
    # Test the dataloader
    data_dir = "/home/cvlab15/project/chyun/4d-recon/kubric/multi_view_output"
    
    # Create dataset
    dataset = MultiViewKubricDataset(
        data_dir=data_dir,
        sequence_length=2,  # Load 2-frame sequences
        load_flows=True,
        load_segmentation=True,
        load_depth=True,
        load_normals=True,
        load_object_coords=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of cameras: {dataset.global_metadata['num_cameras']}")
    print(f"Number of frames: {dataset.global_metadata['num_frames']}")
    print(f"Resolution: {dataset.global_metadata['resolution']}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"RGBA shape: {sample['rgba'].shape}")
        print(f"Camera poses shape: {sample['camera_poses'].shape}")
        print(f"Camera intrinsics shape: {sample['camera_intrinsics'].shape}")
        
        if 'depth' in sample:
            print(f"Depth shape: {sample['depth'].shape}")
        if 'segmentation' in sample:
            print(f"Segmentation shape: {sample['segmentation'].shape}")
        if 'forward_flow' in sample:
            print(f"Forward flow shape: {sample['forward_flow'].shape}")
    
    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=1,
        shuffle=False,
        sequence_length=2
    )
    
    print(f"\nDataloader created with {len(dataloader)} batches")
    
    # Test dataloader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: RGBA shape = {batch['rgba'].shape}")
        if i >= 2:  # Only test first few batches
            break
