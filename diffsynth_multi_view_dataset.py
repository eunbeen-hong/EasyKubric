#!/usr/bin/env python3
"""
DiffSynth-Studio adapter for MultiViewKubricDataset.
Integrates multi-view Kubric data with the DiffSynth-Studio training framework.
"""

import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import os

from multi_view_dataloader import MultiViewKubricDataset

# Import from unified_dataset with fallback
try:
    from diffsynth.trainers.unified_dataset import (
        DataProcessingOperator, DataProcessingPipeline, 
        RouteByType, ToAbsolutePath, LoadImage, ImageCropAndResize,
        SequencialProcess, ToList, LoadTorchPickle
    )
except ImportError:
    # Fallback for when diffsynth is not available
    print("Warning: diffsynth.trainers.unified_dataset not found. Some features may not work.")
    DataProcessingOperator = object
    DataProcessingPipeline = object
    RouteByType = object
    ToAbsolutePath = object
    LoadImage = object
    ImageCropAndResize = object
    SequencialProcess = object
    ToList = object
    LoadTorchPickle = object


class MultiViewKubricOperator(DataProcessingOperator):
    """
    Data processing operator for multi-view Kubric data.
    Handles loading and preprocessing of multi-view sequences.
    """
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: Optional[int] = None,
                 load_flows: bool = True,
                 load_segmentation: bool = True,
                 load_depth: bool = True,
                 load_normals: bool = True,
                 load_object_coords: bool = True,
                 camera_subset: Optional[List[int]] = None,
                 max_pixels: int = 1920*1080,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 height_division_factor: int = 16,
                 width_division_factor: int = 16):
        """
        Initialize the multi-view Kubric operator.
        
        Args:
            data_dir: Path to the multi-view dataset directory
            sequence_length: Number of frames per sequence (None = all frames)
            load_flows: Whether to load optical flow data
            load_segmentation: Whether to load segmentation masks
            load_depth: Whether to load depth maps
            load_normals: Whether to load normal maps
            load_object_coords: Whether to load object coordinates
            camera_subset: List of camera IDs to load (None = all cameras)
            max_pixels: Maximum number of pixels for image resizing
            height: Target height for images (None = auto)
            width: Target width for images (None = auto)
            height_division_factor: Height must be divisible by this factor
            width_division_factor: Width must be divisible by this factor
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.load_flows = load_flows
        self.load_segmentation = load_segmentation
        self.load_depth = load_depth
        self.load_normals = load_normals
        self.load_object_coords = load_object_coords
        self.camera_subset = camera_subset
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        
        # Initialize the underlying dataset
        self.dataset = MultiViewKubricDataset(
            data_dir=data_dir,
            sequence_length=sequence_length,
            load_flows=load_flows,
            load_segmentation=load_segmentation,
            load_depth=load_depth,
            load_normals=load_normals,
            load_object_coords=load_object_coords,
            camera_subset=camera_subset
        )
        
        # Image processing pipeline
        self.image_processor = ImageCropAndResize(
            height=height,
            width=width,
            max_pixels=max_pixels,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor
        )
    
    def __call__(self, data: Union[str, int]) -> Dict[str, Any]:
        """
        Process multi-view Kubric data.
        
        Args:
            data: Either a string path (ignored) or integer index for the dataset
            
        Returns:
            Processed multi-view data dictionary
        """
        if isinstance(data, str):
            # If string is provided, try to extract sequence index from path
            # This allows compatibility with file-based loading
            try:
                # Extract index from path if it contains a sequence identifier
                # For now, use 0 as default
                idx = 0
            except:
                idx = 0
        else:
            idx = data
        
        # Load the sequence from the dataset
        sequence_data = self.dataset[idx]
        
        # Process the data for DiffSynth-Studio format
        processed_data = self._process_sequence(sequence_data)
        
        return processed_data
    
    def _process_sequence(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a multi-view sequence for DiffSynth-Studio format.
        
        Args:
            sequence_data: Raw sequence data from MultiViewKubricDataset
            
        Returns:
            Processed data in DiffSynth-Studio format
        """
        processed = {}
        
        # Process RGB images for each camera
        processed['images'] = self._process_images(sequence_data['rgba'])
        
        # Process camera poses and intrinsics
        processed['camera_poses'] = sequence_data['camera_poses']
        processed['camera_intrinsics'] = sequence_data['camera_intrinsics']
        
        # Process additional data if available
        if 'depth' in sequence_data:
            processed['depth'] = self._process_depth(sequence_data['depth'])
        
        if 'normal' in sequence_data:
            processed['normal'] = self._process_images(sequence_data['normal'])
        
        if 'segmentation' in sequence_data:
            processed['segmentation'] = self._process_segmentation(sequence_data['segmentation'])
        
        if 'forward_flow' in sequence_data:
            processed['forward_flow'] = self._process_flow(sequence_data['forward_flow'])
        
        if 'backward_flow' in sequence_data:
            processed['backward_flow'] = self._process_flow(sequence_data['backward_flow'])
        
        if 'object_coordinates' in sequence_data:
            processed['object_coordinates'] = self._process_images(sequence_data['object_coordinates'])
        
        # Add metadata
        processed['metadata'] = sequence_data['metadata']
        processed['num_cameras'] = sequence_data['metadata']['num_cameras']
        processed['num_frames'] = sequence_data['metadata']['sequence_length']
        
        return processed
    
    def _process_images(self, images: np.ndarray) -> List[List[Image.Image]]:
        """
        Process image arrays to PIL Images for each camera and frame.
        
        Args:
            images: Image array [num_cameras, num_frames, height, width, channels]
            
        Returns:
            List of lists of PIL Images [num_cameras][num_frames]
        """
        processed_images = []
        
        for cam_idx in range(images.shape[0]):
            camera_images = []
            for frame_idx in range(images.shape[1]):
                # Convert numpy array to PIL Image
                img_array = images[cam_idx, frame_idx]
                
                # Ensure proper data type and range
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                
                # Convert to PIL Image
                if img_array.shape[2] == 3:  # RGB
                    pil_image = Image.fromarray(img_array, 'RGB')
                elif img_array.shape[2] == 4:  # RGBA
                    pil_image = Image.fromarray(img_array, 'RGBA')
                else:
                    # Handle other channel counts
                    pil_image = Image.fromarray(img_array)
                
                # Apply image processing (resize, crop, etc.)
                processed_image = self.image_processor(pil_image)
                camera_images.append(processed_image)
            
            processed_images.append(camera_images)
        
        return processed_images
    
    def _process_depth(self, depth: np.ndarray) -> List[List[np.ndarray]]:
        """
        Process depth maps for each camera and frame.
        
        Args:
            depth: Depth array [num_cameras, num_frames, height, width]
            
        Returns:
            List of lists of depth arrays [num_cameras][num_frames]
        """
        processed_depth = []
        
        for cam_idx in range(depth.shape[0]):
            camera_depth = []
            for frame_idx in range(depth.shape[1]):
                depth_array = depth[cam_idx, frame_idx]
                camera_depth.append(depth_array)
            processed_depth.append(camera_depth)
        
        return processed_depth
    
    def _process_segmentation(self, segmentation: np.ndarray) -> List[List[np.ndarray]]:
        """
        Process segmentation masks for each camera and frame.
        
        Args:
            segmentation: Segmentation array [num_cameras, num_frames, height, width]
            
        Returns:
            List of lists of segmentation arrays [num_cameras][num_frames]
        """
        processed_seg = []
        
        for cam_idx in range(segmentation.shape[0]):
            camera_seg = []
            for frame_idx in range(segmentation.shape[1]):
                seg_array = segmentation[cam_idx, frame_idx]
                camera_seg.append(seg_array)
            processed_seg.append(camera_seg)
        
        return processed_seg
    
    def _process_flow(self, flow: np.ndarray) -> List[List[np.ndarray]]:
        """
        Process optical flow for each camera and frame.
        
        Args:
            flow: Flow array [num_cameras, num_frames, height, width, 2]
            
        Returns:
            List of lists of flow arrays [num_cameras][num_frames]
        """
        processed_flow = []
        
        for cam_idx in range(flow.shape[0]):
            camera_flow = []
            for frame_idx in range(flow.shape[1]):
                flow_array = flow[cam_idx, frame_idx]
                camera_flow.append(flow_array)
            processed_flow.append(camera_flow)
        
        return processed_flow


class MultiViewKubricDataset(torch.utils.data.Dataset):
    """
    DiffSynth-Studio compatible dataset for multi-view Kubric data.
    Wraps the MultiViewKubricDataset to work with the unified dataset framework.
    """
    
    def __init__(self,
                 data_dir: str,
                 sequence_length: Optional[int] = None,
                 load_flows: bool = True,
                 load_segmentation: bool = True,
                 load_depth: bool = True,
                 load_normals: bool = True,
                 load_object_coords: bool = True,
                 camera_subset: Optional[List[int]] = None,
                 max_pixels: int = 1920*1080,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 height_division_factor: int = 16,
                 width_division_factor: int = 16,
                 repeat: int = 1):
        """
        Initialize the DiffSynth-Studio compatible multi-view dataset.
        
        Args:
            data_dir: Path to the multi-view dataset directory
            sequence_length: Number of frames per sequence (None = all frames)
            load_flows: Whether to load optical flow data
            load_segmentation: Whether to load segmentation masks
            load_depth: Whether to load depth maps
            load_normals: Whether to load normal maps
            load_object_coords: Whether to load object coordinates
            camera_subset: List of camera IDs to load (None = all cameras)
            max_pixels: Maximum number of pixels for image resizing
            height: Target height for images (None = auto)
            width: Target width for images (None = auto)
            height_division_factor: Height must be divisible by this factor
            width_division_factor: Width must be divisible by this factor
            repeat: Number of times to repeat the dataset
        """
        self.data_dir = data_dir
        self.repeat = repeat
        
        # Initialize the underlying dataset
        self.underlying_dataset = MultiViewKubricDataset(
            data_dir=data_dir,
            sequence_length=sequence_length,
            load_flows=load_flows,
            load_segmentation=load_segmentation,
            load_depth=load_depth,
            load_normals=load_normals,
            load_object_coords=load_object_coords,
            camera_subset=camera_subset
        )
        
        # Initialize the processing operator
        self.processor = MultiViewKubricOperator(
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
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.underlying_dataset) * self.repeat
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Processed multi-view data dictionary
        """
        # Handle repetition
        actual_idx = idx % len(self.underlying_dataset)
        
        # Process the data
        return self.processor(actual_idx)
    
    def get_global_info(self) -> Dict[str, Any]:
        """Get global dataset information."""
        return self.underlying_dataset.get_global_info()
    
    def get_camera_info(self, cam_id: int) -> Dict[str, Any]:
        """Get camera information for a specific camera."""
        return self.underlying_dataset.get_camera_info(cam_id)


class MultiViewKubricUnifiedDataset(torch.utils.data.Dataset):
    """
    Unified dataset that can load multi-view Kubric data alongside other data types.
    Integrates with the DiffSynth-Studio unified dataset framework.
    """
    
    def __init__(self,
                 base_path: str,
                 metadata_path: Optional[str] = None,
                 repeat: int = 1,
                 data_file_keys: tuple = (),
                 main_data_operator=None,
                 special_operator_map: Optional[Dict[str, DataProcessingOperator]] = None,
                 # Multi-view specific parameters
                 multi_view_data_dir: Optional[str] = None,
                 multi_view_sequence_length: Optional[int] = None,
                 multi_view_load_flows: bool = True,
                 multi_view_load_segmentation: bool = True,
                 multi_view_load_depth: bool = True,
                 multi_view_load_normals: bool = True,
                 multi_view_load_object_coords: bool = True,
                 multi_view_camera_subset: Optional[List[int]] = None,
                 multi_view_max_pixels: int = 1920*1080,
                 multi_view_height: Optional[int] = None,
                 multi_view_width: Optional[int] = None,
                 multi_view_height_division_factor: int = 16,
                 multi_view_width_division_factor: int = 16):
        """
        Initialize the unified dataset with multi-view Kubric support.
        
        Args:
            base_path: Base path for the dataset
            metadata_path: Path to metadata file (None for multi-view only)
            repeat: Number of times to repeat the dataset
            data_file_keys: Keys for data files in metadata
            main_data_operator: Main data processing operator
            special_operator_map: Special operators for specific keys
            multi_view_data_dir: Path to multi-view Kubric dataset
            multi_view_sequence_length: Number of frames per sequence
            multi_view_load_flows: Whether to load optical flow
            multi_view_load_segmentation: Whether to load segmentation
            multi_view_load_depth: Whether to load depth maps
            multi_view_load_normals: Whether to load normal maps
            multi_view_load_object_coords: Whether to load object coordinates
            multi_view_camera_subset: List of camera IDs to load
            multi_view_max_pixels: Maximum pixels for image resizing
            multi_view_height: Target height for images
            multi_view_width: Target width for images
            multi_view_height_division_factor: Height division factor
            multi_view_width_division_factor: Width division factor
        """
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.cached_data_operator = LoadTorchPickle()
        
        # Multi-view dataset
        self.multi_view_dataset = None
        if multi_view_data_dir is not None:
            self.multi_view_dataset = MultiViewKubricDataset(
                data_dir=multi_view_data_dir,
                sequence_length=multi_view_sequence_length,
                load_flows=multi_view_load_flows,
                load_segmentation=multi_view_load_segmentation,
                load_depth=multi_view_load_depth,
                load_normals=multi_view_load_normals,
                load_object_coords=multi_view_load_object_coords,
                camera_subset=multi_view_camera_subset,
                max_pixels=multi_view_max_pixels,
                height=multi_view_height,
                width=multi_view_width,
                height_division_factor=multi_view_height_division_factor,
                width_division_factor=multi_view_width_division_factor,
                repeat=repeat
            )
        
        # Regular dataset
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        """Load metadata from file."""
        if metadata_path is None:
            if self.multi_view_dataset is not None:
                # Use multi-view dataset only
                self.data = []
                self.cached_data = []
            else:
                print("No metadata_path and no multi-view dataset. Searching for cached data files.")
                self.search_for_cached_data_files(self.base_path)
                print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            import pandas
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
    
    def search_for_cached_data_files(self, path):
        """Search for cached data files."""
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def __getitem__(self, data_id):
        """Get a sample from the dataset."""
        if self.multi_view_dataset is not None and len(self.data) == 0:
            # Use multi-view dataset only
            return self.multi_view_dataset[data_id]
        elif self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data
    
    def __len__(self):
        """Return the length of the dataset."""
        if self.multi_view_dataset is not None and len(self.data) == 0:
            return len(self.multi_view_dataset)
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat


# Example usage and factory functions
def create_multi_view_kubric_dataset(data_dir: str, **kwargs) -> MultiViewKubricDataset:
    """
    Factory function to create a multi-view Kubric dataset.
    
    Args:
        data_dir: Path to the multi-view dataset directory
        **kwargs: Additional arguments for MultiViewKubricDataset
        
    Returns:
        MultiViewKubricDataset instance
    """
    return MultiViewKubricDataset(data_dir=data_dir, **kwargs)


def create_unified_multi_view_dataset(base_path: str, 
                                    multi_view_data_dir: str,
                                    **kwargs) -> MultiViewKubricUnifiedDataset:
    """
    Factory function to create a unified dataset with multi-view Kubric support.
    
    Args:
        base_path: Base path for the dataset
        multi_view_data_dir: Path to multi-view Kubric dataset
        **kwargs: Additional arguments for MultiViewKubricUnifiedDataset
        
    Returns:
        MultiViewKubricUnifiedDataset instance
    """
    return MultiViewKubricUnifiedDataset(
        base_path=base_path,
        multi_view_data_dir=multi_view_data_dir,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Multi-view dataset only
    dataset = create_multi_view_kubric_dataset(
        data_dir="/path/to/multi_view_output",
        sequence_length=4,
        load_depth=True,
        load_segmentation=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Images shape: {len(sample['images'])} cameras, {len(sample['images'][0])} frames")
    
    # Example 2: Unified dataset with multi-view support
    unified_dataset = create_unified_multi_view_dataset(
        base_path="/path/to/base",
        multi_view_data_dir="/path/to/multi_view_output",
        multi_view_sequence_length=4
    )
    
    print(f"Unified dataset size: {len(unified_dataset)}")
    sample = unified_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
