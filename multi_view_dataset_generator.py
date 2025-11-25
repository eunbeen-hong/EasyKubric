#!/usr/bin/env python3
"""
Multi-view dataset generator using Kubric renderer with MOVi-E format.
Generates renderings from four synchronized camera trajectories for one scene.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import json
import argparse

# Set up logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

class MultiViewDatasetGenerator:
    """Generator for multi-view datasets using Kubric with MOVi-E format."""
    
    def __init__(self, args):
        """
        Initialize the multi-view dataset generator.
        
        Args:
            output_dir: Base directory for output
            resolution: Image resolution (width, height)
            num_frames: Number of frames to render
            frame_rate: Frame rate for rendering
            num_cameras: Number of camera trajectories (default 4)
            samples_per_pixel: Rendering quality parameter
        """
        self.args = args
        self.args.num_frames = self.args.frame_end - self.args.frame_start + 1
        self.args.resolution = tuple(self.args.resolution)
        self.output_dir = Path(args.output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Asset sources
        self.kubasic_assets = "gs://kubric-public/assets/KuBasic/KuBasic.json"
        self.hdri_assets = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
        self.gso_assets = "gs://kubric-public/assets/GSO/GSO.json"
        
        # Scene parameters
        self.static_spawn_region = [(-7, -7, 0), (7, 7, 10)]
        self.dynamic_spawn_region = [(-5, -5, 1), (5, 5, 5)]
        self.velocity_range = [(-4., -4., 0.), (4., 4., 0.)]
        self.samples_per_pixel = 64
        
    def create_camera_trajectories(self) -> List[Dict[str, Any]]:
        """
        Create four synchronized camera trajectories around the scene.
        
        Args:
            rng: Random number generator
            
        Returns:
            List of camera trajectory dictionaries
        """
        
        # Base camera parameters
        focal_length = 35.0
        sensor_width = 32.0
        field_of_view = 2 * np.arctan(sensor_width / (2 * focal_length))
        
        is_panning = self.args.traj_motion == "linear_movement_linear_lookat"
        camera_start, camera_end = self.get_linear_camera_motion_start_end(
            movement_speed=self.rng.uniform(low=0., high=self.args.max_camera_movement)
        )
        if is_panning:
            lookat_start, lookat_end = self.get_linear_lookat_motion_start_end()

        # Create trajectory data
        positions = []
        quaternions = []
        look_ats = []
        
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(self.args.frame_start - 1, self.args.frame_end + 2):
            interp = ((frame - self.args.frame_start + 1) /
                    (self.args.frame_end - self.args.frame_start + 3))
            positions.append((interp * np.array(camera_start) +
                                    (1 - interp) * np.array(camera_end)))
            if is_panning:
                look_ats.append(
                    interp * np.array(lookat_start)
                    + (1 - interp) * np.array(lookat_end)
                )
            else:
                look_ats.append((0, 0, 0))
            
            look_direction = look_ats[-1] - positions[-1]
            # Create rotation matrix and convert to quaternion
            up = np.array([0, 0, 1])
            right = np.cross(look_direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, look_direction)
            
            # Simple quaternion from rotation matrix
            quat = self.rotation_matrix_to_quaternion(
                np.column_stack([right, up, -look_direction]))
            quaternions.append(quat.tolist())

        trajectory = {
            "focal_length": focal_length,
            "sensor_width": sensor_width,
            "field_of_view": field_of_view,
            "positions": positions,
            "quaternions": quaternions,
            "lookat_target": look_ats
        }
        
        return trajectory

    def get_linear_lookat_motion_start_end(
        self,
        inner_radius: float = 1.0,
        outer_radius: float = 4.0,
    ):
        """Sample a linear path which goes through the workspace center."""
        while True:
            # Sample a point near the workspace center that the path travels through
            camera_through = np.array(
                kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0)
            )
            while True:
                # Sample one endpoint of the trajectory
                camera_start = np.array(
                    kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0)
                )
                if camera_start[-1] < inner_radius:
                    break

            # Continue the trajectory beyond the point in the workspace center, so the
            # final path passes through that point.
            continuation = self.rng.rand(1) * 0.5
            camera_end = camera_through + continuation * (camera_through - camera_start)

            # Second point will probably be closer to the workspace center than the
            # first point.  Get extra augmentation by randomly swapping first and last.
            if self.rng.rand(1)[0] < 0.5:
                tmp = camera_start
                camera_start = camera_end
                camera_end = tmp
            return camera_start, camera_end

    def get_linear_camera_motion_start_end(
        self,
        movement_speed: float,
        inner_radius: float = 8.,
        outer_radius: float = 12.,
        z_offset: float = 0.1,
    ):
        """Sample a linear path which starts and ends within a half-sphere shell."""
        while True:
            camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                        outer_radius,
                                                                        z_offset))
            direction = self.rng.rand(3) - 0.5
            movement = direction / np.linalg.norm(direction) * movement_speed
            camera_end = camera_start + movement
            if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                camera_end[2] > z_offset):
                return camera_start, camera_end


    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays to Python lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def setup_scene(self) -> Tuple[kb.Scene, Any, Any, List[Dict]]:
        """
        Set up the scene with objects, lighting, and camera trajectories.
        
        Args:
            rng: Random number generator
            
        Returns:
            Tuple of (scene, simulator, renderer, camera_trajectories)
        """
        # Create scene
        scene = kb.Scene(
            resolution=self.args.resolution,
            frame_start=0,
            frame_end=self.args.num_frames - 1,
            frame_rate=self.args.frame_rate,
            step_rate=240
        )
        
        # Create simulator and renderer
        scratch_dir = tempfile.mkdtemp()
        simulator = PyBullet(scene, scratch_dir)
        renderer = Blender(scene, scratch_dir, use_denoising=True, 
                          samples_per_pixel=self.samples_per_pixel)
        
        # Load asset sources
        kubasic = kb.AssetSource.from_manifest(self.kubasic_assets)
        gso = kb.AssetSource.from_manifest(self.gso_assets)
        hdri_source = kb.AssetSource.from_manifest(self.hdri_assets)
        
        # Setup background
        train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
        hdri_id = self.rng.choice(train_backgrounds)
        background_hdri = hdri_source.create(asset_id=hdri_id)
        scene.metadata["background"] = hdri_id
        renderer._set_ambient_light_hdri(background_hdri.filename)
        
        # Add dome for background
        dome = kubasic.create(asset_id="dome", name="dome",
                            friction=1.0, restitution=0.0,
                            static=True, background=True)
        scene += dome
        
        # Setup lighting
        scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                                   look_at=(0, 0, 0), intensity=1.5)
        
        # Add objects
        train_split, test_split = gso.get_test_split(fraction=0.1)
        active_split = train_split
        
        # Add static objects
        num_static_objects = self.rng.randint(2,6)
        logger.info(f"Adding {num_static_objects} static objects")
        
        for i in range(num_static_objects):
            obj = gso.create(asset_id=self.rng.choice(active_split))
            scale = self.rng.uniform(0.75, 3.0)
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            obj.metadata["scale"] = scale
            scene += obj
            kb.move_until_no_overlap(obj, simulator, 
                                   spawn_region=self.static_spawn_region, rng=self.rng)
            obj.friction = 1.0
            obj.restitution = 0.0
            obj.metadata["is_dynamic"] = False
        
        # Let static objects settle
        logger.info("Running simulation to let static objects settle...")
        simulator.run(frame_start=-100, frame_end=0)
        
        # Reset object properties
        for obj in scene.foreground_assets:
            if hasattr(obj, "velocity"):
                obj.velocity = (0., 0., 0.)
                obj.friction = 0.5
                obj.restitution = 0.5
        
        # Add dynamic objects
        num_dynamic_objects = self.rng.randint(3, 7)
        logger.info(f"Adding {num_dynamic_objects} dynamic objects")
        
        for i in range(num_dynamic_objects):
            obj = gso.create(asset_id=self.rng.choice(active_split))
            scale = self.rng.uniform(0.75, 3.0)
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            obj.metadata["scale"] = scale
            scene += obj
            kb.move_until_no_overlap(obj, simulator,
                                   spawn_region=self.dynamic_spawn_region, rng=self.rng)
            obj.velocity = (self.rng.uniform(*self.velocity_range) - 
                          [obj.position[0], obj.position[1], 0])
            obj.metadata["is_dynamic"] = True
        
        # Run simulation
        logger.info("Running simulation...")
        animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)
        
        # Generate camera trajectories
        camera_trajectory = self.create_camera_trajectories()
        
        return scene, simulator, renderer, camera_trajectory
    
    def render_multi_view(self, scene: kb.Scene, renderer: Any, 
                         camera_trajectory: List[Dict]) -> None:
        """
        Render the scene from multiple camera viewpoints.
        
        Args:
            scene: Kubric scene
            renderer: Blender renderer
            camera_trajecto
            ries: List of camera trajectory data
        """
        # Create directories for each camera
        for cam_id in range(self.args.num_frames):
            cam_dir = self.output_dir / f"camera_{cam_id:02d}"
            cam_dir.mkdir(exist_ok=True)
            
            # Set up camera for this trajectory
            scene.camera = kb.PerspectiveCamera(
                focal_length=camera_trajectory['focal_length'],
                sensor_width=camera_trajectory['sensor_width']
            )
            
            # Set camera keyframes
            for frame in range(self.args.num_frames):
                scene.camera.position = camera_trajectory['positions'][cam_id]
                scene.camera.quaternion = camera_trajectory['quaternions'][cam_id]
                scene.camera.keyframe_insert("position", frame)
                scene.camera.keyframe_insert("quaternion", frame)
            
            # Render the sequence
            logger.info(f"Rendering camera {cam_id}...")
            data_stack = renderer.render()
            
            # Post-process
            kb.compute_visibility(data_stack["segmentation"], scene.assets)
            visible_foreground_assets = [asset for asset in scene.foreground_assets
                                       if np.max(asset.metadata["visibility"]) > 0]
            visible_foreground_assets = sorted(
                visible_foreground_assets,
                key=lambda asset: np.sum(asset.metadata["visibility"]),
                reverse=True)
            
            data_stack["segmentation"] = kb.adjust_segmentation_idxs(
                data_stack["segmentation"], scene.assets, visible_foreground_assets)
            
            # Save rendered data
            kb.write_image_dict(data_stack, cam_dir)
            
            # Compute bounding boxes
            kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                            visible_foreground_assets)
            
            # Save metadata for this camera
            camera_metadata = {
                "camera_id": cam_id,
                "focal_length": camera_trajectory['focal_length'],
                "sensor_width": camera_trajectory['sensor_width'],
                "field_of_view": camera_trajectory['field_of_view'],
                "positions": camera_trajectory['positions'][cam_id],
                "quaternions": camera_trajectory['quaternions'][cam_id],
                "lookat_target": camera_trajectory['lookat_target'],
                "num_frames": self.args.num_frames,
                "resolution": self.args.resolution,
                "num_instances": len(visible_foreground_assets),
                "type": "fixed_trajectory"
            }
            
            with open(cam_dir / "camera_metadata.json", 'w') as f:
                json.dump(camera_metadata, f, indent=2, default=to_serializable)
            
            # Save scene metadata (convert numpy arrays to lists for JSON serialization)
            scene_metadata = {
                "camera": self._convert_numpy_to_list(kb.get_camera_info(scene.camera)),
                "instances": self._convert_numpy_to_list(kb.get_instance_info(scene, visible_foreground_assets)),
                "metadata": self._convert_numpy_to_list(kb.get_scene_metadata(scene))
            }
            
            with open(cam_dir / "scene_metadata.json", 'w') as f:
                json.dump(scene_metadata, f, indent=2)
            
            logger.info(f"Saved camera {cam_id} data to {cam_dir}")
    
    def generate_dataset(self, seed: int = 42) -> None:
        """
        Generate the complete multi-view dataset.
        
        Args:
            seed: Random seed for reproducibility
        """
        logger.info(f"Generating multi-view dataset with seed {seed}")
        
        # Set random seed
        self.rng = np.random.RandomState(seed=seed)
        
        # Setup scene
        scene, simulator, renderer, camera_trajectory = self.setup_scene()
        
        # Render from all camera viewpoints
        self.render_multi_view(scene, renderer, camera_trajectory)
        
        # Save global metadata (convert numpy arrays to lists for JSON serialization)
        global_metadata = {
            "dataset_type": "multi_view_movi_e",
            "num_cameras": self.num_cameras,
            "num_frames": self.num_frames,
            "resolution": self.resolution,
            "frame_rate": self.frame_rate,
            "seed": seed,
            "camera_trajectories": self._convert_numpy_to_list(camera_trajectory)
        }
        
        with open(self.output_dir / "global_metadata.json", 'w') as f:
            json.dump(global_metadata, f, indent=2)
        
        logger.info(f"Multi-view dataset generation complete! Output saved to {self.output_dir}")


def main():
    """Main function to run the multi-view dataset generator."""
    parser = argparse.ArgumentParser(description="Generate multi-view dataset using Kubric")
    parser.add_argument("--output_dir", type=str, default="multi_view_output",
                       help="Output directory for the dataset")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256],
                       help="Image resolution (width height)")
    parser.add_argument("--max_camera_movement", type=float, default=8.0)
    parser.add_argument("--frame_rate", type=int, default=24)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=23)
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--traj_motion", type=str, default="linear_movement_linear_lookat")
    
    args = parser.parse_args()
    
    # Create generator
    generator = MultiViewDatasetGenerator(
        args
    )
    
    # Generate dataset
    generator.generate_dataset(seed=args.seed)


if __name__ == "__main__":
    main()
