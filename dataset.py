import os
import numpy as np
import json
import trimesh
import shutil
from pathlib import Path
import argparse
from PIL import Image
import time
import random
from typing import List, Dict, Tuple, Optional, Union
import traceback
import scipy.io as sio
import pandas as pd
from datetime import datetime, timedelta
import cv2

# Import relevant modules from the existing codebase
from Sensor import Sensor
from Material import Material
from SceneGenerator import SceneGenerator
from LidarSimulation import LiDARSimulator
from OpticalSimulation import OpticalSimulator

class FrustumPointPillarsDatasetGenerator:
    """
    Generate a dataset specifically formatted for the Frustum PointPillars network.
    This extends the original PointPillars dataset generator with:
    1. Camera image generation
    2. 2D bounding box generation for objects (based on camera projection)
    3. Frustum point cloud extraction
    """
    
    def __init__(self, base_folder: str):
        """
        Initialize the dataset generator.
        
        Args:
            base_folder: Base directory for the dataset
        """
        self.base_folder = Path(base_folder)
        
        # Create required directories if they don't exist
        self.base_folder.mkdir(exist_ok=True, parents=True)
        
        # Create the specific directory structure expected by the KITTI format
        self.lidar_fundamental = self.base_folder / "Lidar" / "Fundamental"
        self.lidar_scenes = self.base_folder / "Lidar" / "Scenes"
        self.lidar_test_scenes = self.base_folder / "Lidar" / "TestScenes"
        self.cuboids_folder = self.base_folder / "Cuboids"
        self.input_data = self.base_folder / "InputData"
        self.gt_samples = self.base_folder / "GTsamples"
        
        # Additional directories for Frustum PointPillars
        self.image_dir = self.base_folder / "image_2"
        self.calib_dir = self.base_folder / "calib"
        self.label_2_dir = self.base_folder / "label_2"
        self.frustum_lidar_dir = self.base_folder / "velodyne_frustum"
        
        # Create all required directories
        self.lidar_fundamental.mkdir(exist_ok=True, parents=True)
        self.lidar_scenes.mkdir(exist_ok=True, parents=True)
        self.lidar_test_scenes.mkdir(exist_ok=True, parents=True)
        self.cuboids_folder.mkdir(exist_ok=True)
        self.input_data.mkdir(exist_ok=True)
        self.gt_samples.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        self.calib_dir.mkdir(exist_ok=True)
        self.label_2_dir.mkdir(exist_ok=True)
        self.frustum_lidar_dir.mkdir(exist_ok=True)
        
        # Initialize scene generator
        self.scene_generator = SceneGenerator()
        
        # MATLAB-compatible sensor parameters for LiDAR
        self.lidar_sensor_params = {
            'horizontal_resolution': 1856,
            'vertical_resolution': 64,
            'vertical_fov': [75, -75],  # [max_angle, min_angle] in degrees
            'step_angle': 1.0,          # Angular resolution
            'min_distance': 0.1,
            'max_distance': 130.0
        }
        
        # Camera parameters for image generation (KITTI-like)
        self.camera_params = {
            'width': 1242,
            'height': 375,
            'horizontal_fov': 90.0,
            'vertical_fov': 35.0,
            'position': np.array([0.0, 0.0, 1.6]),  # Camera height similar to KITTI
            'direction': np.array([1.0, 0.0, 0.0])  # Forward-facing
        }
        
        # Class definitions and colors
        self.class_definitions = {}
        
        # Calibration matrix template (will be adjusted per scene)
        self.calib_template = {
            'P0': np.zeros((3, 4)),
            'P1': np.zeros((3, 4)),
            'P2': np.zeros((3, 4)),
            'P3': np.zeros((3, 4)),
            'R0_rect': np.eye(3),
            'Tr_velo_to_cam': np.zeros((3, 4)),
            'Tr_imu_to_velo': np.zeros((3, 4))
        }
    
    def _extend_matrix(self, mat):
        """Extend a 3x4 matrix to 4x4 by adding [0,0,0,1] row"""
        return np.vstack((mat, np.array([0, 0, 0, 1])))
    
    def create_calibration_matrices(self, scene_center: np.ndarray):
        """
        Create calibration matrices for a scene.
        
        Args:
            scene_center: Center point of the scene
            
        Returns:
            Dictionary of calibration matrices in KITTI format
        """
        # Set up a virtual camera with parameters similar to KITTI
        fx = self.camera_params['width'] / (2 * np.tan(np.radians(self.camera_params['horizontal_fov'] / 2)))
        fy = self.camera_params['height'] / (2 * np.tan(np.radians(self.camera_params['vertical_fov'] / 2)))
        cx = self.camera_params['width'] / 2
        cy = self.camera_params['height'] / 2
        
        # Camera intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Camera position and orientation
        # For KITTI-like setup: camera is at origin, LiDAR is offset
        # We'll use a simplified transformation where:
        # - Camera is at a height of 1.6m looking forward
        # - LiDAR is slightly higher than camera (e.g., at 2.0m)
        
        # Translation from velodyne to camera (KITTI-like)
        # In KITTI, LiDAR is slightly behind and above the camera
        t_velo_to_cam = np.array([0.0, 0.0, 0.4])  # LiDAR is 0.4m above camera
        
        # Rotation from velodyne to camera (KITTI-like)
        # In KITTI, rotation changes coordinate system:
        # LiDAR: x-forward, y-left, z-up
        # Camera: x-right, y-down, z-forward
        R_velo_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        
        # Combine rotation and translation
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[:3, :3] = R_velo_to_cam
        Tr_velo_to_cam[:3, 3] = t_velo_to_cam
        
        # Identity rotation for rectification (already rectified)
        R0_rect = np.eye(3)
        
        # Projection matrices (P0, P1, P2, P3)
        # For simplicity, we'll use the same projection for all cameras
        # In reality, KITTI has slightly different parameters for each camera
        P = np.zeros((3, 4))
        P[:3, :3] = K
        
        P0 = P.copy()
        P1 = P.copy()
        P2 = P.copy()
        P3 = P.copy()
        
        # Add a small baseline for P1 (right camera)
        P1[0, 3] = 0.54  # ~54cm baseline similar to KITTI
        
        # Transformation from IMU to velodyne (identity for simplicity)
        Tr_imu_to_velo = np.zeros((3, 4))
        Tr_imu_to_velo[:3, :3] = np.eye(3)
        
        return {
            'P0': P0,
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'R0_rect': R0_rect,
            'Tr_velo_to_cam': Tr_velo_to_cam,
            'Tr_imu_to_velo': Tr_imu_to_velo
        }
    
    def save_calibration_file(self, idx: int, calib_data: Dict[str, np.ndarray]):
        """
        Save calibration data in KITTI format.
        
        Args:
            idx: Scene index
            calib_data: Dictionary of calibration matrices
        """
        calib_path = self.calib_dir / f"{idx:06d}.txt"
        with open(calib_path, 'w') as f:
            f.write(f"P0: {' '.join(map(str, calib_data['P0'].flatten()))}\n")
            f.write(f"P1: {' '.join(map(str, calib_data['P1'].flatten()))}\n")
            f.write(f"P2: {' '.join(map(str, calib_data['P2'].flatten()))}\n")
            f.write(f"P3: {' '.join(map(str, calib_data['P3'].flatten()))}\n")
            f.write(f"R0_rect: {' '.join(map(str, calib_data['R0_rect'].flatten()))}\n")
            f.write(f"Tr_velo_to_cam: {' '.join(map(str, calib_data['Tr_velo_to_cam'].flatten()))}\n")
            f.write(f"Tr_imu_to_velo: {' '.join(map(str, calib_data['Tr_imu_to_velo'].flatten()))}\n")
    
    def create_camera_sensor(self) -> Sensor:
        """
        Create a camera sensor with parameters compatible with KITTI.
        
        Returns:
            Configured Sensor object for camera
        """
        return Sensor(
            position=self.camera_params['position'],
            target_direction=self.camera_params['direction'],
            horizontal_fov=self.camera_params['horizontal_fov'],
            vertical_fov=self.camera_params['vertical_fov']
        )
    
    def create_lidar_sensor(self, position: np.ndarray, target_direction: np.ndarray) -> Sensor:
        """
        Create a LiDAR sensor with parameters compatible with MATLAB's expectations.
        
        Args:
            position: Sensor position
            target_direction: Direction the sensor is pointing
            
        Returns:
            Configured Sensor object for LiDAR
        """
        vfov_range = self.lidar_sensor_params['vertical_fov'][0] - self.lidar_sensor_params['vertical_fov'][1]
        
        return Sensor(
            position=position,
            target_direction=target_direction,
            horizontal_fov=360.0,  # Full horizontal coverage
            vertical_fov=vfov_range,
            step_angle=self.lidar_sensor_params['step_angle'],
            min_distance=self.lidar_sensor_params['min_distance'],
            max_distance=self.lidar_sensor_params['max_distance']
        )
    
    def project_3d_to_2d(self, points_3d: np.ndarray, calib_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Project 3D points to 2D image plane using calibration data.
        
        Args:
            points_3d: Nx3 array of 3D points (in velodyne coordinate system)
            calib_data: Calibration data dictionary
            
        Returns:
            Nx2 array of 2D points in image coordinates
        """
        # Convert points from velodyne to camera coordinate system
        R = calib_data['R0_rect']
        T = calib_data['Tr_velo_to_cam']
        
        # Add homogeneous coordinate to make dimensions match
        points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        
        # Transform to camera coordinate system
        # T is 3x4, so we need to handle the matrix multiplication carefully
        points_cam = np.zeros((points_3d.shape[0], 3))
        for i in range(points_3d_hom.shape[0]):
            points_cam[i] = T[:3, :3] @ points_3d_hom[i, :3] + T[:3, 3]
        
        # Apply rectification (R is 3x3)
        points_rect = points_cam @ R.T
        
        # Project to image plane using P2 (which is 3x4)
        P2 = calib_data['P2']
        
        # Prepare points for projection - add the homogeneous coordinate again
        points_rect_hom = np.hstack((points_rect, np.ones((points_rect.shape[0], 1))))
        points_2d_hom = points_rect_hom @ P2.T
        
        # Normalize by dividing by the third coordinate
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]
        
        return points_2d
    
    def get_3d_box_corners(self, box_center: np.ndarray, dimensions: np.ndarray, rotation_y: float) -> np.ndarray:
        """
        Get the corners of a 3D bounding box.
        
        Args:
            box_center: Center of the box [x, y, z]
            dimensions: Dimensions of the box [length, width, height]
            rotation_y: Rotation around y-axis
            
        Returns:
            8x3 array of corner coordinates
        """
        # Create a box at the origin with the given dimensions
        l, w, h = dimensions
        
        # Define the 8 corners of the box
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, h, h, h, h]
        
        # Combine corners
        corners = np.vstack([x_corners, y_corners, z_corners]).T
        
        # Create rotation matrix around y-axis
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])
        
        # Rotate corners
        corners = np.dot(corners, R.T)
        
        # Translate corners to box center
        corners += box_center
        
        return corners
    
    def get_2d_box_from_3d(self, corners_3d: np.ndarray, calib_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Project 3D box corners to 2D and get the bounding box.
        
        Args:
            corners_3d: 8x3 array of 3D box corners
            calib_data: Calibration data dictionary
            
        Returns:
            4-element array [xmin, ymin, xmax, ymax] representing the 2D box
        """
        # Project 3D corners to 2D
        corners_2d = self.project_3d_to_2d(corners_3d, calib_data)
        
        # Get the bounding box
        xmin = np.min(corners_2d[:, 0])
        ymin = np.min(corners_2d[:, 1])
        xmax = np.max(corners_2d[:, 0])
        ymax = np.max(corners_2d[:, 1])
        
        # Ensure the box is within image boundaries
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.camera_params['width'] - 1, xmax)
        ymax = min(self.camera_params['height'] - 1, ymax)
        
        return np.array([xmin, ymin, xmax, ymax])
    
    def organize_point_cloud(self, points: np.ndarray, normals: np.ndarray, reflectivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Organize point cloud data in a grid structure compatible with MATLAB's expectations.
        
        Args:
            points: (N, 3) array of point locations
            normals: (N, 3) array of point normals
            reflectivity: (N,) array of reflectivity values
            
        Returns:
            Organized point cloud data and reflectivity
        """
        # For organized point cloud in MATLAB, we need to structure the points in a grid
        h_res = self.lidar_sensor_params['horizontal_resolution']
        v_res = self.lidar_sensor_params['vertical_resolution']
        
        # Create azimuth and elevation angles from points (relative to sensor at origin)
        # This assumes points are from a single sensor position at the origin
        r = np.sqrt(np.sum(points**2, axis=1))
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        elevation = np.arcsin(points[:, 2] / np.clip(r, 1e-10, None))
        
        # Convert to degrees
        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)
        
        # Map angles to grid indices
        # Azimuth: -180 to 180 mapped to 0 to h_res-1
        # Elevation: min_angle to max_angle mapped to 0 to v_res-1
        min_elev = self.lidar_sensor_params['vertical_fov'][1]
        max_elev = self.lidar_sensor_params['vertical_fov'][0]
        
        h_indices = ((azimuth_deg + 180) / 360 * h_res).astype(int)
        h_indices = np.clip(h_indices, 0, h_res - 1)
        
        v_indices = ((elevation_deg - min_elev) / (max_elev - min_elev) * v_res).astype(int)
        v_indices = np.clip(v_indices, 0, v_res - 1)
        
        # Create organized point cloud (initialize with NaNs)
        organized_points = np.full((v_res, h_res, 3), np.nan)
        organized_reflectivity = np.full((v_res, h_res), np.nan)
        
        # Fill in the organized arrays
        # In case of multiple points mapping to the same grid cell, use the closest one
        for i in range(len(points)):
            v_idx = v_indices[i]
            h_idx = h_indices[i]
            
            # Check if position is empty or this point is closer
            if np.isnan(organized_points[v_idx, h_idx, 0]) or r[i] < np.sum(organized_points[v_idx, h_idx]**2)**0.5:
                organized_points[v_idx, h_idx] = points[i]
                organized_reflectivity[v_idx, h_idx] = reflectivity[i]
        
        return organized_points, organized_reflectivity
    
    def save_point_cloud_for_matlab(self, points: np.ndarray, reflectivity: np.ndarray, filepath: Path):
        """
        Save point cloud in a format compatible with MATLAB's pcread function.
        
        Args:
            points: Point cloud data
            reflectivity: Reflectivity values
            filepath: Output filepath
        """
        try:
            import open3d as o3d
            
            # Create a point cloud object
            pcd = o3d.geometry.PointCloud()
            
            # Reshape organized points to list of points
            valid_points = ~np.isnan(points[:, :, 0])
            valid_coords = np.where(valid_points)
            
            point_list = points[valid_coords]
            reflectivity_list = reflectivity[valid_coords]
            
            # Set points
            pcd.points = o3d.utility.Vector3dVector(point_list)
            
            # Set intensity values (reflectivity)
            colors = np.zeros((len(reflectivity_list), 3))
            colors[:, 0] = reflectivity_list  # R
            colors[:, 1] = reflectivity_list  # G
            colors[:, 2] = reflectivity_list  # B
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save to file in PCD format
            o3d.io.write_point_cloud(str(filepath), pcd)
        except ImportError:
            print("Warning: open3d not available, using numpy to save point cloud")
            # Fallback to numpy save
            valid_mask = ~np.isnan(points[:, :, 0])
            valid_points = points[valid_mask]
            valid_reflectivity = reflectivity[valid_mask]
            
            # Stack points and reflectivity
            point_cloud_data = np.column_stack((valid_points, valid_reflectivity))
            
            # Save as binary file
            point_cloud_data.astype(np.float32).tofile(str(filepath))
    
    def save_point_cloud_as_bin(self, points: np.ndarray, reflectivity: np.ndarray, filepath: Path):
        """
        Save point cloud in KITTI bin format.
        
        Args:
            points: Nx3 array of points
            reflectivity: N array of reflectivity values
            filepath: Output filepath
        """
        # Stack points and reflectivity
        if len(points.shape) == 3:  # If organized
            valid_mask = ~np.isnan(points[:, :, 0])
            valid_points = points[valid_mask]
            valid_reflectivity = reflectivity[valid_mask].reshape(-1, 1)
        else:  # If unorganized
            valid_points = points
            valid_reflectivity = reflectivity.reshape(-1, 1)
        
        # Create KITTI format point cloud (x, y, z, intensity)
        kitti_cloud = np.hstack((valid_points, valid_reflectivity))
        
        # Save to binary file
        kitti_cloud.astype(np.float32).tofile(str(filepath))
    
    def save_kitti_label(self, idx: int, objects: Dict[str, List], calib_data: Dict[str, np.ndarray]):
        """
        Save object labels in KITTI format.
        
        Args:
            idx: Scene index
            objects: Dictionary mapping class names to lists of objects
            calib_data: Calibration data dictionary
        """
        label_path = self.label_2_dir / f"{idx:06d}.txt"
        
        with open(label_path, 'w') as f:
            for class_name, obj_list in objects.items():
                for obj in obj_list:
                    # Extract 3D box parameters
                    center = obj['center']
                    dimensions = obj['dimensions']
                    rotation_y = obj['rotation_y']
                    
                    # Get 3D corners
                    corners_3d = self.get_3d_box_corners(center, dimensions, rotation_y)
                    
                    # Project to 2D
                    try:
                        box_2d = self.get_2d_box_from_3d(corners_3d, calib_data)
                    except:
                        # Skip if projection fails
                        continue
                    
                    # Check if box is within image
                    if (box_2d[2] - box_2d[0]) <= 0 or (box_2d[3] - box_2d[1]) <= 0:
                        continue
                        
                    # Convert to camera coordinate system for KITTI format
                    R0_rect = calib_data['R0_rect']
                    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
                    
                    # Add homogeneous coordinate
                    center_hom = np.append(center, 1)
                    
                    # Transform to camera coordinate system
                    center_cam = np.dot(Tr_velo_to_cam, center_hom)
                    center_rect = np.dot(R0_rect, center_cam[:3])
                    
                    # KITTI format: 
                    # type truncated occluded alpha xmin ymin xmax ymax height width length x y z rotation_y
                    truncated = 0.0
                    occluded = 0
                    
                    # Calculate alpha (observation angle)
                    alpha = -np.arctan2(center[0], center[2]) + rotation_y
                    
                    # Write KITTI label format
                    f.write(f"{class_name} {truncated:.1f} {occluded} {alpha:.2f} "
                            f"{box_2d[0]:.2f} {box_2d[1]:.2f} {box_2d[2]:.2f} {box_2d[3]:.2f} "
                            f"{dimensions[2]:.2f} {dimensions[1]:.2f} {dimensions[0]:.2f} "
                            f"{center_rect[0]:.2f} {center_rect[1]:.2f} {center_rect[2]:.2f} {rotation_y:.2f}\n")
    
    def generate_frustum_point_cloud(self, idx: int, points: np.ndarray, calib_data: Dict[str, np.ndarray]):
        """
        Generate frustum point clouds based on 2D bounding boxes.
        
        Args:
            idx: Scene index
            points: LiDAR points
            calib_data: Calibration data dictionary
        """
        # Read 2D bounding boxes from label file
        label_path = self.label_2_dir / f"{idx:06d}.txt"
        if not label_path.exists():
            return
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Extract 2D boxes
        boxes_2d = []
        class_names = []
        for line in lines:
            parts = line.strip().split()
            class_name = parts[0]
            box_2d = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            boxes_2d.append(box_2d)
            class_names.append(class_name)
        
        # Project all points to image plane
        points_2d = self.project_3d_to_2d(points[:, :3], calib_data)
        
        # Create frustum point clouds for each box
        for i, (box_2d, class_name) in enumerate(zip(boxes_2d, class_names)):
            xmin, ymin, xmax, ymax = box_2d
            
            # Check which points fall within the 2D box
            mask = ((points_2d[:, 0] >= xmin) & (points_2d[:, 0] <= xmax) & 
                    (points_2d[:, 1] >= ymin) & (points_2d[:, 1] <= ymax))
            
            # Get frustum points
            frustum_points = points[mask]
            
            if len(frustum_points) == 0:
                continue
                
            # Save frustum point cloud
            frustum_file = self.frustum_lidar_dir / f"{idx:06d}_{i:02d}_{class_name}.bin"
            self.save_point_cloud_as_bin(frustum_points[:, :3], frustum_points[:, 3], frustum_file)
    
    def extract_cuboid_parameters(self, obj: Dict, transform: np.ndarray) -> Dict:
        """
        Extract cuboid parameters from an object.
        
        Args:
            obj: Object dictionary
            transform: Transformation matrix
            
        Returns:
            Dictionary of cuboid parameters
        """
        # Extract position (center)
        position = transform[:3, 3]
        
        # Extract dimensions (from scale)
        scale = np.diagonal(transform[:3, :3])
        dimensions = scale * np.array([1.0, 1.0, 1.0])  # Adjust based on your model's dimensions
        
        # Extract rotation
        rotation_matrix = transform[:3, :3] / scale.reshape(3, 1)
        rotation_y = np.arctan2(rotation_matrix[0, 2], rotation_matrix[0, 0])
        
        return {
            'center': position,
            'dimensions': dimensions,
            'rotation_y': rotation_y
        }
    
    def generate_gtruth_mat(self, scene_data: List[Dict]) -> None:
        """
        Generate the gTruth.mat file expected by the MATLAB PointPillars training code.
        
        Args:
            scene_data: List of dictionaries containing scene information.
                        Each scene should have an 'objects' key mapping class names to cuboid parameter lists.
        """
        if not scene_data:
            print("No scene data to generate gTruth.mat")
            return

        # Extract class names from the first scene that contains objects.
        class_names = []
        for scene in scene_data:
            if scene.get('objects'):
                class_names = list(scene['objects'].keys())
                break
        if not class_names:
            class_names = list(self.class_definitions.keys())

        # Create a sequential array of timestamps (in seconds).
        timestamps = np.array([i + 1 for i in range(len(scene_data))], dtype=float)

        # Create the DataSource structure.
        data_source = {
            'Name': "Point Cloud Sequence",
            'Description': "A Point Cloud sequence reader",
            'SourceName': str(self.base_folder),
            'SourceParams': {},
            'SignalName': "Lidar",
            'SignalType': 1,
            'Timestamp': timestamps,  # Stored as a simple numeric array
            'NumSignals': 1
        }

        # Build the LabelDefinitions structure as a table-like dictionary.
        label_definitions_table = {
            'Name': [class_name for class_name in class_names],
            'Type': ['5' for _ in class_names],  # '5' indicates cuboid label type
            'LabelColor': [[random.random(), random.random(), random.random()] for _ in class_names],
            'Group': ['None' for _ in class_names],
            'Description': ['' for _ in class_names]
        }

        # Create the LabelData structure.
        # For each class, store a list of cuboid lists per scene.
        label_data_table = {class_name: [] for class_name in class_names}
        for scene in scene_data:
            for class_name in class_names:
                # If the scene contains objects of this class, use them; otherwise, use an empty list.
                cuboids = scene.get('objects', {}).get(class_name, [])
                label_data_table[class_name].append(cuboids)

        # Combine everything into the gTruth structure.
        # We save LabelData as a struct with fields 'Time' and 'Variables'.
        gtruth = {
            'DataSource': data_source,
            'LabelDefinitions': label_definitions_table,
            'LabelData': {
                'Time': timestamps,
                'Variables': label_data_table
            }
        }

        # Save the resulting structure to a .mat file.
        try:
            output_path = str(self.cuboids_folder / 'gTruth.mat')
            sio.savemat(output_path, {'gTruth': gtruth})
            print(f"  Ground truth data saved to {output_path}")
        except Exception as e:
            print(f"  Error saving gTruth.mat: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_dataset(self,
                         num_scenes: int = 100,
                         object_paths: List[str] = None,
                         num_objects_per_scene: Tuple[int, int] = (3, 8),
                         add_lidar_noise: bool = True,
                         test_split: float = 0.2):
        """
        Generate the complete dataset formatted for Frustum PointPillars.
        
        Args:
            num_scenes: Number of scenes to generate
            object_paths: List of paths to object model files (.obj, .stl)
            num_objects_per_scene: Range of number of objects per scene (min, max)
            add_lidar_noise: Whether to add noise to LiDAR scans
            test_split: Fraction of scenes to use for test set
        """
        if object_paths is None or len(object_paths) == 0:
            raise ValueError("No object model paths provided. Please provide paths to .obj or .stl files.")
        
        print(f"Generating {num_scenes} scenes for the Frustum PointPillars dataset...")
        
        # Map object paths to class names
        class_names = [Path(path).stem for path in object_paths]
        self.class_definitions = {name: path for name, path in zip(class_names, object_paths)}
        
        # Create a default material for simulation
        default_material = Material(albedo=0.7, metallic=0.0, roughness=0.5, ambient=0.2)
        
        scene_data = []
        
        # Calculate number of training and test scenes
        num_test_scenes = int(num_scenes * test_split)
        num_train_scenes = num_scenes - num_test_scenes
        
        for scene_idx in range(num_scenes):
            start_time = time.time()
            print(f"\nGenerating scene {scene_idx+1}/{num_scenes}...")
            
            try:
                # Create scene folder
                scene_folder = self.lidar_scenes if scene_idx < num_train_scenes else self.lidar_test_scenes
                
                # 1. Generate a random scene with objects
                num_objects = random.randint(num_objects_per_scene[0], num_objects_per_scene[1])
                self._generate_random_scene(num_objects, object_paths)
                
                # Get scene bounds for proper sensor placement
                scene_mesh = self.scene_generator.get_combined_mesh()
                scene_bounds = scene_mesh.bounds
                scene_center = (scene_bounds[0] + scene_bounds[1]) / 2
                dimensions = scene_bounds[1] - scene_bounds[0]
                max_dim = max(dimensions)
                
                # 2. Create calibration matrices for this scene
                calib_data = self.create_calibration_matrices(scene_center)
                self.save_calibration_file(scene_idx, calib_data)
                
                # 3. Place LiDAR sensor for the scene
                # Place LiDAR above the scene center looking down (similar to KITTI setup)
                lidar_position = scene_center + np.array([0, 0, max_dim * 1.0])
                lidar_direction = np.array([0, 0, -1])  # Looking downward
                
                lidar_sensor = self.create_lidar_sensor(lidar_position, lidar_direction)
                
                # 4. Run LiDAR simulation
                print(f"  Running LiDAR simulation...")
                lidar_simulator = LiDARSimulator(
                    mesh=scene_mesh,
                    sensors=[lidar_sensor],
                    material=default_material
                )
                
                # Generate the LiDAR point cloud
                lidar_results = lidar_simulator.simulate(add_noise=add_lidar_noise)
                
                # Each scene has only one LiDAR sensor, so get points from the first sensor
                points, normals, reflectivity = lidar_results[0]
                
                # 5. Save the point cloud for MATLAB in organized format
                organized_points, organized_reflectivity = self.organize_point_cloud(points, normals, reflectivity)
                
                scene_pcd_file = f"Scene_{scene_idx:06d}.pcd"
                scene_pcd_path = scene_folder / scene_pcd_file
                
                self.save_point_cloud_for_matlab(organized_points, organized_reflectivity, scene_pcd_path)
                
                # 6. Save point cloud in KITTI bin format for Frustum PointPillars
                velodyne_file = self.base_folder / "velodyne" / f"{scene_idx:06d}.bin"
                velodyne_file.parent.mkdir(exist_ok=True, parents=True)
                self.save_point_cloud_as_bin(points, reflectivity, velodyne_file)
                
                # 7. Configure camera and generate image
                print(f"  Generating camera image...")
                camera_position = scene_center + np.array([0, 0, dimensions[2] * 0.8])  # Place camera at 80% of scene height
                camera_direction = np.array([1, 0, 0])  # Look forward (along X-axis)
                
                camera_sensor = Sensor(
                    position=camera_position,
                    target_direction=camera_direction,
                    horizontal_fov=self.camera_params['horizontal_fov'],
                    vertical_fov=self.camera_params['vertical_fov']
                )
                
                # Run optical simulation
                optical_simulator = OpticalSimulator(
                    mesh=scene_mesh,
                    sensors=[camera_sensor],
                    base_image_width=self.camera_params['width'],
                    background_color=[0, 0, 0],
                    default_color=[150, 150, 150]
                )
                
                # Render the camera image
                images = optical_simulator.render()
                
                if len(images) > 0:
                    # Save the rendered image
                    image_path = self.image_dir / f"{scene_idx:06d}.png"
                    images[0].save(str(image_path))
                else:
                    print(f"  Warning: Failed to render image for scene {scene_idx}")
                
                # 8. Extract object information and generate KITTI labels
                print(f"  Generating labels and frustum point clouds...")
                scene_objects = {}
                
                # Process each object in the scene
                for name, obj in self.scene_generator.objects.items():
                    # Extract class name from object name
                    for class_name in self.class_definitions.keys():
                        if class_name in name:
                            if class_name not in scene_objects:
                                scene_objects[class_name] = []
                            
                            # Create transformation matrix for the object
                            transform = np.eye(4)
                            transform[:3, 3] = obj.position
                            
                            # Apply rotation
                            rot_mat = trimesh.transformations.euler_matrix(
                                obj.rotation[0], obj.rotation[1], obj.rotation[2])
                            transform[:3, :3] = rot_mat[:3, :3]
                            
                            # Apply scale
                            transform[:3, :3] *= obj.scale.reshape(3, 1)
                            
                            # Extract cuboid parameters
                            cuboid_params = self.extract_cuboid_parameters(obj, transform)
                            scene_objects[class_name].append(cuboid_params)
                
                # Save KITTI format labels
                self.save_kitti_label(scene_idx, scene_objects, calib_data)
                
                # 9. Generate frustum point clouds
                # This extracts points that fall within the 2D bounding boxes
                self.generate_frustum_point_cloud(scene_idx, np.column_stack((points, reflectivity)), calib_data)
                
                # 10. Add scene data to collection for MATLAB ground truth
                scene_data.append({
                    'file': scene_pcd_file,
                    'objects': {class_name: [self._convert_to_matlab_format(obj) for obj in objs] 
                               for class_name, objs in scene_objects.items()}
                })
                
                elapsed_time = time.time() - start_time
                print(f"  Scene {scene_idx+1} completed in {elapsed_time:.2f} seconds")
                
            except Exception as e:
                print(f"  Error generating scene {scene_idx+1}: {e}")
                traceback.print_exc()
                print("  Continuing with next scene...")
        
        # Generate the gTruth.mat file for MATLAB
        self.generate_gtruth_mat(scene_data)
        
        print(f"\nDataset generation completed. {num_scenes} scenes created in {self.base_folder}")
        print(f"  - {num_train_scenes} training scenes in {self.lidar_scenes}")
        print(f"  - {num_test_scenes} test scenes in {self.lidar_test_scenes}")
        print(f"  - Ground truth data in {self.cuboids_folder}/gTruth.mat")
        print(f"  - Camera images in {self.image_dir}")
        print(f"  - KITTI labels in {self.label_2_dir}")
        print(f"  - Frustum point clouds in {self.frustum_lidar_dir}")

    def _generate_random_scene(self, num_objects: int, object_paths: List[str]):
        """Generate a random scene with multiple objects."""
        try:
            # Clear existing scene
            self.scene_generator.objects.clear()
            
            # Generate a new random scene
            bounds = (10.0, 10.0, 5.0)  # Default scene bounds
            min_distance = 0.5  # Minimum distance between objects
            max_overlap = 0.1  # Allow some overlap to ensure placement
            
            self.scene_generator.create_random_scene(
                object_paths=object_paths,
                num_objects=num_objects,
                bounds=bounds,
                min_distance=min_distance,
                max_overlap=max_overlap
            )
            
            print(f"  Random scene created with {len(self.scene_generator.objects)} objects")
            
        except Exception as e:
            print(f"  Error generating random scene: {e}")
            # Fall back to creating a simple scene with fewer objects
            self.scene_generator.objects.clear()
            reduced_objects = max(1, num_objects // 2)
            
            try:
                # Create a simpler scene with more tolerance
                self.scene_generator.create_random_scene(
                    object_paths=object_paths,
                    num_objects=reduced_objects,
                    bounds=bounds,
                    min_distance=0.2,
                    max_overlap=0.5  # Allow significant overlap to ensure placement
                )
                print(f"  Fallback: Created scene with {len(self.scene_generator.objects)} objects")
            except Exception as e2:
                print(f"  Critical error in scene generation: {e2}")
                # Last resort: add a single object at the center
                try:
                    # Try to load the first object in the list
                    mesh_path = object_paths[0]
                    name = f"failsafe_object"
                    self.scene_generator.add_object(
                        mesh=mesh_path,
                        position=np.array([0.0, 0.0, 0.0]),
                        rotation=np.array([0.0, 0.0, 0.0]),
                        scale=np.array([1.0, 1.0, 1.0]),
                        name=name,
                        material=Material(),
                        max_overlap=1.0
                    )
                    print("  Emergency fallback: Created scene with a single centered object")
                except Exception as e3:
                    # If even that fails, create an empty scene
                    print("  Fatal error: Unable to create scene, will be empty")
    
    def _convert_to_matlab_format(self, cuboid_params: Dict) -> List[float]:
        """
        Convert the cuboid parameters to MATLAB PointPillars format.
        
        Args:
            cuboid_params: Dictionary with cuboid parameters
            
        Returns:
            List of parameters in MATLAB format
        """
        # Format: [x, y, z, length, width, height, ?, ?, yaw]
        return [
            cuboid_params['center'][0],      # x-center
            cuboid_params['center'][1],      # y-center
            cuboid_params['center'][2],      # z-center
            cuboid_params['dimensions'][0],  # length
            cuboid_params['dimensions'][1],  # width
            cuboid_params['dimensions'][2],  # height
            0.0,                            # placeholder
            -1.78,                          # placeholder (from MATLAB code)
            cuboid_params['rotation_y']     # yaw angle
        ]
    
def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for Frustum PointPillars")
    parser.add_argument("--output", required=True, help="Output directory for the dataset")
    parser.add_argument("--models", required=True, help="Directory containing 3D model files (.obj, .stl)")
    parser.add_argument("--num-scenes", type=int, default=100, help="Number of scenes to generate")
    parser.add_argument("--min-objects", type=int, default=3, help="Minimum objects per scene")
    parser.add_argument("--max-objects", type=int, default=8, help="Maximum objects per scene")
    parser.add_argument("--add-noise", action="store_true", help="Add noise to LiDAR scans")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of scenes to use for test set")
    args = parser.parse_args()
    
    # Find all model files in the models directory
    import os
    model_paths = []
    valid_extensions = ['.obj', '.stl', '.ply']
    for file in os.listdir(args.models):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            model_paths.append(os.path.join(args.models, file))
    
    if not model_paths:
        print(f"Error: No valid model files found in {args.models}")
        return
        
    print(f"Found {len(model_paths)} model files")
    
    # Create the dataset generator
    generator = FrustumPointPillarsDatasetGenerator(args.output)
    
    # Generate the dataset
    generator.generate_dataset(
        num_scenes=args.num_scenes,
        object_paths=model_paths,
        num_objects_per_scene=(args.min_objects, args.max_objects),
        add_lidar_noise=args.add_noise,
        test_split=args.test_split
    )
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()