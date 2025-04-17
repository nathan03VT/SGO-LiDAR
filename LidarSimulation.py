import numpy as np
import trimesh
import torch
from typing import List, Tuple, Optional
from Sensor import Sensor
from Material import Material

class LiDARSimulator:
    def __init__(self, mesh: trimesh.Trimesh, sensors: List[Sensor], material: Optional[Material] = None):
        """        
        Args:
            mesh: Scene mesh to scan
            sensors: List of Sensor objects defining different scan positions
            material: Optional Material object defining surface properties
        """
        self.material = material or Material()
        # Center the mesh at origin
        self.original_centroid = mesh.centroid
        centered_vertices = mesh.vertices - mesh.centroid
        
        # Create new centered mesh
        self.mesh = trimesh.Trimesh(
            vertices=centered_vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals
        )
        
        # Initialize face_labels if not present
        if not hasattr(self.mesh.visual, 'face_labels'):
            # Create default labels (all faces belong to a single object)
            self.mesh.visual.face_labels = np.full(len(self.mesh.faces), "default_object")
        
        self.sensors = sensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Batch size for ray processing to limit memory usage
        self.batch_size = 50000
    
    def world_to_centered(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to centered mesh coordinates"""
        return points - self.original_centroid
    
    def centered_to_world(self, points: np.ndarray) -> np.ndarray:
        """Convert centered mesh coordinates back to world coordinates"""
        return points + self.original_centroid
        
    def _spherical_to_cartesian(self, phi: float, theta: float) -> np.ndarray:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])
    
    def _cartesian_to_spherical(self, direction: np.ndarray) -> Tuple[float, float]:
        direction = direction / np.linalg.norm(direction)
        phi = np.arctan2(direction[1], direction[0])     # azimuth angle
        theta = np.arccos(direction[2])                  # polar angle
        return phi, theta
        
    def _compute_ray_directions(self, target_direction: np.ndarray, sensor: Sensor) -> np.ndarray:
        """
        Compute ray directions for a specific sensor
        """
        # Get base angles for target direction
        center_phi, center_theta = self._cartesian_to_spherical(target_direction)
        
        # Generate angular offsets within FOV
        h_offsets = np.arange(-sensor.horizontal_fov/2, 
                             sensor.horizontal_fov/2,
                             sensor.step_angle)
        v_offsets = np.arange(-sensor.vertical_fov/2,
                             sensor.vertical_fov/2,
                             sensor.step_angle)
        
        # Pre-allocate numpy array
        n_rays = len(h_offsets) * len(v_offsets)
        directions = np.zeros((n_rays, 3))
        
        idx = 0
        for h_offset in h_offsets:
            for v_offset in v_offsets:
                # Apply offsets to base angles
                phi = center_phi + np.radians(h_offset)
                theta = center_theta + np.radians(v_offset)
                
                theta = np.clip(theta, 0, np.pi)
                
                directions[idx] = self._spherical_to_cartesian(phi, theta)
                idx += 1
                
        return directions
    
    def _compute_noise(self, distances: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
        """Compute noise based on distance and reflectivity"""
        # Simplified noise model
        base_noise = np.where(
            distances <= 5.0, 0.0005,
            np.where(distances <= 20.0, 0.0007,
            np.where(distances <= 40.0, 0.0025, 0.005)))
            
        # Scale noise based on reflectivity
        noise_scale = np.where(
            reflectivity <= 0.08, 1.5,  # Black 8%
            np.where(reflectivity <= 0.21, 1.2,  # Gray 21%
            1.0))  # White 89%
            
        return np.random.randn(len(distances)) * base_noise * noise_scale
        
    def simulate(self, add_noise: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform LiDAR simulation from all sensor positions using memory-efficient batching
        
        Args:
            add_noise: Whether to add measurement noise
            
        Returns:
            List of tuples containing (points, normals, reflectivity) for each sensor
        """
        results = []
        
        # Process each sensor
        for sensor in self.sensors:
            # Get sensor position and direction
            position, target_direction = sensor.get_sensor_transform()
            
            # Convert position to centered coordinate system
            centered_position = self.world_to_centered(position)
            
            # Normalize target direction
            target_direction = target_direction / np.linalg.norm(target_direction)
            
            # Compute ray directions for this sensor
            ray_directions = self._compute_ray_directions(target_direction, sensor)
            
            # Initialize arrays to store results
            all_points = []
            all_normals = []
            all_reflectivity = []
            
            # Process rays in batches to reduce memory usage
            num_rays = len(ray_directions)
            num_batches = int(np.ceil(num_rays / self.batch_size))
            
            print(f"Processing {num_rays} rays in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, num_rays)
                
                # Get ray directions for this batch
                batch_directions = ray_directions[start_idx:end_idx]
                
                # Create ray origins for this batch
                batch_origins = np.tile(centered_position, (len(batch_directions), 1))
                
                # Use trimesh's ray.intersects_location directly
                locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                    ray_origins=batch_origins,
                    ray_directions=batch_directions,
                    multiple_hits=False
                )
                
                if len(locations) > 0:
                    # Get face normals for intersections
                    batch_normals = self.mesh.face_normals[index_tri]
                    
                    # Compute reflectivity based on angle between ray and normal
                    batch_ray_dirs = batch_directions[index_ray]
                    dot_products = np.abs(np.sum(batch_normals * (-batch_ray_dirs), axis=1))
                    batch_reflectivity = self.material.albedo * dot_products
                    batch_reflectivity = np.clip(batch_reflectivity, 0.0, 1.0)
                    
                    # Apply noise if requested
                    if add_noise:
                        # Distance-based noise model
                        distances = np.linalg.norm(locations - centered_position, axis=1)
                        
                        # Compute noise based on distance and reflectivity
                        noise = self._compute_noise(distances, batch_reflectivity)
                        
                        # Apply noise in ray direction
                        locations += batch_ray_dirs * noise.reshape(-1, 1)
                        
                        # Add noise to reflectivity
                        batch_reflectivity += np.random.normal(0, 0.05, len(batch_reflectivity))
                        batch_reflectivity = np.clip(batch_reflectivity, 0.0, 1.0)
                    
                    # Add results to the collection
                    all_points.append(locations)
                    all_normals.append(batch_normals)
                    all_reflectivity.append(batch_reflectivity)
            
            # Combine results for this sensor
            if all_points:
                points = np.vstack(all_points)
                normals = np.vstack(all_normals)
                reflectivity = np.concatenate(all_reflectivity)
                
                # Convert points back to world coordinates
                world_points = self.centered_to_world(points)
                
                results.append((
                    world_points,
                    normals,
                    reflectivity
                ))
            else:
                # Return empty arrays if no intersections found
                results.append((
                    np.zeros((0, 3)),
                    np.zeros((0, 3)),
                    np.zeros(0)
                ))
            
        return results
    
    def visualize_results(self, results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], mesh_path: str):
        from VisualizeSimulation import LIDARVisualizer
        
        try:
            # Create visualizer
            visualizer = LIDARVisualizer(mesh_path)
            
            # Combine all point clouds
            all_points = np.concatenate([r[0] for r in results])
            all_normals = np.concatenate([r[1] for r in results])
            all_reflectivity = np.concatenate([r[2] for r in results])
            
            # Get sensor positions
            sensor_positions = np.array([sensor.position for sensor in self.sensors])
            
            # Visualize combined results
            visualizer.visualize_point_cloud(
                points=all_points,
                normals=all_normals,
                scanner_position=sensor_positions[0],
                reflectivity=all_reflectivity
            )
            
            # Add other sensor positions
            for pos in sensor_positions[1:]:
                visualizer.plotter.add_points(
                    pos.reshape(1, 3),
                    color='red',
                    point_size=20,
                    render_points_as_spheres=True
                )
                
            visualizer.show()
            
            # Clean up after showing
            visualizer.close()
            
        except Exception as e:
            print(f"Error in visualize_results: {e}")