import numpy as np
import trimesh
import torch
from typing import List, Tuple, Optional
from Sensor import Sensor
from Material import Material
from RayIntersection import RayMeshIntersector

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
        
    def _compute_ray_directions(self, target_direction: np.ndarray, sensor: Sensor) -> torch.Tensor:
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
                
        return torch.tensor(directions, device=self.device, dtype=torch.float32)
    
    def _compute_noise(self, distances: torch.Tensor, reflectivity: torch.Tensor) -> torch.Tensor:
        """Compute noise based on distance and reflectivity"""
        # Simplified noise model
        base_noise = torch.where(
            distances <= 5.0, 0.0005,
            torch.where(distances <= 20.0, 0.0007,
            torch.where(distances <= 40.0, 0.0025, 0.005)))
            
        # Scale noise based on reflectivity
        noise_scale = torch.where(
            reflectivity <= 0.08, 1.5,  # Black 8%
            torch.where(reflectivity <= 0.21, 1.2,  # Gray 21%
            1.0))  # White 89%
            
        return torch.randn_like(distances) * base_noise * noise_scale
        
    def simulate(self, add_noise: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform LiDAR simulation from all sensor positions simultaneously
        
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
            
            # Setup ray origins
            origins = torch.tensor(centered_position, device=self.device, dtype=torch.float32)
            origins = origins.expand(len(ray_directions), 3)
            
            # Perform ray-casting
            intersector = RayMeshIntersector(self.mesh)
            points, face_indices, ray_indices = intersector.compute_intersections(
                origins, ray_directions,
                min_distance=sensor.min_distance,
                max_distance=sensor.max_distance
            )

            # Only access face_labels if there are any intersections
            if len(face_indices) > 0:
                try:
                    object_labels = self.mesh.visual.face_labels[face_indices]
                except (AttributeError, IndexError):
                    # If face_labels access fails, create a default array
                    object_labels = np.full(len(face_indices), "default_object")
            
            # Compute surface normals and reflectivity
            normals = self._compute_normals(points)
            reflectivity = self._compute_reflectivity(normals, ray_directions[ray_indices])
            
            if add_noise:
                distances = torch.norm(points - origins[ray_indices], dim=1)
                noise = self._compute_noise(distances, reflectivity)
                points += ray_directions[ray_indices] * noise.unsqueeze(1)
            
            # Convert points back to world coordinates
            world_points = self.centered_to_world(points.cpu().numpy())
                
            results.append((
                world_points,
                normals.cpu().numpy(),
                reflectivity.cpu().numpy()
            ))
            
        return results
    
    def _compute_normals(self, points: torch.Tensor) -> torch.Tensor:
        intersector = RayMeshIntersector(self.mesh)
        nearest_faces = self._find_nearest_faces(points)
        return intersector.compute_normals(points, nearest_faces)
    
    def _find_nearest_faces(self, points: torch.Tensor) -> torch.Tensor:
        """Find the nearest face indices for given points"""
        points_np = points.cpu().numpy()
        _, face_indices, _ = self.mesh.nearest.on_surface(points_np)
        return torch.tensor(face_indices, device=self.device, dtype=torch.int64)
    
    def _compute_reflectivity(self, normals: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
        """
        Compute Lambertian reflectivity for each intersection point.
        
        Args:
            normals: (N, 3) tensor of surface normals at intersection points
            ray_dirs: (N, 3) tensor of ray directions
            
        Returns:
            (N,) tensor of reflectivity values in range [0, 1]
        """
        # Normalize vectors
        normals = normals / torch.norm(normals, dim=1, keepdim=True)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=1, keepdim=True)
        
        # Compute cosine of angle between normal and ray direction
        # Use negative ray direction since we want angle with incoming ray
        cos_theta = torch.abs(torch.sum(normals * (-ray_dirs), dim=1))
        
        # Apply Lambertian reflection model
        # Scale to [0,1] range and apply material albedo
        albedo = self.material.albedo[0]
        reflectivity = albedo * cos_theta
        
        # Ensure reflectivity is in valid range
        return torch.clamp(reflectivity, 0.0, 1.0)
    
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