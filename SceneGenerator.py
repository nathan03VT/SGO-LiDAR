import numpy as np
import trimesh
import os
import random
from typing import Dict, List, Tuple, Optional, Union
from Material import Material

class SceneObject:
    def __init__(self, 
                 mesh: Union[str, trimesh.Trimesh],
                 position: np.ndarray,
                 rotation: np.ndarray,
                 scale: np.ndarray,
                 name: str,
                 material: Optional[Material] = None):
        """
        Initialize a scene object with mesh, position, rotation, scale, and material
        
        Args:
            mesh: Path to mesh file or trimesh.Trimesh object
            position: (3,) array representing object position [x, y, z]
            rotation: (3,) array representing rotation in radians [rx, ry, rz]
            scale: (3,) array representing scale factors [sx, sy, sz]
            name: Unique identifier for the object
            material: Optional Material object defining surface properties
        """
        # Store the original mesh path if a string is provided
        if isinstance(mesh, str):
            self.mesh_path = mesh
            self.mesh = trimesh.load(mesh)
        else:
            self.mesh_path = None
            self.mesh = mesh.copy()
            
        self.position = np.asarray(position, dtype=np.float32)
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.name = name
        self.material = material
        
        # Apply the transformation to the mesh
        self.apply_transform()
        
    def apply_transform(self):
        """Apply position, rotation, and scale to the mesh"""
        # Start with identity matrix
        transform = np.eye(4)
        
        # Apply rotation (Euler angles)
        rot_mat = trimesh.transformations.euler_matrix(
            self.rotation[0], self.rotation[1], self.rotation[2])
        transform[:3, :3] = rot_mat[:3, :3]
        
        # Apply scale
        transform[:3, :3] *= self.scale.reshape(3, 1)
        
        # Apply translation
        transform[:3, 3] = self.position
        
        # Reset mesh to original state if we have a path
        if self.mesh_path:
            self.mesh = trimesh.load(self.mesh_path)
        
        # Apply transformation
        self.mesh.apply_transform(transform)
        
    def get_bounds(self) -> np.ndarray:
        """Get the axis-aligned bounding box of the object"""
        return self.mesh.bounds
        
    def check_collision(self, other: 'SceneObject', max_overlap: float = 0.0) -> Tuple[bool, float]:
        """
        Check if this object collides with another object
        
        Args:
            other: Another SceneObject to check collision with
            max_overlap: Maximum allowed overlap distance (0.0 = no overlap allowed)
            
        Returns:
            Tuple of (collision_occurred, overlap_distance)
        """
        # Get bounding boxes
        bounds1 = self.get_bounds()
        bounds2 = other.get_bounds()
        
        # Calculate centers and dimensions
        center1 = (bounds1[0] + bounds1[1]) / 2
        center2 = (bounds2[0] + bounds2[1]) / 2
        dims1 = bounds1[1] - bounds1[0]
        dims2 = bounds2[1] - bounds2[0]
        
        # Calculate distance between centers
        center_dist = np.linalg.norm(center1 - center2)
        
        # Calculate minimum radius needed for each object (half of max dimension)
        radius1 = np.max(dims1) / 2
        radius2 = np.max(dims2) / 2
        
        # Calculate overlap (sum of radii minus center distance)
        overlap = radius1 + radius2 - center_dist
        overlap = max(overlap, 0.0)
        
        # Check if overlap exceeds threshold
        return overlap > max_overlap, overlap


class SceneGenerator:
    def __init__(self):
        """Initialize an empty scene"""
        self.objects: Dict[str, SceneObject] = {}
        
    def add_object(self, 
                  mesh: Union[str, trimesh.Trimesh],
                  position: np.ndarray,
                  rotation: np.ndarray,
                  scale: np.ndarray,
                  name: str,
                  material: Optional[Material] = None,
                  max_overlap: float = 0.0) -> SceneObject:
        """
        Add a new object to the scene
        
        Args:
            mesh: Path to mesh file or trimesh.Trimesh object
            position: (3,) array representing object position [x, y, z]
            rotation: (3,) array representing rotation in radians [rx, ry, rz]
            scale: (3,) array representing scale factors [sx, sy, sz]
            name: Unique identifier for the object
            material: Optional Material object defining surface properties
            max_overlap: Maximum allowed overlap with existing objects
            
        Returns:
            The created SceneObject
            
        Raises:
            ValueError: If position causes a collision exceeding max_overlap
        """
        # Create the new object
        new_object = SceneObject(mesh, position, rotation, scale, name, material)
        
        # Check for collisions with existing objects
        if self.objects:
            for obj_name, existing_obj in self.objects.items():
                collision, overlap = new_object.check_collision(existing_obj, max_overlap)
                if collision:
                    raise ValueError(
                        f"Object '{name}' collides with existing object '{obj_name}' "
                        f"with overlap of {overlap:.3f} units (max allowed: {max_overlap:.3f})")
        
        # Add to our dictionary
        self.objects[name] = new_object
        return new_object
    
    def create_random_scene(self,
                  object_paths: List[str],
                  num_objects: int = 5,
                  bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),
                  min_distance: float = 1.0,
                  max_overlap: float = 0.0,
                  max_attempts: int = 100):
        """
        Create a random scene with multiple objects
        
        Args:
            object_paths: List of paths to mesh files
            num_objects: Number of objects to place
            bounds: Scene bounds (x, y, z) from origin
            min_distance: Minimum distance between object centers
            max_overlap: Maximum allowed overlap between objects
            max_attempts: Maximum placement attempts per object
        """
        # Clear existing objects
        self.objects.clear()
        
        # Track failed placements to retry with scaled-down versions
        failed_placements = []
        
        # First attempt to place each object with normal constraints
        for i in range(num_objects):
            # Select a random mesh
            mesh_path = random.choice(object_paths)
            
            # Load the mesh to get its size
            temp_mesh = trimesh.load(mesh_path)
            mesh_dims = temp_mesh.bounding_box.extents
            
            # Generate random parameters
            success = False
            for attempt in range(max_attempts):
                try:
                    # Random position within bounds
                    position = np.array([
                        random.uniform(-bounds[0] + mesh_dims[0]/2, bounds[0] - mesh_dims[0]/2),
                        random.uniform(-bounds[1] + mesh_dims[1]/2, bounds[1] - mesh_dims[1]/2),
                        random.uniform(0.0, bounds[2] - mesh_dims[2]/2)
                    ])
                    
                    # Random rotation (in radians)
                    rotation = np.array([
                        random.uniform(0, 2 * np.pi),
                        random.uniform(0, 2 * np.pi), 
                        random.uniform(0, 2 * np.pi)
                    ])
                    
                    # Random scale (0.5 to 1.5 times original size)
                    scale_factor = random.uniform(0.5, 1.5)
                    scale = np.array([scale_factor, scale_factor, scale_factor])
                    
                    # Random material properties
                    albedo = random.uniform(0.3, 0.9)
                    material = Material(albedo=albedo)
                    
                    # Generate a unique name
                    name = f"object_{i}_{os.path.basename(mesh_path)}"
                    
                    # Try to add the object
                    self.add_object(
                        mesh=mesh_path,
                        position=position,
                        rotation=rotation,
                        scale=scale,
                        name=name,
                        material=material,
                        max_overlap=max_overlap
                    )
                    
                    # Successfully placed, move to next object
                    success = True
                    break
                    
                except ValueError as e:
                    # Collision detected, try again
                    if attempt == max_attempts - 1:
                        # Add to failed placements list for retry with scaled-down version
                        failed_placements.append((i, mesh_path))
                        print(f"Warning: Could not place object {i} after {max_attempts} attempts")
            
        # Retry failed placements with scaled-down objects
        if failed_placements:
            print(f"Retrying {len(failed_placements)} objects with scaled-down versions...")
            
            # Try different scale factors, decreasing size each time
            scale_factors = [0.8, 0.6, 0.4, 0.3, 0.2]
            
            for i, mesh_path in failed_placements:
                # Try each scale factor in sequence
                success = False
                for scale_factor in scale_factors:
                    # Try multiple attempts with this scale factor
                    for attempt in range(max_attempts // 2):  # Use fewer attempts per scale
                        try:
                            # Load the mesh to get its size
                            temp_mesh = trimesh.load(mesh_path)
                            mesh_dims = temp_mesh.bounding_box.extents
                            
                            # Random position within bounds
                            position = np.array([
                                random.uniform(-bounds[0] + mesh_dims[0]*scale_factor/2, 
                                            bounds[0] - mesh_dims[0]*scale_factor/2),
                                random.uniform(-bounds[1] + mesh_dims[1]*scale_factor/2, 
                                            bounds[1] - mesh_dims[1]*scale_factor/2),
                                random.uniform(0.0, bounds[2] - mesh_dims[2]*scale_factor/2)
                            ])
                            
                            # Random rotation (in radians)
                            rotation = np.array([
                                random.uniform(0, 2 * np.pi),
                                random.uniform(0, 2 * np.pi), 
                                random.uniform(0, 2 * np.pi)
                            ])
                            
                            # Use the reduced scale factor
                            scale = np.array([scale_factor, scale_factor, scale_factor])
                            
                            # Random material properties
                            albedo = random.uniform(0.3, 0.9)
                            material = Material(albedo=albedo)
                            
                            # Generate a unique name
                            name = f"object_{i}_{os.path.basename(mesh_path)}"
                            
                            # Try to add the object
                            self.add_object(
                                mesh=mesh_path,
                                position=position,
                                rotation=rotation,
                                scale=scale,
                                name=name,
                                material=material,
                                max_overlap=max_overlap
                            )
                            
                            print(f"  Successfully placed object {i} with scale factor {scale_factor}")
                            success = True
                            break
                            
                        except ValueError:
                            # Collision detected, continue trying
                            pass
                    
                    # If successful with this scale factor, stop trying smaller ones
                    if success:
                        break
                
                # If still not successful after all scale factors, give up on this object
                if not success:
                    print(f"  Failed to place object {i} even with minimum scale factor {scale_factors[-1]}")
        
        print(f"  Random scene created with {len(self.objects.keys())} objects")
    def get_combined_mesh(self) -> trimesh.Trimesh:
        """
        Combine all object meshes into a single mesh
        
        Returns:
            A single trimesh.Trimesh containing all objects
        """
        if not self.objects:
            return trimesh.Trimesh()
            
        # Collect all meshes
        meshes = []
        
        # For each object, add its mesh
        for name, obj in self.objects.items():
            mesh_copy = obj.mesh.copy()
            
            # Store the object name in face_labels property for material mapping
            if not hasattr(mesh_copy.visual, 'face_labels'):
                face_labels = np.full(len(mesh_copy.faces), name)
                mesh_copy.visual.face_labels = face_labels
            
            meshes.append(mesh_copy)
            
        # Combine into a single mesh
        combined = trimesh.util.concatenate(meshes)
        
        return combined
        
    def export_scene(self, filename: str):
        """
        Export the combined scene to a file
        
        Args:
            filename: Output filename (extension determines format)
        """
        combined_mesh = self.get_combined_mesh()
        combined_mesh.export(filename)

    def analyze_scene_coverage(self, sensor_positions: List[np.ndarray]) -> float:
        """
        Analyze what percentage of the scene is visible from the given sensor positions
        
        Args:
            sensor_positions: List of (3,) arrays representing sensor positions
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        # This is a simplified coverage analysis using bounding box visibility
        # It's less accurate but more compatible with different trimesh versions
        
        # Get combined mesh
        combined_mesh = self.get_combined_mesh()
        if len(combined_mesh.vertices) == 0:
            return 0.0
            
        # Use a simple approximation with random points within the bounding box
        sample_count = 1000
        bounds = combined_mesh.bounds
        
        # Generate random points within the bounding box
        random_points = np.random.uniform(
            low=bounds[0], 
            high=bounds[1], 
            size=(sample_count, 3)
        )
        
        # Check which points are inside or near the mesh (approximation)
        # We'll use bounding box containment for simplicity
        visible_count = 0
        
        for point in random_points:
            # Only consider points that are relatively close to the mesh
            nearest_idx = np.argmin(np.sum((combined_mesh.vertices - point)**2, axis=1))
            nearest_point = combined_mesh.vertices[nearest_idx]
            
            if np.linalg.norm(point - nearest_point) > np.linalg.norm(bounds[1] - bounds[0]) * 0.1:
                # Point is too far from mesh, skip it
                continue
                
            # Check if this point is visible from any sensor
            for sensor_pos in sensor_positions:
                # Simple line-of-sight check
                # In a real implementation, you'd use ray casting
                # For now, just use distance-based visibility
                distance = np.linalg.norm(point - sensor_pos)
                
                # Consider point visible if within a certain distance of a sensor
                # and no major obstructions (simplified)
                if distance < np.linalg.norm(bounds[1] - bounds[0]) * 0.5:
                    visible_count += 1
                    break
        
        # Return approximate coverage percentage
        return visible_count / sample_count if sample_count > 0 else 0.0

    def suggest_sensor_positions(self, num_sensors: int = 3) -> List[np.ndarray]:
        """
        Suggest optimal sensor positions based on scene geometry
        
        Args:
            num_sensors: Number of sensors to place
            
        Returns:
            List of (3,) arrays with suggested sensor positions
        """
        # Get scene bounds
        combined_mesh = self.get_combined_mesh()
        bounds = combined_mesh.bounds
        
        # Get scene center and dimensions
        center = (bounds[0] + bounds[1]) / 2
        dimensions = bounds[1] - bounds[0]
        
        # For a very simple implementation, place sensors at corners looking at center
        suggested_positions = []
        
        # Create a larger bounding box for sensor placement
        sensor_distance = max(dimensions) * 1.5
        
        # Calculate positions for sensors
        if num_sensors >= 1:
            # Top view (above center)
            pos = center + np.array([0, 0, sensor_distance/2])
            suggested_positions.append(pos)
            
        if num_sensors >= 2:
            # Side view 1
            pos = center + np.array([sensor_distance/2, 0, sensor_distance/4])
            suggested_positions.append(pos)
            
        if num_sensors >= 3:
            # Side view 2
            pos = center + np.array([0, sensor_distance/2, sensor_distance/4])
            suggested_positions.append(pos)
            
        if num_sensors >= 4:
            # Side view 3
            pos = center + np.array([-sensor_distance/2, 0, sensor_distance/4])
            suggested_positions.append(pos)
            
        if num_sensors >= 5:
            # Side view 4
            pos = center + np.array([0, -sensor_distance/2, sensor_distance/4])
            suggested_positions.append(pos)
        
        # Add more positions as needed
        while len(suggested_positions) < num_sensors:
            # Add random positions around the perimeter
            angle = random.uniform(0, 2 * np.pi)
            pos = center + np.array([
                np.cos(angle) * sensor_distance/2,
                np.sin(angle) * sensor_distance/2,
                random.uniform(0, sensor_distance/2)
            ])
            suggested_positions.append(pos)
        
        return suggested_positions[:num_sensors]

