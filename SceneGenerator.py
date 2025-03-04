import numpy as np
import trimesh
from typing import List, Dict, Optional, Union, Tuple
from Material import Material
from dataclasses import dataclass

@dataclass
class SceneObject:
    """Represents an object in the scene with its mesh and transform"""
    mesh: trimesh.Trimesh
    position: np.ndarray
    rotation: np.ndarray  # Euler angles in radians
    scale: np.ndarray
    name: str
    material: Optional['Material'] = None

class SceneGenerator:
    def __init__(self):
        """Initialize an empty scene"""
        self.objects: Dict[str, SceneObject] = {}
        
    def check_collision(self, test_mesh: trimesh.Trimesh, 
                   max_overlap: float = 0.0) -> bool:
        """
        Check if test_mesh collides with any existing objects
        
        Args:
            test_mesh: Mesh to test for collisions
            max_overlap: Maximum allowed overlap distance
            
        Returns:
            True if collision detected beyond max_overlap, False otherwise
        """
        if not self.objects:
            return False
            
        # Create manager for proximity queries
        manager = trimesh.collision.CollisionManager()
        
        # Add existing objects to manager
        for obj in self.objects.values():
            manager.add_object(obj.name, obj.mesh)
            
        # Get minimum distance - in newer versions this returns just the distance
        separation_distance = manager.min_distance_single(test_mesh)
        
        # If objects are separated, separation_distance is positive
        # If objects overlap, separation_distance is negative
        # Allow overlap up to max_overlap
        return separation_distance < -max_overlap

    def add_object(self, 
                  mesh: Union[trimesh.Trimesh, str],
                  position: np.ndarray,
                  rotation: np.ndarray = np.zeros(3),
                  scale: np.ndarray = np.ones(3),
                  name: Optional[str] = None,
                  material: Optional['Material'] = None,
                  max_overlap: float = 0.0) -> str:
        """
        Add an object to the scene
        
        Args:
            mesh: Trimesh object or path to mesh file
            position: [x, y, z] position
            rotation: [rx, ry, rz] rotation in radians
            scale: [sx, sy, sz] scale factors
            name: Optional name for the object
            material: Optional Material object
            max_overlap: Maximum allowed overlap with existing objects
            
        Returns:
            Generated object ID
            
        Raises:
            ValueError: If object placement would cause excessive overlap
        """
        # Load mesh if path provided
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)
            
        # Generate unique name if none provided
        if name is None:
            name = f"object_{len(self.objects)}"
            
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, 3] = position
        
        # Apply rotation (Euler angles)
        rot_mat = trimesh.transformations.euler_matrix(
            rotation[0], rotation[1], rotation[2])
        transform[:3, :3] = rot_mat[:3, :3]
        
        # Apply scale
        transform[:3, :3] *= scale.reshape(3, 1)
        
        # Apply transform to mesh
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(transform)
        
        # Check for collisions
        if self.check_collision(transformed_mesh, max_overlap):
            raise ValueError(
                f"Object placement would cause overlap greater than {max_overlap}")
        
        # Create and store scene object
        scene_obj = SceneObject(
            mesh=transformed_mesh,
            position=position,
            rotation=rotation,
            scale=scale,
            name=name,
            material=material
        )
        self.objects[name] = scene_obj
        
        return name

    def create_random_scene(self,
                           object_paths: List[str],
                           num_objects: int,
                           bounds: Tuple[float, float, float] = (5.0, 5.0, 3.0),
                           min_distance: float = 1.0,
                           max_overlap: float = 0.0,
                           max_attempts: int = 100) -> None:
        """
        Create a random scene with multiple objects
        
        Args:
            object_paths: List of paths to mesh files
            num_objects: Number of objects to place
            bounds: (x, y, z) bounds for object placement
            min_distance: Minimum distance between object centers
            max_overlap: Maximum allowed overlap between meshes
            max_attempts: Maximum placement attempts per object
        """
        self.objects.clear()
        
        for i in range(num_objects):
            # Select random mesh
            mesh_path = np.random.choice(object_paths)
            mesh = trimesh.load(mesh_path)
            
            # Try placing the object with different random transforms
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                try:
                    # Generate random position
                    position = np.array([
                        np.random.uniform(-bounds[0]/2, bounds[0]/2),
                        np.random.uniform(-bounds[1]/2, bounds[1]/2),
                        np.random.uniform(0, bounds[2])
                    ])
                    
                    # Check center distance to existing objects
                    too_close = False
                    for obj in self.objects.values():
                        if np.linalg.norm(position - obj.position) < min_distance:
                            too_close = True
                            break
                            
                    if too_close:
                        attempts += 1
                        continue
                    
                    # Random rotation and scale
                    rotation = np.random.rand(3) * 2 * np.pi
                    scale = np.random.uniform(0.8, 1.2, 3)
                    
                    # Try to add object
                    self.add_object(
                        mesh=mesh,
                        position=position,
                        rotation=rotation,
                        scale=scale,
                        max_overlap=max_overlap
                    )
                    placed = True
                    
                except ValueError:
                    attempts += 1
            
            if not placed:
                print(f"Warning: Could not place object {i} after {max_attempts} attempts")
                break

    def get_combined_mesh(self) -> trimesh.Trimesh:
        combined_meshes = []
        face_labels = []  # New array to hold labels for each face
        for name, obj in self.objects.items():
            mesh = obj.mesh.copy()
            n_faces = mesh.faces.shape[0]
            # Create an array of labels for this object’s faces
            labels = np.full(n_faces, name)
            combined_meshes.append(mesh)
            face_labels.append(labels)
        
        # Concatenate meshes and label arrays
        combined = trimesh.util.concatenate(combined_meshes)
        # Store the per-face labels in a custom attribute (e.g. in visual)
        combined.visual.face_labels = np.concatenate(face_labels)
        return combined


    def export_scene(self, filepath: str) -> None:
        """Export the combined scene to a file"""
        combined_mesh = self.get_combined_mesh()
        combined_mesh.export(filepath)