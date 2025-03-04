import numpy as np
import trimesh
import pyvista as pv
from typing import Optional

class LIDARVisualizer:
    def __init__(self, mesh_path: str):
        """
        Initialize the visualizer with a mesh file
        
        Args:
            mesh_path: Path to the mesh file (.obj, .stl, etc.)
        """
        # Load the mesh using trimesh
        self.trimesh = trimesh.load(mesh_path)
        
        # Convert trimesh to PyVista mesh for visualization
        vertices = self.trimesh.vertices
        faces = np.pad(self.trimesh.faces, ((0, 0), (1, 0)), 
                      mode='constant', constant_values=3)
        self.mesh = pv.PolyData(vertices, faces)
        
        # Create the plotter
        self.plotter = pv.Plotter()
        
    def _add_coordinate_axes(self, length: float = 5.0):
        """
        Add prominent coordinate axes to the visualization
        
        Args:
            length: Length of each axis line
        """
        # Create axis lines
        origin = np.array([0, 0, 0])
        
        # Create axis lines with PyVista
        x_axis = pv.Line(origin, [length, 0, 0])
        y_axis = pv.Line(origin, [0, length, 0])
        z_axis = pv.Line(origin, [0, 0, length])
        
        # Add the axes with distinct colors and labels
        self.plotter.add_mesh(x_axis, color='red', line_width=1)
        self.plotter.add_mesh(y_axis, color='green', line_width=1)
        self.plotter.add_mesh(z_axis, color='blue', line_width=1)
        
        # Add axis labels at the ends
        self.plotter.add_point_labels(
            points=[[length, 0, 0], [0, length, 0], [0, 0, length]],
            labels=['X', 'Y', 'Z'],
            font_size=20,
            point_color='black',
            text_color='black'
        )
        
        # Add origin marker
        origin_sphere = pv.Sphere(radius=0.1, center=[0, 0, 0])
        self.plotter.add_mesh(origin_sphere, color='black', label='Origin (0,0,0)')
    
        
    def visualize_point_cloud(self, points: np.ndarray, normals: np.ndarray, 
                        scanner_position: np.ndarray,
                        reflectivity: Optional[np.ndarray] = None,
                        axis_length: float = 5.0):
        """
        Visualize the mesh and point cloud data
        
        Args:
            points: (N, 3) array of point cloud coordinates
            normals: (N, 3) array of point normals
            scanner_position: (3,) array of scanner position [x, y, z]
            reflectivity: Optional (N,) array of reflectivity values
            axis_length: Length of coordinate axes
        """
        try:
            # Clear the plotter first
            self.plotter.clear()
            
            # Add the prominent coordinate axes
            self._add_coordinate_axes(length=axis_length)
            
            # Create point cloud data
            point_cloud = pv.PolyData(points)
            
            # Add the original mesh with transparency
            self.plotter.add_mesh(self.mesh, 
                                style='surface',
                                color='gray',
                                opacity=0.3,
                                label='CAD Model')
            
            # Add the point cloud
            if reflectivity is not None:
                # Color points by reflectivity
                point_cloud['reflectivity'] = reflectivity
                self.plotter.add_points(point_cloud,
                                    render_points_as_spheres=True,
                                    point_size=5,
                                    scalars='reflectivity',
                                    scalar_bar_args={'title': 'Reflectivity'},
                                    label='LiDAR Points')
            else:
                # Use default coloring
                self.plotter.add_points(point_cloud,
                                    render_points_as_spheres=True,
                                    point_size=5,
                                    color='blue',
                                    label='LiDAR Points')
            
            # Add scanner position marker
            scanner_pos = scanner_position.copy()
            self.plotter.add_points(pv.PolyData(scanner_pos.reshape(1, 3)),
                                color='red',
                                point_size=20,
                                render_points_as_spheres=True,
                                label='Scanner Position')
            
            # Add legend
            self.plotter.add_legend()
            
            # Set camera to view all points
            self.plotter.camera_position = 'iso'  # Start with isometric view
            
            # Force an update
            self.plotter.update()
        
        except Exception as e:
            print(f"Error in visualize_point_cloud: {e}")
        
    def show(self):
        """Display the visualization"""
        self.plotter.show()
        
    def clear(self):
        """Clear the current visualization"""
        self.plotter.clear()

    def close(self):
        """Properly clean up resources"""
        try:
            self.plotter.clear()
            self.plotter.close()
        except Exception as e:
            print(f"Error closing LIDARVisualizer: {e}")

    