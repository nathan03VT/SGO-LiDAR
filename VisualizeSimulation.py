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
                    scanner_positions: np.ndarray,  # Changed to accept multiple positions
                    reflectivity: Optional[np.ndarray] = None,
                    sensor_colors: Optional[list] = None,
                    axis_length: float = 5.0):
        """
        Visualize the mesh and point cloud data with multiple sensors
        
        Args:
            points: (N, 3) array of point cloud coordinates
            normals: (N, 3) array of point normals
            scanner_positions: (M, 3) array of scanner positions or a list of (3,) arrays
            reflectivity: Optional (N,) array of reflectivity values
            sensor_colors: Optional list of colors for each sensor (will cycle if not enough)
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
            
            # Ensure scanner_positions is a numpy array with shape (M, 3)
            if isinstance(scanner_positions, list):
                scanner_positions = np.array(scanner_positions)
            
            # Handle single position case
            if scanner_positions.ndim == 1:
                scanner_positions = scanner_positions.reshape(1, 3)
            
            # Default colors if not provided
            if sensor_colors is None:
                # Use a color cycle for multiple sensors
                default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
                sensor_colors = [default_colors[i % len(default_colors)] for i in range(len(scanner_positions))]
            
            # Add scanner position markers for each sensor
            for i, pos in enumerate(scanner_positions):
                color = sensor_colors[i % len(sensor_colors)]
                sensor_label = f'Sensor {i+1}' if i > 0 else 'Primary Sensor'
                
                # Add sensor position marker
                self.plotter.add_points(pv.PolyData(pos.reshape(1, 3)),
                                    color=color,
                                    point_size=20,
                                    render_points_as_spheres=True,
                                    label=sensor_label)
                
                # Add text label
                self.plotter.add_point_labels(
                    points=[pos],
                    labels=[f"{i+1}"],
                    font_size=14,
                    point_color=color,
                    text_color='white',
                    bold=True,
                    italic=False,
                    shadow=True
                )
            
            # Add legend
            self.plotter.add_legend()
            
            # Set camera to view all points
            self.plotter.camera_position = 'iso'
            
            # Force an update
            self.plotter.update()
        
        except Exception as e:
            print(f"Error in visualize_point_cloud: {e}")

    def add_sensor_directions(self, positions: np.ndarray, directions: np.ndarray, 
                         arrow_length: float = 2.0, 
                         colors: Optional[list] = None):
        """
        Add arrows showing sensor directions to the visualization
        
        Args:
            positions: (M, 3) array of sensor positions
            directions: (M, 3) array of direction vectors
            arrow_length: Length of direction arrows
            colors: Optional list of colors for each sensor
        """
        try:
            # Ensure inputs are numpy arrays with correct shapes
            if isinstance(positions, list):
                positions = np.array(positions)
            if isinstance(directions, list):
                directions = np.array(directions)
                
            # Handle single position/direction case
            if positions.ndim == 1:
                positions = positions.reshape(1, 3)
            if directions.ndim == 1:
                directions = directions.reshape(1, 3)
                
            # Default colors if not provided
            if colors is None:
                default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
                colors = [default_colors[i % len(default_colors)] for i in range(len(positions))]
                
            # Add arrows for each sensor direction
            for i, (pos, dir_vec) in enumerate(zip(positions, directions)):
                # Normalize direction vector
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                
                # Calculate end point of arrow
                end_point = pos + (dir_vec * arrow_length)
                
                # Create arrow between points
                arrow = pv.Arrow(start=pos, direction=dir_vec, tip_length=0.2, 
                            tip_radius=0.1, shaft_radius=0.05, scale=arrow_length)
                
                # Add arrow to visualization
                self.plotter.add_mesh(arrow, color=colors[i % len(colors)])
                
        except Exception as e:
            print(f"Error adding sensor directions: {e}")
        
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

    