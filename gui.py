import sys
import numpy as np
import os
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QSpinBox, QHBoxLayout, QDoubleSpinBox,
    QTabWidget, QScrollArea, QGridLayout, QDialog, QComboBox,
    QListWidget, QListWidgetItem, QMessageBox, QStackedWidget,
    QGroupBox, QCheckBox, QSlider, QColorDialog, QLineEdit,
    QSplitter, QSizePolicy, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image, ImageDraw
import io
import trimesh
import pyvista as pv
from pyvistaqt import QtInteractor

from OpticalSimulation import OpticalSimulator
from Sensor import Sensor
from LidarSimulation import LiDARSimulator
from Material import Material
from SceneGenerator import SceneGenerator

# A helper function to convert a PIL image to QPixmap
def pil_to_qpixmap(pil_image):
    """Convert PIL Image to QPixmap"""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    img_data = pil_image.tobytes("raw", "RGB")
    qimage = QImage(img_data, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
    
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def apply_materials_to_mesh(mesh, material_dict=None):
    """
    Apply different materials to mesh components based on face labels
    
    Args:
        mesh: trimesh.Trimesh with face_labels 
        material_dict: Dictionary mapping object names to Material objects
        
    Returns:
        Modified mesh with materials applied
    """
    if not hasattr(mesh.visual, 'face_labels') or mesh.visual.face_labels is None:
        # No face labels, just use default material
        return mesh
        
    if material_dict is None or len(material_dict) == 0:
        # No materials provided
        return mesh
        
    # Create face colors array if it doesn't exist
    if not hasattr(mesh.visual, 'face_colors') or mesh.visual.face_colors is None:
        mesh.visual.face_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * 255
        
    # Apply materials based on face labels
    unique_labels = np.unique(mesh.visual.face_labels)
    
    for label in unique_labels:
        if label in material_dict:
            # Get faces with this label
            mask = mesh.visual.face_labels == label
            material = material_dict[label]
            
            # Apply material properties to these faces
            # For now, just set the color based on albedo
            if material is not None:
                rgba = np.zeros(4, dtype=np.uint8)
                # Convert [0-1] albedo to [0-255] RGBA
                rgba[:3] = (material.albedo * 255).astype(np.uint8)
                rgba[3] = 255  # Full opacity
                mesh.visual.face_colors[mask] = rgba
    
    return mesh

def convert_scene_objects_to_material_dict(scene_objects):
    """
    Convert a dictionary of SceneObjects to a material dictionary for apply_materials_to_mesh
    
    Args:
        scene_objects: Dictionary mapping names to SceneObject instances
        
    Returns:
        Dictionary mapping names to Material instances
    """
    material_dict = {}
    for name, obj in scene_objects.items():
        if obj.material is not None:
            material_dict[name] = obj.material
    return material_dict

class LidarScanThread(QThread):
    """Thread for running LiDAR scan in the background"""
    # Define signals for communication with main thread
    finished = pyqtSignal(list)  # Signal emitted when scan completes, passes results
    progress = pyqtSignal(int)   # Signal for progress updates (0-100)
    error = pyqtSignal(str)      # Signal for error messages
    
    def __init__(self, lidar_simulator, add_noise):
        super().__init__()
        self.lidar_simulator = lidar_simulator
        self.add_noise = add_noise
        
    def run(self):
        """Run the LiDAR scanning process"""
        try:
            # Get total number of rays that will be processed
            total_rays = 0
            for sensor in self.lidar_simulator.sensors:
                # Calculate ray count based on FOV and step angle
                h_rays = int(sensor.horizontal_fov / sensor.step_angle)
                v_rays = int(sensor.vertical_fov / sensor.step_angle)
                total_rays += h_rays * v_rays
            
            # Create a modified simulate method that reports progress
            def simulate_with_progress():
                results = []
                rays_processed = 0
                
                # Process each sensor
                for sensor_idx, sensor in enumerate(self.lidar_simulator.sensors):
                    # Get sensor position and direction
                    position, target_direction = sensor.get_sensor_transform()
                    
                    # Convert position to centered coordinate system
                    centered_position = self.lidar_simulator.world_to_centered(position)
                    
                    # Normalize target direction
                    target_direction = target_direction / np.linalg.norm(target_direction)
                    
                    # Compute ray directions for this sensor
                    ray_directions = self.lidar_simulator._compute_ray_directions(target_direction, sensor)
                    
                    # Initialize arrays to store results
                    all_points = []
                    all_normals = []
                    all_reflectivity = []
                    
                    # Process rays in batches to reduce memory usage
                    num_rays = len(ray_directions)
                    num_batches = int(np.ceil(num_rays / self.lidar_simulator.batch_size))
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * self.lidar_simulator.batch_size
                        end_idx = min((batch_idx + 1) * self.lidar_simulator.batch_size, num_rays)
                        
                        # Get ray directions for this batch
                        batch_directions = ray_directions[start_idx:end_idx]
                        
                        # Create ray origins for this batch
                        batch_origins = np.tile(centered_position, (len(batch_directions), 1))
                        
                        # Use trimesh's ray.intersects_location directly
                        locations, index_ray, index_tri = self.lidar_simulator.mesh.ray.intersects_location(
                            ray_origins=batch_origins,
                            ray_directions=batch_directions,
                            multiple_hits=False
                        )
                        
                        if len(locations) > 0:
                            # Get face normals for intersections
                            batch_normals = self.lidar_simulator.mesh.face_normals[index_tri]
                            
                            # Compute reflectivity based on angle between ray and normal
                            batch_ray_dirs = batch_directions[index_ray]
                            dot_products = np.abs(np.sum(batch_normals * (-batch_ray_dirs), axis=1))
                            batch_reflectivity = self.lidar_simulator.material.albedo * dot_products
                            batch_reflectivity = np.clip(batch_reflectivity, 0.0, 1.0)
                            
                            # Apply noise if requested
                            if self.add_noise:
                                # Distance-based noise model
                                distances = np.linalg.norm(locations - centered_position, axis=1)
                                
                                # Compute noise based on distance and reflectivity
                                noise = self.lidar_simulator._compute_noise(distances, batch_reflectivity)
                                
                                # Apply noise in ray direction
                                locations += batch_ray_dirs * noise.reshape(-1, 1)
                                
                                # Add noise to reflectivity
                                batch_reflectivity += np.random.normal(0, 0.05, len(batch_reflectivity))
                                batch_reflectivity = np.clip(batch_reflectivity, 0.0, 1.0)
                            
                            # Add results to the collection
                            all_points.append(locations)
                            all_normals.append(batch_normals)
                            all_reflectivity.append(batch_reflectivity)
                        
                        # Update progress
                        rays_processed += len(batch_directions)
                        progress_percent = int(rays_processed / total_rays * 100)
                        self.progress.emit(progress_percent)
                        
                        # Small sleep to allow GUI to update
                        self.msleep(1)
                    
                    # Combine results for this sensor
                    if all_points:
                        points = np.vstack(all_points)
                        normals = np.vstack(all_normals)
                        reflectivity = np.concatenate(all_reflectivity)
                        
                        # Convert points back to world coordinates
                        world_points = self.lidar_simulator.centered_to_world(points)
                        
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
            
            # Run the simulation with progress tracking
            results = simulate_with_progress()
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            self.error.emit(f"Error in LiDAR scan: {str(e)}\n{traceback.format_exc()}")


class SceneGeneratorDialog(QDialog):
    """Dialog for creating and editing scenes with multiple objects"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scene Generator")

        # Use a minimum size instead of fixed size
        self.setMinimumSize(600, 400)  # Allow flexibility for resizing
        
        self.scene_generator = SceneGenerator()
        self.model_paths = []
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components using layouts instead of fixed geometries"""
        # Main vertical layout for the dialog
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # QSplitter to manage resizable panels
        splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(splitter)

        # Left panel: Object management
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_panel.setLayout(left_layout)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(left_panel)

        # Model directory selection
        dir_layout = QHBoxLayout()
        self.model_dir_label = QLabel("Model Directory: None")
        dir_layout.addWidget(self.model_dir_label)

        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self.browse_model_directory)
        dir_layout.addWidget(browse_dir_btn)
        left_layout.addLayout(dir_layout)

        # Object list
        left_layout.addWidget(QLabel("Objects in Scene:"))
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.object_selected)
        left_layout.addWidget(self.object_list)

        # Object control buttons
        btn_layout = QHBoxLayout()
        self.add_obj_btn = QPushButton("Add Object")
        self.add_obj_btn.clicked.connect(self.add_object)
        self.add_obj_btn.setEnabled(False)
        btn_layout.addWidget(self.add_obj_btn)

        self.edit_obj_btn = QPushButton("Edit Object")
        self.edit_obj_btn.clicked.connect(self.edit_object)
        self.edit_obj_btn.setEnabled(False)
        btn_layout.addWidget(self.edit_obj_btn)

        self.remove_obj_btn = QPushButton("Remove Object")
        self.remove_obj_btn.clicked.connect(self.remove_object)
        self.remove_obj_btn.setEnabled(False)
        btn_layout.addWidget(self.remove_obj_btn)

        left_layout.addLayout(btn_layout)

        # Random scene generation section
        random_group = QGroupBox("Random Scene Generation")
        random_layout = QVBoxLayout()
        random_group.setLayout(random_layout)

        # Number of objects
        num_obj_layout = QHBoxLayout()
        num_obj_layout.addWidget(QLabel("Number of Objects:"))
        self.num_objects_spin = QSpinBox()
        self.num_objects_spin.setRange(1, 50)
        self.num_objects_spin.setValue(5)
        num_obj_layout.addWidget(self.num_objects_spin)
        random_layout.addLayout(num_obj_layout)

        # Bounds settings
        bounds_layout = QHBoxLayout()
        bounds_layout.addWidget(QLabel("Bounds (X, Y, Z):"))

        self.bounds_x_spin = QDoubleSpinBox()
        self.bounds_x_spin.setRange(1, 100)
        self.bounds_x_spin.setValue(10)
        bounds_layout.addWidget(self.bounds_x_spin)

        self.bounds_y_spin = QDoubleSpinBox()
        self.bounds_y_spin.setRange(1, 100)
        self.bounds_y_spin.setValue(10)
        bounds_layout.addWidget(self.bounds_y_spin)

        self.bounds_z_spin = QDoubleSpinBox()
        self.bounds_z_spin.setRange(1, 100)
        self.bounds_z_spin.setValue(5)
        bounds_layout.addWidget(self.bounds_z_spin)

        random_layout.addLayout(bounds_layout)

        # Min distance and overlap settings
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Min Distance:"))
        self.min_distance_spin = QDoubleSpinBox()
        self.min_distance_spin.setRange(0, 10)
        self.min_distance_spin.setValue(1.0)
        self.min_distance_spin.setSingleStep(0.1)
        spacing_layout.addWidget(self.min_distance_spin)

        spacing_layout.addWidget(QLabel("Max Overlap:"))
        self.max_overlap_spin = QDoubleSpinBox()
        self.max_overlap_spin.setRange(0, 1)
        self.max_overlap_spin.setValue(0.0)
        self.max_overlap_spin.setSingleStep(0.01)
        spacing_layout.addWidget(self.max_overlap_spin)

        random_layout.addLayout(spacing_layout)

        # Generate button
        self.generate_btn = QPushButton("Generate Random Scene")
        self.generate_btn.clicked.connect(self.generate_random_scene)
        self.generate_btn.setEnabled(False)
        random_layout.addWidget(self.generate_btn)

        left_layout.addWidget(random_group)

        # Right panel: 3D preview
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_panel.setLayout(right_layout)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(right_panel)

        right_layout.addWidget(QLabel("Scene Preview:"))
        self.vtk_widget = QtInteractor(right_panel)
        right_layout.addWidget(self.vtk_widget.interactor)

        # Export scene controls
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Scene")
        self.export_btn.clicked.connect(self.export_scene)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        self.use_scene_btn = QPushButton("Use Scene in Simulation")
        self.use_scene_btn.clicked.connect(self.use_scene)
        self.use_scene_btn.setEnabled(False)
        export_layout.addWidget(self.use_scene_btn)

        main_layout.addLayout(export_layout)

        # Initialize the 3D view
        self.setup_3d_view()

        
    def setup_3d_view(self):
        """Set up the 3D preview with coordinate axes"""
        self.vtk_widget.clear()
        self.vtk_widget.set_background("white")
        
        # Add coordinate axes
        origin = np.array([0, 0, 0])
        axis_length = 5.0
        axes = {
            'X': ([0, 0, 0], [axis_length, 0, 0], 'red'),
            'Y': ([0, 0, 0], [0, axis_length, 0], 'green'),
            'Z': ([0, 0, 0], [0, 0, axis_length], 'blue')
        }
        for label, (start, end, color) in axes.items():
            line = pv.Line(np.array(start), np.array(end))
            self.vtk_widget.add_mesh(line, color=color, line_width=3)
            self.vtk_widget.add_point_labels(
                points=[np.array(end)],
                labels=[label],
                font_size=14,
                text_color=color
            )
            
        # Set initial camera position to isometric view
        self.vtk_widget.view_isometric()
        self.vtk_widget.reset_camera()
        
    def browse_model_directory(self):
        """Open directory dialog to select model files directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", "", options=QFileDialog.DontUseNativeDialog
        )
        
        if dir_path:
            # Store the directory path
            self.model_dir_label.setText(f"Model Directory: {dir_path}")
            
            # Find all model files in the directory
            self.model_paths = []
            valid_extensions = ['.obj', '.stl', '.ply']
            
            for file in os.listdir(dir_path):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    self.model_paths.append(os.path.join(dir_path, file))
            
            # Enable or disable controls based on found models
            has_models = len(self.model_paths) > 0
            self.add_obj_btn.setEnabled(has_models)
            self.generate_btn.setEnabled(has_models)
            
            if has_models:
                print(f"Found {len(self.model_paths)} model files")
            else:
                QMessageBox.warning(self, "No Models", 
                                  f"No supported model files found in {dir_path}.\n"
                                  "Please select a directory with .obj, .stl, or .ply files.")
    
    def add_object(self):
        """Add a new object to the scene"""
        if not self.model_paths:
            QMessageBox.warning(self, "No Models", "Please select a directory with model files first.")
            return
            
        # Show dialog to select a model file
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Object")
        dialog_layout = QVBoxLayout()
        dialog.setLayout(dialog_layout)
        
        # Model selection
        dialog_layout.addWidget(QLabel("Select Model:"))
        model_combo = QComboBox()
        for path in self.model_paths:
            model_combo.addItem(os.path.basename(path), path)
        dialog_layout.addWidget(model_combo)
        
        # Position controls
        dialog_layout.addWidget(QLabel("Position (X, Y, Z):"))
        pos_layout = QHBoxLayout()
        pos_x = QDoubleSpinBox()
        pos_x.setRange(-100, 100)
        pos_x.setValue(0)
        pos_layout.addWidget(pos_x)
        
        pos_y = QDoubleSpinBox()
        pos_y.setRange(-100, 100)
        pos_y.setValue(0)
        pos_layout.addWidget(pos_y)
        
        pos_z = QDoubleSpinBox()
        pos_z.setRange(-100, 100)
        pos_z.setValue(0)
        pos_layout.addWidget(pos_z)
        dialog_layout.addLayout(pos_layout)
        
        # Rotation controls
        dialog_layout.addWidget(QLabel("Rotation (degrees):"))
        rot_layout = QHBoxLayout()
        rot_x = QDoubleSpinBox()
        rot_x.setRange(0, 360)
        rot_layout.addWidget(rot_x)
        
        rot_y = QDoubleSpinBox()
        rot_y.setRange(0, 360)
        rot_layout.addWidget(rot_y)
        
        rot_z = QDoubleSpinBox()
        rot_z.setRange(0, 360)
        rot_layout.addWidget(rot_z)
        dialog_layout.addLayout(rot_layout)
        
        # Scale controls
        dialog_layout.addWidget(QLabel("Scale:"))
        scale_layout = QHBoxLayout()
        scale_x = QDoubleSpinBox()
        scale_x.setRange(0.1, 10)
        scale_x.setValue(1.0)
        scale_layout.addWidget(scale_x)
        
        scale_y = QDoubleSpinBox()
        scale_y.setRange(0.1, 10)
        scale_y.setValue(1.0)
        scale_layout.addWidget(scale_y)
        
        scale_z = QDoubleSpinBox()
        scale_z.setRange(0.1, 10)
        scale_z.setValue(1.0)
        scale_layout.addWidget(scale_z)
        dialog_layout.addLayout(scale_layout)
        
        # Material controls (simplistic for now)
        dialog_layout.addWidget(QLabel("Material Albedo (RGB):"))
        color_btn = QPushButton("Choose Color")
        color_value = [200, 200, 200]  # Default color
        
        def choose_color():
            nonlocal color_value
            color = QColorDialog.getColor(QColor(*color_value), dialog, "Choose Object Color")
            if color.isValid():
                color_value = [color.red(), color.green(), color.blue()]
                color_btn.setStyleSheet(f"background-color: rgb({color_value[0]}, {color_value[1]}, {color_value[2]})")
                
        color_btn.clicked.connect(choose_color)
        color_btn.setStyleSheet(f"background-color: rgb({color_value[0]}, {color_value[1]}, {color_value[2]})")
        dialog_layout.addWidget(color_btn)
        
        # Name control
        dialog_layout.addWidget(QLabel("Object Name:"))
        name_edit = QLineEdit(f"Object_{len(self.scene_generator.objects)}")
        dialog_layout.addWidget(name_edit)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Object")
        add_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        dialog_layout.addLayout(btn_layout)
        
        if dialog.exec_():
            # Get the selected values
            model_path = model_combo.currentData()
            position = np.array([pos_x.value(), pos_y.value(), pos_z.value()])
            # Convert degrees to radians for rotation
            rotation = np.array([
                np.radians(rot_x.value()),
                np.radians(rot_y.value()),
                np.radians(rot_z.value())
            ])
            scale = np.array([scale_x.value(), scale_y.value(), scale_z.value()])
            name = name_edit.text()
            
            # Create material
            # Convert RGB to grayscale using standard luminance formula
            grayscale = (0.299 * color_value[0] + 0.587 * color_value[1] + 0.114 * color_value[2]) / 255.0
            material = Material(albedo=grayscale)
            
            try:
                # Add the object to the scene
                self.scene_generator.add_object(
                    mesh=model_path,
                    position=position,
                    rotation=rotation,
                    scale=scale,
                    name=name,
                    material=material,
                    max_overlap=self.max_overlap_spin.value()
                )
                
                # Update UI
                self.object_list.addItem(name)
                self.update_scene_preview()
                self.export_btn.setEnabled(True)
                self.use_scene_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add object: {str(e)}")
    
    def edit_object(self):
        """Edit the selected object"""
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            return
            
        name = selected_items[0].text()
        if name not in self.scene_generator.objects:
            return
            
        obj = self.scene_generator.objects[name]
        
        # Show dialog to edit object properties
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Object: {name}")
        dialog_layout = QVBoxLayout()
        dialog.setLayout(dialog_layout)
        
        # Position controls
        dialog_layout.addWidget(QLabel("Position (X, Y, Z):"))
        pos_layout = QHBoxLayout()
        pos_x = QDoubleSpinBox()
        pos_x.setRange(-100, 100)
        pos_x.setValue(obj.position[0])
        pos_layout.addWidget(pos_x)
        
        pos_y = QDoubleSpinBox()
        pos_y.setRange(-100, 100)
        pos_y.setValue(obj.position[1])
        pos_layout.addWidget(pos_y)
        
        pos_z = QDoubleSpinBox()
        pos_z.setRange(-100, 100)
        pos_z.setValue(obj.position[2])
        pos_layout.addWidget(pos_z)
        dialog_layout.addLayout(pos_layout)
        
        # Rotation controls (convert radians to degrees for display)
        dialog_layout.addWidget(QLabel("Rotation (degrees):"))
        rot_layout = QHBoxLayout()
        rot_x = QDoubleSpinBox()
        rot_x.setRange(0, 360)
        rot_x.setValue(np.degrees(obj.rotation[0]) % 360)
        rot_layout.addWidget(rot_x)
        
        rot_y = QDoubleSpinBox()
        rot_y.setRange(0, 360)
        rot_y.setValue(np.degrees(obj.rotation[1]) % 360)
        rot_layout.addWidget(rot_y)
        
        rot_z = QDoubleSpinBox()
        rot_z.setRange(0, 360)
        rot_z.setValue(np.degrees(obj.rotation[2]) % 360)
        rot_layout.addWidget(rot_z)
        dialog_layout.addLayout(rot_layout)
        
        # Scale controls
        dialog_layout.addWidget(QLabel("Scale:"))
        scale_layout = QHBoxLayout()
        scale_x = QDoubleSpinBox()
        scale_x.setRange(0.1, 10)
        scale_x.setValue(obj.scale[0])
        scale_layout.addWidget(scale_x)
        
        scale_y = QDoubleSpinBox()
        scale_y.setRange(0.1, 10)
        scale_y.setValue(obj.scale[1])
        scale_layout.addWidget(scale_y)
        
        scale_z = QDoubleSpinBox()
        scale_z.setRange(0.1, 10)
        scale_z.setValue(obj.scale[2])
        scale_layout.addWidget(scale_z)
        dialog_layout.addLayout(scale_layout)
        
        # Material controls
        if obj.material:
            dialog_layout.addWidget(QLabel("Material Albedo (RGB):"))
            color_btn = QPushButton("Choose Color")
            
            # For display purposes, convert the single albedo value to grayscale RGB
            gray_value = int(obj.material.albedo * 255)
            color_value = [gray_value, gray_value, gray_value]
            
            def choose_color():
                nonlocal color_value
                color = QColorDialog.getColor(QColor(*color_value), dialog, "Choose Object Color")
                if color.isValid():
                    color_value = [color.red(), color.green(), color.blue()]
                    color_btn.setStyleSheet(f"background-color: rgb({color_value[0]}, {color_value[1]}, {color_value[2]})")
                    
            color_btn.clicked.connect(choose_color)
            color_btn.setStyleSheet(f"background-color: rgb({color_value[0]}, {color_value[1]}, {color_value[2]})")
            dialog_layout.addWidget(color_btn)
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        dialog_layout.addLayout(btn_layout)
        
        if dialog.exec_():
            # Update object properties
            position = np.array([pos_x.value(), pos_y.value(), pos_z.value()])
            # Convert degrees to radians
            rotation = np.array([
                np.radians(rot_x.value()),
                np.radians(rot_y.value()),
                np.radians(rot_z.value())
            ])
            scale = np.array([scale_x.value(), scale_y.value(), scale_z.value()])
            
            # Check if position changed and handle potential collisions
            if not np.array_equal(obj.position, position):
                # Remove object temporarily
                temp_obj = obj
                del self.scene_generator.objects[name]
                
                try:
                    # Try to add it back with new position
                    self.scene_generator.add_object(
                        mesh=temp_obj.mesh,
                        position=position,
                        rotation=rotation,
                        scale=scale,
                        name=name,
                        material=temp_obj.material,
                        max_overlap=self.max_overlap_spin.value()
                    )
                except ValueError as e:
                    # If there's a collision, put it back in its original position
                    self.scene_generator.objects[name] = temp_obj
                    QMessageBox.warning(self, "Position Conflict", 
                                    f"Couldn't move object: {str(e)}\nObject kept at original position.")
                    self.update_scene_preview()
                    return
            else:
                # Just update rotation and scale if position didn't change
                obj.rotation = rotation
                obj.scale = scale
                
                # Apply transform to mesh
                transform = np.eye(4)
                transform[:3, 3] = position
                
                # Apply rotation (Euler angles)
                rot_mat = trimesh.transformations.euler_matrix(
                    rotation[0], rotation[1], rotation[2])
                transform[:3, :3] = rot_mat[:3, :3]
                
                # Apply scale
                transform[:3, :3] *= scale.reshape(3, 1)
                
                # Reset mesh and apply new transform
                model_origin = obj.mesh.copy()
                model_origin.apply_transform(transform)
                obj.mesh = model_origin
            
            # Update material if available
            if obj.material and 'color_value' in locals():
                # Convert RGB to grayscale using standard luminance formula
                grayscale = (0.299 * color_value[0] + 0.587 * color_value[1] + 0.114 * color_value[2]) / 255.0
                obj.material.albedo = grayscale
            
            # Update the view
            self.update_scene_preview()
    
    def remove_object(self):
        """Remove the selected object from the scene"""
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            return
            
        name = selected_items[0].text()
        if name in self.scene_generator.objects:
            # Remove from scene generator
            del self.scene_generator.objects[name]
            
            # Remove from list widget
            row = self.object_list.row(selected_items[0])
            self.object_list.takeItem(row)
            
            # Update controls
            self.edit_obj_btn.setEnabled(False)
            self.remove_obj_btn.setEnabled(False)
            
            # Update the view
            self.update_scene_preview()
            
            # Update export button state
            self.export_btn.setEnabled(len(self.scene_generator.objects) > 0)
            self.use_scene_btn.setEnabled(len(self.scene_generator.objects) > 0)
    
    def object_selected(self, item):
        """Handle object selection in the list"""
        self.edit_obj_btn.setEnabled(True)
        self.remove_obj_btn.setEnabled(True)
    
    def generate_random_scene(self):
        """Generate a random scene with the specified parameters"""
        if not self.model_paths:
            QMessageBox.warning(self, "No Models", "Please select a directory with model files first.")
            return
            
        try:
            # Clear existing scene
            self.scene_generator.objects.clear()
            self.object_list.clear()
            
            # Get parameters
            num_objects = self.num_objects_spin.value()
            bounds = (
                self.bounds_x_spin.value(),
                self.bounds_y_spin.value(),
                self.bounds_z_spin.value()
            )
            min_distance = self.min_distance_spin.value()
            max_overlap = self.max_overlap_spin.value()
            
            # Generate the scene
            self.scene_generator.create_random_scene(
                object_paths=self.model_paths,
                num_objects=num_objects,
                bounds=bounds,
                min_distance=min_distance,
                max_overlap=max_overlap
            )
            
            # Update the list
            for name in self.scene_generator.objects.keys():
                self.object_list.addItem(name)
                
            # Update the view
            self.update_scene_preview()
            
            # Enable export buttons
            self.export_btn.setEnabled(len(self.scene_generator.objects) > 0)
            self.use_scene_btn.setEnabled(len(self.scene_generator.objects) > 0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate scene: {str(e)}")
    
    def update_scene_preview(self):
        """Update the 3D preview with the current scene"""
        # Clear the current view
        self.vtk_widget.clear()
        
        # Re-add coordinate axes
        self.setup_3d_view()
        
        # Get the combined mesh
        try:
            if self.scene_generator.objects:
                combined_mesh = self.scene_generator.get_combined_mesh()
                
                # Convert to PyVista mesh
                vertices = combined_mesh.vertices
                faces = np.hstack([np.full((combined_mesh.faces.shape[0], 1), 3), 
                                 combined_mesh.faces]).flatten()
                pv_mesh = pv.PolyData(vertices, faces)
                
                # Add to the view
                self.vtk_widget.add_mesh(
                    pv_mesh, 
                    color="lightgray", 
                    show_edges=True,
                    opacity=0.8
                )
                
                # Reset camera to see the whole scene
                self.vtk_widget.reset_camera()
        except Exception as e:
            print(f"Error updating scene preview: {e}")
    
    def export_scene(self):
        """Export the current scene to a file"""
        if not self.scene_generator.objects:
            QMessageBox.warning(self, "Empty Scene", "No objects to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Scene", "", "3D Models (*.obj *.stl *.ply)",
            options=QFileDialog.DontUseNativeDialog
        )
        
        if filename:
            try:
                # Export the combined scene
                self.scene_generator.export_scene(filename)
                QMessageBox.information(self, "Export Successful", f"Scene exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export scene: {str(e)}")
    
    def use_scene(self):
        """Use the current scene in the simulation"""
        if not self.scene_generator.objects:
            QMessageBox.warning(self, "Empty Scene", "No objects to use.")
            return
            
        try:
            # Create a temporary file for the combined mesh
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, "combined_scene.obj")
            
            # Export the scene to the temporary file
            self.scene_generator.export_scene(temp_file)
            
            # Accept the dialog and return the path
            self.temp_file_path = temp_file
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate combined scene: {str(e)}")
            
    def get_scene_path(self):
        """Return the path to the generated scene file"""
        if hasattr(self, 'temp_file_path'):
            return self.temp_file_path
        return None
        
    def closeEvent(self, event):
        """Clean up resources when closing"""
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.clear()
        event.accept()

class SensorDialog(QDialog):
    """Dialog for creating a new sensor or editing an existing one"""
    def __init__(self, parent=None, sensor=None):
        super().__init__(parent)
        self.setWindowTitle("Sensor Configuration")
        self.resize(400, 500)
        self.sensor = sensor
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Sensor name
        layout.addWidget(QLabel("Sensor Name:"))
        self.name_edit = QLineEdit()
        if self.sensor and hasattr(self.sensor, 'name'):
            self.name_edit.setText(self.sensor.name)
        else:
            self.name_edit.setText("New Sensor")
        layout.addWidget(self.name_edit)
        
        # Sensor Position
        layout.addWidget(QLabel("Sensor Position (x,y,z):"))
        self.pos_spinboxes = []
        pos_values = self.sensor.position if self.sensor else [0.0, 0.0, 0.0]
        
        for axis, val in zip(['X', 'Y', 'Z'], pos_values):
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setValue(val)
            spin.setSingleStep(1.0)
            layout.addWidget(QLabel(f"Position {axis}:"))
            layout.addWidget(spin)
            self.pos_spinboxes.append(spin)
        
        # Target controls
        layout.addWidget(QLabel("Target Control:"))
        self.target_mode_combo = QComboBox()
        self.target_mode_combo.addItems(["Target Direction", "Look At Point"])
        self.target_mode_combo.currentIndexChanged.connect(self.update_target_mode)
        layout.addWidget(self.target_mode_combo)
        
        # Target Direction or Look At Point
        self.target_stack = QStackedWidget()
        layout.addWidget(self.target_stack)
        
        # Direction panel
        dir_panel = QWidget()
        dir_layout = QVBoxLayout()
        dir_panel.setLayout(dir_layout)
        dir_layout.addWidget(QLabel("Direction Vector (x,y,z):"))
        
        self.dir_spinboxes = []
        dir_values = self.sensor.target_direction if self.sensor else [0.0, 0.0, -1.0]
        
        for axis, val in zip(['X', 'Y', 'Z'], dir_values):
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setDecimals(3)
            spin.setValue(val)
            spin.setSingleStep(0.1)
            dir_layout.addWidget(QLabel(f"Direction {axis}:"))
            dir_layout.addWidget(spin)
            self.dir_spinboxes.append(spin)
            
        # Look-at panel
        lookat_panel = QWidget()
        lookat_layout = QVBoxLayout()
        lookat_panel.setLayout(lookat_layout)
        lookat_layout.addWidget(QLabel("Look At Point (x,y,z):"))
        
        self.lookat_spinboxes = []
        lookat_values = [0.0, 0.0, 0.0]
        
        for axis, val in zip(['X', 'Y', 'Z'], lookat_values):
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setDecimals(3)
            spin.setValue(val)
            spin.setSingleStep(1.0)
            lookat_layout.addWidget(QLabel(f"Look At {axis}:"))
            lookat_layout.addWidget(spin)
            self.lookat_spinboxes.append(spin)
            
        self.target_stack.addWidget(dir_panel)
        self.target_stack.addWidget(lookat_panel)
        
        # FOV controls
        layout.addWidget(QLabel("Horizontal FOV (degrees):"))
        self.hfov_spin = QDoubleSpinBox()
        self.hfov_spin.setRange(1, 180)
        self.hfov_spin.setValue(90.0 if not self.sensor else self.sensor.horizontal_fov)
        self.hfov_spin.setSingleStep(1)
        layout.addWidget(self.hfov_spin)

        layout.addWidget(QLabel("Vertical FOV (degrees):"))
        self.vfov_spin = QDoubleSpinBox()
        self.vfov_spin.setRange(1, 180)
        self.vfov_spin.setValue(60.0 if not self.sensor else self.sensor.vertical_fov)
        self.vfov_spin.setSingleStep(1)
        layout.addWidget(self.vfov_spin)
        
        # LiDAR specific parameters
        layout.addWidget(QLabel("LiDAR Step Angle (degrees):"))
        self.step_angle_spin = QDoubleSpinBox()
        self.step_angle_spin.setRange(0.01, 10.0)
        self.step_angle_spin.setValue(0.1 if not self.sensor else getattr(self.sensor, 'step_angle', 0.1))
        self.step_angle_spin.setSingleStep(0.1)
        layout.addWidget(self.step_angle_spin)
        
        layout.addWidget(QLabel("Min Distance (meters):"))
        self.min_distance_spin = QDoubleSpinBox()
        self.min_distance_spin.setRange(0.001, 10.0)
        self.min_distance_spin.setValue(0.1 if not self.sensor else getattr(self.sensor, 'min_distance', 0.1))
        self.min_distance_spin.setSingleStep(0.1)
        layout.addWidget(self.min_distance_spin)
        
        layout.addWidget(QLabel("Max Distance (meters):"))
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(1.0, 1000.0)
        self.max_distance_spin.setValue(130.0 if not self.sensor else getattr(self.sensor, 'max_distance', 130.0))
        self.max_distance_spin.setSingleStep(10.0)
        layout.addWidget(self.max_distance_spin)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Set initial target mode
        if self.sensor and np.allclose(self.sensor.target_direction, np.array([0.0, 0.0, -1.0])):
            # If using default direction, assume we want to look at origin
            self.target_mode_combo.setCurrentIndex(1)
            self.target_stack.setCurrentIndex(1)
        else:
            self.target_mode_combo.setCurrentIndex(0)
            self.target_stack.setCurrentIndex(0)
            
    def update_target_mode(self, index):
        self.target_stack.setCurrentIndex(index)
    
    def get_sensor_data(self):
        position = np.array([spin.value() for spin in self.pos_spinboxes])
        
        # Get direction based on selected mode
        if self.target_mode_combo.currentIndex() == 0:
            # Using explicit direction
            target_direction = np.array([spin.value() for spin in self.dir_spinboxes])
            
            # Normalize direction if not zero
            dir_norm = np.linalg.norm(target_direction)
            if dir_norm > 1e-10:
                target_direction = target_direction / dir_norm
            else:
                # Default to looking down if zero direction provided
                target_direction = np.array([0.0, 0.0, -1.0])
        else:
            # Using look-at point
            look_at = np.array([spin.value() for spin in self.lookat_spinboxes])
            
            # Calculate direction from position to look-at point
            target_direction = look_at - position
            
            # Normalize if not zero
            dir_norm = np.linalg.norm(target_direction)
            if dir_norm > 1e-10:
                target_direction = target_direction / dir_norm
            else:
                # Default to looking down if sensor position = look at point
                target_direction = np.array([0.0, 0.0, -1.0])
        
        sensor_data = {
            'name': self.name_edit.text(),
            'position': position,
            'target_direction': target_direction,
            'horizontal_fov': self.hfov_spin.value(),
            'vertical_fov': self.vfov_spin.value(),
            'step_angle': self.step_angle_spin.value(),
            'min_distance': self.min_distance_spin.value(),
            'max_distance': self.max_distance_spin.value()
        }
        return sensor_data

class MaterialDialog(QDialog):
    """Dialog for configuring material properties for LiDAR scanning"""
    def __init__(self, parent=None, material=None):
        super().__init__(parent)
        self.setWindowTitle("Material Properties")
        self.resize(400, 300)
        self.material = material or Material()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Albedo slider (changed from color picker to slider)
        albedo_layout = QHBoxLayout()
        albedo_layout.addWidget(QLabel("Albedo:"))
        self.albedo_slider = QSlider(Qt.Horizontal)
        self.albedo_slider.setRange(0, 100)
        self.albedo_slider.setValue(int(self.material.albedo * 100))
        albedo_layout.addWidget(self.albedo_slider)
        self.albedo_label = QLabel(f"{self.material.albedo:.2f}")
        albedo_layout.addWidget(self.albedo_label)
        self.albedo_slider.valueChanged.connect(self.update_albedo_label)
        layout.addLayout(albedo_layout)
        
        # Metallic slider
        metallic_layout = QHBoxLayout()
        metallic_layout.addWidget(QLabel("Metallic:"))
        self.metallic_slider = QSlider(Qt.Horizontal)
        self.metallic_slider.setRange(0, 100)
        self.metallic_slider.setValue(int(self.material.metallic * 100))
        metallic_layout.addWidget(self.metallic_slider)
        self.metallic_label = QLabel(f"{self.material.metallic:.2f}")
        metallic_layout.addWidget(self.metallic_label)
        self.metallic_slider.valueChanged.connect(self.update_metallic_label)
        layout.addLayout(metallic_layout)
        
        # Roughness slider
        roughness_layout = QHBoxLayout()
        roughness_layout.addWidget(QLabel("Roughness:"))
        self.roughness_slider = QSlider(Qt.Horizontal)
        self.roughness_slider.setRange(0, 100)
        self.roughness_slider.setValue(int(self.material.roughness * 100))
        roughness_layout.addWidget(self.roughness_slider)
        self.roughness_label = QLabel(f"{self.material.roughness:.2f}")
        roughness_layout.addWidget(self.roughness_label)
        self.roughness_slider.valueChanged.connect(self.update_roughness_label)
        layout.addLayout(roughness_layout)
        
        # Ambient slider
        ambient_layout = QHBoxLayout()
        ambient_layout.addWidget(QLabel("Ambient:"))
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(int(self.material.ambient * 100))
        ambient_layout.addWidget(self.ambient_slider)
        self.ambient_label = QLabel(f"{self.material.ambient:.2f}")
        ambient_layout.addWidget(self.ambient_label)
        self.ambient_slider.valueChanged.connect(self.update_ambient_label)
        layout.addLayout(ambient_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def update_albedo_label(self, value):
        albedo = value / 100.0
        self.albedo_label.setText(f"{albedo:.2f}")
    
    def update_metallic_label(self, value):
        metallic = value / 100.0
        self.metallic_label.setText(f"{metallic:.2f}")
    
    def update_roughness_label(self, value):
        roughness = value / 100.0
        self.roughness_label.setText(f"{roughness:.2f}")
    
    def update_ambient_label(self, value):
        ambient = value / 100.0
        self.ambient_label.setText(f"{ambient:.2f}")
    
    def get_material(self):
        return Material(
            albedo=self.albedo_slider.value() / 100.0,
            metallic=self.metallic_slider.value() / 100.0,
            roughness=self.roughness_slider.value() / 100.0,
            ambient=self.ambient_slider.value() / 100.0
        )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical and LiDAR Imagery Scan GUI")
        self.resize(1200, 800)

        self.mesh = None
        self.sensors = []
        self.active_sensor_index = -1
        
        # Texture-related attributes
        self.texture_image = None
        self.texture_path = None

        self.optical_simulator = None
        self.lidar_simulator = None
        self.lidar_material = Material()
        
        # LiDAR results storage
        self.lidar_results = None
        self.lidar_file_path = None
        
        # Scene generation flag
        self.is_generated_scene = False

        self.init_ui()
        self.add_scene_generator_controls()

    def init_ui(self):
        # Main container widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel: Controls
        control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        control_panel.setLayout(self.control_layout)
        control_panel.setFixedWidth(300)

        # Button to load CAD model
        self.load_model_btn = QPushButton("Load CAD Model")
        self.load_model_btn.clicked.connect(self.load_cad_model)
        self.control_layout.addWidget(self.load_model_btn)
        
        # Add texture controls in a grouped box
        texture_group = QGroupBox("Texture Controls")
        texture_layout = QVBoxLayout()
        texture_group.setLayout(texture_layout)
        
        # Button to load texture image
        self.load_texture_btn = QPushButton("Load Texture Image")
        self.load_texture_btn.clicked.connect(self.load_texture)
        texture_layout.addWidget(self.load_texture_btn)
        
        # Button to generate procedural texture
        self.gen_texture_btn = QPushButton("Generate Procedural Texture")
        self.gen_texture_btn.clicked.connect(self.generate_texture)
        texture_layout.addWidget(self.gen_texture_btn)
        
        # Texture repeat control
        texture_repeat_layout = QHBoxLayout()
        texture_repeat_layout.addWidget(QLabel("Texture Repeats:"))
        self.texture_repeat_spin = QSpinBox()
        self.texture_repeat_spin.setRange(1, 10)
        self.texture_repeat_spin.setValue(4)
        texture_repeat_layout.addWidget(self.texture_repeat_spin)
        texture_layout.addLayout(texture_repeat_layout)
        
        # Button to apply texture
        self.apply_texture_btn = QPushButton("Apply Texture")
        self.apply_texture_btn.clicked.connect(self.apply_texture)
        self.apply_texture_btn.setEnabled(False)
        texture_layout.addWidget(self.apply_texture_btn)
        
        self.control_layout.addWidget(texture_group)

        # Sensor management grouped box
        sensor_group = QGroupBox("Sensor Management")
        sensor_layout = QVBoxLayout()
        sensor_group.setLayout(sensor_layout)
        
        # List of sensors
        self.sensor_list = QListWidget()
        self.sensor_list.setSelectionMode(QListWidget.SingleSelection)
        self.sensor_list.itemClicked.connect(self.sensor_selected)
        sensor_layout.addWidget(self.sensor_list)
        
        # Sensor control buttons
        sensor_btn_layout = QHBoxLayout()
        
        self.add_sensor_btn = QPushButton("Add Sensor")
        self.add_sensor_btn.clicked.connect(self.add_sensor)
        sensor_btn_layout.addWidget(self.add_sensor_btn)
        
        self.edit_sensor_btn = QPushButton("Edit Sensor")
        self.edit_sensor_btn.clicked.connect(self.edit_sensor)
        self.edit_sensor_btn.setEnabled(False)
        sensor_btn_layout.addWidget(self.edit_sensor_btn)
        
        self.remove_sensor_btn = QPushButton("Remove Sensor")
        self.remove_sensor_btn.clicked.connect(self.remove_sensor)
        self.remove_sensor_btn.setEnabled(False)
        sensor_btn_layout.addWidget(self.remove_sensor_btn)
        
        sensor_layout.addLayout(sensor_btn_layout)
        
        # Button to update 3D preview
        self.update_preview_btn = QPushButton("Update 3D Preview")
        self.update_preview_btn.clicked.connect(self.update_3d_preview)
        sensor_layout.addWidget(self.update_preview_btn)
        
        self.control_layout.addWidget(sensor_group)
        
        # Add LiDAR controls in a grouped box
        lidar_group = QGroupBox("LiDAR Simulation")
        lidar_layout = QVBoxLayout()
        lidar_group.setLayout(lidar_layout)
        
        # Configure material for LiDAR
        self.config_material_btn = QPushButton("Configure Material Properties")
        self.config_material_btn.clicked.connect(self.configure_material)
        lidar_layout.addWidget(self.config_material_btn)
        
        # Add noise checkbox
        self.add_noise_checkbox = QCheckBox("Add Noise to LiDAR Scan")
        self.add_noise_checkbox.setChecked(True)
        lidar_layout.addWidget(self.add_noise_checkbox)
        
        # Button to run LiDAR scan
        self.run_lidar_btn = QPushButton("Run LiDAR Scan")
        self.run_lidar_btn.clicked.connect(self.run_lidar_scan)
        lidar_layout.addWidget(self.run_lidar_btn)
        
        # Button to export LiDAR results
        self.export_lidar_btn = QPushButton("Export LiDAR Point Cloud")
        self.export_lidar_btn.clicked.connect(self.export_lidar_results)
        self.export_lidar_btn.setEnabled(False)
        lidar_layout.addWidget(self.export_lidar_btn)
        
        self.control_layout.addWidget(lidar_group)
        
        # Button to run optical scan
        self.run_scan_btn = QPushButton("Run Optical Scan")
        self.run_scan_btn.clicked.connect(self.run_optical_scan)
        self.control_layout.addWidget(self.run_scan_btn)

        self.control_layout.addStretch()

        main_layout.addWidget(control_panel)

        # Right panel: Tab widget with 3D Preview and Optical Images tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # 3D Preview tab
        self.preview_tab = QWidget()
        preview_layout = QVBoxLayout()
        self.preview_tab.setLayout(preview_layout)
        self.vtk_widget = QtInteractor(self.preview_tab)
        preview_layout.addWidget(self.vtk_widget.interactor)
        self.tabs.addTab(self.preview_tab, "3D Preview")

        # LiDAR Results tab
        self.lidar_tab = QWidget()
        lidar_layout = QVBoxLayout()
        self.lidar_tab.setLayout(lidar_layout)
        self.lidar_vtk_widget = QtInteractor(self.lidar_tab)
        lidar_layout.addWidget(self.lidar_vtk_widget.interactor)
        self.tabs.addTab(self.lidar_tab, "LiDAR Results")

        # Optical Images tab (with scroll area and grid layout)
        self.optical_tab = QWidget()
        optical_layout = QVBoxLayout()
        self.optical_tab.setLayout(optical_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_grid_widget = QWidget()
        self.image_grid = QGridLayout()
        self.image_grid_widget.setLayout(self.image_grid)
        self.scroll_area.setWidget(self.image_grid_widget)
        optical_layout.addWidget(self.scroll_area)
        self.tabs.addTab(self.optical_tab, "Optical Images")

        # Initialize 3D view (add coordinate axes)
        self.setup_3d_view()
        self.setup_lidar_view()
        
    def add_scene_generator_controls(self):
        """Add scene generator button to the control panel"""
        # Create a grouped box for scene generation
        scene_group = QGroupBox("Scene Generation")
        scene_layout = QVBoxLayout()
        scene_group.setLayout(scene_layout)
        
        # Button to open scene generator
        self.scene_gen_btn = QPushButton("Open Scene Generator")
        self.scene_gen_btn.clicked.connect(self.open_scene_generator)
        scene_layout.addWidget(self.scene_gen_btn)
        
        # Add to control layout (insert before the load model button)
        self.control_layout.insertWidget(0, scene_group)
    
    def open_scene_generator(self):
        """Open the scene generator dialog"""
        dialog = SceneGeneratorDialog(self)
        if dialog.exec_():
            # Get the path to the generated scene
            scene_path = dialog.get_scene_path()
            if scene_path and os.path.exists(scene_path):
                # Load the generated scene
                try:
                    self.mesh = trimesh.load(scene_path, force='mesh')
                    self.lidar_file_path = scene_path
                    self.update_3d_model()
                    QMessageBox.information(self, "Scene Loaded", "Generated scene loaded successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load generated scene: {str(e)}")
    
    def setup_3d_view(self):
        # Clear the plotter and add a white background
        self.vtk_widget.clear()
        self.vtk_widget.set_background("white")

        # Add coordinate axes (drawn as lines with labels)
        origin = np.array([0, 0, 0])
        axis_length = 20.0
        axes = {
            'X': ([0, 0, 0], [axis_length, 0, 0], 'red'),
            'Y': ([0, 0, 0], [0, axis_length, 0], 'green'),
            'Z': ([0, 0, 0], [0, 0, axis_length], 'blue')
        }
        for label, (start, end, color) in axes.items():
            line = pv.Line(np.array(start), np.array(end))
            self.vtk_widget.add_mesh(line, color=color, line_width=3)
            self.vtk_widget.add_point_labels(
                points=[np.array(end)],
                labels=[label],
                font_size=14,
                text_color=color
            )

        self.vtk_widget.reset_camera()
    
    def setup_lidar_view(self):
        # Clear the plotter and add a black background for LiDAR visualization
        self.lidar_vtk_widget.clear()
        self.lidar_vtk_widget.set_background("black")

        # Add coordinate axes (drawn as lines with labels)
        origin = np.array([0, 0, 0])
        axis_length = 20.0
        axes = {
            'X': ([0, 0, 0], [axis_length, 0, 0], 'red'),
            'Y': ([0, 0, 0], [0, axis_length, 0], 'green'),
            'Z': ([0, 0, 0], [0, 0, axis_length], 'blue')
        }
        for label, (start, end, color) in axes.items():
            line = pv.Line(np.array(start), np.array(end))
            self.lidar_vtk_widget.add_mesh(line, color=color, line_width=3)
            self.lidar_vtk_widget.add_point_labels(
                points=[np.array(end)],
                labels=[label],
                font_size=14,
                text_color=color
            )

        self.lidar_vtk_widget.reset_camera()

    def load_cad_model(self):
        """Open a file dialog to select a CAD model file (e.g. .obj, .stl)"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CAD Model", "", 
            "CAD Files (*.obj *.stl *.ply)", 
            options=QFileDialog.DontUseNativeDialog
        )        
        if filename:
            try:
                # Check if it's a file from our scene generator (has face_labels)
                self.mesh = trimesh.load(filename, force='mesh')
                self.lidar_file_path = filename
                
                # Check if this is a scene-generated mesh with materials
                if hasattr(self.mesh.visual, 'face_labels'):
                    self.is_generated_scene = True
                    # Store the face_labels for later use in simulation
                    self.mesh_face_labels = self.mesh.visual.face_labels
                    print(f"Loaded scene-generated mesh with {len(np.unique(self.mesh_face_labels))} unique objects")
                else:
                    self.is_generated_scene = False
                    
                self.update_3d_model()
                
                # Update UI
                self.apply_texture_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load CAD model: {str(e)}")
                print("Failed to load CAD model:", e)
    
    def create_optical_simulator(self):
        """Create an optical simulator if one doesn't exist yet"""
        if self.mesh is None:
            # Create a simple placeholder mesh if no mesh is loaded
            self.optical_simulator = OpticalSimulator(
                mesh=trimesh.creation.box(),
                sensors=[],
                base_image_width=1024,
                background_color=[0, 0, 0],
                default_color=[150, 150, 150],  # Use gray as default
                use_default_texture=False  # Don't use brick texture by default
            )
        else:
            self.optical_simulator = OpticalSimulator(
                mesh=self.mesh,
                sensors=[],
                base_image_width=1024,
                background_color=[0, 0, 0],
                default_color=[150, 150, 150],  # Use gray as default 
                use_default_texture=False  # Don't use brick texture by default
            )
            
    def create_lidar_simulator(self):
        """Create a LiDAR simulator using current mesh and materials"""
        if self.mesh is None:
            QMessageBox.warning(self, "No Model", "Please load a CAD model first.")
            return None
            
        if not self.sensors:
            QMessageBox.warning(self, "No Sensors", "Please add at least one sensor first.")
            return None
            
        # For scene-generated meshes, extract materials for each object
        if hasattr(self, 'is_generated_scene') and self.is_generated_scene:
            # Try to load the scene generator and get materials if possible
            try:
                # Create material dictionary by object name
                material_dict = {}
                
                # Check if we have face labels
                if hasattr(self.mesh.visual, 'face_labels'):
                    # Use default material but adjusted for each labeled part
                    unique_labels = np.unique(self.mesh.visual.face_labels)
                    
                    for label in unique_labels:
                        # Get approximate color for this label from face colors 
                        if hasattr(self.mesh.visual, 'face_colors'):
                            mask = self.mesh.visual.face_labels == label
                            if np.any(mask):
                                # Get the first face color for this label
                                face_color = self.mesh.visual.face_colors[mask][0]
                                # Create a material with albedo from this color - convert RGB to grayscale
                                rgb = face_color[:3]
                                grayscale = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0
                                mat = Material(
                                    albedo=grayscale,
                                    metallic=self.lidar_material.metallic,
                                    roughness=self.lidar_material.roughness,
                                    ambient=self.lidar_material.ambient
                                )
                                material_dict[label] = mat
                        else:
                            # If no face colors, use default material with varied albedo
                            color_val = hash(str(label)) % 200 + 55  # Range 55-255
                            albedo = color_val / 255.0  # Single grayscale value
                            mat = Material(
                                albedo=albedo,
                                metallic=self.lidar_material.metallic,
                                roughness=self.lidar_material.roughness,
                                ambient=self.lidar_material.ambient
                            )
                            material_dict[label] = mat
                    
                    # Create simulator with the composite material
                    return LiDARSimulator(
                        mesh=self.mesh,
                        sensors=self.sensors,
                        material=self.lidar_material,
                        material_dict=material_dict  # Pass materials by object
                    )
            except Exception as e:
                print(f"Error setting up materials for generated scene: {e}")
                # Fall back to standard simulator with single material
        
        # Standard case - use global material for the whole mesh
        return LiDARSimulator(
            mesh=self.mesh,
            sensors=self.sensors,
            material=self.lidar_material
        )
        
    def load_texture(self):
        """Open a file dialog to select a texture image file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Texture Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)",
            options=QFileDialog.DontUseNativeDialog
        )
        
        if filename:
            if not hasattr(self, 'optical_simulator') or self.optical_simulator is None:
                self.create_optical_simulator()
            
            texture_img = self.optical_simulator.load_texture_from_file(filename)
            if texture_img:
                # Store the texture path for reference
                self.texture_path = filename
                self.texture_image = texture_img
                self.apply_texture_btn.setEnabled(True)

    def generate_texture(self):
        """Generate a procedural texture based on predefined patterns"""
        # Create a dialog for pattern selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Texture Pattern")
        dialog_layout = QVBoxLayout()
        
        # Pattern selector
        pattern_combo = QComboBox()
        pattern_combo.addItems(["Brick", "Checkerboard", "Gradient", "Noise"])
        dialog_layout.addWidget(QLabel("Select Pattern:"))
        dialog_layout.addWidget(pattern_combo)
        
        # Size selector
        size_combo = QComboBox()
        size_combo.addItems(["256x256", "512x512", "1024x1024"])
        dialog_layout.addWidget(QLabel("Texture Size:"))
        dialog_layout.addWidget(size_combo)
        
        # Create button
        create_btn = QPushButton("Create Texture")
        dialog_layout.addWidget(create_btn)
        
        dialog.setLayout(dialog_layout)
        
        # Function to handle texture creation
        def create_procedural_texture():
            pattern = pattern_combo.currentText()
            size_text = size_combo.currentText()
            width, height = map(int, size_text.split('x'))
            
            # Create optical simulator if needed
            if not hasattr(self, 'optical_simulator') or self.optical_simulator is None:
                self.create_optical_simulator()
            
            # Generate texture based on selected pattern
            if pattern == "Brick":
                self.texture_image = self.optical_simulator.generate_default_texture(width, height)
            elif pattern == "Checkerboard":
                self.texture_image = self.optical_simulator.generate_checkerboard_texture(width, height, squares=8)
            elif pattern == "Gradient":
                self.texture_image = self.optical_simulator.generate_gradient_texture(width, height)
            elif pattern == "Noise":
                self.texture_image = self.optical_simulator.generate_noise_texture(width, height)
            
            print(f"Generated {pattern} texture, size: {width}x{height}")
            self.texture_path = f"generated_{pattern.lower()}.png"
            self.apply_texture_btn.setEnabled(True)
            dialog.accept()
        
        create_btn.clicked.connect(create_procedural_texture)
        dialog.exec_()

    def apply_texture(self):
        """Apply the current texture to the loaded mesh."""
        if self.mesh is None:
            print("Please load a CAD model first.")
            return

        if self.texture_image is None:
            print("Please load or generate a texture first.")
            return

        try:
            # Ensure UV mapping exists
            if not hasattr(self.mesh.visual, 'uv') or self.mesh.visual.uv is None:
                print("Generating UV mapping for mesh...")
                bounds = self.mesh.bounds
                min_xy = bounds[0][:2]
                max_xy = bounds[1][:2]
                size_xy = max_xy - min_xy
                size_xy[size_xy == 0] = 1.0
                
                # Generate UV coordinates from vertex positions
                uvs = (self.mesh.vertices[:, :2] - min_xy) / size_xy
                
                # Scale UVs based on repeat value
                repeats = self.texture_repeat_spin.value()
                uvs = uvs * repeats
                
                # Create new texture visual with UVs
                self.mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)

            if self.texture_image.mode != "RGB":
                self.texture_image = self.texture_image.convert("RGB")
                
            self.mesh.visual.image = self.texture_image
            texture_array = np.array(self.texture_image)
        
            vertices = self.mesh.vertices
            faces = np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten()
            pv_mesh = pv.PolyData(vertices, faces)
            uvs = np.asarray(self.mesh.visual.uv, dtype=np.float32)
            pv_mesh.active_texture_coordinates = uvs
            
            texture = pv.numpy_to_texture(texture_array)

            # Clear the 3D view and add the textured mesh
            self.vtk_widget.clear()
            self.setup_3d_view()
            self.vtk_widget.add_mesh(pv_mesh, texture=texture, show_edges=False)
            self.vtk_widget.reset_camera()

            print("Texture successfully applied.")

        except Exception as e:
            print(f"Failed to apply texture: {e}")
            import traceback
            traceback.print_exc()
    
    def add_sensor(self):
        """Add a new sensor via dialog"""
        dialog = SensorDialog(self)
        if dialog.exec_():
            sensor_data = dialog.get_sensor_data()
            
            sensor = Sensor(
                position=sensor_data['position'],
                target_direction=sensor_data['target_direction'],
                horizontal_fov=sensor_data['horizontal_fov'],
                vertical_fov=sensor_data['vertical_fov']
            )
            # Add name attribute (not in original Sensor class)
            sensor.name = sensor_data['name']
            
            # Set LiDAR-specific properties
            sensor.step_angle = sensor_data['step_angle']
            sensor.min_distance = sensor_data['min_distance']
            sensor.max_distance = sensor_data['max_distance']
            
            self.sensors.append(sensor)
            
            self.sensor_list.addItem(sensor_data['name'])
            
            # Select the newly added sensor
            self.sensor_list.setCurrentRow(len(self.sensors) - 1)
            self.active_sensor_index = len(self.sensors) - 1
            self.edit_sensor_btn.setEnabled(True)
            self.remove_sensor_btn.setEnabled(True)
            
            self.update_3d_preview()

    def edit_sensor(self):
        if self.active_sensor_index >= 0 and self.active_sensor_index < len(self.sensors):
            sensor = self.sensors[self.active_sensor_index]
            dialog = SensorDialog(self, sensor)
            
            if dialog.exec_():
                sensor_data = dialog.get_sensor_data()
                
                # Update the sensor with all properties
                sensor.position = sensor_data['position']
                sensor.target_direction = sensor_data['target_direction']
                sensor.horizontal_fov = sensor_data['horizontal_fov']
                sensor.vertical_fov = sensor_data['vertical_fov']
                sensor.name = sensor_data['name']
                
                # Update additional LiDAR properties
                sensor.step_angle = sensor_data['step_angle']
                sensor.min_distance = sensor_data['min_distance']
                sensor.max_distance = sensor_data['max_distance']
                
                self.sensor_list.item(self.active_sensor_index).setText(sensor_data['name'])
                self.update_3d_preview()

    def remove_sensor(self):
        if self.active_sensor_index >= 0 and self.active_sensor_index < len(self.sensors):
            # Remove from list
            self.sensors.pop(self.active_sensor_index)
            
            # Remove from list widget
            self.sensor_list.takeItem(self.active_sensor_index)
            
            # Update selection
            if len(self.sensors) > 0:
                self.sensor_list.setCurrentRow(0)
                self.active_sensor_index = 0
            else:
                self.active_sensor_index = -1
                self.edit_sensor_btn.setEnabled(False)
                self.remove_sensor_btn.setEnabled(False)
            
            # Update 3D preview
            self.update_3d_preview()

    def sensor_selected(self, item):
        """Handle sensor selection from the list"""
        self.active_sensor_index = self.sensor_list.row(item)
        self.edit_sensor_btn.setEnabled(True)
        self.remove_sensor_btn.setEnabled(True)
        self.update_3d_preview()

    def handle_scan_finished(self, results):
        # Store the results
        self.lidar_results = results
        
        # Re-enable controls
        self.run_lidar_btn.setEnabled(True)
        self.export_lidar_btn.setEnabled(True)
        
        # Show success message
        QMessageBox.information(self, "LiDAR Scan", "LiDAR scan completed successfully.")
        
        # Update the visualization
        self.visualize_lidar_results()
        
        # Switch to the LiDAR Results tab
        self.tabs.setCurrentWidget(self.lidar_tab)

    def handle_scan_error(self, error_message):
        # Re-enable controls
        self.run_lidar_btn.setEnabled(True)
        
        # Show error message
        QMessageBox.critical(self, "LiDAR Scan Error", error_message)
    
    def run_lidar_scan(self):
        if self.mesh is None:
            QMessageBox.warning(self, "No CAD Model", "Please load a CAD model first.")
            return
        
        if not self.sensors:
            QMessageBox.warning(self, "No Sensors", "Please add at least one sensor first.")
            return

        try:
            # Instantiate the simulator
            self.lidar_simulator = self.create_lidar_simulator()
            if not self.lidar_simulator:
                return
            
            # Create the progress dialog
            self.progress_dialog = QProgressDialog("Running LiDAR scan...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setWindowTitle("LiDAR Scan Progress")
            self.progress_dialog.setMinimumDuration(0)  # Show immediately
            self.progress_dialog.setValue(0)
            self.progress_dialog.setAutoClose(True)
            self.progress_dialog.setAutoReset(True)
            
            # Create and configure the worker thread
            self.scan_thread = LidarScanThread(
                lidar_simulator=self.lidar_simulator,
                add_noise=self.add_noise_checkbox.isChecked()
            )
            
            # Connect signals
            self.scan_thread.progress.connect(self.progress_dialog.setValue)
            self.scan_thread.finished.connect(self.handle_scan_finished)
            self.scan_thread.error.connect(self.handle_scan_error)
            self.progress_dialog.canceled.connect(self.scan_thread.terminate)
            
            # Disable controls during scan
            self.run_lidar_btn.setEnabled(False)
            self.export_lidar_btn.setEnabled(False)
            
            # Start the thread
            self.scan_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "LiDAR Scan Error", f"Error starting LiDAR scan: {str(e)}")
                
    def visualize_lidar_results(self):
        """Visualize LiDAR simulation results in the LiDAR tab"""
        if self.lidar_results is None:
            return
            
        # Clear the current view
        self.lidar_vtk_widget.clear()
        
        # Make sure we have a valid context before proceeding
        try:
            self.setup_lidar_view()
            
            # If a model is loaded, add it with transparency
            if self.mesh is not None:
                # Convert trimesh mesh to PyVista PolyData
                vertices = self.mesh.vertices
                faces = np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten()
                pv_mesh = pv.PolyData(vertices, faces)
                
                # Add the mesh with transparency
                self.lidar_vtk_widget.add_mesh(pv_mesh, color="lightgray", opacity=0.2, show_edges=False)
            
            # Process and add all point clouds
            colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
            
            # Combine all point clouds for a single visualization
            all_points = np.vstack([points for points, _, _ in self.lidar_results])
            all_normals = np.vstack([normals for _, normals, _ in self.lidar_results])
            all_reflectivity = np.concatenate([reflectivity for _, _, reflectivity in self.lidar_results])
            
            # Create a point cloud from all points
            point_cloud = pv.PolyData(all_points)
            point_cloud['normals'] = all_normals
            point_cloud['reflectivity'] = all_reflectivity
            
            # Add the point cloud with coloring by reflectivity
            self.lidar_vtk_widget.add_mesh(
                point_cloud, 
                scalars='reflectivity',
                point_size=5,
                render_points_as_spheres=True,
                scalar_bar_args={'title': 'Reflectivity', 'position_x': 0.05, 'position_y': 0.05}
            )
            
            # Add sensor positions as markers
            for i, sensor in enumerate(self.sensors):
                color = colors[i % len(colors)]
                sensor_sphere = pv.Sphere(radius=0.2, center=sensor.position)
                self.lidar_vtk_widget.add_mesh(
                    sensor_sphere, 
                    color=color, 
                    label=f"Sensor {i+1}: {getattr(sensor, 'name', f'Sensor {i+1}')}"
                )
                
                # Add direction arrow
                arrow = pv.Arrow(
                    start=sensor.position,
                    direction=sensor.target_direction,
                    tip_length=0.25,
                    tip_radius=0.1,
                    shaft_radius=0.05
                )
                self.lidar_vtk_widget.add_mesh(arrow, color=color)
                
            # Add a legend
            self.lidar_vtk_widget.add_legend()
            
            # Reset camera to show all points
            self.lidar_vtk_widget.reset_camera()
            
            # Force an update to the VTK widget
            self.lidar_vtk_widget.update()
            
        except Exception as e:
            print(f"Error visualizing LiDAR results: {e}")
        
    def export_lidar_results(self):
        """Export LiDAR point cloud results to file"""
        if self.lidar_results is None:
            QMessageBox.warning(self, "No Results", "No LiDAR scan results to export.")
            return
            
        # Open save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Point Cloud", 
            "", 
            "Point Cloud Files (*.ply *.xyz);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        
        if not filename:
            return
            
        try:
            # Combine all points, normals, and reflectivity
            all_points = np.vstack([points for points, _, _ in self.lidar_results])
            all_normals = np.vstack([normals for _, normals, _ in self.lidar_results])
            all_reflectivity = np.concatenate([reflectivity for _, _, reflectivity in self.lidar_results])
            
            # Create a point cloud
            point_cloud = pv.PolyData(all_points)
            point_cloud['normals'] = all_normals
            point_cloud['reflectivity'] = all_reflectivity
            
            # Save to file
            point_cloud.save(filename)
            QMessageBox.information(self, "Export Successful", f"Point cloud saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export point cloud: {str(e)}")
            print(f"Export error: {e}")

    def configure_material(self):
        """Configure material properties for LiDAR simulation"""
        dialog = MaterialDialog(self, self.lidar_material)
        if dialog.exec_():
            self.lidar_material = dialog.get_material()
            print("Updated material properties:")
            print(f"  Albedo: {self.lidar_material.albedo}")
            print(f"  Metallic: {self.lidar_material.metallic}")
            print(f"  Roughness: {self.lidar_material.roughness}")
            print(f"  Ambient: {self.lidar_material.ambient}")

    def update_3d_model(self):
        if self.mesh is None:
            return
        # Remove previous model (clear and re-add coordinate axes)
        self.setup_3d_view()
        
        vertices = self.mesh.vertices
        faces = np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten()
        pv_mesh = pv.PolyData(vertices, faces)

        # Check if the mesh has texture and apply it correctly
        has_texture = hasattr(self.mesh.visual, 'material') and self.mesh.visual.material is not None
        has_texture_image = hasattr(self.mesh.visual, 'image') and self.mesh.visual.image is not None

        if hasattr(self.mesh.visual, 'uv') and self.mesh.visual.uv is not None:
            # Ensure the UVs are a numpy array of type float32
            uv_array = np.asarray(self.mesh.visual.uv, dtype=np.float32)
            pv_mesh.point_data["tcoords"] = uv_array
            # Set the active texture coordinates directly to the numpy array
            pv_mesh.active_texture_coordinates = uv_array

            if has_texture_image:
                texture_img = np.array(self.mesh.visual.image)
                texture = pv.numpy_to_texture(texture_img)
                self.vtk_widget.add_mesh(pv_mesh, texture=texture, show_edges=False, label="CAD Model")
            else:
                self.vtk_widget.add_mesh(pv_mesh, color="white", opacity=0.8, show_edges=False, label="CAD Model")
        else:
            self.vtk_widget.add_mesh(pv_mesh, color="lightgray", opacity=0.5, show_edges=True, label="CAD Model")
            
        self.vtk_widget.reset_camera()

    def update_3d_preview(self):
        """Update the 3D preview with all sensors"""
        if self.mesh is None:
            return
            
        # Store the current camera position before clearing
        try:
            camera_position = self.vtk_widget.camera_position
        except:
            camera_position = None
        
        # Clear the plotter
        self.vtk_widget.clear()
        
        try:
            # Re-add the coordinate axes
            self.setup_3d_view()
            
            # If a model is loaded, re-add it to the view
            if self.mesh is not None:
                # Calculate the model's bounding box dimensions for proper scaling
                bounding_box = self.mesh.bounds
                diagonal = np.linalg.norm(bounding_box[1] - bounding_box[0])
                marker_radius = diagonal * 0.015 * 0.8
                arrow_length = diagonal * 0.015 * 1.5
                
                # Convert trimesh mesh to PyVista PolyData
                vertices = self.mesh.vertices
                faces = np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten()
                pv_mesh = pv.PolyData(vertices, faces)
                
                # Check if mesh has texture coordinates
                has_uv = hasattr(self.mesh.visual, 'uv') and self.mesh.visual.uv is not None
                # Check if mesh has texture image
                has_texture_image = hasattr(self.mesh.visual, 'image') and self.mesh.visual.image is not None
                
                if has_uv and has_texture_image:
                    # This is a textured mesh
                    try:
                        # Ensure UV coordinates are properly assigned
                        uvs = np.asarray(self.mesh.visual.uv, dtype=np.float32)
                        pv_mesh.active_texture_coordinates = uvs
                        
                        # Convert the texture image to a PyVista texture
                        texture_array = np.array(self.mesh.visual.image)
                        texture = pv.numpy_to_texture(texture_array)
                        
                        # Add the textured mesh
                        self.vtk_widget.add_mesh(pv_mesh, texture=texture, show_edges=False, label="Textured CAD Model")
                    except Exception as e:
                        print(f"Error applying texture: {e}")
                        # Fallback to untextured mesh
                        self.vtk_widget.add_mesh(pv_mesh, color="lightgray", opacity=0.5, show_edges=True, label="CAD Model")
                else:
                    # Add untextured mesh
                    self.vtk_widget.add_mesh(pv_mesh, color="lightgray", opacity=0.5, show_edges=True, label="CAD Model")
            
            # Add all sensors to the view with different colors
            colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
            
            for i, sensor in enumerate(self.sensors):
                # Determine color (cycle through colors if more sensors than colors)
                color = colors[i % len(colors)]
                
                # Add sensor position marker (sphere)
                sensor_sphere = pv.Sphere(radius=marker_radius, center=sensor.position)
                self.vtk_widget.add_mesh(sensor_sphere, color=color, label=f"Sensor {i+1}: {sensor.name}")
                
                # Add direction arrow with proper length adjustment
                # Make sure target_direction is not zero
                if np.linalg.norm(sensor.target_direction) > 1e-10:
                    # Create an arrow with a consistent length regardless of direction magnitude
                    arrow = pv.Arrow(
                        start=sensor.position,
                        direction=sensor.target_direction,
                        tip_length=marker_radius * 0.3,
                        tip_radius=marker_radius * 0.15, 
                        shaft_radius=marker_radius * 0.05, 
                        scale=arrow_length
                    )
                    self.vtk_widget.add_mesh(arrow, color=color)
                
            
            # Add a legend if there are sensors
            if self.sensors:
                self.vtk_widget.add_legend()
            
            # Try to restore the camera position if it was saved
            if camera_position:
                try:
                    self.vtk_widget.camera_position = camera_position
                except:
                    self.vtk_widget.reset_camera()
            else:
                self.vtk_widget.reset_camera()
                
            # Force an update to the VTK widget
            self.vtk_widget.update()
            
        except Exception as e:
            print(f"Error updating 3D preview: {e}")
            
    def run_optical_scan(self):
        """Run optical scan with all configured sensors"""
        if self.mesh is None:
            QMessageBox.warning(self, "No Model", "Please load a CAD model first.")
            return
            
        if not self.sensors:
            QMessageBox.warning(self, "No Sensors", "Please add at least one sensor first.")
            return

        # Create a copy of the current mesh to ensure texture is preserved
        mesh_copy = self.mesh.copy()
        
        # Ensure the mesh copy has the proper visual attributes (texture, uv)
        if hasattr(self.mesh.visual, 'image') and self.mesh.visual.image is not None:
            if not hasattr(mesh_copy.visual, 'image') or mesh_copy.visual.image is None:
                mesh_copy.visual.image = self.mesh.visual.image
                
        if hasattr(self.mesh.visual, 'uv') and self.mesh.visual.uv is not None:
            if not hasattr(mesh_copy.visual, 'uv') or mesh_copy.visual.uv is None:
                mesh_copy.visual.uv = self.mesh.visual.uv

        # Create an OpticalSimulator instance using the current mesh and all sensors
        if hasattr(self, 'optical_simulator') and self.optical_simulator is not None:
            # Update existing optical simulator with current mesh and sensors
            self.optical_simulator.mesh = mesh_copy
            self.optical_simulator.sensors = self.sensors
            
            # Make sure texture is properly transferred
            if hasattr(mesh_copy.visual, 'image') and mesh_copy.visual.image is not None:
                self.optical_simulator.texture_image = mesh_copy.visual.image
        else:
            # Create new optical simulator
            self.optical_simulator = OpticalSimulator(
                mesh=mesh_copy,
                sensors=self.sensors,
                base_image_width=1024,
                background_color=[0, 0, 0],
                default_color=[150, 150, 150],
                use_default_texture=False
            )
            
            # Make sure texture is properly transferred
            if hasattr(mesh_copy.visual, 'image') and mesh_copy.visual.image is not None:
                self.optical_simulator.texture_image = mesh_copy.visual.image
        
        try:
            print("Starting optical rendering...")
            # Debug mesh texture state
            if hasattr(self.optical_simulator.mesh.visual, 'image') and self.optical_simulator.mesh.visual.image is not None:
                print(f"Mesh has texture image, size: {self.optical_simulator.mesh.visual.image.size}")
            else:
                print("Mesh has no texture image")
                
            if hasattr(self.optical_simulator.mesh.visual, 'uv') and self.optical_simulator.mesh.visual.uv is not None:
                print(f"Mesh has UV coordinates, shape: {self.optical_simulator.mesh.visual.uv.shape}")
            else:
                print("Mesh has no UV coordinates")
                
            images = self.optical_simulator.render()
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            QMessageBox.critical(self, "Simulation Error", 
                            f"Error rendering images: {str(e)}\n\nSee console for details.")
            print("Optical simulation error details:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            print(error_traceback)
            return

        # Clear previous images in the grid
        for i in reversed(range(self.image_grid.count())):
            widget = self.image_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Display images in a grid (for example, 2 columns)
        cols = 2
        row = 0
        col = 0
        
        # Create labels for each image with sensor name
        for idx, pil_img in enumerate(images):
            if idx < len(self.sensors):
                sensor_name = self.sensors[idx].name
            else:
                sensor_name = f"Sensor {idx+1}"
                
            # Create container for image and label
            container = QWidget()
            container_layout = QVBoxLayout()
            container.setLayout(container_layout)
            
            # Add sensor name label
            name_label = QLabel(f"Image from: {sensor_name}")
            name_label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(name_label)
            
            # Add image
            img_label = QLabel()
            pixmap = pil_to_qpixmap(pil_img)
            img_label.setPixmap(pixmap.scaledToWidth(400))
            container_layout.addWidget(img_label)
            
            # Add to grid
            self.image_grid.addWidget(container, row, col)
            
            # Update grid position for next image
            col += 1
            if col >= cols:
                col = 0
                row += 1

        # Switch to the Optical Images tab
        self.tabs.setCurrentWidget(self.optical_tab)

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # Clean up VTK resources
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.clear()
            self.vtk_widget.close()
        
        if hasattr(self, 'lidar_vtk_widget'):
            self.lidar_vtk_widget.clear()
            self.lidar_vtk_widget.close()
        
        # Accept the close event
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    