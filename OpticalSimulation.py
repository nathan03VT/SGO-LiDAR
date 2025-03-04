import numpy as np
import trimesh
import pyvista as pv
from PIL import Image, ImageDraw
import io
from pyglet.gl import *

class OpticalSimulator:
    def __init__(self, mesh: trimesh.Trimesh, sensors: list, lights=None,
                 material=None, base_image_width: int = 1024, background_color: list = [0, 0, 0],
                 default_color: list = [150, 150, 150], use_default_texture: bool = False):
        """
        Args:
            mesh: trimesh.Trimesh object representing the scene.
            sensors: List of Sensor objects (from Sensor.py) defining camera positions and FOV.
            lights: List of Light objects for scene lighting.
            material: Optional material (not used in this implementation).
            base_image_width: Base width in pixels for rendering.
            background_color: RGB list for the background color.
            default_color: RGB list for the default mesh color if no texture.
            use_default_texture: Whether to use default brick texture (False = use solid color).
        """
        self.mesh = mesh.copy()
        self.sensors = sensors
        self.base_image_width = base_image_width
        self.background_color = background_color
        self.default_color = default_color
        self.lights = lights if lights is not None else []
        self.texture_image = None
        self.texture_path = None

        # Apply default texture if needed and requested
        if use_default_texture and (not hasattr(self.mesh.visual, 'uv') or self.mesh.visual.uv is None):
            self.generate_uv_mapping(repeats=4)
            default_tex = self.generate_default_texture(width=512, height=512, brick_size=(50, 25))
            self.apply_texture(default_tex)
        # Otherwise, set a solid color 
        elif not use_default_texture:
            if not hasattr(self.mesh.visual, 'face_colors') or self.mesh.visual.face_colors is None:
                # Set solid color for the mesh
                self.mesh.visual.face_colors = np.tile(self.default_color + [255], (len(self.mesh.faces), 1))

    def _apply_lights(self):
        """Apply all lights before rendering."""
        pass

    def generate_uv_mapping(self, repeats: int = 1):
        """Generate a UV mapping that adapts to the model's dominant axis."""
        bounds = self.mesh.bounds
        extents = bounds[1] - bounds[0]

        # Choose the projection plane based on the largest extent
        if extents[0] >= extents[1] and extents[0] >= extents[2]:
            plane = [1, 2]  # Use Y and Z for UV mapping
        elif extents[1] >= extents[0] and extents[1] >= extents[2]:
            plane = [0, 2]  # Use X and Z for UV mapping
        else:
            plane = [0, 1]  # Use X and Y for UV mapping

        min_vals = self.mesh.vertices[:, plane].min(axis=0)
        max_vals = self.mesh.vertices[:, plane].max(axis=0)
        size = max_vals - min_vals
        size[size == 0] = 1.0  # Avoid division by zero

        # Scale UVs and apply repeats
        uvs = (self.mesh.vertices[:, plane] - min_vals) / size * repeats

        # Store the UVs properly
        self.mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=self.texture_image)

    def generate_solid_color_texture(self, width: int = 512, height: int = 512, color: list = None) -> Image.Image:
        """
        Generate a solid color texture as a PIL Image.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            color: RGB list [r,g,b] for the color, uses default_color if None
            
        Returns:
            PIL.Image: The generated solid color texture
        """
        if color is None:
            color = tuple(self.default_color)
        else:
            color = tuple(color)
            
        img = Image.new("RGB", (width, height), color)
        
        self.texture_image = img
        self.texture_path = f"generated_solid_color_{color[0]}_{color[1]}_{color[2]}.png"
        return img

    def load_texture_from_file(self, file_path: str):
        """
        Load a texture image from a file.
        
        Args:
            file_path: Path to the texture image file
            
        Returns:
            PIL.Image: The loaded texture image
        """
        try:
            # Load the texture image using PIL
            self.texture_image = Image.open(file_path)
            print(f"Loaded texture from {file_path}, size: {self.texture_image.size}")
            
            # Store the texture path for reference
            self.texture_path = file_path
            
            return self.texture_image
            
        except Exception as e:
            print(f"Failed to load texture image: {e}")
            return None

    def generate_default_texture(self, width: int = 512, height: int = 512, brick_size: tuple = (50, 25)) -> Image.Image:
        """
        Generate a simple brick texture as a PIL Image.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            brick_size: (width, height) of each brick in pixels
            
        Returns:
            PIL.Image: The generated brick texture
        """
        img = Image.new("RGB", (width, height), (200, 100, 50))
        draw = ImageDraw.Draw(img)
        for y in range(0, height, brick_size[1]):
            offset = (y // brick_size[1]) % 2 * (brick_size[0] // 2)
            for x in range(-offset, width, brick_size[0]):
                draw.rectangle([x, y, x + brick_size[0] - 2, y + brick_size[1] - 2],
                               outline="black")
        
        self.texture_image = img
        self.texture_path = "generated_brick.png"
        return img

    def generate_checkerboard_texture(self, width: int = 512, height: int = 512, squares: int = 8) -> Image.Image:
        """
        Generate a checkerboard texture as a PIL Image.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            squares: Number of squares along each dimension
            
        Returns:
            PIL.Image: The generated checkerboard texture
        """
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        square_size = width // squares
        for i in range(squares):
            for j in range(squares):
                if (i + j) % 2 == 0:
                    continue  # Skip white squares
                x0, y0 = i * square_size, j * square_size
                x1, y1 = (i + 1) * square_size, (j + 1) * square_size
                draw.rectangle([x0, y0, x1, y1], fill=(50, 50, 50))
        
        self.texture_image = img
        self.texture_path = "generated_checkerboard.png"
        return img

    def generate_gradient_texture(self, width: int = 512, height: int = 512) -> Image.Image:
        """
        Generate a gradient texture as a PIL Image.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            
        Returns:
            PIL.Image: The generated gradient texture
        """
        img = Image.new("RGB", (width, height), (255, 255, 255))
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (1 - (x + y) / (width + height)))
                img.putpixel((x, y), (r, g, b))
        
        self.texture_image = img
        self.texture_path = "generated_gradient.png"
        return img

    def generate_noise_texture(self, width: int = 512, height: int = 512) -> Image.Image:
        """
        Generate a noise texture as a PIL Image.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            
        Returns:
            PIL.Image: The generated noise texture
        """
        import random
        img = Image.new("RGB", (width, height), (255, 255, 255))
        for y in range(height):
            for x in range(width):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                img.putpixel((x, y), (r, g, b))
        
        self.texture_image = img
        self.texture_path = "generated_noise.png"
        return img

    def apply_texture(self, texture_img: Image.Image = None, repeats: int = None):
        """
        Apply the given texture image to the mesh using its UV coordinates.
        If no texture is provided, uses the stored texture_image.
        
        Args:
            texture_img: Optional PIL.Image texture to apply
            repeats: Optional number of times to repeat the texture (updates UV mapping)
        """
        if texture_img is not None:
            self.texture_image = texture_img
            
        if repeats is not None:
            self.generate_uv_mapping(repeats=repeats)
            
        if self.texture_image is None:
            print("No texture image available to apply.")
            return
        
        if hasattr(self.mesh.visual, 'uv') and self.mesh.visual.uv is not None:
            self.mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=self.mesh.visual.uv, 
                image=self.texture_image
            )
        else:
            print("Warning: No UV coordinates found on mesh. Generating default UVs.")
            self.generate_uv_mapping(repeats=repeats if repeats is not None else 4)
            self.mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=self.mesh.visual.uv, 
                image=self.texture_image
            )

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
        """
        Create a view matrix (world-to-camera) using the look-at convention.
        Ensures the bottom row is always [0,0,0,1].
        """
        forward = target - eye
        forward_norm = np.linalg.norm(forward)
        forward = forward / forward_norm
        
        # Ensure up vector is not parallel to forward
        up_dot_forward = np.dot(up, forward)
        if abs(abs(up_dot_forward) - 1.0) < 1e-6:
            # If up and forward are nearly parallel, choose a different up vector
            up = np.array([0, 1, 0]) if abs(forward[1]) < 0.9 else np.array([1, 0, 0])
        
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-10:
            # This shouldn't happen with the above checks, but just in case
            if abs(forward[2]) < 0.9:
                right = np.cross(forward, np.array([0, 0, 1]))
            else:
                right = np.cross(forward, np.array([0, 1, 0]))
            right_norm = np.linalg.norm(right)
        
        right = right / right_norm
        true_up = np.cross(right, forward)
        
        view = np.eye(4)  # Start with identity matrix to ensure proper bottom row
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[:3, 3] = -eye @ np.array([right, true_up, -forward])
        
        # Double-check bottom row is correct
        view[3, :] = [0, 0, 0, 1]
        
        return view

    def _compute_camera_transform(self, sensor) -> np.ndarray:
        """
        Compute the camera-to-world transformation matrix for a given sensor.
        The sensor position is used as the eye, and the sensor's target_direction
        defines the look-at point.
        """
        eye = sensor.position
        target = sensor.position + sensor.target_direction
        view_matrix = self._look_at(eye, target, up=np.array([0, 0, 1]))
        camera_transform = np.linalg.inv(view_matrix)
        return camera_transform

    def render(self) -> list:
        """Render an optical image for each sensor in self.sensors."""
        rendered_images = []

        for sensor in self.sensors:
            # Compute camera transform for this specific sensor
            cam_tf = self._compute_camera_transform(sensor)
            
            # Calculate image dimensions based on sensor FOV
            width = self.base_image_width
            height = int(width * (sensor.vertical_fov / sensor.horizontal_fov))

            # Convert trimesh to a PyVista scene
            pv_mesh = pv.PolyData(self.mesh.vertices, np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten())
            
            # Create a fresh plotter for each sensor to avoid camera position issues
            plotter = pv.Plotter(off_screen=True, window_size=[width, height])
            
            try:
                # Check for texture in mesh visual
                has_texture = (hasattr(self.mesh.visual, 'image') and 
                            self.mesh.visual.image is not None)
                has_uv = (hasattr(self.mesh.visual, 'uv') and 
                        self.mesh.visual.uv is not None)
                
                if has_texture and has_uv:
                    # Properly assign UV coordinates to PyVista mesh
                    pv_mesh.active_texture_coordinates = np.asarray(self.mesh.visual.uv, dtype=np.float32)
                    
                    # Convert the texture image to a PyVista texture
                    texture_array = np.array(self.mesh.visual.image)
                    texture = pv.numpy_to_texture(texture_array)
                    
                    # Add textured mesh to the scene
                    plotter.add_mesh(pv_mesh, texture=texture, show_edges=False)
                    
                    print(f"Rendering with texture, size: {texture_array.shape}")
                else:
                    # Use solid color if no texture
                    if hasattr(self.mesh.visual, 'face_colors') and self.mesh.visual.face_colors is not None:
                        # Convert trimesh face colors to cell colors for PyVista
                        pv_mesh.cell_data['colors'] = self.mesh.visual.face_colors
                        plotter.add_mesh(pv_mesh, scalars='colors', rgb=True, show_edges=False)
                        print("Rendering with face colors")
                    else:
                        # Use default color
                        plotter.add_mesh(pv_mesh, color=tuple(c/255 for c in self.default_color), show_edges=False)
                        print(f"Rendering with default color: {self.default_color}")
                
                # Set up the camera based on sensor properties
                position = sensor.position
                target = position + sensor.target_direction
                
                # Calculate camera view up vector (try to maintain a consistent up direction)
                # Default up is Z-axis
                up = np.array([0, 0, 1])
                
                # If looking straight up or down, use Y as up vector
                if abs(np.dot(sensor.target_direction / np.linalg.norm(sensor.target_direction), up)) > 0.95:
                    up = np.array([0, 1, 0])
                
                # Set camera position explicitly
                plotter.camera_position = [position, target, up]
                
                # Set the field of view
                plotter.camera.view_angle = sensor.vertical_fov
                
                # Render the image
                img = plotter.screenshot()
                
                # Convert the screenshot to a PIL image
                image = Image.fromarray(img)
                rendered_images.append(image)
                
            except Exception as e:
                print(f"Error rendering image for sensor: {e}")
                # Append a blank image of the same size
                image = Image.new("RGB", (width, height), self.background_color)
                rendered_images.append(image)
            finally:
                # Always close the plotter to release resources
                plotter.close()

        return rendered_images