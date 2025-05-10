import numpy as np
import torch
from LidarSimulation import LiDARSimulator
from Sensor import Sensor
from Material import Material
import trimesh
from PointCloudInfoGainNet import (
    PointCloudInfoGainNet,
    prepare_partial_cloud,
    prepare_env_cloud,
    prepare_sensor_pose,
    create_coordinate_grid
)


class SimulationInterface:
    """
    Interface to the LiDAR simulation system with point cloud network integration
    """
    def __init__(self, mesh_path: str, material: Material = None, param_bounds: list = None):
        """
        Initialize the simulation interface
        
        Args:
            mesh_path: Path to the CAD model mesh file
            material: Optional Material object for the scene
            param_bounds: List of (min, max) tuples for each parameter
        """
        self.mesh_path = mesh_path
        self.mesh = trimesh.load(mesh_path)
        self.material = material or Material()
        self.param_bounds = param_bounds or []
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize point cloud network
        self.network = PointCloudInfoGainNet().to(self.device)
        
        # Prepare environment cloud once
        self.env_cloud = prepare_env_cloud(self.mesh, num_points=20000).to(self.device)
        
        # Create coordinate grid
        self.x_grid, self.y_grid = create_coordinate_grid(self.mesh.bounds)
        self.x_grid = self.x_grid.to(self.device)
        self.y_grid = self.y_grid.to(self.device)
    
    def set_param_bounds(self, param_bounds: list):
        """Set parameter bounds for configuration sampling"""
        self.param_bounds = param_bounds
    
    def encode_config(self, sensors: list) -> np.ndarray:
        """Convert Sensor objects to flat configuration vector"""
        config = []
        for sensor in sensors:
            # Extract position
            config.extend(sensor.position)
            # Extract orientation (normalized direction)
            config.extend(sensor.target_direction)
            # Extract other parameters
            config.append(sensor.horizontal_fov)
            config.append(sensor.vertical_fov)
            config.append(sensor.step_angle)
        
        return np.array(config)
    
    def decode_config(self, config_vector: np.ndarray) -> list:
        """Convert flat configuration vector to Sensor objects"""
        sensors = []
        
        # Determine parameters per sensor
        params_per_sensor = len(config_vector) // 1  # Default to 1 sensor
        
        # Simple case - just one sensor with 9 parameters
        if params_per_sensor == 9:
            # Extract parameters
            position = config_vector[0:3]
            direction = config_vector[3:6]
            h_fov = config_vector[6]
            v_fov = config_vector[7]
            step_angle = config_vector[8]
            
            # Normalize direction vector
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                direction = np.array([0, 0, -1])  # Default direction
            
            # Create sensor
            sensor = Sensor(
                position=position,
                target_direction=direction,
                horizontal_fov=h_fov,
                vertical_fov=v_fov,
                step_angle=step_angle
            )
            sensors.append(sensor)
        else:
            # Handle multiple sensors
            num_sensors = len(config_vector) // params_per_sensor
            for i in range(num_sensors):
                start_idx = i * params_per_sensor
                # Extract parameters for this sensor
                position = config_vector[start_idx:start_idx+3]
                direction = config_vector[start_idx+3:start_idx+6]
                h_fov = config_vector[start_idx+6]
                v_fov = config_vector[start_idx+7]
                step_angle = config_vector[start_idx+8]
                
                # Normalize direction vector
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    direction = direction / direction_norm
                else:
                    direction = np.array([0, 0, -1])  # Default direction
                
                # Create sensor
                sensor = Sensor(
                    position=position,
                    target_direction=direction,
                    horizontal_fov=h_fov,
                    vertical_fov=v_fov,
                    step_angle=step_angle
                )
                sensors.append(sensor)
        
        return sensors
    
    def evaluate_configuration(self, config_vector: np.ndarray, add_noise: bool = True) -> float:
        """Run LiDAR simulation and evaluate PCCS score"""
        # Decode configuration into Sensor objects
        sensors = self.decode_config(config_vector)
        
        # Create simulator
        simulator = LiDARSimulator(
            mesh=self.mesh,
            sensors=sensors,
            material=self.material
        )
        
        # Run simulation
        results = simulator.simulate(add_noise=add_noise)
        
        # Calculate PCCS score
        pccs = self.calculate_pccs(results, sensors)
        
        return pccs
    
    def predict_information_gain_heatmap(self, sensors: list) -> np.ndarray:
        """
        Use the point cloud network to predict information gain heatmap
        
        Args:
            sensors: List of Sensor objects
            
        Returns:
            heatmap: [H, W] numpy array of information gain values
        """
        self.network.eval()
        
        with torch.no_grad():
            # Get partial cloud from sensors
            if sensors:
                simulator = LiDARSimulator(
                    mesh=self.mesh,
                    sensors=sensors,
                    material=self.material
                )
                results = simulator.simulate(add_noise=False)
                partial_cloud = prepare_partial_cloud(results).to(self.device)
            else:
                partial_cloud = torch.zeros((1, 1000, 3), device=self.device)
            
            # Prepare sensor pose (using the first sensor for now)
            if sensors:
                sensor_pose = prepare_sensor_pose(sensors[0]).to(self.device)
                
                # Calculate EGVS score
                egvs_score = self.calculate_pccs(results, sensors)
                egvs_score = torch.tensor([[egvs_score]], device=self.device)
            else:
                sensor_pose = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, -1.0]], device=self.device)
                egvs_score = torch.tensor([[0.0]], device=self.device)
            
            # Generate heatmap
            heatmap = self.network(
                partial_cloud,
                self.env_cloud,
                sensor_pose,
                egvs_score,
                self.x_grid,
                self.y_grid
            )
            
            # Convert to numpy
            heatmap_np = heatmap[0].cpu().numpy()
            
            return heatmap_np
    
    def calculate_pccs(self, simulation_results: list, sensors: list) -> float:
        """Calculate Point Cloud Coverage Score"""
        # Calculate total rays that could be cast
        total_rays = 0
        for sensor in sensors:
            # Match the exact way rays are generated in _compute_ray_directions
            h_steps = len(np.arange(-sensor.horizontal_fov/2, sensor.horizontal_fov/2, sensor.step_angle))
            v_steps = len(np.arange(-sensor.vertical_fov/2, sensor.vertical_fov/2, sensor.step_angle))
            total_rays += h_steps * v_steps
        
        # Count valid points
        valid_points = 0
        for points, _, _ in simulation_results:
            valid_points += len(points)
        
        # Cap coverage ratio at 1.0 to prevent exceeding theoretical maximum
        coverage_ratio = min(valid_points / total_rays if total_rays > 0 else 0, 1.0)
        
        # Diversity factor calculation
        diversity_factor = 1.0
        if valid_points > 0:
            point_distribution = []
            for points, _, _ in simulation_results:
                if len(points) > 0:
                    point_distribution.append(len(points) / valid_points)
            
            if point_distribution:
                diversity_factor = 1.0 - np.std(point_distribution)
        
        # Cap diversity factor too, as floating point issues might push it slightly above 1.0
        diversity_factor = min(diversity_factor, 1.0)
        
        # Density factor
        density_factor = 1.0
        
        # Final score with guaranteed maximum of 1.0
        pccs = coverage_ratio * diversity_factor * density_factor
        
        return pccs
    
    def visualize_configuration(self, sensors: list):
        """Visualize the point cloud from a specific configuration"""
        # Create simulator
        simulator = LiDARSimulator(
            mesh=self.mesh,
            sensors=sensors,
            material=self.material
        )
        
        # Run simulation
        results = simulator.simulate(add_noise=True)
        
        # Use existing visualization code if available
        from VisualizeSimulation import LIDARVisualizer
        visualizer = LIDARVisualizer(self.mesh_path)
        
        # Combine all point clouds
        all_points = np.concatenate([r[0] for r in results])
        all_normals = np.concatenate([r[1] for r in results])
        all_reflectivity = np.concatenate([r[2] for r in results])
        
        # Visualize
        visualizer.visualize_point_cloud(
            points=all_points,
            normals=all_normals,
            scanner_position=sensors[0].position,
            reflectivity=all_reflectivity
        )
        visualizer.show()


class PointCloudSimulationInterface:
    """
    Specialized interface that uses the point cloud network for optimization
    """
    def __init__(self, mesh_path: str, material: Material = None):
        self.interface = SimulationInterface(mesh_path, material)
        self.heatmap_cache = {}
    
    def get_information_gain_at_position(self, position: np.ndarray, sensors: list = None) -> float:
        """
        Get information gain prediction at a specific position
        
        Args:
            position: [3] array of x, y, z coordinates
            sensors: List of existing sensors
            
        Returns:
            info_gain: Information gain value at this position
        """
        # Generate heatmap
        heatmap = self.interface.predict_information_gain_heatmap(sensors or [])
        
        # Get bounds
        bounds = self.interface.mesh.bounds
        
        # Convert position to grid coordinates
        h, w = heatmap.shape
        grid_x = (position[0] - bounds[0, 0]) / (bounds[1, 0] - bounds[0, 0])
        grid_y = (position[1] - bounds[0, 1]) / (bounds[1, 1] - bounds[0, 1])
        
        # Clamp to valid range
        grid_x = max(0, min(1, grid_x))
        grid_y = max(0, min(1, grid_y))
        
        # Convert to array indices
        grid_x_idx = int(grid_x * (w - 1))
        grid_y_idx = int(grid_y * (h - 1))
        
        return heatmap[grid_y_idx, grid_x_idx]
    
    def find_best_position(self, sensors: list = None, num_candidates: int = 1000) -> np.ndarray:
        """
        Find the best position based on information gain heatmap
        
        Args:
            sensors: List of existing sensors
            num_candidates: Number of candidate positions to evaluate
            
        Returns:
            best_position: [3] array of optimal position coordinates
        """
        # Generate heatmap
        heatmap = self.interface.predict_information_gain_heatmap(sensors or [])
        
        # Find maximum value in heatmap
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Convert back to world coordinates
        bounds = self.interface.mesh.bounds
        h, w = heatmap.shape
        
        world_x = bounds[0, 0] + (max_idx[1] / (w - 1)) * (bounds[1, 0] - bounds[0, 0])
        world_y = bounds[0, 1] + (max_idx[0] / (h - 1)) * (bounds[1, 1] - bounds[0, 1])
        world_z = 5.0  # Default height
        
        return np.array([world_x, world_y, world_z])