import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Union, Any
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist

from Sensor import Sensor
from LidarSimulation import LiDARSimulator
from SceneGenerator import SceneGenerator
from Material import Material
from PointCloudInfoGainNet import (
    PointCloudInfoGainNet, 
    prepare_partial_cloud, 
    prepare_env_cloud, 
    prepare_sensor_pose,
    create_coordinate_grid
)


class SensorOptimizer:
    """
    Optimizes LiDAR sensor placement using the new point cloud based neural network approach.
    """
    def __init__(self, 
                 mesh: trimesh.Trimesh,
                 height: float = 0.0,
                 plane_normal: np.ndarray = np.array([0, 0, 1]),
                 min_sensor_distance: float = 1.0,
                 max_sensors: int = 5,
                 device: str = None):
        """
        Initialize the sensor optimizer
        
        Args:
            mesh: Scene mesh
            height: Height of the candidacy plane for sensors
            plane_normal: Normal vector of the candidacy plane
            min_sensor_distance: Minimum distance between sensors
            max_sensors: Maximum number of sensors to place
            device: Device to run neural network on ('cuda' or 'cpu')
        """
        self.mesh = mesh
        self.height = height
        self.plane_normal = plane_normal / np.linalg.norm(plane_normal)
        self.min_sensor_distance = min_sensor_distance
        self.max_sensors = max_sensors
        
        # Set up device for neural network
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create neural network model (new point cloud based architecture)
        self.model = PointCloudInfoGainNet()
        self.model.to(self.device)
        
        # Information gain calculator
        self.info_gain_calc = LidarInfoGainCalculator(mesh)
        
        # Initialize simulation material
        self.material = Material(albedo=0.7, metallic=0.0, roughness=0.5, ambient=0.2)
        
        # Calculate mesh bounds for creating candidacy region
        self.bounds = mesh.bounds
        self.scene_center = np.mean(self.bounds, axis=0)
        self.scene_dimensions = self.bounds[1] - self.bounds[0]
        
        # Prepare environment point cloud once
        self.env_cloud = prepare_env_cloud(mesh).to(self.device)
        
        # Create coordinate grid for heatmap generation
        self.x_grid, self.y_grid = create_coordinate_grid(self.bounds)
        self.x_grid = self.x_grid.to(self.device)
        self.y_grid = self.y_grid.to(self.device)
        
        # Store optimization results
        self.optimization_history = {
            'info_gain': [],
            'candidacy_size': [],
            'sensor_positions': []
        }
        
        # Initial sensor positions
        self.sensors = []
        
        # Default LiDAR sensor parameters
        self.sensor_params = {
            'horizontal_fov': 90.0,
            'vertical_fov': 60.0,
            'step_angle': 0.5,
            'min_distance': 0.1,
            'max_distance': 100.0
        }
    
    def _evaluate_candidates_pointcloud(self, 
                                       candidates: np.ndarray, 
                                       existing_sensors: List[Sensor]) -> np.ndarray:
        """
        Evaluate candidacy positions using the point cloud neural network
        
        Args:
            candidates: Array of candidate positions to evaluate
            existing_sensors: List of already placed sensors
            
        Returns:
            Information gain predictions for each candidate
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get partial cloud from existing sensors
        if existing_sensors:
            # Create simulator with existing sensors
            simulator = LiDARSimulator(
                mesh=self.mesh,
                sensors=existing_sensors,
                material=self.material
            )
            
            # Run simulation to get partial cloud
            results = simulator.simulate(add_noise=False)
            partial_cloud = prepare_partial_cloud(results).to(self.device)
        else:
            # No existing sensors, empty partial cloud
            partial_cloud = torch.zeros((1, 1000, 3), device=self.device)
        
        # Prepare predictions array
        predictions = np.zeros(len(candidates))
        
        with torch.no_grad():
            for i, pos in enumerate(candidates):
                # Create temporary sensor at this position
                direction = self.scene_center - pos
                direction = direction / np.linalg.norm(direction)
                
                temp_sensor = Sensor(
                    position=pos,
                    target_direction=direction,
                    horizontal_fov=self.sensor_params['horizontal_fov'],
                    vertical_fov=self.sensor_params['vertical_fov'],
                    step_angle=self.sensor_params['step_angle'],
                    min_distance=self.sensor_params['min_distance'],
                    max_distance=self.sensor_params['max_distance']
                )
                
                # Prepare sensor pose
                sensor_pose = prepare_sensor_pose(temp_sensor).to(self.device)
                
                # Calculate EGVS score for initial configuration
                all_sensors = existing_sensors + [temp_sensor]
                simulator = LiDARSimulator(
                    mesh=self.mesh,
                    sensors=all_sensors,
                    material=self.material
                )
                
                # Quick evaluation for EGVS (simplified)
                egvs_score = self.info_gain_calc.calculate_coverage(simulator, all_sensors)
                egvs_score = torch.tensor([[egvs_score]], device=self.device, dtype=torch.float32)
                
                # Run network prediction
                heatmap = self.model(
                    partial_cloud,
                    self.env_cloud,
                    sensor_pose,
                    egvs_score,
                    self.x_grid,
                    self.y_grid
                )
                
                # Extract prediction at the sensor position
                h, w = heatmap.shape[1], heatmap.shape[2]
                grid_x = (pos[0] - self.bounds[0, 0]) / (self.bounds[1, 0] - self.bounds[0, 0])
                grid_y = (pos[1] - self.bounds[0, 1]) / (self.bounds[1, 1] - self.bounds[0, 1])
                
                # Convert to grid coordinates
                grid_x_idx = int(grid_x * (w - 1))
                grid_y_idx = int(grid_y * (h - 1))
                
                # Get prediction at this position
                predictions[i] = heatmap[0, grid_y_idx, grid_x_idx].item()
        
        return predictions
    
    def _train_pointcloud_model(self, 
                               train_positions: np.ndarray, 
                               train_info_gains: np.ndarray,
                               existing_sensors: List[Sensor],
                               epochs: int = 50) -> None:
        """
        Train the point cloud neural network on collected data
        
        Args:
            train_positions: Positions used for training
            train_info_gains: Actual information gain values for positions
            existing_sensors: List of existing sensors
        """
        # Prepare training data
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create data loader
        dataset = []
        for pos, info_gain in zip(train_positions, train_info_gains):
            # Create temporary sensor at this position
            direction = self.scene_center - pos
            direction = direction / np.linalg.norm(direction)
            
            temp_sensor = Sensor(
                position=pos,
                target_direction=direction,
                horizontal_fov=self.sensor_params['horizontal_fov'],
                vertical_fov=self.sensor_params['vertical_fov'],
                step_angle=self.sensor_params['step_angle'],
                min_distance=self.sensor_params['min_distance'],
                max_distance=self.sensor_params['max_distance']
            )
            
            # Prepare data
            if existing_sensors:
                # Get partial cloud
                simulator = LiDARSimulator(
                    mesh=self.mesh,
                    sensors=existing_sensors,
                    material=self.material
                )
                results = simulator.simulate(add_noise=False)
                partial_cloud = prepare_partial_cloud(results).to(self.device)
            else:
                partial_cloud = torch.zeros((1, 1000, 3), device=self.device)
            
            sensor_pose = prepare_sensor_pose(temp_sensor).to(self.device)
            
            # Calculate EGVS
            all_sensors = existing_sensors + [temp_sensor]
            simulator = LiDARSimulator(
                mesh=self.mesh,
                sensors=all_sensors,
                material=self.material
            )
            egvs_score = self.info_gain_calc.calculate_coverage(simulator, all_sensors)
            egvs_score = torch.tensor([[egvs_score]], device=self.device, dtype=torch.float32)
            
            target_heatmap = torch.zeros_like(self.x_grid, device=self.device).unsqueeze(0)
            # Set the target value at the sensor position
            h, w = target_heatmap.shape[1], target_heatmap.shape[2]
            grid_x = (pos[0] - self.bounds[0, 0]) / (self.bounds[1, 0] - self.bounds[0, 0])
            grid_y = (pos[1] - self.bounds[0, 1]) / (self.bounds[1, 1] - self.bounds[0, 1])
            grid_x_idx = int(grid_x * (w - 1))
            grid_y_idx = int(grid_y * (h - 1))
            target_heatmap[0, grid_y_idx, grid_x_idx] = info_gain
            
            dataset.append((partial_cloud, sensor_pose, egvs_score, target_heatmap))
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for partial_cloud, sensor_pose, egvs_score, target_heatmap in dataset:
                optimizer.zero_grad()
                
                # Forward pass
                predicted_heatmap = self.model(
                    partial_cloud,
                    self.env_cloud,
                    sensor_pose,
                    egvs_score,
                    self.x_grid,
                    self.y_grid
                )
                
                # Calculate loss
                loss = criterion(predicted_heatmap, target_heatmap)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.6f}")
    
    def _generate_heatmap_visualization(self, iteration: int) -> None:
        """
        Generate and save a heatmap visualization of information gain
        
        Args:
            iteration: Current iteration number
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get partial cloud from existing sensors
            if self.sensors:
                simulator = LiDARSimulator(
                    mesh=self.mesh,
                    sensors=self.sensors,
                    material=self.material
                )
                results = simulator.simulate(add_noise=False)
                partial_cloud = prepare_partial_cloud(results).to(self.device)
            else:
                partial_cloud = torch.zeros((1, 1000, 3), device=self.device)
            
            # Generate heatmap for a representative sensor pose
            if self.sensors:
                sensor_pose = prepare_sensor_pose(self.sensors[-1]).to(self.device)
                egvs_score = torch.tensor([[0.5]], device=self.device, dtype=torch.float32)  # Placeholder
            else:
                # Default sensor pose
                sensor_pose = torch.tensor([[0.0, 0.0, self.height, 0.0, 0.0, -1.0]], device=self.device)
                egvs_score = torch.tensor([[0.0]], device=self.device, dtype=torch.float32)
            
            # Generate heatmap
            heatmap = self.model(
                partial_cloud,
                self.env_cloud,
                sensor_pose,
                egvs_score,
                self.x_grid,
                self.y_grid
            )
            
            # Convert to numpy and visualize
            heatmap_np = heatmap[0].cpu().numpy()
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            im = plt.imshow(heatmap_np, cmap='viridis', origin='lower', 
                           extent=[self.bounds[0, 0], self.bounds[1, 0], 
                                  self.bounds[0, 1], self.bounds[1, 1]])
            plt.colorbar(im, label='Information Gain')
            
            # Add sensor positions
            if self.sensors:
                sensor_positions = np.array([s.position for s in self.sensors])
                plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
                           c='red', s=100, marker='*', label='Placed Sensors')
                plt.legend()
            
            plt.title(f'Information Gain Heatmap - Iteration {iteration}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            
            os.makedirs('visualizations', exist_ok=True)
            plt.savefig(f'visualizations/pointcloud_heatmap_iter_{iteration}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def optimize(self, 
                max_iterations: int = 10, 
                max_sensors: int = None, 
                visualize: bool = True) -> List[Sensor]:
        """
        Run the optimization process using the point cloud neural network
        
        Args:
            max_iterations: Maximum number of iterations
            max_sensors: Maximum number of sensors to place (overrides instance value if provided)
            visualize: Whether to generate visualizations
            
        Returns:
            List of optimally placed sensors
        """
        if max_sensors is not None:
            self.max_sensors = max_sensors
            
        print(f"Starting sensor placement optimization with point cloud network for up to {self.max_sensors} sensors")
        
        # [Rest of the method implementation would follow the same structure as before,
        # but using the new _evaluate_candidates_pointcloud and _generate_heatmap_visualization methods
        # and training the point cloud model instead of the standard neural network]
        
        # Main optimization loop to place each sensor
        for sensor_idx in range(self.max_sensors):
            print(f"\n--- Placing sensor {sensor_idx + 1} ---")
            
            # Generate candidate positions
            candidates = self._generate_initial_candidacy_region()
            print(f"Initial candidacy region: {len(candidates)} positions")
            
            # Training data collection
            train_positions = []
            train_info_gains = []
            
            # Refinement iterations for each sensor
            info_gain_history = []
            
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration + 1}")
                print(f"Candidacy region: {len(candidates)} positions")
                
                if iteration > 0 and len(train_positions) >= 20:
                    # Train point cloud network on collected data
                    print("Training point cloud network...")
                    self._train_pointcloud_model(
                        np.array(train_positions),
                        np.array(train_info_gains),
                        self.sensors,
                        epochs=50
                    )
                    
                    # Use point cloud network to predict information gain
                    print("Evaluating candidates with point cloud network...")
                    predicted_info_gains = self._evaluate_candidates_pointcloud(candidates, self.sensors)
                    
                    # Select top candidates for actual evaluation
                    top_k = min(20, len(candidates))
                    top_indices = np.argsort(predicted_info_gains)[-top_k:]
                    top_candidates = candidates[top_indices]
                    
                    # Calculate actual information gain for top candidates
                    print(f"Calculating actual information gain for top {top_k} candidates...")
                    sample_info_gains = self._calculate_actual_info_gain(
                        top_candidates, self.sensors
                    )
                    
                    # Create full info gain array
                    info_gains = predicted_info_gains.copy()
                    info_gains[top_indices] = sample_info_gains
                    
                    # Add to training dataset
                    train_positions.extend(top_candidates)
                    train_info_gains.extend(sample_info_gains)
                    
                else:
                    # For first iterations, calculate actual info gain for a subset
                    sample_size = min(50, len(candidates))
                    print(f"Calculating actual information gain for {sample_size} random candidates...")
                    sample_indices = np.random.choice(len(candidates), sample_size, replace=False)
                    sample_candidates = candidates[sample_indices]
                    
                    sample_info_gains = self._calculate_actual_info_gain(
                        sample_candidates, self.sensors
                    )
                    
                    # Create full info gain array
                    info_gains = np.zeros(len(candidates))
                    info_gains[sample_indices] = sample_info_gains
                    
                    # Add to training dataset
                    train_positions.extend(sample_candidates)
                    train_info_gains.extend(sample_info_gains)
                
                # Select best position
                best_position, best_info_gain = self._select_best_position(
                    candidates, info_gains, self.sensors
                )
                
                print(f"Best position: {best_position}")
                print(f"Best information gain: {best_info_gain:.4f}")
                
                info_gain_history.append(best_info_gain)
                
                # Generate visualization
                if visualize:
                    self._generate_heatmap_visualization(iteration)
                
                # Check termination criteria
                if self._check_termination(iteration, info_gain_history, len(candidates), max_iterations):
                    break
                
                # Refine candidacy region for next iteration
                candidates = self._refine_candidacy_region(candidates, info_gains, iteration)
            
            # Create and add the best sensor
            best_sensor = self._create_sensor_at_position(best_position)
            self.sensors.append(best_sensor)
            
            # Store optimization results
            self.optimization_history['info_gain'].append(best_info_gain)
            self.optimization_history['sensor_positions'].append(best_position.copy())
            
            print(f"Placed sensor {sensor_idx + 1} at {best_position} with info gain {best_info_gain:.4f}")
            
            # If this is the last sensor, or info gain is very high, stop optimization
            if sensor_idx >= self.max_sensors - 1 or best_info_gain > 0.95:
                break
        
        print("\nSensor placement optimization complete!")
        print(f"Placed {len(self.sensors)} sensors")
        
        for i, sensor in enumerate(self.sensors):
            print(f"Sensor {i+1}: Position={sensor.position}, Direction={sensor.target_direction}")
        
        return self.sensors
    
    def _generate_initial_candidacy_region(self):
        """Generate initial candidate positions"""
        # Create candidates within scene bounds on the candidacy plane
        bounds = self.bounds
        step = 1.0
        
        x_range = np.arange(bounds[0, 0], bounds[1, 0], step)
        y_range = np.arange(bounds[0, 1], bounds[1, 1], step)
        
        candidates = []
        for x in x_range:
            for y in y_range:
                # Calculate z based on plane equation (normal dot product)
                # For simplicity, use fixed height
                z = self.height
                candidates.append(np.array([x, y, z]))
        
        return np.array(candidates)

    def _calculate_actual_info_gain(self, candidates, existing_sensors):
        """Calculate actual information gain for candidate positions"""
        info_gains = []
        
        for position in candidates:
            # Create temporary sensor at this position
            direction = self.scene_center - position
            direction = direction / np.linalg.norm(direction)
            
            temp_sensor = Sensor(
                position=position,
                target_direction=direction,
                horizontal_fov=self.sensor_params['horizontal_fov'],
                vertical_fov=self.sensor_params['vertical_fov'],
                step_angle=self.sensor_params['step_angle']
            )
            
            # Simulate with this sensor configuration
            all_sensors = existing_sensors + [temp_sensor]
            simulator = LiDARSimulator(
                mesh=self.mesh,
                sensors=all_sensors,
                material=self.material
            )
            
            # Calculate info gain
            info_gain = self.info_gain_calc.calculate_information_gain(simulator, all_sensors)
            info_gains.append(info_gain)
        
        return np.array(info_gains)

    def _select_best_position(self, candidates, info_gains, existing_sensors):
        """Select the best position from candidates"""
        best_idx = np.argmax(info_gains)
        best_position = candidates[best_idx]
        best_info_gain = info_gains[best_idx]
        
        return best_position, best_info_gain

    def _check_termination(self, iteration, info_gain_history, candidacy_size, max_iterations):
        """Check if optimization should terminate"""
        # Terminate if max iterations reached
        if iteration >= max_iterations - 1:
            return True
        
        # Terminate if improvement is small
        if iteration > 1:
            improvement = info_gain_history[-1] - info_gain_history[-2]
            if improvement < 1e-4:
                return True
        
        return False

    def _refine_candidacy_region(self, candidates, info_gains, iteration):
        """Refine the candidacy region based on information gains"""
        # For simplicity, just reduce the region around the best candidates
        top_k = min(100, len(candidates))
        top_indices = np.argsort(info_gains)[-top_k:]
        
        # Create new candidates around the best positions
        new_candidates = []
        for idx in top_indices:
            center = candidates[idx]
            # Add candidates in a smaller radius around this center
            radius = max(0.5, 5.0 / (iteration + 1))
            for _ in range(10):
                angle = np.random.uniform(0, 2*np.pi)
                dist = np.random.uniform(0, radius)
                new_pos = center.copy()
                new_pos[0] += dist * np.cos(angle)
                new_pos[1] += dist * np.sin(angle)
                new_candidates.append(new_pos)
        
        return np.array(new_candidates)

    def _create_sensor_at_position(self, position):
        """Create a sensor at the given position"""
        direction = self.scene_center - position
        direction = direction / np.linalg.norm(direction)
        
        sensor = Sensor(
            position=position,
            target_direction=direction,
            horizontal_fov=self.sensor_params['horizontal_fov'],
            vertical_fov=self.sensor_params['vertical_fov'],
            step_angle=self.sensor_params['step_angle']
        )
        
        return sensor

    def get_optimization_history(self):
        """Get the optimization history"""
        return self.optimization_history
    
class LidarInfoGainCalculator:
    """
    Calculates information gain metrics for LiDAR sensor placement optimization.
    """
    def __init__(self, 
                mesh: trimesh.Trimesh, 
                alpha: float = 0.6, 
                beta: float = 0.3, 
                gamma: float = 0.1):
        """
        Initialize the information gain calculator
        
        Args:
            mesh: Scene mesh
            alpha: Coverage weight in information gain formula
            beta: Point density weight in information gain formula
            gamma: View angle diversity weight in information gain formula
        """
        self.mesh = mesh
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        
        # Generate scene point cloud
        self.scene_points = mesh.sample(20000)  # Sample points from mesh surface
        self.scene_point_count = len(self.scene_points)
        
        # Store the total number of points for coverage calculation
        self.total_points = self.scene_point_count
        
        # Calculate scene bounds for visualization
        self.bounds = mesh.bounds
        self.scene_center = np.mean(self.bounds, axis=0)
        self.scene_dimensions = self.bounds[1] - self.bounds[0]
        
    def calculate_visibility(self, 
                           simulator: LiDARSimulator, 
                           sensors: List[Sensor]) -> np.ndarray:
        """
        Calculate the visibility of scene points from given sensors
        
        Args:
            simulator: LiDAR simulator instance
            sensors: List of sensor positions to evaluate
            
        Returns:
            Visibility mask for scene points
        """
        # Run simulation to get point cloud coverage
        lidar_results = simulator.simulate(add_noise=False)
        
        # Initialize visibility mask
        visibility = np.zeros(self.scene_point_count, dtype=bool)
        
        # For each sensor's point cloud, mark visible points
        for points, normals, reflectivity in lidar_results:
            if len(points) == 0:
                continue
                
            # Find which scene points are visible from this sensor
            # For each simulated point, find the closest scene point
            distances = cdist(points, self.scene_points)
            min_distances = np.min(distances, axis=0)
            
            # Points are considered visible if they are close to a simulated point
            # Threshold based on scene dimensions (0.5% of max dimension)
            threshold = 0.005 * np.max(self.scene_dimensions)
            visible_indices = np.where(min_distances < threshold)[0]
            
            # Mark these points as visible
            visibility[visible_indices] = True
            
        return visibility
        
    def calculate_point_density(self, 
                              simulator: LiDARSimulator, 
                              sensors: List[Sensor]) -> float:
        """
        Calculate point density from the current sensor setup
        
        Args:
            simulator: LiDAR simulator instance
            sensors: List of sensors to evaluate
            
        Returns:
            Point density metric
        """
        # Run simulation 
        lidar_results = simulator.simulate(add_noise=False)
        
        # Count total points captured
        total_points = sum(len(points) for points, _, _ in lidar_results)
        
        if total_points == 0:
            return 0.0
            
        # Calculate spatial distribution of points
        all_points = np.vstack([points for points, _, _ in lidar_results if len(points) > 0])
        
        if len(all_points) == 0:
            return 0.0
            
        # Calculate average minimum distance between points (lower is better density)
        min_distances = []
        sample_size = min(5000, len(all_points))  # Limit computation for large point clouds
        sampled_indices = np.random.choice(len(all_points), sample_size, replace=False)
        sampled_points = all_points[sampled_indices]
        
        # Calculate distances between sampled points
        distance_matrix = cdist(sampled_points, sampled_points)
        np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distance
        min_dists = np.min(distance_matrix, axis=1)
        
        # Average minimum distance (lower is better)
        avg_min_dist = np.mean(min_dists)
        
        # Normalize based on scene dimensions (0 to 1, where higher is better)
        max_dist = np.max(self.scene_dimensions)
        normalized_density = 1.0 - (avg_min_dist / max_dist)
        normalized_density = np.clip(normalized_density, 0.0, 1.0)
        
        return normalized_density
        
    def calculate_view_angle_diversity(self, 
                                     simulator: LiDARSimulator, 
                                     sensors: List[Sensor]) -> float:
        """
        Calculate view angle diversity from the current sensor setup
        
        Args:
            simulator: LiDAR simulator instance
            sensors: List of sensors to evaluate
            
        Returns:
            View angle diversity metric
        """
        if len(sensors) <= 1:
            return 0.0  # No diversity with 0 or 1 sensor
            
        # Calculate angles between sensors relative to scene center
        sensor_positions = np.array([sensor.position for sensor in sensors])
        vectors_to_center = self.scene_center - sensor_positions
        
        # Normalize vectors
        normalized_vectors = vectors_to_center / np.linalg.norm(vectors_to_center, axis=1, keepdims=True)
        
        # Calculate cosine similarities between all pairs of vectors
        similarities = np.dot(normalized_vectors, normalized_vectors.T)
        
        # Clip to valid range [-1, 1]
        similarities = np.clip(similarities, -1.0, 1.0)
        
        # Calculate angles in degrees
        angles = np.arccos(similarities) * 180 / np.pi
        
        # Set diagonal to zero (angles between same vector)
        np.fill_diagonal(angles, 0)
        
        # Average angle between sensors (higher is better)
        avg_angle = np.sum(angles) / (len(sensors) * (len(sensors) - 1))
        
        # Normalize to [0, 1] where 1 is perfect diversity (180 degrees)
        normalized_diversity = avg_angle / 180.0
        
        return normalized_diversity
    
    def calculate_coverage(self, 
                          simulator: LiDARSimulator, 
                          sensors: List[Sensor]) -> float:
        """
        Calculate coverage as the fraction of scene points that are visible
        
        Args:
            simulator: LiDAR simulator instance
            sensors: List of sensors to evaluate
            
        Returns:
            Coverage ratio (0 to 1)
        """
        # Get visibility mask
        visibility = self.calculate_visibility(simulator, sensors)
        
        # Calculate coverage ratio
        coverage = np.sum(visibility) / self.total_points
        
        return coverage
    
    def calculate_information_gain(self, 
                                 simulator: LiDARSimulator, 
                                 sensors: List[Sensor]) -> float:
        """
        Calculate the overall information gain metric
        
        Args:
            simulator: LiDAR simulator instance
            sensors: List of sensors to evaluate
            
        Returns:
            Information gain value (0 to 1)
        """
        # Calculate individual metrics
        coverage = self.calculate_coverage(simulator, sensors)
        point_density = self.calculate_point_density(simulator, sensors)
        angle_diversity = self.calculate_view_angle_diversity(simulator, sensors)
        
        # Combine metrics with weights
        information_gain = (
            self.alpha * coverage + 
            self.beta * point_density + 
            self.gamma * angle_diversity
        )
        
        return information_gain