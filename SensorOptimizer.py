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

class SensorPlacementNN(nn.Module):
    """
    Neural network model for sensor placement optimization.
    Takes point cloud and partial sensor coverage as input,
    predicts information gain for potential positions.
    """
    def __init__(self, 
                 point_feature_dim: int = 3, 
                 hidden_dim: int = 64,
                 output_dim: int = 1):
        super(SensorPlacementNN, self).__init__()
        
        # Feature extraction layers
        self.point_feature_extractor = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Spatial mapping layers
        self.spatial_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Information gain prediction
        self.info_gain_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, output_dim),
            nn.Sigmoid()  # Output is normalized information gain (0-1)
        )
    
    def forward(self, points, visibility=None):
        """
        Forward pass through the network
        
        Args:
            points: [B, N, F] batch of point clouds where B is batch size, 
                   N is number of points, F is feature dimension
            visibility: [B, N, S] optional visibility mask where S is number 
                        of existing sensors
        
        Returns:
            [B, 1] predicted information gain
        """
        batch_size, num_points, _ = points.shape
        
        # Extract point features
        point_features = self.point_feature_extractor(points)
        
        # If visibility information is provided, concatenate it
        if visibility is not None:
            visibility_expanded = visibility.float().unsqueeze(-1)
            point_features = torch.cat([point_features, visibility_expanded], dim=-1)
            
        # Process features
        processed_features = self.feature_processor(point_features)
        
        # Max pooling over points for permutation invariance
        global_features = torch.max(processed_features, dim=1)[0]
        
        # Spatial mapping
        spatial_features = self.spatial_mapper(global_features)
        
        # Predict information gain
        info_gain = self.info_gain_predictor(spatial_features)
        
        return info_gain

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

class SensorOptimizer:
    """
    Optimizes LiDAR sensor placement using an iterative neural network approach.
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
        
        # Create neural network model
        self.model = SensorPlacementNN(point_feature_dim=3, hidden_dim=64, output_dim=1)
        self.model.to(self.device)
        
        # Information gain calculator
        self.info_gain_calc = LidarInfoGainCalculator(mesh)
        
        # Initialize simulation material
        self.material = Material(albedo=0.7, metallic=0.0, roughness=0.5, ambient=0.2)
        
        # Calculate mesh bounds for creating candidacy region
        self.bounds = mesh.bounds
        self.scene_center = np.mean(self.bounds, axis=0)
        self.scene_dimensions = self.bounds[1] - self.bounds[0]
        
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
    
    def _generate_initial_candidacy_region(self) -> np.ndarray:
        """
        Generate initial candidate positions on a horizontal plane
        
        Returns:
            Array of candidate positions
        """
        # Create a grid of candidate positions around the scene center
        # Adjust the height to the specified plane height
        max_dimension = np.max(self.scene_dimensions)
        grid_size = max(10, int(max_dimension / self.min_sensor_distance))
        
        # Create grid on XY plane
        x = np.linspace(self.bounds[0, 0], self.bounds[1, 0], grid_size)
        y = np.linspace(self.bounds[0, 1], self.bounds[1, 1], grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Create positions with fixed height
        positions = np.column_stack((xx.flatten(), yy.flatten(), 
                                     np.ones_like(xx.flatten()) * self.height))
        
        # Filter positions to keep only those outside the mesh
        # This prevents placing sensors inside objects
        inside_mesh = np.zeros(len(positions), dtype=bool)
        
        if hasattr(self.mesh, 'contains') and callable(getattr(self.mesh, 'contains')):
            try:
                inside_mesh = self.mesh.contains(positions)
            except Exception as e:
                print(f"Warning: Could not check if positions are inside mesh: {e}")
                print("Using all candidate positions")
        
        # Keep positions outside the mesh or use all if contains check failed
        valid_positions = positions[~inside_mesh]
        
        if len(valid_positions) == 0:
            print("Warning: No valid positions outside mesh. Using all positions.")
            valid_positions = positions
        
        return valid_positions
    
    def _refine_candidacy_region(self, 
                                best_positions: np.ndarray, 
                                info_gain_map: np.ndarray,
                                iteration: int) -> np.ndarray:
        """
        Refine the candidacy region based on neural network predictions
        
        Args:
            best_positions: Current best positions to focus around
            info_gain_map: Information gain values for previous candidates
            iteration: Current iteration number
        
        Returns:
            Refined candidate positions
        """
        # Threshold for selecting high information gain candidates
        # This threshold gets higher with each iteration
        base_threshold = 0.5 
        threshold = base_threshold + (iteration * 0.05)
        threshold = min(threshold, 0.9)  # Cap at 0.9
        
        # Select candidates with high information gain
        high_gain_indices = np.where(info_gain_map > threshold)[0]
        
        if len(high_gain_indices) < 3:
            # If too few high gain positions, take top 20%
            high_gain_count = max(3, int(0.2 * len(info_gain_map)))
            high_gain_indices = np.argsort(info_gain_map)[-high_gain_count:]
        
        high_gain_positions = best_positions[high_gain_indices]
        
        # Calculate centroid of high gain positions
        centroid = np.mean(high_gain_positions, axis=0)
        
        # Calculate standard deviation to determine search radius
        # This shrinks with each iteration to focus the search
        stdev = np.std(high_gain_positions, axis=0)
        search_radius = np.mean(stdev) * (1.0 - (iteration * 0.1))
        search_radius = max(search_radius, self.min_sensor_distance)
        
        # Generate new candidates around high gain positions
        # Number of new candidates reduces with iterations to focus search
        candidates_per_point = max(5, 50 - (iteration * 5))
        
        new_candidates = []
        for pos in high_gain_positions:
            # Generate candidates in a circle around this position
            for _ in range(candidates_per_point):
                # Random direction on the plane
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, search_radius)
                
                # Calculate offset in the plane
                plane_offset = np.array([
                    r * np.cos(theta),
                    r * np.sin(theta),
                    0  # No offset in normal direction
                ])
                
                # Generate new candidate
                new_pos = pos + plane_offset
                
                # Ensure the height is maintained
                new_pos[2] = self.height
                
                new_candidates.append(new_pos)
        
        new_candidates = np.array(new_candidates)
        
        # Add the centroid and best positions to ensure good coverage
        candidates = np.vstack([
            new_candidates,
            centroid.reshape(1, 3),
            high_gain_positions
        ])
        
        # Filter positions outside scene bounds
        in_bounds = np.all((candidates >= self.bounds[0]) & 
                         (candidates <= self.bounds[1]), axis=1)
        candidates = candidates[in_bounds]
        
        # Remove duplicates
        candidates = np.unique(candidates, axis=0)
        
        return candidates
    
    def _evaluate_candidates(self, 
                           candidates: np.ndarray, 
                           existing_sensors: List[Sensor]) -> np.ndarray:
        """
        Evaluate candidacy positions using the neural network
        
        Args:
            candidates: Array of candidate positions to evaluate
            existing_sensors: List of already placed sensors
            
        Returns:
            Information gain predictions for each candidate
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Create temporary sensor for each candidate
        candidate_sensors = []
        for pos in candidates:
            # Calculate direction towards scene center
            direction = self.scene_center - pos
            direction = direction / np.linalg.norm(direction)
            
            # Create sensor
            sensor = Sensor(
                position=pos,
                target_direction=direction,
                horizontal_fov=self.sensor_params['horizontal_fov'],
                vertical_fov=self.sensor_params['vertical_fov'],
                step_angle=self.sensor_params['step_angle'],
                min_distance=self.sensor_params['min_distance'],
                max_distance=self.sensor_params['max_distance']
            )
            candidate_sensors.append(sensor)
        
        # Prepare batch data for neural network
        batch_size = min(16, len(candidates))  # Process in batches
        predictions = np.zeros(len(candidates))
        
        with torch.no_grad():
            for i in range(0, len(candidates), batch_size):
                end_idx = min(i + batch_size, len(candidates))
                batch_candidates = candidates[i:end_idx]
                
                # Normalize point coordinates
                normalized_points = (batch_candidates - self.bounds[0]) / self.scene_dimensions
                
                # Convert to tensor
                points_tensor = torch.tensor(normalized_points, dtype=torch.float32).unsqueeze(1)
                points_tensor = points_tensor.to(self.device)
                
                # Get predictions
                batch_predictions = self.model(points_tensor)
                predictions[i:end_idx] = batch_predictions.cpu().numpy().flatten()
        
        return predictions
    
    def _train_neural_network(self, 
                            train_positions: np.ndarray, 
                            train_info_gains: np.ndarray,
                            epochs: int = 100) -> None:
        """
        Train the neural network on collected data
        
        Args:
            train_positions: Positions used for training
            train_info_gains: Actual information gain values for positions
        """
        # Normalize point coordinates
        normalized_positions = (train_positions - self.bounds[0]) / self.scene_dimensions
        
        # Convert to tensors
        X = torch.tensor(normalized_positions, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(train_info_gains, dtype=torch.float32).unsqueeze(1)
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train model
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")
    
    def _calculate_actual_info_gain(self, 
                                  candidates: np.ndarray, 
                                  existing_sensors: List[Sensor],
                                  sample_size: int = None) -> np.ndarray:
        """
        Calculate actual information gain for a subset of candidates
        
        Args:
            candidates: Array of candidate positions
            existing_sensors: List of already placed sensors
            sample_size: Number of candidates to sample (None for all)
            
        Returns:
            Array of information gain values
        """
        if sample_size is not None and sample_size < len(candidates):
            # Sample a subset of candidates for efficiency
            indices = np.random.choice(len(candidates), sample_size, replace=False)
            eval_candidates = candidates[indices]
        else:
            eval_candidates = candidates
            indices = np.arange(len(candidates))
        
        # Initialize results array for all candidates
        info_gains = np.zeros(len(candidates))
        
        # Calculate information gain for each candidate
        for i, idx in enumerate(indices):
            pos = eval_candidates[i]
            
            # Calculate direction towards scene center
            direction = self.scene_center - pos
            direction = direction / np.linalg.norm(direction)
            
            # Create temporary sensor at this position
            temp_sensor = Sensor(
                position=pos,
                target_direction=direction,
                horizontal_fov=self.sensor_params['horizontal_fov'],
                vertical_fov=self.sensor_params['vertical_fov'],
                step_angle=self.sensor_params['step_angle'],
                min_distance=self.sensor_params['min_distance'],
                max_distance=self.sensor_params['max_distance']
            )
            
            # Create combined sensor list
            eval_sensors = existing_sensors + [temp_sensor]
            
            # Create simulator
            simulator = LiDARSimulator(
                mesh=self.mesh,
                sensors=eval_sensors,
                material=self.material
            )
            
            # Calculate information gain
            info_gain = self.info_gain_calc.calculate_information_gain(simulator, eval_sensors)
            
            # Store result
            info_gains[idx] = info_gain
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i+1}/{len(indices)} candidates")
        
        return info_gains
    
    def _select_best_position(self, 
                            candidates: np.ndarray, 
                            info_gains: np.ndarray,
                            existing_sensors: List[Sensor]) -> Tuple[np.ndarray, float]:
        """
        Select the best position from candidates based on info gain
        and minimum distance constraint
        
        Args:
            candidates: Array of candidate positions
            info_gains: Array of information gain values
            existing_sensors: List of already placed sensors
            
        Returns:
            Tuple of (best position, info gain)
        """
        # Create mask for candidates that satisfy minimum distance constraint
        valid_mask = np.ones(len(candidates), dtype=bool)
        
        if existing_sensors:
            existing_positions = np.array([s.position for s in existing_sensors])
            
            for i, pos in enumerate(candidates):
                # Calculate distances to existing sensors
                distances = np.linalg.norm(existing_positions - pos, axis=1)
                
                # Check if any distance is below minimum
                if np.any(distances < self.min_sensor_distance):
                    valid_mask[i] = False
        
        # If no valid candidates, relax constraint by 50%
        if not np.any(valid_mask):
            relaxed_distance = self.min_sensor_distance * 0.5
            print(f"No valid candidates found. Relaxing minimum distance to {relaxed_distance}")
            
            valid_mask = np.ones(len(candidates), dtype=bool)
            if existing_sensors:
                existing_positions = np.array([s.position for s in existing_sensors])
                
                for i, pos in enumerate(candidates):
                    # Calculate distances to existing sensors
                    distances = np.linalg.norm(existing_positions - pos, axis=1)
                    
                    # Check if any distance is below relaxed minimum
                    if np.any(distances < relaxed_distance):
                        valid_mask[i] = False
        
        # If still no valid candidates, return the one with highest info gain
        if not np.any(valid_mask):
            print("Warning: Could not find candidates satisfying distance constraint.")
            valid_mask = np.ones(len(candidates), dtype=bool)
        
        # Filter candidates and their info gains
        valid_candidates = candidates[valid_mask]
        valid_info_gains = info_gains[valid_mask]
        
        # Get index of best candidate
        best_idx = np.argmax(valid_info_gains)
        best_position = valid_candidates[best_idx]
        best_info_gain = valid_info_gains[best_idx]
        
        return best_position, best_info_gain
    
    def _create_sensor_at_position(self, position: np.ndarray) -> Sensor:
        """
        Create a Sensor object at the given position
        
        Args:
            position: 3D position for the sensor
            
        Returns:
            Configured Sensor object
        """
        # Calculate direction towards scene center
        direction = self.scene_center - position
        direction = direction / np.linalg.norm(direction)
        
        # Create sensor
        sensor = Sensor(
            position=position,
            target_direction=direction,
            horizontal_fov=self.sensor_params['horizontal_fov'],
            vertical_fov=self.sensor_params['vertical_fov'],
            step_angle=self.sensor_params['step_angle'],
            min_distance=self.sensor_params['min_distance'],
            max_distance=self.sensor_params['max_distance']
        )
        
        return sensor
    
    def _check_termination(self, 
                         iteration: int, 
                         info_gain_history: List[float], 
                         candidacy_size: int,
                         max_iterations: int = 10) -> bool:
        """
        Check if optimization should terminate
        
        Args:
            iteration: Current iteration number
            info_gain_history: History of best information gain values
            candidacy_size: Current size of candidacy region
            max_iterations: Maximum number of iterations
            
        Returns:
            True if optimization should terminate, False otherwise
        """
        # Check maximum iterations
        if iteration >= max_iterations:
            print(f"Reached maximum iterations ({max_iterations})")
            return True
        
        # Check information gain plateau
        if len(info_gain_history) >= 3:
            # Check if info gain improvement is less than 1%
            if (info_gain_history[-1] - info_gain_history[-2]) / info_gain_history[-2] < 0.01:
                if (info_gain_history[-2] - info_gain_history[-3]) / info_gain_history[-3] < 0.01:
                    print("Information gain plateau detected")
                    return True
        
        # Check candidacy region size
        if candidacy_size < 10:
            print(f"Candidacy region too small ({candidacy_size} candidates)")
            return True
        
        return False
    
    def _generate_info_gain_heatmap(self, 
                                  positions: np.ndarray, 
                                  info_gains: np.ndarray,
                                  iteration: int) -> None:
        """
        Generate and save a heatmap visualization of information gain
        
        Args:
            positions: Candidate positions
            info_gains: Information gain values for positions
            iteration: Current iteration number
        """
        # Create directory for visualizations if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Create scatter plot of positions with info gain as color
        plt.figure(figsize=(10, 8))
        
        # Extract x and y coordinates for 2D plot
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Create colormap for information gain
        scatter = plt.scatter(x, y, c=info_gains, cmap='viridis', 
                             s=50, alpha=0.8, edgecolors='k')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Information Gain')
        
        # Add existing sensor positions if any
        if self.sensors:
            sensor_positions = np.array([s.position for s in self.sensors])
            plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
                       c='red', s=100, marker='*', label='Placed Sensors')
            
        # Add scene bounds
        plt.xlim(self.bounds[0, 0], self.bounds[1, 0])
        plt.ylim(self.bounds[0, 1], self.bounds[1, 1])
        
        # Add title and labels
        plt.title(f'Information Gain Map - Iteration {iteration}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(f'visualizations/info_gain_map_iter_{iteration}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def optimize(self, 
                max_iterations: int = 10, 
                max_sensors: int = None, 
                visualize: bool = True) -> List[Sensor]:
        """
        Run the optimization process to find optimal sensor placement
        
        Args:
            max_iterations: Maximum number of iterations
            max_sensors: Maximum number of sensors to place (overrides instance value if provided)
            visualize: Whether to generate visualizations
            
        Returns:
            List of optimally placed sensors
        """
        if max_sensors is not None:
            self.max_sensors = max_sensors
            
        print(f"Starting sensor placement optimization for up to {self.max_sensors} sensors")
        
        # Reset optimization history
        self.optimization_history = {
            'info_gain': [],
            'candidacy_size': [],
            'sensor_positions': []
        }
        
        # Reset sensor list
        self.sensors = []
        
        # Training dataset for neural network
        train_positions = []
        train_info_gains = []
        
        # Generate initial candidacy region
        candidates = self._generate_initial_candidacy_region()
        print(f"Initial candidacy region: {len(candidates)} positions")
        
        # Main optimization loop to place each sensor
        for sensor_idx in range(self.max_sensors):
            print(f"\n--- Placing sensor {sensor_idx + 1} ---")
            
            # Refinement iterations for each sensor
            info_gain_history = []
            
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration + 1}")
                print(f"Candidacy region: {len(candidates)} positions")
                self.optimization_history['candidacy_size'].append(len(candidates))
                
                if iteration > 0 and len(train_positions) >= 20:
                    # Train neural network on collected data
                    print("Training neural network...")
                    self._train_neural_network(
                        np.array(train_positions),
                        np.array(train_info_gains),
                        epochs=100
                    )
                    
                    # Use neural network to predict information gain for all candidates
                    print("Evaluating candidates with neural network...")
                    predicted_info_gains = self._evaluate_candidates(candidates, self.sensors)
                    
                    # Select top candidates for actual evaluation
                    top_k = min(20, len(candidates))
                    top_indices = np.argsort(predicted_info_gains)[-top_k:]
                    top_candidates = candidates[top_indices]
                    
                    # Calculate actual information gain for top candidates
                    print(f"Calculating actual information gain for top {top_k} candidates...")
                    sample_info_gains = self._calculate_actual_info_gain(
                        top_candidates, self.sensors
                    )
                    
                    # Create full info gain array using predictions for non-evaluated candidates
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
                    self._generate_info_gain_heatmap(candidates, info_gains, iteration)
                
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
                
            # Generate new candidacy region for next sensor
            candidates = self._generate_initial_candidacy_region()
        
        print("\nSensor placement optimization complete!")
        print(f"Placed {len(self.sensors)} sensors")
        
        for i, sensor in enumerate(self.sensors):
            print(f"Sensor {i+1}: Position={sensor.position}, Direction={sensor.target_direction}")
        
        return self.sensors
    
    def get_optimization_history(self) -> Dict:
        """Get the optimization history"""
        return self.optimization_history
    
    def visualize_results(self, output_path: str = None) -> None:
        """
        Visualize the optimization results
        
        Args:
            output_path: Path to save visualization
        """
        if not self.optimization_history['info_gain']:
            print("No optimization results to visualize")
            return
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot information gain over sensors
        sensor_indices = np.arange(1, len(self.optimization_history['info_gain']) + 1)
        ax1.plot(sensor_indices, self.optimization_history['info_gain'], 'o-', linewidth=2)
        ax1.set_xlabel('Number of Sensors')
        ax1.set_ylabel('Information Gain')
        ax1.set_title('Information Gain vs. Number of Sensors')
        ax1.grid(True)
        
        # Plot final sensor positions
        sensor_positions = np.array([s.position for s in self.sensors])
        ax2.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
                   c='red', s=100, marker='*', label='Sensors')
        
        # Add scene bounds
        ax2.set_xlim(self.bounds[0, 0], self.bounds[1, 0])
        ax2.set_ylim(self.bounds[0, 1], self.bounds[1, 1])
        
        # Add title and labels
        ax2.set_title('Optimized Sensor Positions')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add sensor indices as labels
        for i, pos in enumerate(sensor_positions):
            ax2.annotate(f"{i+1}", (pos[0], pos[1]), 
                        fontsize=12, weight='bold',
                        xytext=(10, 10), textcoords='offset points')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()

def main():
    """Example usage of the SensorOptimizer class"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize LiDAR sensor placement")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.obj, .stl, etc.)")
    parser.add_argument("--height", type=float, default=5.0, 
                       help="Height of the candidacy plane for sensors")
    parser.add_argument("--max-sensors", type=int, default=3, 
                       help="Maximum number of sensors to place")
    parser.add_argument("--min-distance", type=float, default=1.0, 
                       help="Minimum distance between sensors")
    parser.add_argument("--max-iterations", type=int, default=5, 
                       help="Maximum iterations for each sensor placement")
    parser.add_argument("--output", default="optimization_results.png", 
                       help="Path to save visualization")
    parser.add_argument("--no-visualize", action="store_true", 
                       help="Disable visualization during optimization")
    
    args = parser.parse_args()
    
    try:
        # Load mesh
        print(f"Loading mesh from {args.mesh}...")
        mesh = trimesh.load(args.mesh, force='mesh')
        
        # Create optimizer
        optimizer = SensorOptimizer(
            mesh=mesh,
            height=args.height,
            min_sensor_distance=args.min_distance,
            max_sensors=args.max_sensors
        )
        
        # Run optimization
        sensors = optimizer.optimize(
            max_iterations=args.max_iterations,
            visualize=not args.no_visualize
        )
        
        # Visualize results
        optimizer.visualize_results(args.output)
        
        print(f"Optimization complete. Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()