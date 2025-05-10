import torch
import numpy as np
import os

from SurrogateModel import create_surrogate_model, create_attention_surrogate_model
from OptimizerLoop import SurrogateOptimizer
from SimulationInterface import SimulationInterface

# Parameter definitions
NUM_SENSORS = 1
SCENE_PATH = "./structured_scene.obj"
DATASET_PATH = "./results/optimization_dataset.pkl"

# Parameter bounds for a single sensor (position, direction, FOV, etc.)
# Flattened list of (min, max) tuples
PARAM_BOUNDS = [
    (-10.0, 10.0),  # position_x
    (-10.0, 10.0),  # position_y
    (0.0, 5.0),     # position_z
    (-1.0, 1.0),    # direction_x
    (-1.0, 1.0),    # direction_y
    (-1.0, 1.0),    # direction_z
    (30.0, 120.0),  # horizontal_fov
    (20.0, 90.0),   # vertical_fov
    (0.05, 0.5)     # step_angle
]

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create simulation interface
    simulator = SimulationInterface(mesh_path=SCENE_PATH, param_bounds=PARAM_BOUNDS)
    
    # Calculate input dimension based on parameters and number of sensors
    input_dim = len(PARAM_BOUNDS) * NUM_SENSORS
    
    # Choose model type
    if NUM_SENSORS > 1:
        # For multiple sensors, use attention-based model
        surrogate_model = create_attention_surrogate_model(
            num_sensors=NUM_SENSORS,
            features_per_sensor=len(PARAM_BOUNDS) // NUM_SENSORS
        )
    else:
        # For single sensor, use standard model
        surrogate_model = create_surrogate_model(input_dim=input_dim)
    
    # Create optimizer
    optimizer = SurrogateOptimizer(
        surrogate_model=surrogate_model,
        simulator=simulator,
        dataset_path=DATASET_PATH,
        device=device
    )
    
    # Check if we have existing data
    if len(optimizer.configs) == 0:
        print("Initializing dataset with random configurations...")
        optimizer.initialize_dataset(num_samples=5, bounds=PARAM_BOUNDS)
    
    # Train initial surrogate model
    print("Training surrogate model...")
    optimizer.train_surrogate()
    
    # Run optimization
    print("Running surrogate-guided optimization...")
    best_config, best_score = optimizer.optimize(
        iterations=20,
        candidates_per_iter=100,
        evals_per_iter=5
    )
    
    # Print results
    print(f"Optimization complete!")
    print(f"Best configuration found: {best_config}")
    print(f"Best PCCS score: {best_score:.4f}")
    
    # Save the best configuration
    np.save("best_sensor_configuration.npy", best_config)
    
    # Create and visualize the best point cloud
    best_sensors = simulator.decode_config(best_config)
    simulator.visualize_configuration(best_sensors)

if __name__ == "__main__":
    main()