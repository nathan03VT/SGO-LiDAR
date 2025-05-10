import os
import argparse
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('.')
from models.spatial_info_gain_net import SpatialInfoGainNet
from Sensor import Sensor

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with SpatialInfoGainNet')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to input CAD model')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--grid_size', type=int, default=64, help='Resolution of information gain map')
    parser.add_argument('--z_height', type=float, default=5.0, help='Height of sensor plane')
    parser.add_argument('--num_sensors', type=int, default=3, help='Number of sensors to place')
    parser.add_argument('--min_distance', type=float, default=1.0, help='Minimum distance between sensors')
    return parser.parse_args()

def generate_info_gain_map(model, mesh_file, grid_size, z_height, device='cuda'):
    """
    Generate information gain map for a 3D model
    
    Args:
        model: Trained SpatialInfoGainNet model
        mesh_file: Path to CAD model file
        grid_size: Resolution of information gain map
        z_height: Height of sensor plane
        device: Device to run inference on
        
    Returns:
        info_gain_map: Generated information gain map
        bounds: Scene bounds
    """
    # Load mesh
    mesh = trimesh.load(mesh_file, force='mesh')
    
    # Get scene bounds
    bounds = mesh.bounds
    center = np.mean(bounds, axis=0)
    scale = np.max(bounds[1] - bounds[0])
    
    # Sample point cloud
    point_cloud = mesh.sample(10000)
    
    # Normalize to unit cube
    normalized_points = (point_cloud - center) / scale
    
    # Convert to tensor
    points_tensor = torch.FloatTensor(normalized_points).unsqueeze(0).to(device)
    
    # Generate information gain map
    model.eval()
    with torch.no_grad():
        info_gain_map = model(points_tensor).squeeze().cpu().numpy()
    
    return info_gain_map, bounds

def select_sensor_positions(info_gain_map, bounds, z_height, num_sensors, min_distance, grid_size):
    """
    Select optimal sensor positions based on information gain map
    
    Args:
        info_gain_map: Information gain map
        bounds: Scene bounds
        z_height: Height of sensor plane
        num_sensors: Number of sensors to place
        min_distance: Minimum distance between sensors in world units
        grid_size: Resolution of information gain map
        
    Returns:
        sensors: List of selected Sensor objects
        sensor_positions: List of selected positions
    """
    # Map grid coordinates to world coordinates
    x = np.linspace(bounds[0, 0], bounds[1, 0], grid_size)
    y = np.linspace(bounds[0, 1], bounds[1, 1], grid_size)
    
    # Calculate minimum grid distance from world distance
    grid_cell_size_x = (bounds[1, 0] - bounds[0, 0]) / grid_size
    grid_cell_size_y = (bounds[1, 1] - bounds[0, 1]) / grid_size
    grid_min_distance = min_distance / min(grid_cell_size_x, grid_cell_size_y)
    
    # Greedy selection of sensor positions
    selected_positions = []
    selected_grid_positions = []
    flat_info_gain = info_gain_map.flatten()
    
    # Select position with highest information gain first
    max_idx = np.argmax(flat_info_gain)
    grid_i, grid_j = np.unravel_index(max_idx, info_gain_map.shape)
    selected_grid_positions.append((grid_i, grid_j))
    world_pos = np.array([x[grid_j], y[grid_i], z_height])  # Note the swap of i,j for x,y
    selected_positions.append(world_pos)
    
    # Compute scene center for sensor direction
    scene_center = np.mean(bounds, axis=0)
    
    # Create sensors list
    sensors = []
    
    # Create first sensor
    direction = scene_center - world_pos
    direction = direction / np.linalg.norm(direction)
    sensors.append(Sensor(
        position=world_pos,
        target_direction=direction,
        horizontal_fov=90.0,
        vertical_fov=60.0
    ))
    
    # Create a mask of valid positions
    valid_mask = np.ones_like(flat_info_gain, dtype=bool)
    
    # Mark the selected position and its neighborhood as invalid
    for i in range(grid_size):
        for j in range(grid_size):
            for si, sj in selected_grid_positions:
                # Calculate grid distance
                grid_dist = np.sqrt((i - si)**2 + (j - sj)**2)
                if grid_dist < grid_min_distance:
                    valid_mask[i * grid_size + j] = False
    
    # Select remaining sensors
    for _ in range(1, num_sensors):
        # Find position with highest information gain among valid positions
        valid_info_gain = flat_info_gain.copy()
        valid_info_gain[~valid_mask] = -1
        
        if np.max(valid_info_gain) < 0:
            print(f"Warning: Couldn't place all {num_sensors} sensors with minimum distance {min_distance}")
            break
            
        max_idx = np.argmax(valid_info_gain)
        grid_i, grid_j = np.unravel_index(max_idx, info_gain_map.shape)
        selected_grid_positions.append((grid_i, grid_j))
        world_pos = np.array([x[grid_j], y[grid_i], z_height])
        selected_positions.append(world_pos)
        
        # Create sensor
        direction = scene_center - world_pos
        direction = direction / np.linalg.norm(direction)
        sensors.append(Sensor(
            position=world_pos,
            target_direction=direction,
            horizontal_fov=90.0,
            vertical_fov=60.0
        ))
        
        # Update valid mask
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate grid distance
                grid_dist = np.sqrt((i - grid_i)**2 + (j - grid_j)**2)
                if grid_dist < grid_min_distance:
                    valid_mask[i * grid_size + j] = False
    
    return sensors, selected_positions

def visualize_results(info_gain_map, bounds, selected_positions, output_path):
    """
    Visualize information gain map and selected sensor positions
    
    Args:
        info_gain_map: Information gain map
        bounds: Scene bounds
        selected_positions: List of selected sensor positions
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create extent for proper world coordinates in plot
    extent = [bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1]]
    
    # Plot information gain map
    im = ax.imshow(info_gain_map, cmap='viridis', origin='lower', extent=extent)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Information Gain')
    
    # Plot selected sensor positions
    sensor_x = [pos[0] for pos in selected_positions]
    sensor_y = [pos[1] for pos in selected_positions]
    ax.scatter(sensor_x, sensor_y, c='red', s=80, marker='*', edgecolors='black', linewidths=1, label='Sensors')
    
    # Add sensor indices
    for i, (x, y) in enumerate(zip(sensor_x, sensor_y)):
        ax.text(x, y, f" {i+1}", fontsize=12, weight='bold')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Information Gain Map with Optimal Sensor Positions')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

def save_sensor_positions(sensors, output_path):
    """
    Save sensor positions and directions to CSV file
    
    Args:
        sensors: List of Sensor objects
        output_path: Path to save CSV file
    """
    with open(output_path, 'w') as f:
        f.write("sensor_id,x,y,z,dir_x,dir_y,dir_z\n")
        for i, sensor in enumerate(sensors):
            pos = sensor.position
            dir = sensor.target_direction
            f.write(f"{i+1},{pos[0]},{pos[1]},{pos[2]},{dir[0]},{dir[1]},{dir[2]}\n")
    
    print(f"Sensor positions saved to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = SpatialInfoGainNet(grid_size=args.grid_size, z_height=args.z_height)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f'Model loaded from {args.model_path}')
    
    # Generate information gain map
    print('Generating information gain map...')
    info_gain_map, bounds = generate_info_gain_map(
        model, args.mesh_file, args.grid_size, args.z_height, device
    )
    
    # Select sensor positions
    print(f'Selecting {args.num_sensors} sensor positions...')
    sensors, selected_positions = select_sensor_positions(
        info_gain_map, bounds, args.z_height, args.num_sensors, args.min_distance, args.grid_size
    )
    
    # Visualize results
    print('Visualizing results...')
    output_image_path = os.path.join(args.output_dir, 'info_gain_map.png')
    visualize_results(info_gain_map, bounds, selected_positions, output_image_path)
    
    # Save sensor positions
    output_csv_path = os.path.join(args.output_dir, 'sensor_positions.csv')
    save_sensor_positions(sensors, output_csv_path)
    
    print('Done!')

if __name__ == '__main__':
    main()