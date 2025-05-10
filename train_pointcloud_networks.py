#!/usr/bin/env python
"""
Training script for the PointCloudInfoGainNet.
This script demonstrates how to train the network on synthetic data
generated from LiDAR simulations.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import trimesh
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

from LidarSimulation import LiDARSimulator
from Sensor import Sensor
from Material import Material
from PointCloudInfoGainNet import (
    PointCloudInfoGainNet,
    prepare_partial_cloud,
    prepare_env_cloud,
    prepare_sensor_pose,
    create_coordinate_grid
)
from SensorOptimizer import LidarInfoGainCalculator


class LiDARTrainingDataset(Dataset):
    """Dataset for training the PointCloudInfoGainNet."""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def generate_training_data(mesh_path: str, 
                         num_samples: int = 1000,
                         max_sensors: int = 3,
                         device: str = 'cuda'):
    """
    Generate synthetic training data for the point cloud network.
    
    Args:
        mesh_path: Path to the CAD model
        num_samples: Number of training samples to generate
        max_sensors: Maximum number of sensors per sample
        device: Device to run on
        
    Returns:
        training_samples: List of training samples
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    material = Material(albedo=0.7, metallic=0.0, roughness=0.5, ambient=0.2)
    
    # Prepare static data
    env_cloud = prepare_env_cloud(mesh).to(device)
    bounds = mesh.bounds
    x_grid, y_grid = create_coordinate_grid(bounds)
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)
    
    # Info gain calculator
    info_gain_calc = LidarInfoGainCalculator(mesh)
    
    training_samples = []
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Generating sample {i}/{num_samples}")
        
        # Generate random number of sensors for this sample
        num_existing_sensors = np.random.randint(0, max_sensors)
        existing_sensors = []
        
        # Generate random existing sensors
        for _ in range(num_existing_sensors):
            # Random position within scene bounds
            position = np.random.uniform(
                low=[bounds[0, 0], bounds[0, 1], 0.0],
                high=[bounds[1, 0], bounds[1, 1], 5.0]
            )
            
            # Random direction toward scene center
            scene_center = np.mean(bounds, axis=0)
            direction = scene_center - position
            direction = direction / np.linalg.norm(direction)
            
            sensor = Sensor(
                position=position,
                target_direction=direction,
                horizontal_fov=90.0,
                vertical_fov=60.0,
                step_angle=0.5
            )
            existing_sensors.append(sensor)
        
        # Get partial cloud from existing sensors
        if existing_sensors:
            simulator = LiDARSimulator(
                mesh=mesh,
                sensors=existing_sensors,
                material=material
            )
            results = simulator.simulate(add_noise=False)
            partial_cloud = prepare_partial_cloud(results).to(device)
        else:
            partial_cloud = torch.zeros((1, 1000, 3), device=device)
        
        # Generate target heatmap
        heatmap_shape = (x_grid.shape[0], x_grid.shape[1])
        target_heatmap = torch.zeros(heatmap_shape, device=device)
        
        # Calculate information gain for random positions
        sample_positions = []
        sample_info_gains = []
        
        # Sample positions across the grid
        for h in range(0, heatmap_shape[0], max(1, heatmap_shape[0] // 32)):
            for w in range(0, heatmap_shape[1], max(1, heatmap_shape[1] // 32)):
                # Convert grid coordinates to world coordinates
                world_x = x_grid[h, w].item()
                world_y = y_grid[h, w].item()
                world_z = 5.0  # Fixed height
                
                pos = np.array([world_x, world_y, world_z])
                
                # Create temporary sensor at this position
                direction = scene_center - pos
                direction = direction / np.linalg.norm(direction)
                
                temp_sensor = Sensor(
                    position=pos,
                    target_direction=direction,
                    horizontal_fov=90.0,
                    vertical_fov=60.0,
                    step_angle=0.5
                )
                
                # Calculate information gain
                all_sensors = existing_sensors + [temp_sensor]
                simulator = LiDARSimulator(
                    mesh=mesh,
                    sensors=all_sensors,
                    material=material
                )
                
                info_gain = info_gain_calc.calculate_information_gain(simulator, all_sensors)
                
                # Set value in target heatmap
                target_heatmap[h, w] = info_gain
                
                sample_positions.append(pos)
                sample_info_gains.append(info_gain)
        
        # Store training sample
        training_samples.append({
            'partial_cloud': partial_cloud,
            'env_cloud': env_cloud,
            'target_heatmap': target_heatmap.unsqueeze(0),  # Add batch dimension
            'x_grid': x_grid,
            'y_grid': y_grid
        })
    
    return training_samples


def train_network(training_samples: list,
                 output_dir: str = 'checkpoints',
                 epochs: int = 10,
                 batch_size: int = 4,
                 learning_rate: float = 0.001,
                 device: str = 'cuda'):
    """
    Train the PointCloudInfoGainNet.
    
    Args:
        training_samples: List of training samples
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to run on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = LiDARTrainingDataset(training_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize network and optimizer
    network = PointCloudInfoGainNet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        network.train()
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Prepare batch data
            batch_partial_clouds = []
            batch_env_clouds = []
            batch_target_heatmaps = []
            
            for sample in batch:
                batch_partial_clouds.append(sample['partial_cloud'])
                batch_env_clouds.append(sample['env_cloud'])
                batch_target_heatmaps.append(sample['target_heatmap'])
            
            # Stack batches
            partial_clouds = torch.cat(batch_partial_clouds, dim=0)
            env_clouds = torch.cat(batch_env_clouds, dim=0)
            target_heatmaps = torch.cat(batch_target_heatmaps, dim=0)
            
            # Get grids from first sample (all should be the same)
            x_grid = batch[0]['x_grid']
            y_grid = batch[0]['y_grid']
            
            # Dummy sensor pose and EGVS for training
            batch_poses = torch.zeros((len(batch), 6), device=device)
            batch_egvs = torch.zeros((len(batch), 1), device=device)
            
            # Forward pass
            predicted_heatmaps = network(
                partial_clouds,
                env_clouds,
                batch_poses,
                batch_egvs,
                x_grid,
                y_grid
            )
            
            # Calculate loss
            loss = criterion(predicted_heatmaps, target_heatmaps)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        # Save epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs} complete, Average Loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"pointcloud_net_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "pointcloud_net_final.pth")
    torch.save({
        'model_state_dict': network.state_dict(),
        'losses': losses,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    return network


def main():
    parser = argparse.ArgumentParser(description="Train PointCloudInfoGainNet")
    parser.add_argument("--mesh", required=True, help="Path to CAD model")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    
    args = parser.parse_args()
    
    print(f"Training PointCloudInfoGainNet on {args.device}")
    print(f"Mesh: {args.mesh}")
    print(f"Samples: {args.num_samples}")
    print(f"Epochs: {args.epochs}")
    
    # Generate training data
    print("Generating training data...")
    training_samples = generate_training_data(
        mesh_path=args.mesh,
        num_samples=args.num_samples,
        device=args.device
    )
    
    # Train network
    print("Training network...")
    network = train_network(
        training_samples=training_samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()