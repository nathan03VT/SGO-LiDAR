#!/usr/bin/env python
"""
Optimize LiDAR sensor placement using the new point cloud neural network.
This script uses the PointCloudInfoGainNet to predict information gain directly
from 3D point cloud data without requiring BEV image conversion.
"""

import os
import sys
import numpy as np
import trimesh
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from Sensor import Sensor
from LidarSimulation import LiDARSimulator
from Material import Material
from ModifiedSensorOptimizer import SensorOptimizer
from ModifiedSimulationInterface import SimulationInterface, PointCloudSimulationInterface

def visualize_final_results(optimizer, output_dir, timestamp):
    """
    Create comprehensive visualizations of the final optimization results
    
    Args:
        optimizer: Configured SensorOptimizer object
        output_dir: Directory to save visualization files
        timestamp: Timestamp string for filenames
    """
    print("Creating final visualizations...")
    
    # Generate final heatmap visualization
    try:
        optimizer._generate_heatmap_visualization(99)  # Use 99 to indicate final iteration
        
        # Get the last generated heatmap
        heatmap_path = os.path.join('visualizations', 'pointcloud_heatmap_iter_99.png')
        if os.path.exists(heatmap_path):
            # Copy to output directory with better name
            import shutil
            final_heatmap_path = os.path.join(output_dir, f'final_heatmap_{timestamp}.png')
            shutil.copy(heatmap_path, final_heatmap_path)
            print(f"Final heatmap saved to {final_heatmap_path}")
    except Exception as e:
        print(f"Error generating final heatmap: {e}")
    
    # Create summary visualization
    try:
        history = optimizer.get_optimization_history()
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Information gain over iterations
        ax1 = fig.add_subplot(gs[0, :])
        sensor_indices = np.arange(1, len(history['info_gain']) + 1)
        ax1.plot(sensor_indices, history['info_gain'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sensor Number')
        ax1.set_ylabel('Information Gain')
        ax1.set_title('Information Gain per Sensor')
        ax1.grid(True, alpha=0.3)
        
        # Candidacy region size over iterations
        ax2 = fig.add_subplot(gs[1, 0])
        iteration_indices = range(len(history['candidacy_size']))
        ax2.plot(iteration_indices, history['candidacy_size'], 'o-', color='green', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Candidacy Region Size')
        ax2.set_title('Search Space Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3D sensor placement visualization
        ax3 = fig.add_subplot(gs[1, 1], projection='3d')
        sensor_positions = np.array(history['sensor_positions'])
        
        # Plot sensors
        scatter = ax3.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2],
                             c=range(len(sensor_positions)), cmap='viridis', s=200, marker='*')
        
        # Add sensor labels
        for i, pos in enumerate(sensor_positions):
            ax3.text(pos[0], pos[1], pos[2], f"{i+1}", fontsize=12, weight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, pad=0.1)
        cbar.set_label('Sensor Placement Order')
        
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.set_zlabel('Z Position')
        ax3.set_title('3D Sensor Placement')
        
        plt.tight_layout()
        
        # Save summary visualization
        summary_path = os.path.join(output_dir, f'optimization_summary_{timestamp}.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Optimization summary saved to {summary_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating summary visualization: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize LiDAR sensor placement using point cloud network")
    parser.add_argument("--mesh", required=True, 
                      help="Path to mesh file (.obj, .stl, etc.)")
    parser.add_argument("--height", type=float, default=5.0, 
                      help="Height of the candidacy plane for sensors")
    parser.add_argument("--max-sensors", type=int, default=3, 
                      help="Maximum number of sensors to place")
    parser.add_argument("--min-distance", type=float, default=1.0, 
                      help="Minimum distance between sensors")
    parser.add_argument("--max-iterations", type=int, default=5, 
                      help="Maximum iterations for each sensor placement")
    parser.add_argument("--output-dir", default="optimization_results", 
                      help="Directory to save results")
    parser.add_argument("--no-visualize", action="store_true", 
                      help="Disable visualization during optimization")
    parser.add_argument("--device", default=None,
                      help="Device to run neural network on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load mesh
        print(f"Loading mesh from {args.mesh}...")
        mesh = trimesh.load(args.mesh, force='mesh')
        
        # Create optimizer with point cloud network
        optimizer = SensorOptimizer(
            mesh=mesh,
            height=args.height,
            min_sensor_distance=args.min_distance,
            max_sensors=args.max_sensors,
            device=args.device
        )
        
        # Run optimization
        start_time = datetime.now()
        print(f"Starting optimization at {start_time}")
        print("Using point cloud neural network for information gain prediction...")
        
        sensors = optimizer.optimize(
            max_iterations=args.max_iterations,
            visualize=not args.no_visualize
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Optimization completed in {duration}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save sensor positions to CSV file
        csv_path = os.path.join(args.output_dir, f"sensor_positions_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("sensor_id,x,y,z,dir_x,dir_y,dir_z\n")
            for i, sensor in enumerate(sensors):
                pos = sensor.position
                dir = sensor.target_direction
                f.write(f"{i+1},{pos[0]},{pos[1]},{pos[2]},{dir[0]},{dir[1]},{dir[2]}\n")
        
        print(f"Sensor positions saved to {csv_path}")
        
        # Generate final visualizations
        visualize_final_results(optimizer, args.output_dir, timestamp)
        
        # Save optimization history
        history_path = os.path.join(args.output_dir, f"optimization_history_{timestamp}.npz")
        history = optimizer.get_optimization_history()
        np.savez(history_path, 
                info_gain=history['info_gain'],
                candidacy_size=history['candidacy_size'],
                sensor_positions=history['sensor_positions'])
        print(f"Optimization history saved to {history_path}")
        
        # Run a simulation with the optimized sensors to visualize coverage
        material = Material(albedo=0.7, metallic=0.0, roughness=0.5, ambient=0.2)
        simulator = LiDARSimulator(
            mesh=mesh,
            sensors=sensors,
            material=material
        )
        
        print("Running LiDAR simulation to visualize coverage...")
        results = simulator.simulate(add_noise=False)
        
        # Combine all point clouds
        all_points = np.vstack([points for points, _, _ in results if len(points) > 0])
        all_normals = np.vstack([normals for _, normals, _ in results if len(normals) > 0])
        all_reflectivity = np.concatenate([reflectivity for _, _, reflectivity in results if len(reflectivity) > 0])
        
        # Save point cloud
        pcd_path = os.path.join(args.output_dir, f"optimized_coverage_{timestamp}.ply")
        
        try:
            pc = trimesh.PointCloud(
                vertices=all_points,
                colors=np.column_stack([all_reflectivity * 255] * 3).astype(np.uint8)
            )
            pc.export(pcd_path)
            print(f"Coverage point cloud saved to {pcd_path}")
        except Exception as e:
            print(f"Error saving point cloud with trimesh: {e}")
            # Fallback to numpy save
            np.save(os.path.join(args.output_dir, f"coverage_points_{timestamp}.npy"), all_points)
        
        print(f"\nOptimization complete! All results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())