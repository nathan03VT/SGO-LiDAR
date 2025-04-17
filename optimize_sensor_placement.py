#!/usr/bin/env python
"""
Optimize LiDAR sensor placement for a given mesh file.
This script uses the SensorOptimizer to find the optimal placement for multiple LiDAR sensors.
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
from SceneGenerator import SceneGenerator
from Material import Material
from SensorOptimizer import SensorOptimizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize LiDAR sensor placement")
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
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        start_time = datetime.now()
        print(f"Starting optimization at {start_time}")
        
        sensors = optimizer.optimize(
            max_iterations=args.max_iterations,
            visualize=not args.no_visualize
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Optimization completed in {duration}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output_dir, f"sensor_optimization_{timestamp}.png")
        optimizer.visualize_results(results_path)
        
        # Also save sensor positions to CSV file
        csv_path = os.path.join(args.output_dir, f"sensor_positions_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("sensor_id,x,y,z,dir_x,dir_y,dir_z\n")
            for i, sensor in enumerate(sensors):
                pos = sensor.position
                dir = sensor.target_direction
                f.write(f"{i+1},{pos[0]},{pos[1]},{pos[2]},{dir[0]},{dir[1]},{dir[2]}\n")
        
        print(f"Results saved to {results_path} and {csv_path}")
        
        # Run a simulation with the optimized sensors to visualize coverage
        visualize_sensor_coverage(mesh, sensors, args.output_dir, timestamp)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def visualize_sensor_coverage(mesh, sensors, output_dir, timestamp):
    """
    Visualize the coverage of the optimized sensor placement.
    
    Args:
        mesh: The scene mesh
        sensors: List of optimized sensors
        output_dir: Directory to save results
        timestamp: Timestamp string for filenames
    """
    print("Running LiDAR simulation to visualize coverage...")
    
    # Create material
    material = Material(albedo=0.7, metallic=0.0, roughness=0.5, ambient=0.2)
    
    # Create simulator
    simulator = LiDARSimulator(
        mesh=mesh,
        sensors=sensors,
        material=material
    )
    
    # Run simulation
    results = simulator.simulate(add_noise=False)
    
    # Combine all point clouds
    all_points = np.vstack([points for points, _, _ in results if len(points) > 0])
    all_normals = np.vstack([normals for _, normals, _ in results if len(normals) > 0])
    all_reflectivity = np.concatenate([reflectivity for _, _, reflectivity in results if len(reflectivity) > 0])
    
    # Save point cloud
    pcd_path = os.path.join(output_dir, f"optimized_coverage_{timestamp}.ply")
    
    try:
        # Try to save using trimesh
        pc = trimesh.PointCloud(
            vertices=all_points,
            colors=np.column_stack([all_reflectivity * 255] * 3).astype(np.uint8)
        )
        pc.export(pcd_path)
    except Exception as e:
        print(f"Error saving point cloud with trimesh: {e}")
        try:
            # Fallback to numpy save
            np.save(os.path.join(output_dir, f"optimized_coverage_{timestamp}.npy"), all_points)
        except Exception as e2:
            print(f"Error saving point cloud with numpy: {e2}")
    
    print(f"Coverage visualization saved to {pcd_path}")
    
    # Create a coverage visualization image
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Sample points to avoid overcrowding the plot
        max_points = 5000
        if len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            plot_points = all_points[indices]
            plot_reflectivity = all_reflectivity[indices]
        else:
            plot_points = all_points
            plot_reflectivity = all_reflectivity
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot point cloud
        scatter = ax.scatter(
            plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
            c=plot_reflectivity, cmap='viridis',
            s=2, alpha=0.5
        )
        
        # Plot sensor positions
        sensor_positions = np.array([s.position for s in sensors])
        ax.scatter(
            sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2],
            color='red', s=100, marker='*', label='Sensors'
        )
        
        # Add sensor indices as labels
        for i, pos in enumerate(sensor_positions):
            ax.text(pos[0], pos[1], pos[2], f"{i+1}", fontsize=12, color='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Reflectivity')
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title
        ax.set_title(f'LiDAR Coverage with {len(sensors)} Optimized Sensors')
        
        # Add legend
        ax.legend()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"coverage_visualization_{timestamp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Coverage visualization image saved to {fig_path}")
        
    except Exception as e:
        print(f"Error creating coverage visualization: {e}")

if __name__ == "__main__":
    sys.exit(main())