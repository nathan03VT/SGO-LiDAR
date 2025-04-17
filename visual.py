import open3d as o3d
import argparse

def view_obj(file_path):
    # Load the .obj file
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    if not mesh:
        print("Failed to load the .obj file. Check the file path.")
        return
    
    mesh.compute_vertex_normals()
    
    # Visualize using Open3D scene previewer
    o3d.visualization.draw_geometries([mesh],
                                      window_name="Open3D OBJ Viewer",
                                      mesh_show_wireframe=True)

def visualize_pcd(file_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Check if the file was loaded successfully
    if not pcd:
        print("Error loading point cloud. Please check the file path.")
        return
    
    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd], window_name="PCD Viewer",
                                       width=800, height=600,
                                       left=50, top=50,
                                       point_show_normal=False,
                                       mesh_show_wireframe=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 3D model (.obj or .pcd file) using Open3D.")
    parser.add_argument("file_path", type=str, help="Path to the 3D file")
    parser.add_argument("--type", choices=["obj", "pcd"], required=True, help="Specify the file type: 'obj' for mesh, 'pcd' for point cloud")
    args = parser.parse_args()
    
    if args.type == "obj":
        view_obj(args.file_path)
    elif args.type == "pcd":
        visualize_pcd(args.file_path)
