import argparse
import trimesh
import pathlib

def main():
    parser = argparse.ArgumentParser(description="Visualize a .stp (STEP) file using trimesh.")
    parser.add_argument('file', type=str, help="Path to the .stp file.")
    args = parser.parse_args()

    # Normalize the file path for any OS
    file_path = pathlib.Path(args.file)

    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return

    # Load the mesh
    try:
        mesh = trimesh.load(str(file_path), force='mesh')
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return

    # Check if the mesh is valid
    if mesh.is_empty:
        print(f"The file {file_path} didn't contain a valid mesh.")
        return

    # Create a simple gray material
    material = trimesh.visual.material.SimpleMaterial(color=[200, 200, 200, 255])  # light gray

    # Apply material to the mesh
    if hasattr(mesh, 'visual'):
        mesh.visual = trimesh.visual.TextureVisuals(material=material)

    # Show the mesh
    mesh.show()

if __name__ == "__main__":
    main()
