import torch
import numpy as np
import trimesh
from typing import Tuple, Optional

class RayMeshIntersector:
    def __init__(self, mesh: trimesh.Trimesh):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert mesh data to PyTorch tensors on GPU
        self.vertices = torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32)
        self.faces = torch.tensor(mesh.faces, device=self.device, dtype=torch.int64)
        self.face_normals = torch.tensor(mesh.face_normals, device=self.device, dtype=torch.float32)
        
    def compute_intersections(self, 
                            origins: torch.Tensor, 
                            directions: torch.Tensor,
                            min_distance: float = 0.0001,
                            max_distance: float = 1000.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ray-mesh intersections using Möller–Trumbore algorithm
        
        Args:
            origins: (N, 3) tensor of ray origins
            directions: (N, 3) tensor of ray directions (normalized)
            min_distance: minimum valid intersection distance
            max_distance: maximum valid intersection distance
            
        Returns:
            points: (M, 3) tensor of intersection points
            face_indices: (M,) tensor of intersected face indices
            ray_indices: (M,) tensor of ray indices that hit
        """
        N_rays = origins.shape[0]
        N_faces = self.faces.shape[0]
        
        # Get face vertices (N_faces, 3, 3)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]] 
        v2 = self.vertices[self.faces[:, 2]]
        
        # Compute edge vectors
        edge1 = v1 - v0  # (N_faces, 3)
        edge2 = v2 - v0  # (N_faces, 3)
        
        # Expand rays for vectorized computation
        dirs = directions.unsqueeze(1).expand(-1, N_faces, 3)  # (N_rays, N_faces, 3)
        origs = origins.unsqueeze(1).expand(-1, N_faces, 3)    # (N_rays, N_faces, 3)
        
        # Compute determinant
        h = torch.cross(dirs, edge2.unsqueeze(0), dim=2)  # (N_rays, N_faces, 3)
        det = torch.sum(edge1.unsqueeze(0) * h, dim=2)    # (N_rays, N_faces)
        
        # Find valid intersections (non-parallel rays)
        valid_det = torch.abs(det) > 1e-8
        
        # Compute distance from v0 to ray origin
        s = origs - v0.unsqueeze(0)  # (N_rays, N_faces, 3)
        
        # Compute u parameter
        u = torch.sum(s * h, dim=2) / (det + 1e-8)  # (N_rays, N_faces)
        
        # Compute v parameter
        q = torch.cross(s, edge1.unsqueeze(0), dim=2)  # (N_rays, N_faces, 3)
        v = torch.sum(dirs * q, dim=2) / (det + 1e-8)  # (N_rays, N_faces)
        
        # Compute t (distance along ray)
        t = torch.sum(edge2.unsqueeze(0) * q, dim=2) / (det + 1e-8)  # (N_rays, N_faces)
        
        # Find valid intersections
        valid = (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & valid_det & \
                (t >= min_distance) & (t <= max_distance)
                
        # Get closest intersection per ray
        distances = torch.where(valid, t, torch.tensor(float('inf'), device=self.device, dtype=torch.float32))
        closest_face_idx = torch.argmin(distances, dim=1)  # (N_rays,)
        min_distances = torch.min(distances, dim=1)[0]     # (N_rays,)
        
        # Filter rays that hit
        hit_mask = min_distances < float('inf')
        ray_indices = torch.nonzero(hit_mask).squeeze(1)
        face_indices = closest_face_idx[hit_mask]
        
        # Compute intersection points
        points = origins[hit_mask] + directions[hit_mask] * min_distances[hit_mask].unsqueeze(1)
        
        return points, face_indices, ray_indices
        
    def compute_normals(self, 
                       points: torch.Tensor,
                       face_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute interpolated normals at intersection points
        
        Args:
            points: (N, 3) tensor of intersection points
            face_indices: (N,) tensor of intersected face indices
            
        Returns:
            normals: (N, 3) tensor of interpolated normals
        """
        # Get vertices of intersected faces
        v0 = self.vertices[self.faces[face_indices, 0]]
        v1 = self.vertices[self.faces[face_indices, 1]]
        v2 = self.vertices[self.faces[face_indices, 2]]
        
        # Compute barycentric coordinates
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Vector from v0 to intersection point
        c = points - v0
        
        # Compute areas using cross products
        va = torch.cross(edge1, edge2, dim=1)
        v1 = torch.cross(edge1, c, dim=1)
        v2 = torch.cross(c, edge2, dim=1)
        
        # Normalize to get barycentric coordinates
        area = torch.norm(va, dim=1, keepdim=True)
        w = torch.stack([
            torch.sum(v1 * va, dim=1) / (area.squeeze(1) + 1e-8),
            torch.sum(v2 * va, dim=1) / (area.squeeze(1) + 1e-8)
        ], dim=1)
        w = torch.cat([w, 1 - w.sum(dim=1, keepdim=True)], dim=1)
        
        # Get face normals
        face_normals = self.face_normals[face_indices]
        
        # For this implementation we'll just use face normals
        # For smoother results, you could interpolate vertex normals using barycentric coords
        return face_normals