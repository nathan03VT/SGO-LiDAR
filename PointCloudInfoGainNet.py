import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Sample points from point cloud given indices.
    
    Args:
        points: [B, N, C] batch of point clouds
        idx: [B, S] index tensor
        
    Returns:
        indexed_points: [B, S, C] sampled points
    """
    batch_size = points.shape[0]
    device = points.device
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    indexed_points = points[batch_indices, idx, :]
    
    return indexed_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for downsampling point clouds.
    
    Args:
        xyz: [B, N, 3] coordinates of points
        npoint: number of points to sample
        
    Returns:
        idx: [B, npoint] index of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Select random starting points
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        idx[:, i] = farthest
        
        # Get the position of current farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        
        # Compute distance to all points
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        # Update distances
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Get the farthest point for next iteration
        farthest = torch.max(distance, -1)[1]
    
    return idx


def ball_query(xyz: torch.Tensor, new_xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    """
    Ball query for finding neighbors within a radius.
    
    Args:
        xyz: [B, N, 3] original point cloud
        new_xyz: [B, S, 3] query points
        radius: search radius
        nsample: max number of points to sample per query
        
    Returns:
        idx: [B, S, nsample] indices of neighbors
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # Calculate pairwise distances
    new_xyz = new_xyz.unsqueeze(2)  # [B, S, 1, 3]
    xyz = xyz.unsqueeze(1)  # [B, 1, N, 3]
    
    sqrdists = torch.sum((new_xyz - xyz) ** 2, -1)  # [B, S, N]
    
    # Get indices of points within radius
    idx = torch.arange(N, device=device).view(1, 1, N).repeat([B, S, 1])
    within_radius = sqrdists < radius * radius
    
    # Get nsample nearest points within radius
    # First, mask distances outside radius with infinity
    masked_dists = sqrdists.clone()
    masked_dists[~within_radius] = float('inf')
    
    # Sort and get the first nsample indices
    _, indices = torch.sort(masked_dists, dim=-1)
    idx = indices[:, :, :nsample]
    
    # Handle edge case where there are fewer than nsample points within radius
    # Repeat the closest point to fill the requirement
    for i in range(B):
        for j in range(S):
            if torch.all(within_radius[i, j] == False):
                # No points within radius, use the closest point
                closest_idx = torch.argmin(sqrdists[i, j])
                idx[i, j, :] = closest_idx
            else:
                valid_count = torch.sum(within_radius[i, j]).item()
                if valid_count < nsample:
                    # Fill with repeating valid indices
                    valid_indices = idx[i, j, :valid_count]
                    repeats = nsample // valid_count
                    remainder = nsample % valid_count
                    idx[i, j, :] = torch.cat([valid_indices.repeat(repeats), valid_indices[:remainder]])
    
    return idx


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer."""
    
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3  # +3 for coordinate features
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Set Abstraction layer.
        
        Args:
            xyz: [B, N, 3] coordinates
            points: [B, N, D] features (optional)
            
        Returns:
            new_xyz: [B, S, 3] sampled coordinates
            new_points: [B, S, mlp[-1]] aggregated features
        """
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            new_points = torch.cat([xyz, points], dim=-1) if points is not None else xyz
            new_points = new_points.permute(0, 2, 1).unsqueeze(-1)
        else:
            # Sample points
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)
            
            # Group points
            idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # [B, npoint, nsample]
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
            
            # Relative coordinates
            grouped_xyz -= new_xyz.unsqueeze(2)
            
            if points is not None:
                grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, npoint, nsample, D+3]
            else:
                grouped_points = grouped_xyz
            
            # Reshape for convolution: [B, npoint, nsample, D+3] -> [B, D+3, npoint, nsample]
            new_points = grouped_points.permute(0, 3, 1, 2)
        
        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling
        if self.group_all:
            # Max pool over all points (dimension 2)
            new_points = torch.max(new_points, dim=2, keepdim=True)[0]  # [B, mlp[-1], 1, 1]
            new_points = torch.squeeze(new_points, -1)  # [B, mlp[-1], 1]
            new_points = new_points.permute(0, 2, 1)  # [B, 1, mlp[-1]]
        else:
            new_points = torch.max(new_points, dim=-1)[0]  # [B, mlp[-1], npoint]
            new_points = new_points.permute(0, 2, 1)  # [B, npoint, mlp[-1]]
        
        return new_xyz, new_points


class PartialScanEncoder(nn.Module):
    """Encoder for partial LiDAR scan data."""
    
    def __init__(self):
        super().__init__()
        
        # SA Layer 1: 1024 points, radius 0.5m, K=32
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.5, nsample=32, 
            in_channel=0, mlp=[64, 64, 128]
        )
        
        # SA Layer 2: 256 points, radius 1.0m, K=32  
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=1.0, nsample=32,
            in_channel=128, mlp=[128, 128, 256]
        )
        
        # Global max pooling layer
        self.global_sa = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256, mlp=[256, 512, 1024], group_all=True
        )
        
        # Feature compression MLP - Fix the dimensionality
        self.comp_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xyz: [B, N, 3] point cloud coordinates
            
        Returns:
            features: [B, 256] global feature vector
        """
        # SA Layers
        l1_xyz, l1_points = self.sa1(xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # Global feature
        _, l3_points = self.global_sa(l2_xyz, l2_points)
        global_features = l3_points.reshape(l3_points.size(0), -1)  # [B, 1024]
        
        # Compress to 256-D
        output = self.comp_mlp(global_features)
        
        return output


class EnvironmentEncoder(nn.Module):
    """Encoder for CAD environment model."""
    
    def __init__(self):
        super().__init__()
        
        # SA Layer 1: 2048 points, radius 1.0m, K=32
        self.sa1 = PointNetSetAbstraction(
            npoint=2048, radius=1.0, nsample=32,
            in_channel=0, mlp=[64, 64, 128]
        )
        
        # SA Layer 2: 512 points, radius 2.0m, K=32
        self.sa2 = PointNetSetAbstraction(
            npoint=512, radius=2.0, nsample=32,
            in_channel=128, mlp=[128, 128, 256]
        )
        
        # Global pooling
        self.global_sa = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256, mlp=[256, 512, 1024], group_all=True
        )
        
        # Feature compression - Fix the dimensionality
        self.comp_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xyz: [B, N, 3] CAD model point cloud coordinates
            
        Returns:
            features: [B, 256] global feature vector
        """
        # SA Layers  
        l1_xyz, l1_points = self.sa1(xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # Global feature
        _, l3_points = self.global_sa(l2_xyz, l2_points)
        global_features = l3_points.reshape(l3_points.size(0), -1)  # [B, 1024] - Remove the last dimension
        
        # Compress to 256-D
        output = self.comp_mlp(global_features)
        
        return output


class SensorPoseEncoder(nn.Module):
    """Encoder for sensor pose and EGVS quality score."""
    
    def __init__(self):
        super().__init__()
        
        # Pose encoder MLP (3D position + 3D direction)
        self.pose_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # EGVS encoder
        self.egvs_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, pose: torch.Tensor, egvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pose: [B, 6] sensor pose (3 position + 3 direction)
            egvs: [B, 1] EGVS quality score
            
        Returns:
            pose_feat: [B, 64] encoded pose features
            egvs_feat: [B, 64] encoded EGVS features
        """
        pose_feat = self.pose_mlp(pose)
        egvs_feat = self.egvs_mlp(egvs)
        
        return pose_feat, egvs_feat


class FeatureFusionModule(nn.Module):
    """Fusion module for combining features from different streams."""
    
    def __init__(self):
        super().__init__()
        
        # Self-attention for feature refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=640, num_heads=8, batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(640)
        
        # Feature compression
        self.fusion_mlp = nn.Sequential(
            nn.Linear(640, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
    
    def forward(self, partial_feat: torch.Tensor, env_feat: torch.Tensor, 
                pose_feat: torch.Tensor, egvs_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            partial_feat: [B, 256] partial scan features
            env_feat: [B, 256] environment features
            pose_feat: [B, 64] pose features
            egvs_feat: [B, 64] EGVS features
            
        Returns:
            fused_feat: [B, 512] fused global features
        """
        print(f"partial_feat shape: {partial_feat.shape}")
        print(f"env_feat shape: {env_feat.shape}")
        print(f"pose_feat shape: {pose_feat.shape}")
        print(f"egvs_feat shape: {egvs_feat.shape}")
        # Concatenate all features
        fused = torch.cat([partial_feat, env_feat, pose_feat, egvs_feat], dim=1)  # [B, 640]
        
        # Self-attention refinement
        fused_unsqueezed = fused.unsqueeze(1)  # [B, 1, 640]
        attended, _ = self.attention(fused_unsqueezed, fused_unsqueezed, fused_unsqueezed)
        attended = attended.squeeze(1)  # [B, 640]
        
        # Residual connection
        fused = self.layer_norm(fused + attended)
        
        # Feature compression
        return self.fusion_mlp(fused)


class HeatmapDecoder(nn.Module):
    """Decoder for generating 2D information gain heatmap."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        # Positional encoding for 2D coordinates
        self.pos_encoding_dim = 32
        
        # MLP for per-coordinate prediction
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(feature_dim + self.pos_encoding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def positional_encoding(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Positional encoding for 2D coordinates.
        
        Args:
            x: [B, H, W] x coordinates
            y: [B, H, W] y coordinates
            
        Returns:
            pos_enc: [B, H, W, pos_encoding_dim] encoded positions
        """
        # Using sine/cosine positional encoding
        d = self.pos_encoding_dim // 4
        
        # Normalize coordinates to [-1, 1] range
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min() + 1e-10) - 1
        y_normalized = 2 * (y - y.min()) / (y.max() - y.min() + 1e-10) - 1
        
        pos_enc = []
        for freq in range(d):
            scale = 2 ** freq
            pos_enc.append(torch.sin(scale * np.pi * x_normalized))
            pos_enc.append(torch.cos(scale * np.pi * x_normalized))
            pos_enc.append(torch.sin(scale * np.pi * y_normalized))
            pos_enc.append(torch.cos(scale * np.pi * y_normalized))
        
        return torch.stack(pos_enc, dim=-1)
    
    def forward(self, global_features: torch.Tensor, x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            global_features: [B, 512] global context features
            x_grid: [H, W] x coordinates of heatmap grid
            y_grid: [H, W] y coordinates of heatmap grid
            
        Returns:
            heatmap: [B, H, W] information gain heatmap
        """
        B = global_features.shape[0]
        H, W = x_grid.shape
        
        # Expand coordinates to match batch size
        x_grid_batch = x_grid.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]
        y_grid_batch = y_grid.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]
        
        # Positional encoding
        pos_enc = self.positional_encoding(x_grid_batch, y_grid_batch)  # [B, H, W, pos_encoding_dim]
        
        # Reshape for per-coordinate processing
        pos_enc_flat = pos_enc.reshape(B, H * W, self.pos_encoding_dim)  # [B, H*W, pos_encoding_dim]
        
        # Expand global features for each coordinate
        global_features_expanded = global_features.unsqueeze(1).repeat(1, H * W, 1)  # [B, H*W, 512]
        
        # Concatenate features and position encoding
        input_features = torch.cat([global_features_expanded, pos_enc_flat], dim=2)  # [B, H*W, 512+pos_encoding_dim]
        
        # Process each coordinate through MLP
        input_features_flat = input_features.reshape(B * H * W, -1)
        heatmap_flat = self.coordinate_mlp(input_features_flat)  # [B*H*W, 1]
        
        # Reshape to heatmap
        heatmap = heatmap_flat.reshape(B, H, W)
        
        return heatmap


class PointCloudInfoGainNet(nn.Module):
    """
    Complete network for predicting information gain heatmap from LiDAR point clouds.
    """
    
    def __init__(self):
        super().__init__()
        
        self.partial_encoder = PartialScanEncoder()
        self.env_encoder = EnvironmentEncoder()
        self.pose_encoder = SensorPoseEncoder()
        self.fusion = FeatureFusionModule()
        self.decoder = HeatmapDecoder()
        
    def forward(self, partial_cloud: torch.Tensor, env_cloud: torch.Tensor, 
                sensor_pose: torch.Tensor, egvs_score: torch.Tensor,
                x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            partial_cloud: [B, N1, 3] partial LiDAR scan points
            env_cloud: [B, N2, 3] CAD environment points
            sensor_pose: [B, 6] sensor pose (position + direction)
            egvs_score: [B, 1] EGVS quality score
            x_grid: [H, W] x coordinates for heatmap
            y_grid: [H, W] y coordinates for heatmap
            
        Returns:
            heatmap: [B, H, W] information gain heatmap
        """
        # Encode partial scan
        partial_feat = self.partial_encoder(partial_cloud)
        
        # Encode environment
        env_feat = self.env_encoder(env_cloud)
        
        # Encode pose and EGVS
        pose_feat, egvs_feat = self.pose_encoder(sensor_pose, egvs_score)
        
        # Fuse features
        global_feat = self.fusion(partial_feat, env_feat, pose_feat, egvs_feat)
        
        # Generate heatmap
        heatmap = self.decoder(global_feat, x_grid, y_grid)
        
        return heatmap


# Utility functions for data preparation
def prepare_partial_cloud(lidar_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> torch.Tensor:
    """
    Prepare partial cloud data from LiDAR simulation results.
    
    Args:
        lidar_results: List of (points, normals, reflectivity) tuples
        
    Returns:
        partial_cloud: [B, N, 3] partial cloud tensor
    """
    # Combine all LiDAR results
    all_points = []
    for points, _, _ in lidar_results:
        if len(points) > 0:
            all_points.append(points)
    
    if not all_points:
        return torch.zeros((1, 1000, 3))  # Empty point cloud
    
    # Stack and convert to tensor
    partial_cloud = np.vstack(all_points)
    
    # Downsample if too many points (for computational efficiency)
    if len(partial_cloud) > 10000:
        indices = np.random.choice(len(partial_cloud), 10000, replace=False)
        partial_cloud = partial_cloud[indices]
    
    # Add batch dimension and convert to tensor
    partial_cloud = torch.from_numpy(partial_cloud).float().unsqueeze(0)
    
    return partial_cloud


def prepare_env_cloud(mesh: 'trimesh.Trimesh', num_points: int = 20000) -> torch.Tensor:
    """
    Prepare environment cloud from CAD mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        num_points: number of points to sample
        
    Returns:
        env_cloud: [B, N, 3] environment cloud tensor
    """
    # Sample points from mesh surface
    env_cloud = mesh.sample(num_points, return_index=False)  # Fixed: removed unpacking
    
    # Add batch dimension and convert to tensor
    env_cloud = torch.from_numpy(env_cloud).float().unsqueeze(0)
    
    return env_cloud


def prepare_sensor_pose(sensor: 'Sensor') -> torch.Tensor:
    """
    Prepare sensor pose for the network.
    
    Args:
        sensor: Sensor object
        
    Returns:
        pose: [B, 6] pose tensor (position + direction)
    """
    position = sensor.position
    direction = sensor.target_direction
    
    # Concatenate position and direction
    pose = np.concatenate([position, direction])
    
    # Add batch dimension and convert to tensor
    pose = torch.from_numpy(pose).float().unsqueeze(0)
    
    return pose


def create_coordinate_grid(bounds: np.ndarray, resolution: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create coordinate grid for heatmap generation.
    
    Args:
        bounds: [2, 3] array of [min, max] bounds
        resolution: grid resolution in meters
        
    Returns:
        x_grid: [H, W] x coordinates
        y_grid: [H, W] y coordinates
    """
    # Create coordinate ranges
    x_range = np.arange(bounds[0, 0], bounds[1, 0] + resolution, resolution)
    y_range = np.arange(bounds[0, 1], bounds[1, 1] + resolution, resolution)
    
    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # Convert to tensors
    x_grid = torch.from_numpy(x_grid).float()
    y_grid = torch.from_numpy(y_grid).float()
    
    return x_grid, y_grid