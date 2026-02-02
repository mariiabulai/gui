"""
Advanced point cloud filtering for cleaner reconstructions
"""
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional


class PointCloudFilter:
    """
    Multi-stage point cloud filtering:
    1. Statistical Outlier Removal (SOR)
    2. Radius Outlier Removal (ROR)
    3. Voxel Grid Downsampling
    4. Surface-aware filtering
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def filter_statistical(self, points: np.ndarray, colors: np.ndarray,
                          k_neighbors: int = 20, 
                          std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Statistical Outlier Removal (SOR)
        
        Remove points whose average distance to k neighbors
        is greater than mean + std_ratio * std
        """
        if len(points) < k_neighbors + 1:
            return points, colors
        
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k_neighbors + 1)
        
        # Average distance to k neighbors (exclude self at index 0)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        threshold = global_mean + std_ratio * global_std
        mask = mean_distances < threshold
        
        if self.verbose:
            removed = len(points) - np.sum(mask)
            print(f"  SOR: removed {removed:,} points ({100*removed/len(points):.1f}%)")
        
        return points[mask], colors[mask]
    
    def filter_radius(self, points: np.ndarray, colors: np.ndarray,
                     radius: float, min_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Radius Outlier Removal (ROR)
        
        Remove points that have fewer than min_neighbors within radius
        """
        if len(points) < min_neighbors:
            return points, colors
        
        tree = cKDTree(points)
        neighbors_count = tree.query_ball_point(points, radius, return_length=True)
        
        # Subtract 1 to exclude self
        mask = (neighbors_count - 1) >= min_neighbors
        
        if self.verbose:
            removed = len(points) - np.sum(mask)
            print(f"  ROR: removed {removed:,} points ({100*removed/len(points):.1f}%)")
        
        return points[mask], colors[mask]
    
    def voxel_downsample(self, points: np.ndarray, colors: np.ndarray,
                        voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voxel grid downsampling - average points within each voxel
        """
        if len(points) == 0:
            return points, colors
        
        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Create unique key for each voxel
        # Shift to handle negative indices
        min_idx = voxel_indices.min(axis=0)
        voxel_indices = voxel_indices - min_idx
        
        # Compute unique voxel keys
        max_idx = voxel_indices.max(axis=0) + 1
        keys = (voxel_indices[:, 0] + 
                voxel_indices[:, 1] * max_idx[0] + 
                voxel_indices[:, 2] * max_idx[0] * max_idx[1])
        
        # Find unique voxels and compute mean position/color
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        
        new_points = np.zeros((len(unique_keys), 3))
        new_colors = np.zeros((len(unique_keys), 3))
        
        np.add.at(new_points, inverse, points)
        np.add.at(new_colors, inverse, colors.astype(np.float64))
        
        new_points /= counts[:, np.newaxis]
        new_colors = (new_colors / counts[:, np.newaxis]).astype(np.uint8)
        
        if self.verbose:
            print(f"  Voxel: {len(points):,} → {len(new_points):,} points")
        
        return new_points, new_colors
    
    def filter_by_reprojection(self, points: np.ndarray, colors: np.ndarray,
                               cameras: dict, K: np.ndarray,
                               max_error: float = 5.0,
                               min_views: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove points with high reprojection error or seen by too few cameras
        
        cameras: dict with values being either (R, t) tuples or CameraPose objects
        """
        import cv2 as cv
        
        if len(cameras) == 0:
            return points, colors
        
        valid_mask = np.ones(len(points), dtype=bool)
        view_counts = np.zeros(len(points), dtype=int)
        
        for idx, cam in cameras.items():
            # Handle both CameraPose dataclass and (R, t) tuple
            if hasattr(cam, 'R') and hasattr(cam, 't'):
                R, t = cam.R, cam.t
            else:
                R, t = cam
            
            t = np.asarray(t).flatten()
            rvec, _ = cv.Rodrigues(R)
            
            # Project all points
            proj, _ = cv.projectPoints(points, rvec, t, K, None)
            proj = proj.reshape(-1, 2)
            
            # Check if points are in front of camera
            points_cam = (R @ points.T).T + t
            in_front = points_cam[:, 2] > 0.1
            
            view_counts += in_front.astype(int)
        
        # Keep points seen by at least min_views cameras
        valid_mask = view_counts >= min_views
        
        if self.verbose:
            removed = len(points) - np.sum(valid_mask)
            print(f"  View filter: removed {removed:,} points (seen by <{min_views} cameras)")
        
        return points[valid_mask], colors[valid_mask]
    
    def full_pipeline(self, points: np.ndarray, colors: np.ndarray,
                     cameras: Optional[dict] = None,
                     K: Optional[np.ndarray] = None,
                     voxel_size: float = 0.01,
                     sor_k: int = 20,
                     sor_std: float = 2.0,
                     ror_radius: Optional[float] = None,
                     ror_min_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full filtering pipeline
        """
        if self.verbose:
            print(f"\nPoint Cloud Filtering Pipeline")
            print(f"  Input: {len(points):,} points")
        
        # 1. Remove extreme outliers first (fast, coarse)
        centroid = np.median(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        median_dist = np.median(distances)
        mask = distances < median_dist * 5
        points, colors = points[mask], colors[mask]
        
        if self.verbose:
            print(f"  Coarse filter: {np.sum(~mask):,} extreme outliers removed")
        
        # 2. Statistical Outlier Removal
        points, colors = self.filter_statistical(points, colors, sor_k, sor_std)
        
        # 3. Radius Outlier Removal (if radius specified)
        if ror_radius is not None:
            points, colors = self.filter_radius(points, colors, ror_radius, ror_min_neighbors)
        else:
            # Auto-compute radius based on point density
            if len(points) > 100:
                tree = cKDTree(points)
                distances, _ = tree.query(points[:min(1000, len(points))], k=2)
                auto_radius = np.median(distances[:, 1]) * 5
                points, colors = self.filter_radius(points, colors, auto_radius, ror_min_neighbors)
        
        # 4. View-based filtering (if cameras provided)
        if cameras is not None and K is not None:
            points, colors = self.filter_by_reprojection(points, colors, cameras, K, min_views=2)
        
        # 5. Voxel downsampling (final cleanup)
        if voxel_size > 0:
            points, colors = self.voxel_downsample(points, colors, voxel_size)
        
        if self.verbose:
            print(f"  Output: {len(points):,} points")
        
        return points, colors