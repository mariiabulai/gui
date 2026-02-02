"""
Dense Point Cloud via Multi-View Triangulation

Instead of using stereo matching (SGBM), this module creates a dense point cloud
by triangulating ALL matched keypoints between ALL pairs of cameras.

This approach gives geometrically accurate points because:
1. Each point is triangulated from actual feature correspondences
2. Points are filtered by reprojection error across multiple views
3. No "fronto-parallel bias" that stereo matching suffers from

The result is fewer points than SGBM but each point is geometrically correct.

Author: GUI & Computer Graphics Course Project
"""

import numpy as np
import cv2 as cv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class CameraPose:
    """Camera pose (R, t) where X_cam = R @ X_world + t"""
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    
    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates"""
        return -self.R.T @ self.t.ravel()
    
    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix (without K)"""
        return np.hstack([self.R, self.t.reshape(3, 1)])


class DenseTriangulator:
    """
    Creates dense point cloud by triangulating all matched features
    between all pairs of cameras with known poses.
    """
    
    def __init__(self, K: np.ndarray, dist: np.ndarray = None):
        """
        Args:
            K: 3x3 camera intrinsic matrix
            dist: Distortion coefficients (for undistorting keypoints)
        """
        self.K = K.astype(np.float64)
        self.dist = dist if dist is not None else np.zeros(5)
        
        # SIFT detector with MAXIMUM keypoints for densest cloud
        # nfeatures=32000: more points per image
        # contrastThreshold=0.005: detect in lower contrast areas
        # edgeThreshold=12: filter edge responses
        self.sift = cv.SIFT_create(nfeatures=32000, contrastThreshold=0.005, edgeThreshold=12)
        
        # FLANN matcher
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=200)  # More checks = better matches
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        
        # Filtering parameters - relaxed for more points while maintaining quality
        self.min_parallax_deg = 0.25  # Lower = more points from nearby cameras
        self.max_reproj_error = 6.0  # Higher = accept more points
        self.min_depth = 0.1
        self.max_depth = 100.0
    
    def reconstruct(self, images: List[dict], 
                   camera_poses: Dict[int, CameraPose],
                   matches_dict: Dict[Tuple[int, int], List] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct dense point cloud from images and camera poses.
        
        Args:
            images: List of {'image': ndarray, 'gray': ndarray, 'path': str}
            camera_poses: Dict of {image_idx: CameraPose}
            matches_dict: Optional pre-computed matches {(i,j): matches}
            
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
        """
        print(f"  Dense triangulation: {len(camera_poses)} cameras")
        
        # Extract keypoints for all images
        print("  Extracting keypoints...")
        t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        for idx in camera_poses.keys():
            if idx >= len(images):
                continue
            kp, desc = self.sift.detectAndCompute(images[idx]['gray'], None)
            if desc is not None:
                keypoints[idx] = kp
                descriptors[idx] = desc
        
        print(f"    {sum(len(kp) for kp in keypoints.values()):,} total keypoints ({time.time()-t0:.1f}s)")
        
        # Generate camera pairs for matching
        camera_indices = sorted(camera_poses.keys())
        pairs = []
        
        # Match nearby cameras (within window of 40 for maximum coverage)
        for i, idx1 in enumerate(camera_indices):
            for j, idx2 in enumerate(camera_indices):
                if j <= i:
                    continue
                # Match cameras within window, plus loop closure
                if abs(i - j) <= 40 or abs(i - j) >= len(camera_indices) - 40:
                    pairs.append((idx1, idx2))
        
        print(f"  Matching {len(pairs)} pairs...")
        
        # Triangulate all matches
        all_points = []
        all_colors = []
        
        t0 = time.time()
        for pair_idx, (idx1, idx2) in enumerate(pairs):
            if idx1 not in descriptors or idx2 not in descriptors:
                continue
            
            # Match features
            matches = self._match_features(descriptors[idx1], descriptors[idx2])
            
            if len(matches) < 10:
                continue
            
            # Get matched keypoint coordinates
            pts1 = np.float32([keypoints[idx1][m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints[idx2][m.trainIdx].pt for m in matches])
            
            # Triangulate
            pose1 = camera_poses[idx1]
            pose2 = camera_poses[idx2]
            
            points_3d, valid_mask = self._triangulate_points(
                pose1, pose2, pts1, pts2
            )
            
            if len(points_3d) == 0:
                continue
            
            # Get colors from first image
            colors = self._get_colors(images[idx1]['image'], pts1[valid_mask])
            
            all_points.append(points_3d)
            all_colors.append(colors)
            
            if (pair_idx + 1) % 50 == 0:
                total_pts = sum(len(p) for p in all_points)
                print(f"    [{pair_idx+1}/{len(pairs)}] {total_pts:,} points ({time.time()-t0:.1f}s)")
        
        if not all_points:
            return np.array([]), np.array([])
        
        # Merge all points
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        print(f"  Raw points: {len(points):,}")
        
        # Apply basic filtering (SOR + ROR) without voxel downsampling
        # Voxel downsampling will be done by --filter flag if needed
        from point_cloud_filter import PointCloudFilter
        
        pf = PointCloudFilter(verbose=True)
        
        points, colors = pf.full_pipeline(
            points, colors,
            cameras=camera_poses,
            K=self.K,
            voxel_size=0,  # Disable voxel here, let --filter handle it
            sor_k=30,
            sor_std=2.0  # Less aggressive
        )
        
        print(f"  ✓ Final: {len(points):,} points")
        
        return points, colors
    
    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features with ratio test"""
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        
        try:
            matches = self.flann.knnMatch(desc1, desc2, k=2)
        except cv.error:
            return []
        
        # Ratio test - relaxed for more matches
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.85 * n.distance:  # 0.85 = even more matches
                    good.append(m)
        
        return good
    
    def _triangulate_points(self, pose1, pose2,
                           pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate points between two views with filtering.
        
        Returns:
            points_3d: Nx3 valid triangulated points
            valid_mask: Boolean mask of which input points were valid
        """
        if len(pts1) == 0:
            return np.array([]), np.array([], dtype=bool)
        
        # Build projection matrices
        # Handle both CameraPose types (from pipeline.py and our dataclass)
        if hasattr(pose1, 'P'):
            P1 = self.K @ pose1.P
            P2 = self.K @ pose2.P
            C1 = pose1.center
            C2 = pose2.center
        else:
            # CameraPose from pipeline.py has R, t but no P property
            P1 = self.K @ np.hstack([pose1.R, pose1.t.reshape(3, 1)])
            P2 = self.K @ np.hstack([pose2.R, pose2.t.reshape(3, 1)])
            C1 = -pose1.R.T @ pose1.t.ravel()
            C2 = -pose2.R.T @ pose2.t.ravel()
        
        # Triangulate
        points_4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        # Filter points
        valid_mask = np.ones(len(points_3d), dtype=bool)
        
        for i, pt in enumerate(points_3d):
            # Check depth in both cameras
            pt_cam1 = pose1.R @ pt + pose1.t.ravel()
            pt_cam2 = pose2.R @ pt + pose2.t.ravel()
            
            if pt_cam1[2] <= self.min_depth or pt_cam2[2] <= self.min_depth:
                valid_mask[i] = False
                continue
            
            if pt_cam1[2] > self.max_depth or pt_cam2[2] > self.max_depth:
                valid_mask[i] = False
                continue
            
            # Check parallax angle
            ray1 = pt - C1
            ray2 = pt - C2
            cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-8)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            if angle_deg < self.min_parallax_deg:
                valid_mask[i] = False
                continue
            
            # Check reprojection error
            pt_proj1 = P1 @ np.append(pt, 1)
            pt_proj1 = pt_proj1[:2] / pt_proj1[2]
            err1 = np.linalg.norm(pt_proj1 - pts1[i])
            
            pt_proj2 = P2 @ np.append(pt, 1)
            pt_proj2 = pt_proj2[:2] / pt_proj2[2]
            err2 = np.linalg.norm(pt_proj2 - pts2[i])
            
            if err1 > self.max_reproj_error or err2 > self.max_reproj_error:
                valid_mask[i] = False
                continue
        
        return points_3d[valid_mask], valid_mask
    
    def _get_colors(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Get RGB colors for points from image"""
        if len(pts) == 0:
            return np.array([])
        
        h, w = image.shape[:2]
        colors = []
        
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            bgr = image[y, x]
            colors.append(bgr[::-1])  # BGR to RGB
        
        return np.array(colors)
    
    def _filter_and_deduplicate(self, points: np.ndarray, 
                                colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicates and outliers"""
        if len(points) == 0:
            return points, colors
        
        from scipy.spatial import cKDTree
        
        # 1. Remove extreme outliers
        centroid = np.median(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        median_dist = np.median(distances)
        
        mask = distances < median_dist * 5
        points = points[mask]
        colors = colors[mask]
        
        if len(points) == 0:
            return points, colors
        
        # 2. Voxel grid downsampling to remove duplicates
        voxel_size = median_dist / 200  # Smaller voxels = more points retained
        
        # Quantize to voxel grid
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Use dictionary to keep one point per voxel
        voxel_dict = {}
        for i, voxel in enumerate(voxel_indices):
            key = tuple(voxel)
            if key not in voxel_dict:
                voxel_dict[key] = (points[i], colors[i])
        
        points = np.array([v[0] for v in voxel_dict.values()])
        colors = np.array([v[1] for v in voxel_dict.values()])
        
        # 3. Statistical outlier removal
        if len(points) > 100:
            tree = cKDTree(points)
            k = min(20, len(points) - 1)
            distances, _ = tree.query(points, k=k+1)
            mean_dist = np.mean(distances[:, 1:], axis=1)
            
            threshold = np.mean(mean_dist) + 2 * np.std(mean_dist)
            mask = mean_dist < threshold
            points = points[mask]
            colors = colors[mask]
        
        return points, colors
    
    def match_all_pairs(self, features, images=None, ratio_test=0.7, min_matches=30, 
                      sequential_window=5, full_pairwise=False):
        """
        Adaptive matching with quality checks
        """
        # НОВОЕ: Адаптивный ratio test
        # Если мало матчей с текущим порогом, ослабляем
        def match_with_adaptive_ratio(desc_i, desc_j, initial_ratio=0.7):
            for ratio in [initial_ratio, 0.75, 0.8, 0.85]:
                matches = []
                raw = self.flann.knnMatch(desc_i, desc_j, k=2)
                
                for m_n in raw:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            matches.append(m)
                
                if len(matches) >= min_matches:
                    return matches, ratio
            
            return matches, ratio  # Return whatever we got
        
        # В цикле по парам:
        matches, used_ratio = match_with_adaptive_ratio(features[0], features[1], ratio_test)
        
        if used_ratio != ratio_test and self.verbose:
            print(f"  [Adaptive] Used ratio={used_ratio:.2f} for pair ({0},{1})")
        
        return matches


def create_dense_cloud(images: List[dict], 
                       camera_poses: Dict[int, 'CameraPose'],
                       K: np.ndarray,
                       dist: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create dense point cloud.
    
    Args:
        images: List of image dicts from SfM
        camera_poses: Dict of camera poses from SfM
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        
    Returns:
        points: Nx3 point cloud
        colors: Nx3 RGB colors
    """
    triangulator = DenseTriangulator(K, dist)
    return triangulator.reconstruct(images, camera_poses)
