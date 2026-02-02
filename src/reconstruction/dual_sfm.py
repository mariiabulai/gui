"""
Dual-Cluster SfM for 360° Object Reconstruction

When photographing an object in a full circle, cameras on opposite sides
may not share enough visible 3D points for PnP to work. This module
runs SfM twice from different starting pairs and merges the results.

Cluster 1: Starts from cameras 5-10 (front side)
Cluster 2: Starts from cameras 35-45 (back side)

The clusters are then aligned using ICP or common camera poses.

Author: GUI & Computer Graphics Course Project
"""

import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import copy


@dataclass
class SfMCluster:
    """Result of one SfM cluster"""
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]]  # {idx: (R, t)}
    points_3d: Dict[int, np.ndarray]
    point_colors: Dict[int, np.ndarray]
    observations: dict
    observation_index: dict
    start_pair: Tuple[int, int]


def find_best_pair_in_range(match_graph, features, K, start_idx: int, end_idx: int, 
                             min_parallax: float = 2.0, max_parallax: float = 30.0):
    """
    Find best initialization pair within a specific image range.
    
    Args:
        match_graph: Feature matches graph
        features: Features per image
        K: Camera intrinsic matrix
        start_idx: Start of range
        end_idx: End of range (exclusive)
        min_parallax: Minimum parallax angle in degrees
        max_parallax: Maximum parallax angle in degrees
    
    Returns:
        Best (i, j, matches, F, R, t, parallax) or None
    """
    candidates = []
    
    for i in match_graph:
        if not (start_idx <= i < end_idx):
            continue
            
        for j, matches in match_graph[i]:
            if not (start_idx <= j < end_idx):
                continue
            if len(matches) < 100:
                continue
                
            kp1 = features[i]['keypoints']
            kp2 = features[j]['keypoints']
            
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Compute F matrix
            F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.5, 0.99)
            if F is None:
                continue
            
            inlier_mask = mask.ravel() == 1
            pts1_in = pts1[inlier_mask]
            pts2_in = pts2[inlier_mask]
            matches_in = [m for m, inl in zip(matches, inlier_mask) if inl]
            
            if len(matches_in) < 100:
                continue
            
            # Essential matrix
            E = K.T @ F @ K
            _, R, t, _ = cv.recoverPose(E, pts1_in, pts2_in, K)
            
            # Estimate parallax
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t])
            
            pts4d = cv.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            
            # Calculate parallax
            C2 = -R.T @ t.ravel()
            angles = []
            for pt in pts3d:
                if pt[2] > 0:
                    ray1 = pt / np.linalg.norm(pt)
                    ray2 = (pt - C2) / np.linalg.norm(pt - C2)
                    cos_angle = np.clip(np.dot(ray1, ray2), -1, 1)
                    angles.append(np.degrees(np.arccos(cos_angle)))
            
            if len(angles) < 20:
                continue
                
            median_parallax = np.median(angles)
            
            if min_parallax <= median_parallax <= max_parallax:
                candidates.append({
                    'pair': (i, j),
                    'matches': matches_in,
                    'F': F,
                    'R': R,
                    't': t,
                    'parallax': median_parallax,
                    'n_inliers': len(matches_in),
                    'pts1': pts1_in,
                    'pts2': pts2_in
                })
    
    if not candidates:
        return None
    
    # Sort by inliers
    candidates.sort(key=lambda x: x['n_inliers'], reverse=True)
    return candidates[0]


def run_sfm_from_pair(sfm_reconstructor, images, features, match_graph, 
                      init_pair: Tuple[int, int], init_data: dict) -> Optional[SfMCluster]:
    """
    Run SfM starting from a specific pair.
    
    Args:
        sfm_reconstructor: SfMReconstructor instance (will be modified)
        images: List of image data
        features: List of features
        match_graph: Matches graph
        init_pair: (idx1, idx2) to start from
        init_data: Contains 'R', 't', 'matches', 'pts1', 'pts2'
    
    Returns:
        SfMCluster with results
    """
    # Reset state
    sfm_reconstructor.camera_poses = {}
    sfm_reconstructor.points_3d = {}
    sfm_reconstructor.point_colors = {}
    sfm_reconstructor.observations = {}
    sfm_reconstructor.observation_index = {}
    
    from collections import defaultdict
    sfm_reconstructor.observations = defaultdict(list)
    
    idx1, idx2 = init_pair
    R, t = init_data['R'], init_data['t']
    matches = init_data['matches']
    pts1, pts2 = init_data['pts1'], init_data['pts2']
    
    # Set first camera as origin
    sfm_reconstructor.camera_poses[idx1] = (np.eye(3), np.zeros((3, 1)))
    sfm_reconstructor.camera_poses[idx2] = (R, t)
    
    # Triangulate initial points
    P1 = sfm_reconstructor.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = sfm_reconstructor.K @ np.hstack([R, t])
    
    pts3d = sfm_reconstructor.triangulate(P1, P2, pts1, pts2)
    pts3d_filtered, valid_mask = sfm_reconstructor.filter_triangulated_points(
        pts3d, R, t, pts1, pts2, thresh=100.0
    )
    
    matches_filtered = [m for m, v in zip(matches, valid_mask) if v]
    
    # Store initial points
    for i, match in enumerate(matches_filtered):
        sfm_reconstructor.points_3d[i] = pts3d_filtered[i]
        sfm_reconstructor.add_observation(i, idx1, match.queryIdx)
        sfm_reconstructor.add_observation(i, idx2, match.trainIdx)
        
        kp = features[idx1]['keypoints'][match.queryIdx]
        x, y = int(kp.pt[0]), int(kp.pt[1])
        img = images[idx1]['image']
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            sfm_reconstructor.point_colors[i] = img[y, x][::-1]
        else:
            sfm_reconstructor.point_colors[i] = np.array([127, 127, 127])
    
    print(f"  Initialized cluster from ({idx1}, {idx2}) with {len(sfm_reconstructor.points_3d)} points")
    
    # Now run incremental reconstruction
    from collections import defaultdict
    failed_images = set()
    failure_counts = defaultdict(int)
    max_failures = 3
    
    while True:
        next_img = sfm_reconstructor.find_next_image(features, match_graph, failed_images)
        if next_img is None:
            break
        
        pose = sfm_reconstructor.estimate_pose_pnp(next_img, features, match_graph)
        
        if pose is None:
            failure_counts[next_img] += 1
            if failure_counts[next_img] >= max_failures:
                failed_images.add(next_img)
            continue
        
        failure_counts[next_img] = 0
        R_new, t_new = pose
        sfm_reconstructor.camera_poses[next_img] = (R_new, t_new)
        
        # Triangulate new points with all previous images
        _triangulate_new_points(sfm_reconstructor, next_img, features, match_graph, images)
    
    return SfMCluster(
        camera_poses=dict(sfm_reconstructor.camera_poses),
        points_3d=dict(sfm_reconstructor.points_3d),
        point_colors=dict(sfm_reconstructor.point_colors),
        observations=dict(sfm_reconstructor.observations),
        observation_index=dict(sfm_reconstructor.observation_index),
        start_pair=init_pair
    )


def _triangulate_new_points(sfm, next_img, features, match_graph, images):
    """
    Triangulate new 3D points between the new image and all previous images.
    This is extracted from SfMReconstructor.incremental_reconstruction.
    """
    reconstructed_images = [idx for idx in sfm.camera_poses.keys() if idx != next_img]
    
    total_new_points = 0
    
    for prev_img in reconstructed_images:
        matches = None
        
        # Find matches between prev_img and next_img
        if prev_img in match_graph:
            for matched_idx, match_list in match_graph[prev_img]:
                if matched_idx == next_img:
                    matches = match_list
                    break
        
        if matches is None or len(matches) < 20:
            continue
        
        # Get camera poses
        R_prev, t_prev = sfm.camera_poses[prev_img]
        R_new, t_new = sfm.camera_poses[next_img]
        
        # Build projection matrices
        P_prev = sfm.K @ np.hstack([R_prev, t_prev])
        P_new = sfm.K @ np.hstack([R_new, t_new])
        
        # Get corresponding points
        kp_prev = features[prev_img]['keypoints']
        kp_new = features[next_img]['keypoints']
        
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_new = np.float32([kp_new[m.trainIdx].pt for m in matches])
        
        # Triangulate
        points_3d = sfm.triangulate(P_prev, P_new, pts_prev, pts_new)
        
        # Filter
        R_rel = R_new @ R_prev.T
        t_rel = t_new - R_rel @ t_prev
        points_3d_filtered, valid_mask = sfm.filter_triangulated_points(
            points_3d, R_rel, t_rel, pts_prev, pts_new, thresh=100.0
        )
        
        if len(points_3d_filtered) == 0:
            continue
        
        # Add new points
        new_point_count = 0
        current_point_id = max(sfm.points_3d.keys()) + 1 if sfm.points_3d else 0
        
        for i, match in enumerate(matches):
            if not valid_mask[i]:
                continue
            
            # Check if observation already exists (using fast index)
            if (prev_img, match.queryIdx) in sfm.observation_index or \
               (next_img, match.trainIdx) in sfm.observation_index:
                continue
            
            # Find correct index in filtered array
            filtered_idx = int(np.sum(valid_mask[:i+1])) - 1
            
            if filtered_idx < 0 or filtered_idx >= len(points_3d_filtered):
                continue
            
            sfm.points_3d[current_point_id] = points_3d_filtered[filtered_idx]
            
            # Add observations with index
            sfm.add_observation(current_point_id, prev_img, match.queryIdx)
            sfm.add_observation(current_point_id, next_img, match.trainIdx)
            
            # Color
            kp = kp_prev[match.queryIdx]
            x, y = int(kp.pt[0]), int(kp.pt[1])
            img = images[prev_img]['image']
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                sfm.point_colors[current_point_id] = img[y, x][::-1]
            else:
                sfm.point_colors[current_point_id] = np.array([127, 127, 127], dtype=np.uint8)
            
            current_point_id += 1
            new_point_count += 1
        
        total_new_points += new_point_count
    
    if total_new_points > 0:
        print(f"  ✓ Added {total_new_points} new points total")


def merge_clusters(cluster1: SfMCluster, cluster2: SfMCluster, 
                   K: np.ndarray, features: list) -> Tuple[Dict, Dict, Dict]:
    """
    Merge two SfM clusters into one.
    
    Strategy: Find overlapping cameras and compute alignment transform.
    """
    # Find common cameras
    common_cameras = set(cluster1.camera_poses.keys()) & set(cluster2.camera_poses.keys())
    
    print(f"\n  Merging clusters:")
    print(f"    Cluster 1: {len(cluster1.camera_poses)} cameras, {len(cluster1.points_3d)} points")
    print(f"    Cluster 2: {len(cluster2.camera_poses)} cameras, {len(cluster2.points_3d)} points")
    print(f"    Common cameras: {len(common_cameras)}")
    
    if len(common_cameras) >= 3:
        # Compute alignment using common camera centers
        centers1 = []
        centers2 = []
        for idx in common_cameras:
            R1, t1 = cluster1.camera_poses[idx]
            R2, t2 = cluster2.camera_poses[idx]
            centers1.append(-R1.T @ t1.ravel())
            centers2.append(-R2.T @ t2.ravel())
        
        centers1 = np.array(centers1)
        centers2 = np.array(centers2)
        
        # Procrustes alignment: find R, t, s such that centers1 ≈ s * R @ centers2 + t
        # Simplified: use SVD-based alignment
        mu1 = centers1.mean(axis=0)
        mu2 = centers2.mean(axis=0)
        
        c1 = centers1 - mu1
        c2 = centers2 - mu2
        
        H = c2.T @ c1
        U, S, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T
        
        # Ensure proper rotation
        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T
        
        scale = np.linalg.norm(c1) / (np.linalg.norm(c2) + 1e-8)
        t_align = mu1 - scale * R_align @ mu2
        
        print(f"    Alignment scale: {scale:.4f}")
        
        # Transform cluster2 points and cameras to cluster1 frame
        transformed_cameras = {}
        for idx, (R, t) in cluster2.camera_poses.items():
            if idx not in cluster1.camera_poses:
                # Transform: R_new = R_align @ R, t_new = scale * R_align @ t + t_align
                R_new = R_align @ R
                t_new = scale * (R_align @ t.reshape(3, 1)) + t_align.reshape(3, 1)
                transformed_cameras[idx] = (R_new, t_new)
        
        # Transform cluster2 points
        transformed_points = {}
        max_id = max(cluster1.points_3d.keys()) + 1 if cluster1.points_3d else 0
        for pid, pt in cluster2.points_3d.items():
            pt_new = scale * (R_align @ pt) + t_align
            transformed_points[max_id + pid] = pt_new
        
        # Merge
        merged_cameras = dict(cluster1.camera_poses)
        merged_cameras.update(transformed_cameras)
        
        merged_points = dict(cluster1.points_3d)
        merged_points.update(transformed_points)
        
        merged_colors = dict(cluster1.point_colors)
        for pid, color in cluster2.point_colors.items():
            merged_colors[max_id + pid] = color
        
        print(f"    Merged: {len(merged_cameras)} cameras, {len(merged_points)} points")
        
        return merged_cameras, merged_points, merged_colors
    
    else:
        # No common cameras - just return larger cluster
        print("    Warning: No common cameras, returning larger cluster")
        if len(cluster1.camera_poses) >= len(cluster2.camera_poses):
            return cluster1.camera_poses, cluster1.points_3d, cluster1.point_colors
        else:
            return cluster2.camera_poses, cluster2.points_3d, cluster2.point_colors


def dual_cluster_sfm(sfm_reconstructor, images, features, match_graph,
                     range1: Tuple[int, int] = (0, 25),
                     range2: Tuple[int, int] = (35, 60)):
    """
    Run dual-cluster SfM for 360° object reconstruction.
    
    Args:
        sfm_reconstructor: SfMReconstructor instance
        images: List of images
        features: List of features
        match_graph: Match graph
        range1: Image range for cluster 1 (front)
        range2: Image range for cluster 2 (back)
    
    Returns:
        (camera_poses, points_3d, point_colors) merged result
    """
    K = sfm_reconstructor.K
    n_images = len(images)
    
    print("\n" + "="*60)
    print("DUAL-CLUSTER SfM")
    print("="*60)
    
    # Find best pair for cluster 1
    print(f"\n[1/4] Finding initialization for Cluster 1 (images {range1[0]}-{range1[1]})...")
    pair1_data = find_best_pair_in_range(match_graph, features, K, range1[0], min(range1[1], n_images))
    
    if pair1_data is None:
        print("  ✗ No good pair found for cluster 1")
        return None, None, None
    
    print(f"  ✓ Best pair: {pair1_data['pair']} ({pair1_data['n_inliers']} inliers, {pair1_data['parallax']:.1f}° parallax)")
    
    # Find best pair for cluster 2
    print(f"\n[2/4] Finding initialization for Cluster 2 (images {range2[0]}-{range2[1]})...")
    pair2_data = find_best_pair_in_range(match_graph, features, K, range2[0], min(range2[1], n_images))
    
    if pair2_data is None:
        print("  ✗ No good pair found for cluster 2, running single-cluster SfM")
        # Fall back to single cluster
        cluster1 = run_sfm_from_pair(sfm_reconstructor, images, features, match_graph,
                                     pair1_data['pair'], pair1_data)
        return cluster1.camera_poses, cluster1.points_3d, cluster1.point_colors
    
    print(f"  ✓ Best pair: {pair2_data['pair']} ({pair2_data['n_inliers']} inliers, {pair2_data['parallax']:.1f}° parallax)")
    
    # Run cluster 1
    print(f"\n[3/4] Running Cluster 1 SfM...")
    import copy
    sfm1 = copy.deepcopy(sfm_reconstructor)
    cluster1 = run_sfm_from_pair(sfm1, images, features, match_graph,
                                 pair1_data['pair'], pair1_data)
    print(f"  ✓ Cluster 1: {len(cluster1.camera_poses)} cameras, {len(cluster1.points_3d)} points")
    
    # Run cluster 2
    print(f"\n[4/4] Running Cluster 2 SfM...")
    sfm2 = copy.deepcopy(sfm_reconstructor)
    cluster2 = run_sfm_from_pair(sfm2, images, features, match_graph,
                                 pair2_data['pair'], pair2_data)
    print(f"  ✓ Cluster 2: {len(cluster2.camera_poses)} cameras, {len(cluster2.points_3d)} points")
    
    # Merge clusters
    merged_cameras, merged_points, merged_colors = merge_clusters(
        cluster1, cluster2, K, features
    )
    
    return merged_cameras, merged_points, merged_colors
