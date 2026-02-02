import numpy as np
import cv2 as cv
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict
import traceback
import time
import pickle

# Allow running both as module (-m) and as script (python sfm.py)
if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    for p in (CURRENT_DIR, PARENT_DIR):
        if str(p) not in sys.path:
            sys.path.append(str(p))

try:
    from .bundle_adjustment import BundleAdjuster  # type: ignore
    from .dense_triangulation import DenseTriangulator  # type: ignore
    from .point_cloud_filter import PointCloudFilter  # type: ignore
    from .image_quality import ImageQualityChecker  # type: ignore
except Exception:
    from bundle_adjustment import BundleAdjuster  # type: ignore
    from dense_triangulation import DenseTriangulator  # type: ignore
    from point_cloud_filter import PointCloudFilter  # type: ignore
    from image_quality import ImageQualityChecker  # type: ignore

class SfMReconstructor:
    def __init__(self, calibration_file):
        """Load camera calibration parameters"""
        data = np.load(calibration_file)
        self.K = data['mtx']
        self.dist = data['dist']
        self.K_inv = np.linalg.inv(self.K)
        
        # Global reconstruction state
        self.camera_poses = {}
        self.points_3d = {}
        self.point_colors = {}
        self.observations = defaultdict(list)
        self.observation_index = {}
    
    def _prepare_matching(self, image_dir: str):
        """
        Prepare images, features and matches for dual-cluster SfM.
        
        This is a separate method so dual_sfm.py can access intermediate state.
        
        Args:
            image_dir: Path to images folder
        
        Returns:
            (features, match_graph) - ready for SfM
        """
        # Load images
        self.images = self.load_images(image_dir)
        
        # Detect features
        features = self.detect_features(self.images, detector='auto')
        
        # Match features
        match_graph = self.match_all_pairs(features, self.images, 
                                           ratio_test=0.7, min_matches=30,
                                           sequential_window=5)
        
        return features, match_graph
    
    def load_images(self, image_dir, max_images=None):
        """Load and undistort images from directory"""
        image_paths = set(
            glob.glob(os.path.join(image_dir, '*.jpg')) +
            glob.glob(os.path.join(image_dir, '*.JPG')) +
            glob.glob(os.path.join(image_dir, '*.png')) +
            glob.glob(os.path.join(image_dir, '*.PNG'))
        )
        image_paths = sorted(image_paths)
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        images = []
        for path in image_paths:
            img = cv.imread(path)
            if img is None:
                print(f"Failed to load: {path}")
                continue
            
            # Apply undistortion for better geometric accuracy
            img_undistorted = cv.undistort(img, self.K, self.dist)
            
            images.append({
                'path': path,
                'image': img_undistorted,
                'gray': cv.cvtColor(img_undistorted, cv.COLOR_BGR2GRAY)
            })
        
        print(f"Loaded {len(images)} images (undistortion: OFF)")
        return images
    
    def detect_features(self, images, detector='sift', nfeatures=8000):
        """
        Detect keypoints and compute descriptors using SIFT
        """
        print(f"Using CPU SIFT detector (nfeatures={nfeatures})")
        feature_detector = cv.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=10)
        features = []
        
        for idx, img_dict in enumerate(images):
            kp, desc = feature_detector.detectAndCompute(img_dict['gray'], None)
            if desc is None:
                features.append({'keypoints': [], 'descriptors': np.array([])})
            else:
                features.append({'keypoints': kp, 'descriptors': desc})
        
        for idx, feat in enumerate(features):
            print(f"[{idx}] {len(feat['keypoints'])} keypoints in {os.path.basename(images[idx]['path'])}")
        
        return features
    
    def match_all_pairs(self, features, images=None, ratio_test=0.75, min_matches=30, 
                          sequential_window=5, full_pairwise=False):
        """
        Match features between image pairs.
        
        Args:
            features: feature dictionaries
            images: image data (not used, kept for API compatibility)
            ratio_test: Lowe's ratio test threshold
            min_matches: minimum matches to keep pair
            sequential_window: match i with i+1..i+window (default 5)
            full_pairwise: if True, do O(N²) matching (slow but thorough)
        """
        match_graph = defaultdict(list)
        n_images = len(features)
        
        # FLANN matcher - faster than BFMatcher for large descriptor sets
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        
        # Build pairs to match
        if full_pairwise:
            pairs = [(i, j) for i in range(n_images) for j in range(i+1, n_images)]
            print(f"Full pairwise matching: {len(pairs)} pairs")
        else:
            pairs = set()
            
            # Sequential neighbors (wider window)
            for i in range(n_images):
                for offset in range(1, sequential_window + 1):
                    j = i + offset
                    if j < n_images:
                        pairs.add((i, j))
            
            # LOOP CLOSURE: connect last images to first images
            # Critical for circular camera motion around object
            loop_window = min(15, n_images // 3)  # Increased from 10
            for i in range(loop_window):
                for j in range(n_images - loop_window, n_images):
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))
            
            # DENSE SKIP CONNECTIONS: wider range for better coverage
            for i in range(0, n_images):
                for offset in [6, 9, 12, 15, 20, 25, 30]:  # Extended range
                    j = i + offset
                    if j < n_images:
                        pairs.add((i, j))
            
            # BRIDGE CONNECTIONS: connect "middle" images to both ends
            # This ensures images 15-30 have matches with 0-14 AND 55-67
            mid_start = n_images // 4  # ~17 for 68 images
            mid_end = 3 * n_images // 4  # ~51 for 68 images
            for i in range(mid_start, mid_end):
                # Connect to first quarter
                for j in range(min(10, mid_start)):
                    pairs.add((j, i))
                # Connect to last quarter
                for j in range(max(mid_end, n_images - 10), n_images):
                    pairs.add((i, j))
            
            pairs = sorted(pairs)
            print(f"Smart matching (window={sequential_window}, loop+skip): {len(pairs)} pairs")
        
        total_pairs = len(pairs)
        matched_count = 0
        
        for idx, (i, j) in enumerate(pairs):
            desc_i = features[i].get('descriptors')
            desc_j = features[j].get('descriptors')
            
            if desc_i is None or desc_j is None or len(desc_i) < 2 or len(desc_j) < 2:
                continue
            
            t0 = time.time()
            
            try:
                # FLANN matching with Lowe's ratio test
                desc_i_f = desc_i.astype(np.float32)
                desc_j_f = desc_j.astype(np.float32)
                raw = flann.knnMatch(desc_i_f, desc_j_f, k=2)
                
                matches = []
                for m_n in raw:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < ratio_test * n.distance:
                            matches.append(m)
                elapsed = time.time() - t0
                
                if matches and len(matches) >= min_matches:
                    match_graph[i].append((j, matches))
                    matched_count += 1
                    print(f"[{idx+1}/{total_pairs}] ✓ {len(matches):4d} matches: {i:2d} ↔ {j:2d} ({elapsed:.2f}s)")
                else:
                    n_matches = len(matches) if matches else 0
                    print(f"[{idx+1}/{total_pairs}] ✗ {n_matches:4d} matches: {i:2d} ↔ {j:2d} (skipped)")
                    
            except Exception as e:
                print(f"[{idx+1}/{total_pairs}] ✗ Pair ({i},{j}) failed: {e}")
        
        print(f"\nMatching complete: {matched_count}/{total_pairs} pairs matched")
        return match_graph
    
    def compute_fundamental_matrix_ransac(self, kp1, kp2, matches, thresh=1.0):
        """Compute F matrix with RANSAC"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 
                                        ransacReprojThreshold=thresh, confidence=0.999)
        
        if F is None:
            return None, None, None, None
        
        inliers = mask.ravel() == 1
        inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
        
        pts1_inliers = pts1[inliers]
        pts2_inliers = pts2[inliers]
        
        return F, inlier_matches, pts1_inliers, pts2_inliers
    
    def triangulate(self, P1, P2, pts1, pts2):
        """Triangulate 3D points from two views"""
        points_4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        return points_3d
    
    def filter_triangulated_points(self, points_3d, R, t, pts1, pts2, thresh=50.0):
        """
        Filter triangulated points by:
        1. Positive depth in both cameras
        2. Reasonable depth range
        3. Parallax angle (CRITICAL for accuracy)
        4. Reprojection error
        """
        if len(points_3d) == 0:
            return np.array([]), np.array([])
        
        n_points = len(points_3d)
        valid_mask = np.ones(n_points, dtype=bool)
        
        # Debug counters
        fail_z1 = 0
        fail_z2 = 0
        fail_depth = 0
        fail_parallax = 0
        fail_reproj1 = 0
        fail_reproj2 = 0
        
        # Camera centers
        C1 = np.zeros(3)  # First camera at origin
        C2 = -R.T @ t.ravel()  # Second camera center
        baseline = np.linalg.norm(C2 - C1)
        
        for i in range(n_points):
            pt = points_3d[i]
            
            # 1. Depth in camera 1
            z1 = pt[2]
            if z1 <= 0.01:
                valid_mask[i] = False
                fail_z1 += 1
                continue
            
            # 2. Depth in camera 2
            pt_cam2 = R @ pt + t.ravel()
            z2 = pt_cam2[2]
            if z2 <= 0.01:
                valid_mask[i] = False
                fail_z2 += 1
                continue
            
            # 3. Reasonable depth range (relaxed for indoor scenes)
            max_depth = max(thresh, baseline * 500)  # Increased from 100
            if z1 > max_depth or z2 > max_depth:
                valid_mask[i] = False
                fail_depth += 1
                continue
            
            # 4. PARALLAX ANGLE CHECK (relaxed)
            ray1 = pt - C1
            ray2 = pt - C2
            cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            
            # Require at least 0.1 degree parallax (more relaxed)
            if angle_deg < 0.1:
                valid_mask[i] = False
                fail_parallax += 1
                continue
            
            # 5. Reprojection error check (relaxed to 10px)
            proj1 = self.K @ pt.reshape(3, 1)
            if proj1[2, 0] > 0:
                px1 = proj1[0, 0] / proj1[2, 0]
                py1 = proj1[1, 0] / proj1[2, 0]
                err1 = np.sqrt((px1 - pts1[i, 0])**2 + (py1 - pts1[i, 1])**2)
                if err1 > 10.0:  # Relaxed from 5.0
                    valid_mask[i] = False
                    fail_reproj1 += 1
                    continue
            
            proj2 = self.K @ pt_cam2.reshape(3, 1)
            if proj2[2, 0] > 0:
                px2 = proj2[0, 0] / proj2[2, 0]
                py2 = proj2[1, 0] / proj2[2, 0]
                err2 = np.sqrt((px2 - pts2[i, 0])**2 + (py2 - pts2[i, 1])**2)
                if err2 > 10.0:  # Relaxed from 5.0
                    valid_mask[i] = False
                    fail_reproj2 += 1
                    continue
        
        print(f"  FILTER DEBUG: z1={fail_z1}, z2={fail_z2}, depth={fail_depth}, parallax={fail_parallax}, reproj1={fail_reproj1}, reproj2={fail_reproj2}, pass={valid_mask.sum()}")
        
        return points_3d[valid_mask], valid_mask
    
    def add_observation(self, point_id, img_idx, kp_idx):
        """Add observation and update index"""
        self.observations[point_id].append((img_idx, kp_idx))
        self.observation_index[(img_idx, kp_idx)] = point_id
    
    def initialize_reconstruction(self, images, features, match_graph):
        """
        Initialize reconstruction with the best image pair.
        
        Selection criteria (in order of importance):
        1. Sufficient inliers (>100)
        2. Good parallax angle (not too small, not too large)
        3. Prefer pairs closer to middle of sequence for better coverage
        """
        candidates = []
        n_images = len(features)
        
        for i in match_graph:
            for j, matches in match_graph[i]:
                kp1 = features[i]['keypoints']
                kp2 = features[j]['keypoints']
                
                F, inlier_matches, pts1, pts2 = self.compute_fundamental_matrix_ransac(
                    kp1, kp2, matches, thresh=1.5
                )
                
                if F is None or len(inlier_matches) < 100:
                    continue
                
                # Compute parallax by recovering pose
                E = self.K.T @ F @ self.K
                _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, self.K)
                
                # Quick triangulation to check parallax
                P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                P2 = self.K @ np.hstack([R, t])
                
                # Sample a few points to estimate median parallax
                sample_indices = np.linspace(0, len(pts1)-1, min(50, len(pts1)), dtype=int)
                sample_pts1 = pts1[sample_indices]
                sample_pts2 = pts2[sample_indices]
                
                points_4d = cv.triangulatePoints(P1, P2, sample_pts1.T, sample_pts2.T)
                points_3d = (points_4d[:3] / points_4d[3]).T
                
                # Calculate parallax angles
                C1 = np.zeros(3)
                C2 = -R.T @ t.flatten()
                
                parallax_angles = []
                for pt in points_3d:
                    if pt[2] > 0:  # Positive depth
                        ray1 = pt - C1
                        ray2 = pt - C2
                        cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-8)
                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                        parallax_angles.append(angle_deg)
                
                if len(parallax_angles) < 10:
                    continue
                    
                median_parallax = np.median(parallax_angles)
                
                # Prefer pairs with:
                # - Parallax between 2 and 30 degrees (good triangulation geometry)
                # - More inliers is better
                # - Pairs closer to middle of sequence (for better coverage)
                
                if median_parallax < 1.0 or median_parallax > 45.0:
                    continue  # Bad geometry
                
                # Score: balance inliers, parallax quality, and position
                parallax_score = 1.0 if 3 < median_parallax < 20 else 0.5
                center_distance = abs((i + j) / 2 - n_images / 2) / (n_images / 2)
                position_score = 1.0 - center_distance * 0.3  # Small penalty for being far from center
                
                score = len(inlier_matches) * parallax_score * position_score
                
                # Apply mask_pose from recoverPose to filter points
                pose_valid = mask_pose.ravel() > 0
                pts1_filtered = pts1[pose_valid]
                pts2_filtered = pts2[pose_valid]
                matches_filtered = [m for k, m in enumerate(inlier_matches) if pose_valid[k]]
                
                candidates.append({
                    'pair': (i, j),
                    'matches': matches_filtered,  # Filtered by Essential matrix
                    'pts1': pts1_filtered,
                    'pts2': pts2_filtered,
                    'F': F,
                    'R': R,
                    't': t,
                    'score': score,
                    'n_inliers': len(matches_filtered),  # Count after E filtering
                    'parallax': median_parallax
                })
        
        if not candidates:
            print("Failed to find suitable initialization pair!")
            return False
        
        # Sort by score and pick best
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        idx1, idx2 = best['pair']
        print(f"\n✓ Initializing with pair ({idx1}, {idx2}): {best['n_inliers']} inliers, "
              f"parallax={best['parallax']:.1f}°")
        
        # Set first camera as world origin
        self.camera_poses[idx1] = (np.eye(3), np.zeros((3, 1)))
        self.camera_poses[idx2] = (best['R'], best['t'])
        
        # Extract data from best candidate
        R, t = best['R'], best['t']
        pts1, pts2 = best['pts1'], best['pts2']
        matches = best['matches']
        
        print(f"  DEBUG: pts1={len(pts1)}, pts2={len(pts2)}, matches={len(matches)}")
        
        # Triangulate initial points
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        points_3d = self.triangulate(P1, P2, pts1, pts2)
        print(f"  DEBUG: triangulated {len(points_3d)} points")
        
        points_3d_filtered, valid_mask = self.filter_triangulated_points(
            points_3d, R, t, pts1, pts2, thresh=100.0
        )
        print(f"  DEBUG: after filter {len(points_3d_filtered)} points (valid_mask sum={valid_mask.sum() if len(valid_mask) > 0 else 0})")
        
        # Store 3D points and observations
        matches_filtered = [matches[i] for i in range(len(matches)) if valid_mask[i]]
        
        point_id = 0
        for i, match in enumerate(matches_filtered):
            self.points_3d[point_id] = points_3d_filtered[i]
            
            # Add observations with index
            self.add_observation(point_id, idx1, match.queryIdx)
            self.add_observation(point_id, idx2, match.trainIdx)
            
            # Get color from first image
            kp = features[idx1]['keypoints'][match.queryIdx]
            x, y = int(kp.pt[0]), int(kp.pt[1])
            img = images[idx1]['image']
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                self.point_colors[point_id] = img[y, x][::-1]  # BGR to RGB
            else:
                self.point_colors[point_id] = np.array([127, 127, 127], dtype=np.uint8)
            
            point_id += 1
        
        print(f"  Initialized with {len(self.points_3d)} 3D points")
        return True
    
    def find_next_image(self, features, match_graph, failed_images=set()):
        """Find next image to add (one with most 2D-3D correspondences)"""
        reconstructed_images = set(self.camera_poses.keys())
        
        best_img = None
        best_count = 0
        
        for img_idx in range(len(features)):
            if img_idx in reconstructed_images or img_idx in failed_images:
                continue
            
            # Count 2D-3D correspondences using fast index
            count_2d_3d = 0
            
            for other_idx in reconstructed_images:
                # Check forward direction
                if img_idx in match_graph:
                    for matched_idx, matches in match_graph[img_idx]:
                        if matched_idx == other_idx:
                            # Count how many of these matches have existing 3D points
                            for match in matches:
                                if (other_idx, match.trainIdx) in self.observation_index:
                                    count_2d_3d += 1
                
                # Check reverse direction
                if other_idx in match_graph:
                    for matched_idx, matches in match_graph[other_idx]:
                        if matched_idx == img_idx:
                            for match in matches:
                                if (other_idx, match.queryIdx) in self.observation_index:
                                    count_2d_3d += 1
            
            if count_2d_3d > best_count:
                best_count = count_2d_3d
                best_img = img_idx
        
        if best_img is None or best_count < 20:
            return None
        
        print(f"\n→ Adding image {best_img} ({best_count} 2D-3D correspondences)")
        return best_img
    
    def estimate_pose_pnp(self, img_idx, features, match_graph):
        """
        Estimate camera pose using PnP with RANSAC (OPTIMIZED)
        """
        points_3d_list = []
        points_2d_list = []
        
        reconstructed_images = set(self.camera_poses.keys())
        
        # Collect 2D-3D correspondences using fast index lookup
        for other_idx in reconstructed_images:
            matches = None
            
            # Check forward direction
            if img_idx in match_graph:
                for matched_idx, match_list in match_graph[img_idx]:
                    if matched_idx == other_idx:
                        matches = match_list
                        break
            
            # Check reverse direction (FIX: don't shadow variable)
            if matches is None and other_idx in match_graph:
                for matched_idx, match_list in match_graph[other_idx]:
                    if matched_idx == img_idx:
                        # Reverse queryIdx/trainIdx
                        matches = [cv.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in match_list]
                        break
            
            if matches is None:
                continue
            
            # For each match, check if observation exists in other_idx
            for match in matches:
                key = (other_idx, match.trainIdx)
                if key in self.observation_index:
                    point_id = self.observation_index[key]
                    point_3d = self.points_3d[point_id]
                    
                    kp = features[img_idx]['keypoints'][match.queryIdx]
                    points_2d_list.append(kp.pt)
                    points_3d_list.append(point_3d)
        
        if len(points_3d_list) < 10:
            print(f"  Not enough 2D-3D correspondences: {len(points_3d_list)}")
            return None
        
        points_3d_arr = np.array(points_3d_list, dtype=np.float32)
        points_2d_arr = np.array(points_2d_list, dtype=np.float32)
        
        # Solve PnP with RANSAC (balanced parameters - not too permissive to avoid bad poses)
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            points_3d_arr, points_2d_arr, self.K, None,
            iterationsCount=3000, reprojectionError=8.0, confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < 10:
            print(f"  PnP failed: success={success}, inliers={len(inliers) if inliers is not None else 0}/{len(points_3d_list)}")
            return None
        
        R, _ = cv.Rodrigues(rvec)
        
        print(f"  ✓ PnP: {len(inliers)}/{len(points_3d_list)} inliers")
        return R, tvec
    
    def incremental_reconstruction(self, images, features, match_graph):
        """Incrementally add images and triangulate new 3D points"""
        if not self.initialize_reconstruction(images, features, match_graph):
            return False
        
        # Track failed images to avoid infinite loops
        failed_images = set()
        max_failures = 3  # Allow retry up to 3 times per image
        failure_counts = defaultdict(int)
        
        # Add remaining images one by one
        while True:
            next_img = self.find_next_image(features, match_graph, failed_images)
            if next_img is None:
                print(f"\nNo more images to add. Failed images: {failed_images}")
                break
            
            pose = self.estimate_pose_pnp(next_img, features, match_graph)
            
            if pose is None:
                failure_counts[next_img] += 1
                if failure_counts[next_img] >= max_failures:
                    print(f"  ✗ Image {next_img} failed {max_failures} times, skipping permanently")
                    failed_images.add(next_img)
                continue
            
            # Reset failure count on success
            failure_counts[next_img] = 0
            
            R, t = pose
            self.camera_poses[next_img] = (R, t)
            
            # Triangulate new points with all previous images
            reconstructed_images = [idx for idx in self.camera_poses.keys() if idx != next_img]
            
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
                R_prev, t_prev = self.camera_poses[prev_img]
                R_new, t_new = self.camera_poses[next_img]
                
                # Build projection matrices
                P_prev = self.K @ np.hstack([R_prev, t_prev])
                P_new = self.K @ np.hstack([R_new, t_new])
                
                # Get corresponding points
                kp_prev = features[prev_img]['keypoints']
                kp_new = features[next_img]['keypoints']
                
                pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
                pts_new = np.float32([kp_new[m.trainIdx].pt for m in matches])
                
                # Triangulate
                points_3d = self.triangulate(P_prev, P_new, pts_prev, pts_new)
                
                # Filter
                R_rel = R_new @ R_prev.T
                t_rel = t_new - R_rel @ t_prev
                points_3d_filtered, valid_mask = self.filter_triangulated_points(
                    points_3d, R_rel, t_rel, pts_prev, pts_new, thresh=100.0
                )
                
                if len(points_3d_filtered) == 0:
                    continue
                
                # Add new points
                new_point_count = 0
                current_point_id = max(self.points_3d.keys()) + 1 if self.points_3d else 0
                
                for i, match in enumerate(matches):
                    if not valid_mask[i]:
                        continue
                    
                    # Check if observation already exists (using fast index)
                    if (prev_img, match.queryIdx) in self.observation_index or \
                       (next_img, match.trainIdx) in self.observation_index:
                        continue
                    
                    # Find correct index in filtered array
                    filtered_idx = int(np.sum(valid_mask[:i+1])) - 1
                    
                    if filtered_idx < 0 or filtered_idx >= len(points_3d_filtered):
                        continue
                    
                    self.points_3d[current_point_id] = points_3d_filtered[filtered_idx]
                    
                    # Add observations with index
                    self.add_observation(current_point_id, prev_img, match.queryIdx)
                    self.add_observation(current_point_id, next_img, match.trainIdx)
                    
                    # Color
                    kp = kp_prev[match.queryIdx]
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    img = images[prev_img]['image']
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        self.point_colors[current_point_id] = img[y, x][::-1]
                    else:
                        self.point_colors[current_point_id] = np.array([127, 127, 127], dtype=np.uint8)
                    
                    current_point_id += 1
                    new_point_count += 1
                
                total_new_points += new_point_count
            
            if total_new_points > 0:
                print(f"  ✓ Added {total_new_points} new points total")
        
        print(f"\n✓ Reconstruction complete:")
        print(f"  Cameras: {len(self.camera_poses)}/{len(images)}")
        print(f"  3D points: {len(self.points_3d)}")
        
        return True
    
    def normalize_reconstruction(self):
        """
        Normalize 3D points and camera poses to reasonable scale
        CRITICAL: Must maintain consistent coordinate system
        """
        if len(self.points_3d) == 0:
            return
        
        # Get all 3D points
        points_array = np.array([self.points_3d[pid] for pid in self.points_3d.keys()])
        
        # Compute centroid in world coordinates
        centroid = points_array.mean(axis=0)
        
        # Center points around origin
        points_centered = points_array - centroid
        
        # Compute scale based on mean distance from origin
        distances = np.linalg.norm(points_centered, axis=1)
        mean_distance = distances.mean()
        
        if mean_distance < 0.01:
            mean_distance = 1.0
        
        # Scale factor: normalize to ~10 units (reasonable for BA)
        scale_factor = 10.0 / mean_distance
        
        print(f"\nNormalization:")
        print(f"  Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        print(f"  Mean distance: {mean_distance:.4f}")
        print(f"  Scale factor: {scale_factor:.6f}")
        print(f"  Final scale: ~{mean_distance * scale_factor:.2f} units")
        
        # Apply to 3D points
        for pt_id in self.points_3d.keys():
            self.points_3d[pt_id] = (self.points_3d[pt_id] - centroid) * scale_factor
        
        # Apply to camera poses
        # CRITICAL: Camera center is at C = -R^T * t
        # After transformation: C_new = (C_old - centroid) * scale
        # So: t_new = -R * C_new = -R * ((−R^T * t_old − centroid) * scale)
        
        for img_idx in self.camera_poses.keys():
            R, t = self.camera_poses[img_idx]
            
            # Compute camera center in world coordinates
            C = -R.T @ t  # Camera center: C = -R^T * t
            
            # Transform camera center
            C_new = (C.ravel() - centroid) * scale_factor
            
            # Compute new translation: t_new = -R * C_new
            t_new = -R @ C_new.reshape(3, 1)
            
            self.camera_poses[img_idx] = (R, t_new)
        
        print(f"  ✓ Applied to {len(self.points_3d)} points and {len(self.camera_poses)} cameras")
        
        # Verify transformation
        camera_centers = []
        for img_idx in self.camera_poses.keys():
            R, t = self.camera_poses[img_idx]
            C = -R.T @ t
            camera_centers.append(C.ravel())
        
        if camera_centers:
            camera_centers = np.array(camera_centers)
            camera_scale = np.linalg.norm(camera_centers, axis=1).mean()
            print(f"  Verification: Camera distance from origin: {camera_scale:.4f}")
        
        return scale_factor
    
    def prune_by_reprojection(self, features, max_error=5.0):
        """
        Remove 3D points with high reprojection error
        """
        if len(self.points_3d) == 0:
            return
        
        points_to_remove = []
        
        for point_id, point_3d in self.points_3d.items():
            errors = []
            
            for img_idx, kp_idx in self.observations[point_id]:
                if img_idx not in self.camera_poses:
                    continue
                
                R, t = self.camera_poses[img_idx]
                
                # Project point to image
                point_cam = R @ point_3d.reshape(3, 1) + t
                
                if point_cam[2, 0] <= 0:
                    errors.append(1000.0)  # Behind camera
                    continue
                
                point_proj = self.K @ point_cam
                px = point_proj[0, 0] / point_proj[2, 0]
                py = point_proj[1, 0] / point_proj[2, 0]
                
                # Get observed keypoint
                kp = features[img_idx]['keypoints'][kp_idx]
                ox, oy = kp.pt
                
                error = np.sqrt((px - ox)**2 + (py - oy)**2)
                errors.append(error)
            
            if len(errors) > 0:
                mean_error = np.mean(errors)
                if mean_error > max_error:
                    points_to_remove.append(point_id)
        
        # Remove bad points
        for point_id in points_to_remove:
            del self.points_3d[point_id]
            if point_id in self.point_colors:
                del self.point_colors[point_id]
            # Clean up observations
            for img_idx, kp_idx in self.observations[point_id]:
                if (img_idx, kp_idx) in self.observation_index:
                    del self.observation_index[(img_idx, kp_idx)]
            del self.observations[point_id]
        
        print(f"  Pruned {len(points_to_remove)} points with reprojection error > {max_error}px")
        print(f"  Remaining: {len(self.points_3d)} points")
    
    def export_ply(self, output_file):
        """Export point cloud to PLY format"""
        if len(self.points_3d) == 0:
            print("No points to export!")
            return
        
        with open(output_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point_id in self.points_3d.keys():
                p = self.points_3d[point_id]
                c = self.point_colors.get(point_id, np.array([127, 127, 127]))
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        
        print(f"\n✓ Exported {len(self.points_3d)} points to {output_file}")
    
    def reconstruct(self, image_dir, output_file='point_cloud.ply', max_images=None, 
                    use_ba=True, full_pairwise=False, check_quality=False):
        """Full incremental SfM pipeline with optional Bundle Adjustment"""
        total_start = time.time()
        
        images = self.load_images(image_dir, max_images)
        if len(images) < 2:
            print("Need at least 2 images!")
            return None
        
        # Optional quality filtering
        if check_quality:
            print(f"\n{'='*60}")
            print("IMAGE QUALITY CHECK")
            print(f"{'='*60}")
            qc = ImageQualityChecker()
            original_count = len(images)
            images = qc.filter_images(images, verbose=True)
            if len(images) < 2:
                print("Not enough usable images after quality filter!")
                return None
            print(f"  ✓ {len(images)}/{original_count} images passed quality check")
        
        # Store images for dense triangulation later
        self.images = images
        
        print(f"\n{'='*60}")
        print("FEATURE DETECTION")
        print(f"{'='*60}")
        t0 = time.time()
        features = self.detect_features(images, detector='auto')
        print(f"Detection time: {time.time() - t0:.2f}s")
        
        print(f"\n{'='*60}")
        print("FEATURE MATCHING")
        print(f"{'='*60}")
        t0 = time.time()
        match_graph = self.match_all_pairs(features, images, full_pairwise=full_pairwise)
        print(f"Matching time: {time.time() - t0:.2f}s")
        
        if len(match_graph) == 0:
            print("No matches found!")
            return
        
        print(f"\n{'='*60}")
        print("INCREMENTAL RECONSTRUCTION")
        print(f"{'='*60}")
        success = self.incremental_reconstruction(images, features, match_graph)
        
        if not success:
            print("Reconstruction failed!")
            return
        
        # Normalize before BA
        print(f"\n{'='*60}")
        print("NORMALIZATION")
        print(f"{'='*60}")
        self.normalize_reconstruction()
        
        # Prune outliers by reprojection error
        print(f"\n{'='*60}")
        print("OUTLIER PRUNING")
        print(f"{'='*60}")
        self.prune_by_reprojection(features, max_error=8.0)
        
        # Bundle Adjustment
        if use_ba and len(self.points_3d) > 0:
            print(f"\n{'='*60}")
            print("BUNDLE ADJUSTMENT")
            print(f"{'='*60}")
            
            ba = BundleAdjuster(self.K)
            optimized = ba.optimize(
                self.camera_poses,
                self.points_3d,
                self.observations,
                features,
                images
            )
            
            if optimized:
                self.camera_poses, self.points_3d = optimized
                print("  ✓ Bundle Adjustment complete")
                
                # Prune again after BA
                self.prune_by_reprojection(features, max_error=5.0)
        
        # Export result
        print(f"\n{'='*60}")
        print("EXPORT")
        print(f"{'='*60}")
        self.export_ply(output_file)
        
        print(f"\n{'='*60}")
        print("RECONSTRUCTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Images processed: {len(self.camera_poses)}/{len(images)}")
        print(f"  3D points: {len(self.points_3d)}")
        print(f"  Output: {output_file}")
        
        # Return data for dense reconstruction
        points_array = np.array([self.points_3d[i] for i in sorted(self.points_3d.keys())])
        colors_array = np.array([self.point_colors.get(i, [127,127,127]) for i in sorted(self.points_3d.keys())])
        
        # Camera poses as dict of (R, t) tuples - already in this format
        cameras_dict = dict(self.camera_poses)
        
        return points_array, colors_array, cameras_dict


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Structure from Motion reconstruction')
    parser.add_argument('dataset', type=str, help='Dataset name (subfolder in data/samples/)')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--no-ba', action='store_true', help='Disable Bundle Adjustment')
    parser.add_argument('--output', type=str, default=None, help='Output PLY file path')
    parser.add_argument('--full-match', action='store_true', help='Full O(N²) pairwise matching (slow)')
    parser.add_argument('--dense', action='store_true', help='Run dense triangulation after sparse SfM')
    parser.add_argument('--filter', action='store_true', help='Apply point cloud filtering (SOR + ROR + voxel)')
    parser.add_argument('--check-quality', action='store_true', help='Filter low-quality images before processing')
    parser.add_argument('--voxel-divider', type=int, default=500, 
                        help='Voxel grid divider (higher = more points). 500=coarse, 1000=medium, 2000=fine, 0=disable')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    calibration_file = project_root / 'src' / 'calibration' / 'calibration_results' / 'calibration_data.npz'
    image_dir = project_root / 'data' / 'samples' / args.dataset
    
    if args.output:
        output_file = args.output
    else:
        output_file = project_root / 'data' / 'samples' / args.dataset / 'point_cloud.ply'
    
    if not calibration_file.exists():
        print(f"Calibration file not found: {calibration_file}")
        exit(1)
    
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        exit(1)
    
    print(f"Calibration: {calibration_file}")
    print(f"Images: {image_dir}")
    print(f"Output: {output_file}")
    
    # Run reconstruction
    reconstructor = SfMReconstructor(str(calibration_file))
    
    result = reconstructor.reconstruct(
        str(image_dir),
        str(output_file),
        max_images=args.max_images,
        use_ba=not args.no_ba,
        full_pairwise=args.full_match,
        check_quality=args.check_quality
    )
    
    if result is None:
        print("Reconstruction failed!")
        exit(1)
    
    points, colors, cameras = result
    
    # Dense triangulation
    if args.dense and len(cameras) >= 2:
        print(f"\n{'='*60}")
        print("DENSE TRIANGULATION")
        print(f"{'='*60}")
        
        # Load images for dense
        dense_images = reconstructor.images
        
        # Convert cameras to CameraPose format
        from dense_triangulation import CameraPose
        camera_poses = {}
        for idx, (R, t) in cameras.items():
            camera_poses[idx] = CameraPose(R=R, t=t.reshape(3, 1) if t.shape == (3,) else t)
        
        # Run dense triangulation
        triangulator = DenseTriangulator(reconstructor.K, reconstructor.dist)
        dense_points, dense_colors = triangulator.reconstruct(dense_images, camera_poses)
        
        if len(dense_points) > 0:
            points = dense_points
            colors = dense_colors
            print(f"  ✓ Dense points: {len(points):,}")
    
    # Point cloud filtering
    if args.filter and len(points) > 100:
        print(f"\n{'='*60}")
        print("POINT CLOUD FILTERING")
        print(f"{'='*60}")
        
        pf = PointCloudFilter(verbose=True)
        
        # Compute voxel size based on scene scale and user preference
        scene_scale = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
        if args.voxel_divider > 0:
            voxel_size = scene_scale / args.voxel_divider
            print(f"  Voxel divider: {args.voxel_divider} (size={voxel_size:.4f})")
        else:
            voxel_size = 0  # Disable voxel downsampling
            print(f"  Voxel downsampling: DISABLED")
        
        points, colors = pf.full_pipeline(
            points, colors,
            voxel_size=voxel_size,
            sor_k=30,
            sor_std=1.5
        )
        
        print(f"  ✓ Filtered points: {len(points):,}")
        
        # Re-export with filtered points
        output_path = Path(output_file)
        if args.dense:
            final_output = output_path.parent / "dense_filtered.ply"
        else:
            final_output = output_path.parent / "sparse_filtered.ply"
        
        # Save filtered result
        with open(final_output, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i].astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        
        print(f"  → Saved: {final_output}")
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")