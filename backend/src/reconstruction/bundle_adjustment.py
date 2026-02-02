"""
Bundle Adjustment implementation
Optimizes camera poses and 3D points jointly by minimizing reprojection error
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2 as cv

class BundleAdjuster:
    """
    Bundle Adjustment: Global optimization of camera poses and 3D structure
    """
    
    def __init__(self, K, dist=None):
        self.K = K
        self.dist = dist if dist is not None else np.zeros(5)
    
    def project_point(self, X, rvec, tvec):
        """Project 3D point to 2D image coordinates"""
        R, _ = cv.Rodrigues(rvec)
        X_cam = R @ X + tvec
        
        if X_cam[2] <= 0:
            return None
        
        x_normalized = X_cam[:2] / X_cam[2]
        
        r2 = np.sum(x_normalized**2)
        k1, k2, p1, p2, k3 = self.dist.ravel()
        
        radial_distortion = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        x_distorted = x_normalized * radial_distortion
        
        x_distorted[0] += 2*p1*x_normalized[0]*x_normalized[1] + p2*(r2 + 2*x_normalized[0]**2)
        x_distorted[1] += p1*(r2 + 2*x_normalized[1]**2) + 2*p2*x_normalized[0]*x_normalized[1]
        
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        
        x_pixel = np.array([
            fx * x_distorted[0] + cx,
            fy * x_distorted[1] + cy
        ])
        
        return x_pixel
    
    def residuals(self, params_normalized, n_cameras, n_points, camera_indices, point_indices, points_2d,
                  camera_scale, point_scale):
        """
        Compute residuals with DENORMALIZED parameters
        
        Args:
            camera_scale: shape (n_cameras * 6,)
            point_scale: scalar value (single scale for all points)
        """
        # Denormalize camera parameters
        camera_params = (params_normalized[:n_cameras * 6] * camera_scale).reshape((n_cameras, 6))
        
        # Denormalize point parameters - point_scale is a SCALAR
        point_params = params_normalized[n_cameras * 6:] * point_scale
        points_3d = point_params.reshape((n_points, 3))
        
        residuals = []
        
        for i in range(len(camera_indices)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            
            rvec = camera_params[cam_idx, :3]
            tvec = camera_params[cam_idx, 3:]
            X = points_3d[pt_idx]
            
            projected = self.project_point(X, rvec, tvec)
            
            if projected is None:
                residuals.extend([1000.0, 1000.0])
            else:
                observed = points_2d[i]
                error = projected - observed
                residuals.extend([error[0], error[1]])
        
        return np.array(residuals)
    
    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        """Build sparsity structure of Jacobian matrix"""
        n_observations = len(camera_indices)
        m = n_observations * 2
        n = n_cameras * 6 + n_points * 3
        
        A = lil_matrix((m, n), dtype=int)
        
        i = np.arange(n_observations)
        
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1
        
        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
        
        return A
    
    def optimize(self, camera_poses, points_3d, observations, features, images, 
                 max_nfev=100, verbose=2):
        """Run bundle adjustment with parameter normalization"""
        
        # Create index mappings
        camera_idx_map = {img_idx: i for i, img_idx in enumerate(sorted(camera_poses.keys()))}
        point_idx_map = {pt_id: i for i, pt_id in enumerate(sorted(points_3d.keys()))}
        
        n_cameras = len(camera_poses)
        n_points = len(points_3d)
        
        print(f"\n{'='*60}")
        print("BUNDLE ADJUSTMENT")
        print(f"{'='*60}")
        print(f"Cameras: {n_cameras}")
        print(f"3D points: {n_points}")
        
        # Initialize parameter vector
        camera_params = np.zeros(n_cameras * 6)
        point_params = np.zeros(n_points * 3)
        
        # Pack camera parameters
        for img_idx, (R, t) in camera_poses.items():
            cam_i = camera_idx_map[img_idx]
            rvec, _ = cv.Rodrigues(R)
            camera_params[cam_i * 6:cam_i * 6 + 3] = rvec.ravel()
            camera_params[cam_i * 6 + 3:cam_i * 6 + 6] = t.ravel()
        
        # Pack 3D point parameters
        for pt_id, pt_3d in points_3d.items():
            pt_i = point_idx_map[pt_id]
            point_params[pt_i * 3:pt_i * 3 + 3] = pt_3d
        
        # Compute scale factors
        # Camera scale: per-parameter scaling
        camera_scale = np.ones(n_cameras * 6)
        for cam_i in range(n_cameras):
            tvec_magnitude = np.abs(camera_params[cam_i * 6 + 3:cam_i * 6 + 6]).mean()
            if tvec_magnitude < 0.01:
                tvec_magnitude = 1.0
            camera_scale[cam_i * 6 + 3:cam_i * 6 + 6] = tvec_magnitude
        
        # Point scale: SINGLE scalar for all points
        point_scale = np.abs(point_params).mean() if n_points > 0 else 1.0
        if point_scale < 0.01:
            point_scale = 1.0
        
        print(f"Scale factors:")
        print(f"  Camera rotation scale: {camera_scale[0]:.4f}")
        print(f"  Camera translation scale (avg): {camera_scale[3::6].mean():.4f}")
        print(f"  Point scale: {point_scale:.4f}")
        
        # Normalize parameters
        camera_params_norm = camera_params / camera_scale
        point_params_norm = point_params / point_scale
        
        x0 = np.hstack([camera_params_norm, point_params_norm])
        
        # Build observation arrays
        camera_indices = []
        point_indices = []
        points_2d = []
        
        for pt_id, obs_list in observations.items():
            if pt_id not in point_idx_map:
                continue
            pt_i = point_idx_map[pt_id]
            
            for img_idx, kp_idx in obs_list:
                if img_idx not in camera_idx_map:
                    continue
                
                cam_i = camera_idx_map[img_idx]
                kp = features[img_idx]['keypoints'][kp_idx]
                pt_2d = np.array(kp.pt)
                
                camera_indices.append(cam_i)
                point_indices.append(pt_i)
                points_2d.append(pt_2d)
        
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        points_2d = np.array(points_2d)
        
        n_observations = len(camera_indices)
        print(f"Observations: {n_observations}")
        print(f"Parameters to optimize: {len(x0)}")
        
        # Compute sparsity pattern
        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        
        # Initial residual
        res_initial = self.residuals(x0, n_cameras, n_points,
                                    camera_indices, point_indices, points_2d,
                                    camera_scale, point_scale)
        
        rmse_initial = np.sqrt(np.mean(res_initial**2))
        print(f"\nInitial RMSE: {rmse_initial:.4f} pixels")
        
        # Check for numerical issues
        if rmse_initial > 1e6:
            print(f"\n⚠ WARNING: Very high initial RMSE ({rmse_initial:.2e})")
            print("  This indicates poor initial reconstruction.")
            print("  Skipping Bundle Adjustment to avoid divergence.")
            return camera_poses, points_3d
        
        # Run optimization
        print("\nOptimizing...")
        result = least_squares(
            self.residuals,
            x0,
            jac_sparsity=A,
            verbose=verbose,
            x_scale='jac',
            ftol=1e-4,
            xtol=1e-6,
            gtol=1e-4,
            method='trf',
            max_nfev=max_nfev,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d,
                  camera_scale, point_scale)
        )
        
        # Extract optimized parameters
        x_opt = result.x
        
        # Denormalize
        camera_params_opt = (x_opt[:n_cameras * 6] * camera_scale).reshape((n_cameras, 6))
        point_params_opt = (x_opt[n_cameras * 6:] * point_scale).reshape((n_points, 3))
        
        # Compute final residual
        rmse_final = np.sqrt(np.mean(result.fun**2))
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULT")
        print(f"{'='*60}")
        print(f"Initial RMSE: {rmse_initial:.4f} pixels")
        print(f"Final RMSE:   {rmse_final:.4f} pixels")
        
        if rmse_initial > 0:
            improvement = 100 * (rmse_initial - rmse_final) / rmse_initial
            print(f"Improvement:  {rmse_initial - rmse_final:.4f} pixels ({improvement:.1f}%)")
        
        print(f"Function evaluations: {result.nfev}")
        print(f"Jacobian evaluations: {result.njev}")
        print(f"Status: {result.message}")
        
        # Unpack optimized camera poses
        camera_poses_opt = {}
        for img_idx, cam_i in camera_idx_map.items():
            rvec = camera_params_opt[cam_i, :3]
            tvec = camera_params_opt[cam_i, 3:]
            R, _ = cv.Rodrigues(rvec)
            camera_poses_opt[img_idx] = (R, tvec.reshape(3, 1))
        
        # Unpack optimized 3D points
        points_3d_opt = {}
        for pt_id, pt_i in point_idx_map.items():
            points_3d_opt[pt_id] = point_params_opt[pt_i]
        
        return camera_poses_opt, points_3d_opt