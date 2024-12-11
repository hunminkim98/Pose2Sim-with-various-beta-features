#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bundle adjustment module for optimizing camera parameters and 3D point positions.
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def project_points(points_3d, camera_params, camera_matrix, dist_coeffs):
    """
    Project 3D points to 2D using camera parameters.
    
    Args:
        points_3d: (N, 3) array of 3D points
        camera_params: Camera parameters (rvec, tvec, focal_length)
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    
    Returns:
        points_2d: (N, 2) array of projected 2D points
    """
    rvec = camera_params[:3]
    tvec = camera_params[3:6]
    focal_length = camera_params[6] if len(camera_params) > 6 else camera_matrix[0, 0]
    
    # Update camera matrix with current focal length
    camera_matrix_current = camera_matrix.copy()
    camera_matrix_current[0, 0] = focal_length  # fx
    camera_matrix_current[1, 1] = focal_length  # fy
    
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix_current, dist_coeffs)
    return points_2d.reshape(-1, 2)

def compute_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices, optimize_points=True):
    """
    Compute the sparsity structure of the Jacobian matrix.
    
    Args:
        optimize_points: If True, optimize 3D points. If False, only optimize camera parameters.
    """
    m = len(camera_indices) * 2 
    n = n_cameras * 7  # camera parameters (rvec, tvec, focal_length)
    if optimize_points:
        n += n_points * 3  # Add 3D points if optimizing them
    
    sparsity = np.zeros((m, n), dtype=bool)
    
    for idx, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        # Mark camera parameters (including focal_length)
        sparsity[2*idx:2*idx + 2, cam_idx*7:(cam_idx + 1)*7] = True
        
        # Mark point parameters if optimizing points
        if optimize_points:
            sparsity[2*idx:2*idx + 2, n_cameras*7 + pt_idx*3:n_cameras*7 + (pt_idx + 1)*3] = True
    
    return sparsity

def cost_function(params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_matrices, dist_coeffs, points_3d_fixed=None, optimize_points=True):
    """
    Compute residuals for bundle adjustment.
    
    Args:
        optimize_points: If True, optimize 3D points. If False, use points_3d_fixed.
    """
    # Extract camera parameters
    camera_params = params[:n_cameras * 7].reshape((n_cameras, 7))
    
    # Get 3D points
    if not optimize_points:
        points_3d = points_3d_fixed
    else:
        points_3d = params[n_cameras * 7:].reshape((n_points, 3))

    # Project 3D points to 2D for all cameras
    points_proj = np.zeros_like(points_2d)
    for i in range(len(points_2d)):
        cam_idx = camera_indices[i]
        point_idx = point_indices[i]
        point_3d = points_3d[point_idx]
        point_proj = project_points(point_3d[None], 
                                  camera_params[cam_idx], 
                                  camera_matrices[cam_idx], 
                                  dist_coeffs[cam_idx])
        points_proj[i] = point_proj
    
    # Compute residuals
    residuals = (points_proj - points_2d).ravel()
    return residuals

def run_bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, point_indices, camera_matrices, dist_coeffs, 
                         optimize_points=True, max_iterations=5, convergence_threshold=1e-6):
    """
    Run bundle adjustment optimization.
    
    Args:
        optimize_points: If True, optimize both camera parameters and 3D points.
                       If False, only optimize camera parameters.
    Returns:
        camera_params: Camera parameters [rvec (3), tvec (3), focal_length (1)]
        points_3d: 3D points if optimize_points=True, else input points_3d
        res: Optimization results
    """
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    
    print("\nStarting iterative bundle adjustment:")
    print(f"Maximum iterations: {max_iterations}")
    print(f"Convergence threshold: {convergence_threshold}")
    print(f"Optimizing {'camera parameters and 3D points' if optimize_points else 'camera parameters only'}")
    
    # Initialize parameters with current focal lengths
    current_camera_params = np.zeros((n_cameras, 7))
    current_camera_params[:, :6] = camera_params  # Copy rotation and translation
    initial_camera_params = camera_params.copy()  # Store initial values
    initial_points_3d = points_3d.copy() if optimize_points else None  # Store initial 3D points
    
    for i in range(n_cameras):
        current_camera_params[i, 6] = camera_matrices[i][0, 0]  # Initialize with current focal length
    
    # Define parameter scales for better conditioning
    x_scales = np.ones(n_cameras * 7)
    for i in range(n_cameras):
        x_scales[i*7:i*7+3] = 1.0      # rotation (radians)
        x_scales[i*7+3:i*7+6] = 1.0    # translation (mm)
        x_scales[i*7+6] = 100.0        # focal length (pixels)
    
    if optimize_points:
        current_points_3d = points_3d.copy()
        x_scales = np.hstack((x_scales, np.ones(n_points * 3)))  # 3D point scales (mm)
    
    previous_cost = float('inf')
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Create sparsity matrix
        sparsity = compute_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices, optimize_points)
        
        # Prepare optimization parameters
        if optimize_points:
            x0 = np.hstack((current_camera_params.ravel(), current_points_3d.ravel()))
        else:
            x0 = current_camera_params.ravel()
        
        # Run single iteration of optimization
        res = least_squares(cost_function, 
                          x0,
                          jac_sparsity=sparsity,
                          verbose=2,
                          x_scale=x_scales,
                          ftol=1e-4,
                          gtol=1e-4,
                          xtol=1e-4,
                          loss='huber',
                          method='trf',
                          args=(n_cameras, n_points, camera_indices, point_indices, 
                                points_2d, camera_matrices, dist_coeffs, points_3d, optimize_points))
        
        # Extract updated parameters
        current_camera_params = res.x[:n_cameras * 7].reshape((n_cameras, 7))
        if optimize_points:
            current_points_3d = res.x[n_cameras * 7:].reshape((n_points, 3))
        
        # Print parameter changes
        print("\nCamera parameter changes:")
        for i in range(n_cameras):
            # Get rotation changes in degrees
            init_rvec = initial_camera_params[i, :3]
            curr_rvec = current_camera_params[i, :3]
            init_R, _ = cv2.Rodrigues(init_rvec)
            curr_R, _ = cv2.Rodrigues(curr_rvec)
            R_diff = np.dot(curr_R.T, init_R)
            angles_diff, _ = cv2.Rodrigues(R_diff)
            angle_change_deg = np.linalg.norm(angles_diff) * 180 / np.pi
            
            # Get translation changes in mm
            init_tvec = initial_camera_params[i, 3:6]
            curr_tvec = current_camera_params[i, 3:6]
            trans_change = curr_tvec - init_tvec
            trans_change_norm = np.linalg.norm(trans_change)
            
            print(f"\nCamera {i+1}:")
            print(f"Focal length: {current_camera_params[i, 6]:.1f} (change: {current_camera_params[i, 6] - camera_matrices[i][0, 0]:.1f})")
            print(f"Rotation change: {angle_change_deg:.2f} degrees")
            print(f"Translation change: {trans_change_norm:.2f} mm (dX={trans_change[0]:.2f}, dY={trans_change[1]:.2f}, dZ={trans_change[2]:.2f})")
        
        if optimize_points and iteration == 0:  # Only print point changes after first iteration
            point_changes = current_points_3d - initial_points_3d
            point_change_norms = np.linalg.norm(point_changes, axis=1) / 1000  # Convert to meters
            print("\n3D Point changes (meters):")
            for i in range(n_points):
                change = point_changes[i] / 1000
                print(f"Point {i}: dX={change[0]:.6f}, dY={change[1]:.6f}, dZ={change[2]:.6f}, total={point_change_norms[i]:.6f}")
        
        # Check convergence
        current_cost = res.cost
        if previous_cost > 0:
            relative_change = abs(previous_cost - current_cost) / previous_cost
        elif current_cost == 0 and previous_cost == 0:
            relative_change = 0
        else:
            relative_change = float('inf')
        
        print(f"\nCurrent cost: {current_cost:.6f}")
        print(f"Relative change: {relative_change:.6f}")
        
        if relative_change < convergence_threshold:
            print(f"\nConverged after {iteration + 1} iterations")
            break
            
        previous_cost = current_cost
    
    if optimize_points:
        return current_camera_params, current_points_3d, res  # 전체 camera_params 반환 (focal length 포함)
    else:
        return current_camera_params, points_3d, res  # 전체 camera_params 반환 (focal length 포함)
