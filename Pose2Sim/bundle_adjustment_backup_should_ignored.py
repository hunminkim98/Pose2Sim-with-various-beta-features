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

def cost_function(params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_matrices, dist_coeffs):
    """
    Compute residuals for bundle adjustment.
    
    Args:
        params: Array containing camera parameters (rotation, translation, focal_length) and 3D points
        n_cameras: Number of cameras
        n_points: Number of 3D points
        camera_indices: Array of camera indices for each observation
        point_indices: Array of point indices for each observation
        points_2d: Array of observed 2D points
        camera_matrices: List of camera matrices
        dist_coeffs: List of distortion coefficients
    
    Returns:
        residuals: Array of residuals
    """
    # Extract camera parameters and 3D points from params
    camera_params = params[:n_cameras * 7].reshape((n_cameras, 7))  # Now includes focal_length
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

def compute_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Compute the sparsity structure of the Jacobian matrix.
    """
    m = len(camera_indices) * 2 
    n = n_cameras * 7 + n_points * 3  # 7 parameters per camera (rvec, tvec, focal_length)
    
    sparsity = np.zeros((m, n), dtype=bool)
    
    for idx, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        # Mark camera parameters (including focal_length)
        sparsity[2*idx:2*idx + 2, cam_idx*7:(cam_idx + 1)*7] = True
        
        # Mark point parameters
        sparsity[2*idx:2*idx + 2, n_cameras*7 + pt_idx*3:n_cameras*7 + (pt_idx + 1)*3] = True
    
    return sparsity

def run_bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, point_indices, camera_matrices, dist_coeffs, max_iterations=5, convergence_threshold=1e-6):
    """
    Run bundle adjustment optimization.
    """
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    
    print("\nStarting iterative bundle adjustment:")
    print(f"Maximum iterations: {max_iterations}")
    print(f"Convergence threshold: {convergence_threshold}")
    
    # Initialize parameters with current focal lengths
    current_camera_params = np.zeros((n_cameras, 7))
    current_camera_params[:, :6] = camera_params  # Copy rotation and translation
    for i in range(n_cameras):
        current_camera_params[i, 6] = camera_matrices[i][0, 0]  # Initialize with current focal length
    
    current_points_3d = points_3d.copy()
    previous_cost = float('inf')
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Create sparsity matrix
        sparsity = compute_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices)
        
        # Run single iteration of optimization
        res = least_squares(cost_function, 
                          np.hstack((current_camera_params.ravel(), current_points_3d.ravel())),
                          jac_sparsity=sparsity,
                          verbose=2,
                          x_scale='jac',
                          ftol=1e-15,
                          gtol=1e-15,
                          xtol=1e-15,
                          loss='huber',
                          f_scale=1.0,
                          method='trf',
                          args=(n_cameras, n_points, camera_indices, point_indices, 
                                points_2d, camera_matrices, dist_coeffs))
        
        # Extract updated parameters
        current_camera_params = res.x[:n_cameras * 7].reshape((n_cameras, 7))
        current_points_3d = res.x[n_cameras * 7:].reshape((n_points, 3))
        
        # Print focal lengths
        print("\nFocal lengths:")
        for i in range(n_cameras):
            print(f"Camera {i+1}: {current_camera_params[i, 6]:.1f} (initial: {camera_matrices[i][0, 0]:.1f})")
        
        # Check convergence
        current_cost = res.cost
        relative_change = abs(previous_cost - current_cost) / previous_cost if previous_cost != 0 else float('inf')
        
        print(f"Current cost: {current_cost:.6f}")
        print(f"Relative change: {relative_change:.6f}")
        
        if relative_change < convergence_threshold:
            print(f"\nConverged after {iteration + 1} iterations")
            break
            
        previous_cost = current_cost
    
    # Return only rotation and translation for compatibility
    return current_camera_params[:, :6], current_points_3d, res
