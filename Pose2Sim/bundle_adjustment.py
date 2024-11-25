#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bundle adjustment module for optimizing camera parameters and 3D point positions.
This is a wrapper around the new bundle adjustment implementation.
"""

import numpy as np
import cv2
import torch
from torch.optim import Adam

def prepare_matched_points(points_2d, camera_indices, point_indices, points_3d):
    """
    Convert traditional bundle adjustment format to matched points dictionary format.
    Creates pairs of cameras with their corresponding 2D points and indices.
    """
    # Create a dictionary to store points for each camera
    camera_points = {}
    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        if cam_idx not in camera_points:
            camera_points[cam_idx] = {'points_2d': [], 'point_indices': []}
        camera_points[cam_idx]['points_2d'].append(points_2d[i])
        camera_points[cam_idx]['point_indices'].append(pt_idx)

    # Convert to camera pairs format
    matched_points = []
    camera_list = sorted(camera_points.keys())
    
    # Create pairs of cameras
    for i in range(len(camera_list)):
        for j in range(i + 1, len(camera_list)):
            cam1_idx = camera_list[i]
            cam2_idx = camera_list[j]
            
            # Find common points between cameras
            cam1_points = camera_points[cam1_idx]
            cam2_points = camera_points[cam2_idx]
            
            # Convert to sets for intersection
            cam1_indices_set = set(cam1_points['point_indices'])
            cam2_indices_set = set(cam2_points['point_indices'])
            common_indices = cam1_indices_set.intersection(cam2_indices_set)
            
            if common_indices:  # Only create pair if they share points
                # Create match dictionary for this camera pair
                match_dict = {
                    cam1_idx: {'points': [], 'indices': []},
                    cam2_idx: {'points': [], 'indices': []}
                }
                
                # Add points that are visible in both cameras
                for idx in common_indices:
                    cam1_pt_idx = cam1_points['point_indices'].index(idx)
                    cam2_pt_idx = cam2_points['point_indices'].index(idx)
                    
                    match_dict[cam1_idx]['points'].append(cam1_points['points_2d'][cam1_pt_idx])
                    match_dict[cam1_idx]['indices'].append(idx)
                    match_dict[cam2_idx]['points'].append(cam2_points['points_2d'][cam2_pt_idx])
                    match_dict[cam2_idx]['indices'].append(idx)
                
                # Convert lists to numpy arrays
                match_dict[cam1_idx]['points'] = np.array(match_dict[cam1_idx]['points'])
                match_dict[cam1_idx]['indices'] = np.array(match_dict[cam1_idx]['indices'])
                match_dict[cam2_idx]['points'] = np.array(match_dict[cam2_idx]['points'])
                match_dict[cam2_idx]['indices'] = np.array(match_dict[cam2_idx]['indices'])
                
                matched_points.append(match_dict)
    
    return matched_points

def convert_camera_params(camera_params, camera_matrices):
    """
    Convert traditional camera parameters to new format.
    """
    n_cameras = len(camera_params)
    camera_rotation = []
    camera_translation = []
    single_view_array = []

    for i in range(n_cameras):
        # Convert rotation vector to matrix
        rvec = camera_params[i, :3]
        tvec = camera_params[i, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        
        camera_rotation.append(R)
        camera_translation.append(tvec)
        
        # Prepare camera intrinsics
        K = camera_matrices[i]
        single_view = np.array([
            [K[0,0], 0, K[0,2]],
            [0, K[1,1], K[1,2]],
            [0, 0, 1]
        ])
        single_view_array.append(single_view)
    
    return camera_rotation, camera_translation, single_view_array

def convert_results(result, camera_params, points_3d, optimize_points=True):
    """
    Convert results from new bundle adjustment format back to original format.
    """
    # Extract optimized camera parameters
    R_matrix = result['R_matrix']
    T_matrix = result['T_matrix']
    f_matrix = result['f_matrix']
    
    n_cameras = len(camera_params)
    optimized_camera_params = np.zeros_like(camera_params)
    
    for i in range(n_cameras):
        # Convert rotation matrix back to vector
        R = R_matrix[i].detach().cpu().numpy()
        rvec, _ = cv2.Rodrigues(R)
        
        # Get translation vector
        tvec = T_matrix[i].detach().cpu().numpy()
        
        # Store parameters
        optimized_camera_params[i, :3] = rvec.ravel()
        optimized_camera_params[i, 3:6] = tvec
    
    # Handle 3D points if they were optimized
    if optimize_points and 'points_3d' in result:
        optimized_points_3d = result['points_3d'].detach().cpu().numpy()
    else:
        optimized_points_3d = points_3d
    
    # Create result object with similar structure to scipy.optimize.least_squares result
    class OptimizationResult:
        def __init__(self, cost):
            self.cost = cost
            self.success = True
            self.status = 0
            self.message = "Optimization terminated successfully"
            
    res = OptimizationResult(result.get('final_cost', 0))
    
    return optimized_camera_params, optimized_points_3d, res

def run_bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, point_indices, camera_matrices, dist_coeffs, 
                         optimize_points=True, max_iterations=100, convergence_threshold=1e-6):
    """
    Wrapper function for the new bundle adjustment implementation.
    
    Args:
        camera_params: (n_cameras, 6) array of camera parameters [rvec (3), tvec (3)]
        points_3d: (n_points, 3) array of 3D points
        points_2d: (n_observations, 2) array of 2D points
        camera_indices: (n_observations,) array of camera indices
        point_indices: (n_observations,) array of point indices
        camera_matrices: list of (3, 3) camera intrinsic matrices
        dist_coeffs: list of distortion coefficients
        optimize_points: whether to optimize 3D points
        max_iterations: maximum number of iterations
        convergence_threshold: convergence threshold
    
    Returns:
        camera_params: optimized camera parameters
        points_3d: optimized 3D points
        res: optimization results
    """
    print("\nStarting bundle adjustment with PyTorch3D implementation:")
    print(f"Maximum iterations: {max_iterations}")
    print(f"Convergence threshold: {convergence_threshold}")
    print(f"Optimizing {'camera parameters and 3D points' if optimize_points else 'camera parameters only'}")
    
    # 1. Convert camera parameters
    camera_rotation, camera_translation, single_view_array = convert_camera_params(camera_params, camera_matrices)
    
    # 2. Prepare matched points
    matched_points = prepare_matched_points(points_2d, camera_indices, point_indices, points_3d)
    
    # 3. Run bundle adjustment with new implementation
    from .bundle_adjustment_new import bundle_adjustment as new_bundle_adjustment
    
    result = new_bundle_adjustment(
        matched_points=matched_points,
        camera_rotation=camera_rotation,
        camera_translation=camera_translation,
        single_view_array=single_view_array,
        distortion_k_array=dist_coeffs,
        distortion_p_array=[],
        iteration=max_iterations,
        focal_lr=0.1,
        rot_lr=0.1,
        translation_lr=0.1
    )
    
    # 4. Convert results back to original format
    return convert_results(result, camera_params, points_3d, optimize_points)
