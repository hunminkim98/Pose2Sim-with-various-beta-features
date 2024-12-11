import torch
from torch.func import jacrev, vmap, jacfwd
from einops import einsum
from typing import Callable

#@torch.compile(disable = sys.platform == "windows" or True)
def schur_solve(cam_block: torch.Tensor, cross_block: torch.Tensor, point_block: torch.Tensor, g_cam: torch.Tensor, g_points: torch.Tensor):
    N, D, _ = point_block.shape # N, D, D
    M, th, _ = cam_block.shape # M, th, th
    K, _, _, _ = cross_block.shape # K, M, th, D
    g_cam.shape # M, th
    
    
    assert N == K, "Assuming K is equal to N"
    point_block_inv = torch.linalg.inv(point_block) # N, D, D
    rhs = einsum(point_block_inv, cross_block, "k d1 d2, k m2 th2 d2 -> k m2 th2 d1")
    BTA_invB = einsum(cross_block, rhs, "k m1 th1 d1, k m2 th2 d1 -> m1 th1 m2 th2").reshape(M*th, M*th)
    S = torch.block_diag(*cam_block) - BTA_invB
    
    part_of_camera_rhs = einsum(point_block_inv, g_points,"n d1 d2, n d2 -> n d1")
    camera_rhs = (g_cam - einsum(cross_block, part_of_camera_rhs, "n m th d1, n d1 -> m th")).reshape(M*th)
    camera_params = torch.linalg.solve(S, camera_rhs).reshape(M, th)
    
    points_rhs = g_points - einsum(cross_block, camera_params, "n m th d2, m th -> n d2")
    point_coordinates = einsum(point_block_inv, points_rhs,"n d1 d2, n d2 -> n d1")
    return camera_params, point_coordinates

def compute_jacobians_residuals_and_loss(jacobian_operator_x, jacobian_operator_theta,
                      observations, theta, f, X, 
                      K, N, M, D, C, device = "cuda", dtype = torch.float32):
    # Create accumulator for x jacobians, each 3D point projects into (assumed) all images
    J_x = torch.zeros((N, M, 2, D), device = device, dtype = dtype)# TODO: obviously each point X is not seen in every image
    # Accumulator for theta jacobians, each theta produces residuals in its own image 
    J_theta = torch.zeros((K, M,  2, C), device = device, dtype = dtype)
    
    # Residuals
    loss = 0
    residuals = torch.zeros((K, M, 2), device = device, dtype = dtype)
    for image_idx in range(len(observations)):
        x_im, inds = observations[image_idx]
        theta_im = theta[image_idx]
        residuals_im = f(X[inds], theta_im, x_im) # list with residuals for each image
        loss += (residuals_im**2).sum()
        residuals[inds, image_idx] = residuals_im
        J_x[inds, image_idx] = jacobian_operator_x(X[inds], theta_im, x_im)
        J_theta[inds, image_idx] = jacobian_operator_theta(X[inds], theta_im, x_im)
    return J_x, J_theta, residuals, loss

#@torch.compile(disable = sys.platform == "windows" or True)
def compute_jacobian_operators(f) -> tuple[Callable, Callable]:
    jacobian_operator_x = vmap(jacrev(f, 0), in_dims = (0, None, 0))
    jacobian_operator_theta = jacfwd(f, 1)
    return jacobian_operator_x, jacobian_operator_theta

def lm_optimize(f, X_0, theta_0, observations, num_steps = 100, L_0 = 10, device = "cuda", dtype = torch.float64):
    # See https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm for details
    N, D = X_0.shape
    K = N # Assume num observations is equal to num 3D points
    M = len(observations)
    C = theta_0.shape[1]
    X = X_0.clone()
    theta = theta_0.clone()
    
    # 첫 번째 카메라의 초기 파라미터 저장
    ref_camera_params = theta[0].clone()
    
    L = L_0
    prev_loss = float('inf')
    min_L, max_L = 1e-15, 1e15
    convergence_threshold = 1e-8
    
    jacobian_operator_x, jacobian_operator_theta = compute_jacobian_operators(f)
    
    for step in range(num_steps):
        
        J_x, J_theta, residuals, loss = compute_jacobians_residuals_and_loss(
            jacobian_operator_x, jacobian_operator_theta,
            observations, theta, f, X, 
            K, N, M, D, C, 
            device = device, dtype = dtype)
        
        # Check convergence
        if abs(prev_loss - loss) < convergence_threshold * prev_loss:
            print(f"Converged at step {step}")
            break
            
        damp_x = L * torch.eye(D, device = device, dtype = dtype)[None]
        damp_theta = L * torch.eye(C, device = device, dtype = dtype)[None]
        
        # Compute J^T J terms
        camera_block = einsum(J_theta, J_theta, "k m obs th1, k m obs th2 -> m th1 th2") + damp_theta
        points_block = einsum(J_x, J_x, "n m obs d1, n m obs d2 -> n d1 d2") + damp_x
        cross_block = einsum(J_theta, J_x, "k m obs th, k m obs d -> k m th d")
        
        g_theta = -einsum(J_theta, residuals, "k m obs th, k m obs -> m th")
        g_points = -einsum(J_x, residuals, "n m obs d, n m obs -> n d")
        
        delta_theta, delta_x = schur_solve(camera_block, cross_block, points_block, g_theta, g_points)
        
        # 첫 번째 카메라의 업데이트는 0으로 설정 (고정)
        delta_theta[0].zero_()
        
        # 첫 번째 카메라 파라미터 복원
        theta_new = theta.clone()
        theta_new[0] = ref_camera_params
        theta_new += delta_theta
        
        X_new = X + delta_x
        
        loss_new = 0
        for image_idx in range(len(observations)):
            x_im, inds = observations[image_idx]
            theta_im = theta_new[image_idx]
            residuals_im = f(X_new[inds], theta_im, x_im, image_idx)
            loss_new += (residuals_im**2).sum()
            
        if loss_new < loss:
            L = max(min_L, L/10)
            theta = theta_new  # theta 업데이트 추가
            X = X_new
            prev_loss = loss
        else:
            L = min(max_L, L*10)
        print(f"Step {step}: loss={loss_new:.6f} prev_loss={loss:.6f} L={L:.6e}")
    return X, theta