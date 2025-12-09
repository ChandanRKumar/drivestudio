import torch
import math

def project_point_fisheye(points_3d, camera):
    """
    Projects a batch of 3D points to 2D pixels using a fisheye camera model.
    """
    # From 3dgrut/threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh
    eps = torch.finfo(torch.float32).eps
    rho = torch.linalg.norm(points_3d[..., :2], dim=-1)
    rho = torch.max(rho, eps)

    theta_full = torch.atan2(rho, points_3d[..., 2])

    # For now, we assume the angle is always within the valid range
    theta = theta_full

    theta2 = theta * theta

    # The radial distortion coefficients are stored in the `distortions` attribute of the camera
    k1, k2, k3, k4 = camera.distortions[0][:4]

    # Horner's method for polynomial evaluation
    radial_poly = k4
    radial_poly = radial_poly * theta2 + k3
    radial_poly = radial_poly * theta2 + k2
    radial_poly = radial_poly * theta2 + k1

    delta = (theta * (radial_poly * theta2 + 1.0)) / rho

    focal_length = torch.tensor([camera.intrinsics[0][0,0], camera.intrinsics[0][1,1]], device=points_3d.device)
    principal_point = torch.tensor([camera.intrinsics[0][0,2], camera.intrinsics[0][1,2]], device=points_3d.device)

    projected = focal_length * points_3d[..., :2] * delta[..., None] + principal_point

    return projected

def unscented_transform(means, covariances, camera):
    """
    A vectorized Python implementation of the Unscented Transform.
    """
    # From 3dgrut/threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutProjector.cuh

    num_gaussians = means.shape[0]

    # Parameters for the Unscented Transform
    D = 3  # Dimensionality of the state space
    alpha = 0.5
    kappa = 0.0
    beta = 2.0

    lambda_ = alpha**2 * (D + kappa) - D

    # Generate sigma points
    sigma_points = torch.zeros((num_gaussians, 2 * D + 1, D), device=means.device)
    sigma_points[:, 0, :] = means

    sqrt_covs = torch.linalg.cholesky((D + lambda_) * covariances)

    for i in range(D):
        sigma_points[:, i + 1, :] = means + sqrt_covs[:, i, :]
        sigma_points[:, i + D + 1, :] = means - sqrt_covs[:, i, :]

    # Project sigma points
    projected_sigma_points = project_point_fisheye(sigma_points.view(-1, D), camera).view(num_gaussians, 2 * D + 1, 2)

    # Calculate weights
    weights_mean = torch.zeros(2 * D + 1, device=means.device)
    weights_cov = torch.zeros(2 * D + 1, device=means.device)

    weights_mean[0] = lambda_ / (D + lambda_)
    weights_cov[0] = lambda_ / (D + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2 * D + 1):
        weights_mean[i] = 1 / (2 * (D + lambda_))
        weights_cov[i] = 1 / (2 * (D + lambda_))

    # Calculate projected mean and covariance
    projected_means = torch.sum(weights_mean[None, :, None] * projected_sigma_points, dim=1)

    centered_points = projected_sigma_points - projected_means[:, None, :]
    projected_covariances = torch.sum(weights_cov[None, :, None, None] * (centered_points[..., :, None] @ centered_points[..., None, :]), dim=1)

    return projected_means, projected_covariances
