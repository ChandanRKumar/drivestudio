import torch

def rasterize_fisheye(
    means2d,
    covs2d,
    rgbs,
    opacities,
    width: int,
    height: int,
    bg_color: torch.Tensor,
):
    """
    A pure PyTorch, but very slow, implementation of a 2D Gaussian rasterizer.

    Args:
        means2d (torch.Tensor): The 2D projected means of the Gaussians [N, 2].
        covs2d (torch.Tensor): The 2D projected covariances of the Gaussians [N, 2, 2].
        rgbs (torch.Tensor): The colors of the Gaussians [N, 3].
        opacities (torch.Tensor): The opacities of the Gaussians [N, 1].
        width (int): The width of the output image.
        height (int): The height of the output image.
        bg_color (torch.Tensor): The background color [3].
    """

    # Initialize an empty image and alpha buffer
    rendered_image = torch.zeros((height, width, 3), device=means2d.device)
    alpha_buffer = torch.zeros((height, width, 1), device=means2d.device)

    # Pre-calculate the inverse of the 2D covariances
    inv_covs2d = torch.linalg.inv(covs2d)

    # Create a grid of pixel coordinates
    px, py = torch.meshgrid(
        torch.arange(width, device=means2d.device),
        torch.arange(height, device=means2d.device),
        indexing="xy"
    )
    pixel_coords = torch.stack([px, py], dim=-1).float() + 0.5 # N_pixels, 2

    # Reshape for broadcasting
    pixel_coords = pixel_coords.view(-1, 2) # (H*W, 2)

    # Sort Gaussians by depth (approximated by assuming they are pre-sorted)
    # A real implementation would need to sort them based on their Z-value in camera space.
    # For this reference, we process them in the given order.

    # Iterate over each Gaussian (this is the slow part)
    for i in range(means2d.shape[0]):
        mean = means2d[i] # [2]
        inv_cov = inv_covs2d[i] # [2, 2]
        rgb = rgbs[i] # [3]
        opacity = opacities[i] # [1]

        # Calculate the Mahalanobis distance for all pixels relative to the current Gaussian
        delta = pixel_coords - mean # (H*W, 2)

        # This is equivalent to: (delta @ inv_cov) * delta
        mahalanobis_dist = torch.sum(
            (delta @ inv_cov) * delta,
            dim=-1
        )

        # Calculate the per-pixel alpha contribution from this Gaussian
        # The weight is an unnormalized Gaussian distribution
        gauss_weight = torch.exp(-0.5 * mahalanobis_dist)

        # Modulate by the learned opacity
        alpha_i = opacity * gauss_weight.view(height, width, 1)

        # Perform standard front-to-back alpha blending
        # T_i = 1 - alpha_buffer (Transmittance up to this point)
        # new_color = old_color + T_i * alpha_i * color_i
        rendered_image = rendered_image + (1.0 - alpha_buffer) * alpha_i * rgb

        # Update the alpha buffer
        # new_alpha = old_alpha + (1 - old_alpha) * alpha_i
        alpha_buffer = alpha_buffer + (1.0 - alpha_buffer) * alpha_i

    # Add background color where the scene is not opaque
    final_image = rendered_image + (1.0 - alpha_buffer) * bg_color

    return final_image, alpha_buffer
