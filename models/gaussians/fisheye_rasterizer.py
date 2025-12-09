import torch

def rasterize_fisheye(projected_means, projected_covariances, rgbs, opacities, width, height):
    """
    A simplified rasterizer for fisheye cameras.
    """
    image = torch.zeros((height, width, 3), device=projected_means.device)

    for i in range(height):
        for j in range(width):
            pixel_coord = torch.tensor([j + 0.5, i + 0.5], device=projected_means.device)

            total_alpha = 0.0
            color = torch.zeros(3, device=projected_means.device)

            for k in range(projected_means.shape[0]):
                mean = projected_means[k]
                cov = projected_covariances[k]

                # Calculate the Mahalanobis distance
                delta = pixel_coord - mean
                mahalanobis_dist = torch.exp(-0.5 * (delta.T @ torch.inverse(cov) @ delta))

                alpha = opacities[k] * mahalanobis_dist

                # Alpha blending
                color = color + (1 - total_alpha) * alpha * rgbs[k]
                total_alpha = total_alpha + (1 - total_alpha) * alpha

            image[i, j] = color

    return image
