"""Noise reduction filters for document images.

Provides Gaussian blur and bilateral filtering to reduce noise
while preserving text edges in scanned documents.
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def denoise_gaussian(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise.

    Args:
        image: Input image as a numpy array.
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Denoised image.
    """
    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    logger.debug("Applied Gaussian denoise with kernel_size=%d", kernel_size)
    return result


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: int = 75,
    sigma_space: int = 75,
) -> np.ndarray:
    """Apply bilateral filter to reduce noise while preserving edges.

    Args:
        image: Input image as a numpy array.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        Denoised image with edges preserved.
    """
    result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    logger.debug("Applied bilateral denoise with d=%d", d)
    return result


def denoise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Apply noise reduction using the specified method.

    Args:
        image: Input image as a numpy array.
        method: Denoising method, either ``"gaussian"`` or ``"bilateral"``.

    Returns:
        Denoised image.

    Raises:
        ValueError: If an unsupported method is specified.
    """
    if method == "gaussian":
        return denoise_gaussian(image)
    if method == "bilateral":
        return denoise_bilateral(image)
    raise ValueError(f"Unsupported denoise method: {method}")
