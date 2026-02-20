"""Binarization and contrast enhancement for document images.

Provides Otsu's thresholding, adaptive thresholding, and CLAHE
contrast enhancement to improve text readability for OCR.
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale if it has color channels.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """Binarize an image using Otsu's automatic thresholding.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Binary image with pixel values 0 or 255.
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug("Applied Otsu binarization")
    return binary


def binarize_adaptive(
    image: np.ndarray, block_size: int = 11, c: int = 2
) -> np.ndarray:
    """Binarize an image using adaptive Gaussian thresholding.

    Args:
        image: Input image (BGR or grayscale).
        block_size: Size of the pixel neighborhood for threshold calculation.
        c: Constant subtracted from the mean.

    Returns:
        Binary image with pixel values 0 or 255.
    """
    gray = _to_gray(image)
    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    logger.debug("Applied adaptive binarization (block=%d, c=%d)", block_size, c)
    return result


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image (BGR or grayscale).
        clip_limit: Threshold for contrast limiting.
        tile_size: Size of the grid for histogram equalization.

    Returns:
        Contrast-enhanced grayscale image.
    """
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    result = clahe.apply(gray)
    logger.debug("Applied CLAHE (clip=%.1f, tile=%d)", clip_limit, tile_size)
    return result
