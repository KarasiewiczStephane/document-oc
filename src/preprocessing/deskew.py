"""Deskew correction for scanned document images.

Detects and corrects rotational skew using Hough line transform
to improve OCR accuracy on tilted scans.
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_skew_angle(image: np.ndarray) -> float:
    """Detect the skew angle of a document image.

    Uses Hough line transform on edge-detected image to find
    dominant line angles and returns the median angle.

    Args:
        image: Input image as a numpy array (BGR or grayscale).

    Returns:
        Estimated skew angle in degrees.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        logger.debug("No lines detected for skew estimation")
        return 0.0

    angles = [
        np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in lines[:, 0]
    ]
    median_angle = float(np.median(angles))
    logger.debug("Detected skew angle: %.2f degrees", median_angle)
    return median_angle


def deskew(image: np.ndarray, angle_threshold: float = 0.5) -> np.ndarray:
    """Correct rotational skew in a document image.

    Args:
        image: Input image as a numpy array (BGR or grayscale).
        angle_threshold: Minimum angle (degrees) to trigger correction.

    Returns:
        Deskewed image with the same shape and dtype as input.
    """
    angle = detect_skew_angle(image)

    if abs(angle) < angle_threshold:
        logger.debug("Skew angle below threshold, skipping correction")
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("Applied deskew correction: %.2f degrees", angle)
    return result
