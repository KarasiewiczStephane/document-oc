"""Configurable image preprocessing pipeline for document OCR.

Orchestrates deskew, denoise, contrast enhancement, and binarization
steps with quality metrics tracking.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import PreprocessingConfig
from src.utils.logger import get_logger

from .binarize import apply_clahe, binarize_adaptive, binarize_otsu
from .denoise import denoise
from .deskew import deskew

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Before/after image quality measurements."""

    sharpness_before: float
    sharpness_after: float
    contrast_before: float
    contrast_after: float


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Sharpness score (higher means sharper).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate image contrast as the standard deviation of pixel intensities.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Contrast score (higher means more contrast).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(gray.std())


class PreprocessingPipeline:
    """Configurable document image preprocessing pipeline.

    Applies a series of image processing steps based on the provided
    configuration, measuring quality before and after processing.

    Args:
        config: Preprocessing configuration controlling which steps to apply.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def process(self, image: np.ndarray) -> tuple[np.ndarray, QualityMetrics]:
        """Run the full preprocessing pipeline on an image.

        Args:
            image: Input document image (BGR or grayscale).

        Returns:
            Tuple of (processed_image, quality_metrics).
        """
        metrics = QualityMetrics(
            sharpness_before=calculate_sharpness(image),
            contrast_before=calculate_contrast(image),
            sharpness_after=0.0,
            contrast_after=0.0,
        )

        result = image.copy()

        if self.config.deskew_enabled:
            result = deskew(result)

        if self.config.denoise_enabled:
            result = denoise(result, method=self.config.denoise_method)

        if self.config.contrast_enabled:
            result = apply_clahe(
                result,
                clip_limit=self.config.clahe_clip_limit,
                tile_size=self.config.clahe_tile_size,
            )

        if self.config.binarize_enabled:
            if self.config.binarize_method == "otsu":
                result = binarize_otsu(result)
            else:
                result = binarize_adaptive(result)

        metrics.sharpness_after = calculate_sharpness(result)
        metrics.contrast_after = calculate_contrast(result)

        logger.info(
            "Preprocessing complete: sharpness %.1f->%.1f, contrast %.1f->%.1f",
            metrics.sharpness_before,
            metrics.sharpness_after,
            metrics.contrast_before,
            metrics.contrast_after,
        )
        return result, metrics
