"""Tests for the image preprocessing pipeline."""

import numpy as np
import pytest

from src.preprocessing.binarize import (
    apply_clahe,
    binarize_adaptive,
    binarize_otsu,
)
from src.preprocessing.denoise import (
    denoise,
    denoise_bilateral,
    denoise_gaussian,
)
from src.preprocessing.deskew import deskew, detect_skew_angle
from src.preprocessing.pipeline import (
    PreprocessingPipeline,
    QualityMetrics,
    calculate_contrast,
    calculate_sharpness,
)
from src.utils.config import PreprocessingConfig


def _make_noisy_image(height: int = 200, width: int = 300) -> np.ndarray:
    """Create a synthetic noisy grayscale image for testing."""
    rng = np.random.default_rng(42)
    base = np.zeros((height, width), dtype=np.uint8)
    base[50:150, 50:250] = 200
    noise = rng.integers(0, 50, size=(height, width), dtype=np.uint8)
    return np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(
        np.uint8
    )


def _make_color_image(height: int = 200, width: int = 300) -> np.ndarray:
    """Create a synthetic BGR color image for testing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[50:150, 50:250] = (200, 200, 200)
    return image


class TestDeskew:
    """Tests for skew detection and correction."""

    def test_detect_skew_angle_no_lines(self) -> None:
        blank = np.zeros((100, 100), dtype=np.uint8)
        angle = detect_skew_angle(blank)
        assert angle == 0.0

    def test_detect_skew_angle_with_color_image(self) -> None:
        image = _make_color_image()
        angle = detect_skew_angle(image)
        assert isinstance(angle, float)

    def test_deskew_returns_same_shape(self) -> None:
        image = _make_noisy_image()
        result = deskew(image)
        assert result.shape == image.shape

    def test_deskew_no_correction_below_threshold(self) -> None:
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = deskew(blank, angle_threshold=0.5)
        np.testing.assert_array_equal(result, blank)

    def test_deskew_color_image(self) -> None:
        image = _make_color_image()
        result = deskew(image)
        assert result.shape == image.shape


class TestDenoise:
    """Tests for noise reduction functions."""

    def test_gaussian_reduces_variance(self) -> None:
        noisy = _make_noisy_image()
        denoised = denoise_gaussian(noisy, kernel_size=5)
        assert denoised.shape == noisy.shape
        assert denoised.var() <= noisy.var()

    def test_bilateral_preserves_shape(self) -> None:
        image = _make_noisy_image()
        result = denoise_bilateral(image)
        assert result.shape == image.shape

    def test_denoise_method_gaussian(self) -> None:
        image = _make_noisy_image()
        result = denoise(image, method="gaussian")
        assert result.shape == image.shape

    def test_denoise_method_bilateral(self) -> None:
        image = _make_noisy_image()
        result = denoise(image, method="bilateral")
        assert result.shape == image.shape

    def test_denoise_invalid_method_raises(self) -> None:
        image = _make_noisy_image()
        with pytest.raises(ValueError, match="Unsupported denoise method"):
            denoise(image, method="magic")


class TestBinarize:
    """Tests for binarization and contrast enhancement."""

    def test_otsu_produces_binary(self) -> None:
        image = _make_noisy_image()
        binary = binarize_otsu(image)
        unique_values = set(np.unique(binary))
        assert unique_values.issubset({0, 255})

    def test_otsu_with_color_image(self) -> None:
        image = _make_color_image()
        binary = binarize_otsu(image)
        assert len(binary.shape) == 2

    def test_adaptive_produces_binary(self) -> None:
        image = _make_noisy_image()
        binary = binarize_adaptive(image, block_size=11, c=2)
        unique_values = set(np.unique(binary))
        assert unique_values.issubset({0, 255})

    def test_adaptive_with_color_image(self) -> None:
        image = _make_color_image()
        binary = binarize_adaptive(image)
        assert len(binary.shape) == 2

    def test_clahe_enhances_contrast(self) -> None:
        image = _make_noisy_image()
        enhanced = apply_clahe(image, clip_limit=2.0, tile_size=8)
        assert enhanced.shape == image.shape[:2]
        assert enhanced.std() >= 0

    def test_clahe_with_color_image(self) -> None:
        image = _make_color_image()
        enhanced = apply_clahe(image)
        assert len(enhanced.shape) == 2


class TestQualityMetrics:
    """Tests for image quality measurement functions."""

    def test_sharpness_positive(self) -> None:
        image = _make_noisy_image()
        sharpness = calculate_sharpness(image)
        assert sharpness >= 0

    def test_sharpness_color_image(self) -> None:
        image = _make_color_image()
        sharpness = calculate_sharpness(image)
        assert isinstance(sharpness, float)

    def test_contrast_positive(self) -> None:
        image = _make_noisy_image()
        contrast = calculate_contrast(image)
        assert contrast >= 0

    def test_contrast_color_image(self) -> None:
        image = _make_color_image()
        contrast = calculate_contrast(image)
        assert isinstance(contrast, float)

    def test_blank_image_low_sharpness(self) -> None:
        blank = np.zeros((100, 100), dtype=np.uint8)
        assert calculate_sharpness(blank) == 0.0


class TestPreprocessingPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_pipeline_all_enabled(self) -> None:
        config = PreprocessingConfig()
        pipeline = PreprocessingPipeline(config)
        image = _make_noisy_image()
        result, metrics = pipeline.process(image)
        assert isinstance(result, np.ndarray)
        assert isinstance(metrics, QualityMetrics)
        assert metrics.sharpness_before >= 0
        assert metrics.contrast_before >= 0

    def test_pipeline_all_disabled(self) -> None:
        config = PreprocessingConfig(
            deskew_enabled=False,
            denoise_enabled=False,
            binarize_enabled=False,
            contrast_enabled=False,
        )
        pipeline = PreprocessingPipeline(config)
        image = _make_noisy_image()
        result, metrics = pipeline.process(image)
        np.testing.assert_array_equal(result, image)

    def test_pipeline_color_image(self) -> None:
        config = PreprocessingConfig()
        pipeline = PreprocessingPipeline(config)
        image = _make_color_image()
        result, metrics = pipeline.process(image)
        assert isinstance(result, np.ndarray)

    def test_pipeline_otsu_method(self) -> None:
        config = PreprocessingConfig(
            deskew_enabled=False,
            denoise_enabled=False,
            contrast_enabled=False,
            binarize_method="otsu",
        )
        pipeline = PreprocessingPipeline(config)
        image = _make_noisy_image()
        result, _ = pipeline.process(image)
        unique_values = set(np.unique(result))
        assert unique_values.issubset({0, 255})

    def test_pipeline_metrics_populated(self) -> None:
        config = PreprocessingConfig(
            deskew_enabled=False,
            denoise_enabled=True,
            contrast_enabled=True,
            binarize_enabled=False,
        )
        pipeline = PreprocessingPipeline(config)
        image = _make_noisy_image()
        _, metrics = pipeline.process(image)
        assert metrics.sharpness_after >= 0
        assert metrics.contrast_after >= 0
