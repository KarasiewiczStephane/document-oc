"""Shared test fixtures for the document OCR test suite."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a simple synthetic grayscale test image."""
    image = np.zeros((200, 300), dtype=np.uint8)
    image[50:150, 50:250] = 255
    return image


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """Create a simple synthetic BGR test image."""
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[50:150, 50:250] = (255, 255, 255)
    return image


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the configs directory path."""
    return project_root / "configs"
