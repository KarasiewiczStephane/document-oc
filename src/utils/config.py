"""Configuration management for the document OCR system.

Loads and validates YAML configuration with sensible defaults
for preprocessing, OCR, extraction, and validation settings.
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for image preprocessing pipeline."""

    deskew_enabled: bool = True
    denoise_enabled: bool = True
    denoise_method: str = "bilateral"
    binarize_enabled: bool = True
    binarize_method: str = "adaptive"
    contrast_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8


class OCRConfig(BaseModel):
    """Configuration for Tesseract OCR engine."""

    tesseract_cmd: str | None = None
    default_lang: str = "eng"
    psm: int = 3
    pdf_dpi: int = 300


class ExtractionConfig(BaseModel):
    """Configuration for field extraction."""

    use_ml: bool = True
    model_name: str = "microsoft/layoutlm-base-uncased"
    ml_weight: float = 0.6
    rule_weight: float = 0.4
    confidence_threshold: float = 0.5


class ValidationConfig(BaseModel):
    """Configuration for validation rules engine."""

    rules_path: str = "configs/validation_rules.yaml"
    templates_path: str = "configs/templates.yaml"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    log_level: str = "INFO"


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.
            Defaults to configs/config.yaml.

    Returns:
        Validated application configuration.
    """
    if path is None:
        path = Path("configs/config.yaml")

    if path.exists():
        logger.info("Loading configuration from %s", path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)

    logger.info("No config file found at %s, using defaults", path)
    return AppConfig()
