"""Tests for configuration loading and validation."""

from pathlib import Path

import yaml

from src.utils.config import (
    AppConfig,
    ExtractionConfig,
    OCRConfig,
    PreprocessingConfig,
    ValidationConfig,
    load_config,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.deskew_enabled is True
        assert cfg.denoise_enabled is True
        assert cfg.binarize_enabled is True
        assert cfg.contrast_enabled is True
        assert cfg.denoise_method == "bilateral"
        assert cfg.clahe_clip_limit == 2.0

    def test_override(self) -> None:
        cfg = PreprocessingConfig(deskew_enabled=False, clahe_clip_limit=3.5)
        assert cfg.deskew_enabled is False
        assert cfg.clahe_clip_limit == 3.5


class TestOCRConfig:
    """Tests for OCRConfig defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = OCRConfig()
        assert cfg.default_lang == "eng"
        assert cfg.psm == 3
        assert cfg.pdf_dpi == 300
        assert cfg.tesseract_cmd is None

    def test_custom_lang(self) -> None:
        cfg = OCRConfig(default_lang="fra", psm=6)
        assert cfg.default_lang == "fra"
        assert cfg.psm == 6


class TestExtractionConfig:
    """Tests for ExtractionConfig defaults."""

    def test_defaults(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.use_ml is True
        assert cfg.ml_weight == 0.6
        assert cfg.rule_weight == 0.4
        assert cfg.confidence_threshold == 0.5


class TestValidationConfig:
    """Tests for ValidationConfig defaults."""

    def test_defaults(self) -> None:
        cfg = ValidationConfig()
        assert cfg.rules_path == "configs/validation_rules.yaml"


class TestAppConfig:
    """Tests for the top-level AppConfig."""

    def test_defaults(self) -> None:
        cfg = AppConfig()
        assert isinstance(cfg.preprocessing, PreprocessingConfig)
        assert isinstance(cfg.ocr, OCRConfig)
        assert isinstance(cfg.extraction, ExtractionConfig)
        assert isinstance(cfg.validation, ValidationConfig)
        assert cfg.log_level == "INFO"

    def test_nested_override(self) -> None:
        cfg = AppConfig(
            preprocessing=PreprocessingConfig(deskew_enabled=False),
            log_level="DEBUG",
        )
        assert cfg.preprocessing.deskew_enabled is False
        assert cfg.log_level == "DEBUG"


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_default_config(self) -> None:
        cfg = load_config(Path("configs/config.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.ocr.default_lang == "eng"

    def test_load_missing_file_returns_defaults(self) -> None:
        cfg = load_config(Path("/nonexistent/path/config.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.ocr.default_lang == "eng"

    def test_load_custom_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "preprocessing": {"deskew_enabled": False},
            "ocr": {"default_lang": "deu", "psm": 6},
            "log_level": "DEBUG",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_file)
        assert cfg.preprocessing.deskew_enabled is False
        assert cfg.ocr.default_lang == "deu"
        assert cfg.ocr.psm == 6
        assert cfg.log_level == "DEBUG"

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert isinstance(cfg, AppConfig)

    def test_load_none_defaults_to_standard_path(self) -> None:
        cfg = load_config()
        assert isinstance(cfg, AppConfig)
