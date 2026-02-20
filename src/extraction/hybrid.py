"""Hybrid extraction pipeline combining ML and rule-based approaches.

Merges results from both extraction engines with weighted confidence
scoring and intelligent fallback strategies.
"""

from dataclasses import dataclass

import numpy as np

from src.utils.config import ExtractionConfig
from src.utils.logger import get_logger

from .ml_extractor import LayoutLMExtractor, MLExtractedField
from .rule_extractor import ExtractedField, RuleExtractor

logger = get_logger(__name__)


@dataclass
class HybridField:
    """A field produced by the hybrid extraction pipeline."""

    field_name: str
    value: str
    confidence: float
    source: str
    ml_value: str | None = None
    rule_value: str | None = None
    ml_confidence: float = 0.0
    rule_confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Complete extraction result from the hybrid pipeline."""

    fields: list[HybridField]
    raw_text: str
    document_type: str | None = None
    overall_confidence: float = 0.0


class HybridExtractor:
    """Combines ML and rule-based extraction with smart merging.

    Uses weighted confidence scoring to resolve conflicts between
    ML and rule-based extraction results, with ML as optional
    enhancement and rule-based as reliable fallback.

    Args:
        config: Extraction configuration with weights and thresholds.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        self.rule_extractor = RuleExtractor()
        self._ml_extractor: LayoutLMExtractor | None = None
        self.ml_weight = config.ml_weight
        self.rule_weight = config.rule_weight
        self.confidence_threshold = config.confidence_threshold

    def _get_ml_extractor(self) -> LayoutLMExtractor:
        """Lazily initialize the ML extractor on first use.

        Returns:
            Initialized LayoutLMExtractor instance.
        """
        if self._ml_extractor is None:
            self._ml_extractor = LayoutLMExtractor(self.config.model_name)
        return self._ml_extractor

    def extract(
        self,
        image: np.ndarray | None,
        ocr_text: str,
        ocr_words: list | None = None,
        ocr_confidence: float = 1.0,
        use_ml: bool = True,
        image_shape: tuple[int, ...] | None = None,
    ) -> ExtractionResult:
        """Run hybrid extraction combining ML and rule-based results.

        Args:
            image: Document image (required for ML extraction).
            ocr_text: Full OCR text for rule-based extraction.
            ocr_words: OCR word objects with bounding boxes.
            ocr_confidence: Average OCR confidence score.
            use_ml: Whether to attempt ML extraction.
            image_shape: Shape of the source image for box normalization.

        Returns:
            Combined extraction results.
        """
        rule_fields = self.rule_extractor.extract(ocr_text)
        logger.debug("Rule extraction found %d fields", len(rule_fields))

        ml_fields: list[MLExtractedField] = []
        if use_ml and self.config.use_ml and image is not None and ocr_words:
            try:
                words = [w.text for w in ocr_words]
                boxes = self._normalize_boxes(ocr_words, image_shape or image.shape)
                ml_extractor = self._get_ml_extractor()
                ml_fields = ml_extractor.extract(image, words, boxes)
                logger.debug("ML extraction found %d fields", len(ml_fields))
            except Exception:
                logger.warning("ML extraction failed, using rule-only fallback")

        merged = self._merge_fields(rule_fields, ml_fields, ocr_confidence)
        overall_conf = (
            sum(f.confidence for f in merged) / len(merged) if merged else 0.0
        )

        return ExtractionResult(
            fields=merged,
            raw_text=ocr_text,
            overall_confidence=overall_conf,
        )

    def _normalize_boxes(
        self, words: list, image_shape: tuple[int, ...]
    ) -> list[tuple[int, int, int, int]]:
        """Normalize OCR bounding boxes to 0-1000 range for LayoutLM.

        Args:
            words: OCR word objects with bbox attributes.
            image_shape: ``(height, width, ...)`` of the source image.

        Returns:
            List of normalized ``(x1, y1, x2, y2)`` boxes.
        """
        h, w = image_shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for word in words:
            x1 = int(word.bbox.x * 1000 / w) if w else 0
            y1 = int(word.bbox.y * 1000 / h) if h else 0
            x2 = int((word.bbox.x + word.bbox.width) * 1000 / w) if w else 0
            y2 = int((word.bbox.y + word.bbox.height) * 1000 / h) if h else 0
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _merge_fields(
        self,
        rule_fields: list[ExtractedField],
        ml_fields: list[MLExtractedField],
        ocr_confidence: float,
    ) -> list[HybridField]:
        """Merge ML and rule-based extractions with weighted confidence.

        Args:
            rule_fields: Results from rule-based extraction.
            ml_fields: Results from ML extraction.
            ocr_confidence: OCR confidence factor.

        Returns:
            Merged and filtered hybrid fields.
        """
        merged: dict[str, HybridField] = {}

        for rf in rule_fields:
            key = rf.field_name
            weighted_conf = rf.confidence * self.rule_weight * ocr_confidence
            if key not in merged or rf.confidence > merged[key].rule_confidence:
                if key in merged:
                    merged[key].rule_value = rf.value
                    merged[key].rule_confidence = rf.confidence
                    if merged[key].source == "rule":
                        merged[key].value = rf.value
                        merged[key].confidence = weighted_conf
                else:
                    merged[key] = HybridField(
                        field_name=key,
                        value=rf.value,
                        confidence=weighted_conf,
                        source="rule",
                        rule_value=rf.value,
                        rule_confidence=rf.confidence,
                    )

        for mf in ml_fields:
            key = mf.field_name
            weighted_conf = mf.confidence * self.ml_weight * ocr_confidence
            if key not in merged:
                merged[key] = HybridField(
                    field_name=key,
                    value=mf.value,
                    confidence=weighted_conf,
                    source="ml",
                    ml_value=mf.value,
                    ml_confidence=mf.confidence,
                )
            else:
                existing = merged[key]
                existing.ml_value = mf.value
                existing.ml_confidence = mf.confidence
                existing.source = "hybrid"

                ml_score = mf.confidence * self.ml_weight
                rule_score = existing.rule_confidence * self.rule_weight

                if ml_score > rule_score:
                    existing.value = mf.value
                existing.confidence = (ml_score + rule_score) * ocr_confidence

        return [f for f in merged.values() if f.confidence >= self.confidence_threshold]
