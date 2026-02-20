"""Tests for the hybrid extraction pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.extraction.hybrid import ExtractionResult, HybridExtractor, HybridField
from src.extraction.ml_extractor import MLExtractedField
from src.extraction.rule_extractor import ExtractedField
from src.utils.config import ExtractionConfig


class TestHybridField:
    """Tests for the HybridField data class."""

    def test_creation(self) -> None:
        field = HybridField(
            field_name="date",
            value="2024-01-15",
            confidence=0.85,
            source="hybrid",
        )
        assert field.field_name == "date"
        assert field.source == "hybrid"
        assert field.ml_value is None
        assert field.rule_value is None


class TestExtractionResult:
    """Tests for the ExtractionResult data class."""

    def test_creation(self) -> None:
        result = ExtractionResult(
            fields=[],
            raw_text="test",
            overall_confidence=0.0,
        )
        assert result.fields == []
        assert result.document_type is None


class TestHybridExtractor:
    """Tests for the HybridExtractor class."""

    def _make_extractor(
        self, ml_weight: float = 0.6, rule_weight: float = 0.4
    ) -> HybridExtractor:
        config = ExtractionConfig(
            use_ml=True,
            ml_weight=ml_weight,
            rule_weight=rule_weight,
            confidence_threshold=0.1,
        )
        return HybridExtractor(config)

    def test_rule_only_extraction(self) -> None:
        extractor = self._make_extractor()
        result = extractor.extract(
            image=None,
            ocr_text="Invoice #INV-001\nTotal: $500.00\nDate: 01/15/2024",
            use_ml=False,
        )
        assert isinstance(result, ExtractionResult)
        assert len(result.fields) > 0
        field_names = {f.field_name for f in result.fields}
        assert "amount" in field_names or "date" in field_names

    def test_rule_only_no_ml_when_disabled(self) -> None:
        config = ExtractionConfig(use_ml=False, confidence_threshold=0.1)
        extractor = HybridExtractor(config)
        result = extractor.extract(
            image=np.zeros((100, 200, 3), dtype=np.uint8),
            ocr_text="Total: $100.00",
        )
        assert all(f.source == "rule" for f in result.fields)

    def test_empty_text_returns_empty(self) -> None:
        extractor = self._make_extractor()
        result = extractor.extract(image=None, ocr_text="", use_ml=False)
        assert result.fields == []
        assert result.overall_confidence == 0.0

    def test_merge_rule_fields_only(self) -> None:
        extractor = self._make_extractor()
        rule_fields = [
            ExtractedField("date", "01/15/2024", 0.9, 0, 10, "regex"),
            ExtractedField("amount", "500.00", 0.95, 15, 25, "regex"),
        ]
        merged = extractor._merge_fields(rule_fields, [], 1.0)
        assert len(merged) == 2
        assert all(f.source == "rule" for f in merged)

    def test_merge_ml_fields_only(self) -> None:
        extractor = self._make_extractor()
        ml_fields = [
            MLExtractedField("date", "2024-01-15", 0.88, (10, 20, 100, 40)),
        ]
        merged = extractor._merge_fields([], ml_fields, 1.0)
        assert len(merged) == 1
        assert merged[0].source == "ml"

    def test_merge_hybrid_prefers_higher_confidence(self) -> None:
        extractor = self._make_extractor(ml_weight=0.6, rule_weight=0.4)
        rule_fields = [
            ExtractedField("date", "01/15/2024", 0.5, 0, 10, "regex"),
        ]
        ml_fields = [
            MLExtractedField("date", "January 15 2024", 0.95, (10, 20, 100, 40)),
        ]
        merged = extractor._merge_fields(rule_fields, ml_fields, 1.0)
        assert len(merged) == 1
        assert merged[0].source == "hybrid"
        assert merged[0].value == "January 15 2024"  # ML wins

    def test_merge_hybrid_rule_wins_when_higher(self) -> None:
        extractor = self._make_extractor(ml_weight=0.4, rule_weight=0.6)
        rule_fields = [
            ExtractedField("date", "01/15/2024", 0.95, 0, 10, "regex"),
        ]
        ml_fields = [
            MLExtractedField("date", "January 15 2024", 0.3, (10, 20, 100, 40)),
        ]
        merged = extractor._merge_fields(rule_fields, ml_fields, 1.0)
        assert len(merged) == 1
        assert merged[0].value == "01/15/2024"  # Rule wins

    def test_confidence_threshold_filters(self) -> None:
        config = ExtractionConfig(
            use_ml=False, confidence_threshold=0.5, rule_weight=0.4
        )
        extractor = HybridExtractor(config)
        rule_fields = [
            ExtractedField("vendor", "ACME", 0.3, 0, 4, "regex"),
        ]
        merged = extractor._merge_fields(rule_fields, [], 1.0)
        assert len(merged) == 0  # 0.3 * 0.4 = 0.12 < 0.5 threshold

    def test_ocr_confidence_scales_results(self) -> None:
        extractor = self._make_extractor()
        rule_fields = [
            ExtractedField("date", "01/15/2024", 0.9, 0, 10, "regex"),
        ]
        merged_high = extractor._merge_fields(rule_fields, [], 1.0)
        merged_low = extractor._merge_fields(rule_fields, [], 0.5)

        assert merged_high[0].confidence > merged_low[0].confidence

    def test_overall_confidence_calculation(self) -> None:
        extractor = self._make_extractor()
        result = extractor.extract(
            image=None,
            ocr_text="Total: $500.00",
            use_ml=False,
            ocr_confidence=1.0,
        )
        if result.fields:
            assert result.overall_confidence > 0.0

    def test_normalize_boxes(self) -> None:
        extractor = self._make_extractor()
        word = MagicMock()
        word.bbox.x = 100
        word.bbox.y = 50
        word.bbox.width = 200
        word.bbox.height = 30

        boxes = extractor._normalize_boxes([word], (1000, 2000))
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert x1 == 50
        assert y1 == 50
        assert x2 == 150
        assert y2 == 80

    def test_normalize_boxes_zero_dimensions(self) -> None:
        extractor = self._make_extractor()
        word = MagicMock()
        word.bbox.x = 10
        word.bbox.y = 10
        word.bbox.width = 50
        word.bbox.height = 20
        boxes = extractor._normalize_boxes([word], (0, 0))
        assert boxes == [(0, 0, 0, 0)]

    @patch("src.extraction.hybrid.LayoutLMExtractor")
    def test_ml_failure_falls_back_to_rules(self, mock_ml_cls: MagicMock) -> None:
        mock_ml = mock_ml_cls.return_value
        mock_ml.extract.side_effect = RuntimeError("Model failed")

        extractor = self._make_extractor()
        extractor._ml_extractor = mock_ml

        word = MagicMock()
        word.text = "Total"
        word.bbox.x = 10
        word.bbox.y = 10
        word.bbox.width = 50
        word.bbox.height = 20

        result = extractor.extract(
            image=np.zeros((100, 200, 3), dtype=np.uint8),
            ocr_text="Total: $500.00",
            ocr_words=[word],
            use_ml=True,
        )
        assert all(f.source == "rule" for f in result.fields)

    def test_multiple_rule_extractions_best_wins(self) -> None:
        extractor = self._make_extractor()
        rule_fields = [
            ExtractedField("date", "01/15/2024", 0.7, 0, 10, "regex"),
            ExtractedField("date", "2024-01-15", 0.9, 20, 30, "regex"),
        ]
        merged = extractor._merge_fields(rule_fields, [], 1.0)
        dates = [f for f in merged if f.field_name == "date"]
        assert len(dates) == 1
        assert dates[0].rule_confidence == 0.9
