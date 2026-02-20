"""Tests for the ML-based field extraction module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.extraction.ml_extractor import LABEL_MAP, LayoutLMExtractor, MLExtractedField


class TestLabelMap:
    """Tests for the BIO label mapping."""

    def test_contains_outside_label(self) -> None:
        assert LABEL_MAP[0] == "O"

    def test_contains_begin_labels(self) -> None:
        begin_labels = [v for v in LABEL_MAP.values() if v.startswith("B-")]
        assert len(begin_labels) >= 4

    def test_all_begin_have_inside(self) -> None:
        begin_fields = {v[2:] for v in LABEL_MAP.values() if v.startswith("B-")}
        inside_fields = {v[2:] for v in LABEL_MAP.values() if v.startswith("I-")}
        assert begin_fields == inside_fields


class TestMLExtractedField:
    """Tests for the MLExtractedField data class."""

    def test_creation(self) -> None:
        field = MLExtractedField(
            field_name="date",
            value="2024-01-15",
            confidence=0.92,
            bbox=(10, 20, 100, 40),
        )
        assert field.field_name == "date"
        assert field.value == "2024-01-15"
        assert field.confidence == 0.92


class TestLayoutLMExtractor:
    """Tests for the LayoutLMExtractor class (mocked model)."""

    @patch("src.extraction.ml_extractor.AutoModelForTokenClassification")
    @patch("src.extraction.ml_extractor.AutoProcessor")
    def _make_extractor(
        self, mock_proc_cls: MagicMock, mock_model_cls: MagicMock
    ) -> LayoutLMExtractor:
        """Create an extractor with mocked model and processor."""
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_proc

        extractor = LayoutLMExtractor(
            model_name="mock-model",
            device="cpu",
        )
        return extractor

    def test_extract_empty_words(self) -> None:
        extractor = self._make_extractor()
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = extractor.extract(image, [], [])
        assert result == []

    def test_aggregate_single_begin_field(self) -> None:
        extractor = self._make_extractor()
        words = ["2024-01-15"]
        boxes = [(100, 200, 300, 220)]
        predictions = [1]  # B-DATE
        confidences = [0.95]

        fields = extractor._aggregate_fields(words, boxes, predictions, confidences)
        assert len(fields) == 1
        assert fields[0].field_name == "date"
        assert fields[0].value == "2024-01-15"

    def test_aggregate_multi_token_field(self) -> None:
        extractor = self._make_extractor()
        words = ["January", "15", "2024"]
        boxes = [(100, 200, 150, 220)] * 3
        predictions = [1, 2, 2]  # B-DATE, I-DATE, I-DATE
        confidences = [0.9, 0.85, 0.88]

        fields = extractor._aggregate_fields(words, boxes, predictions, confidences)
        assert len(fields) == 1
        assert fields[0].value == "January 15 2024"
        assert fields[0].confidence == pytest.approx(0.8766, abs=0.01)

    def test_aggregate_multiple_different_fields(self) -> None:
        extractor = self._make_extractor()
        words = ["ACME", "Corp", "100.00"]
        boxes = [(50, 50, 150, 70)] * 3
        predictions = [3, 4, 5]  # B-VENDOR, I-VENDOR, B-TOTAL
        confidences = [0.9, 0.85, 0.95]

        fields = extractor._aggregate_fields(words, boxes, predictions, confidences)
        assert len(fields) == 2
        vendor = next(f for f in fields if f.field_name == "vendor")
        total = next(f for f in fields if f.field_name == "total")
        assert vendor.value == "ACME Corp"
        assert total.value == "100.00"

    def test_aggregate_outside_tokens_ignored(self) -> None:
        extractor = self._make_extractor()
        words = ["the", "date", "is", "2024-01-15"]
        boxes = [(10, 10, 50, 30)] * 4
        predictions = [0, 0, 0, 1]  # O, O, O, B-DATE
        confidences = [0.5, 0.5, 0.5, 0.95]

        fields = extractor._aggregate_fields(words, boxes, predictions, confidences)
        assert len(fields) == 1
        assert fields[0].field_name == "date"

    def test_aggregate_mismatched_inside_breaks_field(self) -> None:
        extractor = self._make_extractor()
        words = ["ACME", "100.00"]
        boxes = [(10, 10, 50, 30)] * 2
        predictions = [3, 6]  # B-VENDOR, I-TOTAL (mismatch)
        confidences = [0.9, 0.8]

        fields = extractor._aggregate_fields(words, boxes, predictions, confidences)
        assert len(fields) == 1
        assert fields[0].field_name == "vendor"
        assert fields[0].value == "ACME"

    def test_create_field(self) -> None:
        extractor = self._make_extractor()
        field = extractor._create_field("DATE", ["Jan", "2024"], [0.9, 0.8])
        assert field.field_name == "date"
        assert field.value == "Jan 2024"
        assert field.confidence == pytest.approx(0.85)

    def test_create_field_empty_confidences(self) -> None:
        extractor = self._make_extractor()
        field = extractor._create_field("VENDOR", [], [])
        assert field.confidence == 0.0

    @patch("src.extraction.ml_extractor.AutoModelForTokenClassification")
    @patch("src.extraction.ml_extractor.AutoProcessor")
    def test_extract_with_mocked_output(
        self, mock_proc_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        logits = torch.zeros(1, 3, 11)
        logits[0, 0, 1] = 5.0  # B-DATE
        logits[0, 1, 2] = 5.0  # I-DATE
        logits[0, 2, 0] = 5.0  # O

        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.__call__ = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.return_value = {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        mock_proc_cls.from_pretrained.return_value = mock_proc

        extractor = LayoutLMExtractor(model_name="mock", device="cpu")
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        fields = extractor.extract(
            image,
            ["Jan", "2024", "text"],
            [(100, 100, 200, 120)] * 3,
        )
        assert len(fields) == 1
        assert fields[0].field_name == "date"

    def test_device_selection_cpu(self) -> None:
        extractor = self._make_extractor()
        assert extractor.device == "cpu"
