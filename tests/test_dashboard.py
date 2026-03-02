"""Tests for the document OCR dashboard data generators."""

import numpy as np
import pandas as pd
from PIL import Image

from src.dashboard.app import (
    DEMO_OCR_TEXT,
    create_demo_document_image,
    draw_bounding_boxes,
    generate_demo_bounding_boxes,
    generate_demo_extracted_fields,
)


class TestGenerateDemoExtractedFields:
    """Tests for generate_demo_extracted_fields."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_demo_extracted_fields()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "field_name",
            "value",
            "confidence",
            "source",
            "is_valid",
        }
        assert expected_cols == set(df.columns)

    def test_field_count(self):
        """Contains expected number of extracted fields."""
        df = generate_demo_extracted_fields()
        assert len(df) >= 8

    def test_confidence_range(self):
        """Confidence values are between 0 and 1."""
        df = generate_demo_extracted_fields()
        assert (df["confidence"] >= 0).all()
        assert (df["confidence"] <= 1).all()

    def test_source_values(self):
        """Source column contains valid extractor types."""
        df = generate_demo_extracted_fields()
        valid_sources = {"rule_extractor", "ml_extractor"}
        assert set(df["source"]).issubset(valid_sources)

    def test_has_invoice_number(self):
        """Contains an invoice number field."""
        df = generate_demo_extracted_fields()
        assert "Invoice Number" in df["field_name"].values


class TestDemoOcrText:
    """Tests for the DEMO_OCR_TEXT constant."""

    def test_not_empty(self):
        """OCR text is not empty."""
        assert len(DEMO_OCR_TEXT.strip()) > 0

    def test_contains_invoice_marker(self):
        """Contains the word INVOICE."""
        assert "INVOICE" in DEMO_OCR_TEXT

    def test_contains_amount(self):
        """Contains a dollar amount."""
        assert "$" in DEMO_OCR_TEXT


class TestGenerateDemoBoundingBoxes:
    """Tests for generate_demo_bounding_boxes."""

    def test_returns_list(self):
        """Returns a non-empty list."""
        boxes = generate_demo_bounding_boxes()
        assert isinstance(boxes, list)
        assert len(boxes) > 0

    def test_box_structure(self):
        """Each box has required keys."""
        boxes = generate_demo_bounding_boxes()
        required_keys = {"label", "x", "y", "width", "height", "confidence"}
        for box in boxes:
            assert required_keys == set(box.keys())

    def test_box_values_positive(self):
        """Box coordinates and dimensions are positive."""
        boxes = generate_demo_bounding_boxes()
        for box in boxes:
            assert box["width"] > 0
            assert box["height"] > 0
            assert box["x"] >= 0
            assert box["y"] >= 0


class TestCreateDemoDocumentImage:
    """Tests for create_demo_document_image."""

    def test_returns_pil_image(self):
        """Returns a PIL Image."""
        img = create_demo_document_image()
        assert isinstance(img, Image.Image)

    def test_image_dimensions(self):
        """Image has expected dimensions."""
        img = create_demo_document_image()
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_image_mode_rgb(self):
        """Image is in RGB mode."""
        img = create_demo_document_image()
        assert img.mode == "RGB"


class TestDrawBoundingBoxes:
    """Tests for draw_bounding_boxes."""

    def test_returns_pil_image(self):
        """Returns a PIL Image."""
        img = create_demo_document_image()
        boxes = generate_demo_bounding_boxes()
        result = draw_bounding_boxes(img, boxes)
        assert isinstance(result, Image.Image)

    def test_does_not_modify_original(self):
        """Original image is not modified."""
        img = create_demo_document_image()
        original_data = np.array(img).copy()
        boxes = generate_demo_bounding_boxes()
        draw_bounding_boxes(img, boxes)
        np.testing.assert_array_equal(np.array(img), original_data)

    def test_empty_boxes(self):
        """Works with empty box list."""
        img = create_demo_document_image()
        result = draw_bounding_boxes(img, [])
        assert isinstance(result, Image.Image)

    def test_without_labels(self):
        """Works with show_labels=False."""
        img = create_demo_document_image()
        boxes = generate_demo_bounding_boxes()
        result = draw_bounding_boxes(img, boxes, show_labels=False)
        assert isinstance(result, Image.Image)
