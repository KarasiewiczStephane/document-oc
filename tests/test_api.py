"""Tests for the FastAPI REST endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.ocr.document_processor import DocumentResult, PageResult
from src.ocr.tesseract_engine import BoundingBox, OCRResult, OCRWord
from src.preprocessing.pipeline import QualityMetrics


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


def _make_test_image_bytes() -> bytes:
    """Create a minimal PNG image as bytes."""
    import io

    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_doc_result() -> DocumentResult:
    """Create a mock DocumentResult for testing."""
    ocr_result = OCRResult(
        text="Invoice #001\nTotal: $500.00\nDate: 01/15/2024",
        words=[
            OCRWord(
                text="Invoice",
                bbox=BoundingBox(10, 10, 60, 20),
                confidence=0.95,
                block_num=1,
                line_num=1,
                word_num=1,
            ),
        ],
        language="eng",
        confidence=0.9,
    )
    return DocumentResult(
        source_file="test.png",
        page_count=1,
        pages=[
            PageResult(
                page_number=1,
                ocr_result=ocr_result,
                text_blocks=[],
                quality_metrics=QualityMetrics(100.0, 120.0, 50.0, 60.0),
            )
        ],
        combined_text=ocr_result.text,
    )


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert isinstance(data["tesseract_available"], bool)
        assert isinstance(data["gpu_available"], bool)


class TestTemplatesEndpoint:
    """Tests for the /templates endpoint."""

    def test_list_templates(self, client: TestClient) -> None:
        response = client.get("/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert len(data["templates"]) == 3

    def test_templates_have_required_fields(self, client: TestClient) -> None:
        response = client.get("/templates")
        data = response.json()
        for template in data["templates"]:
            assert "name" in template
            assert "description" in template
            assert "supported_fields" in template


class TestExtractEndpoint:
    """Tests for the /extract endpoint."""

    @patch("src.api.app._get_components")
    def test_extract_success(
        self, mock_components: MagicMock, client: TestClient
    ) -> None:
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            fields=[], raw_text="test", overall_confidence=0.0
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(
            all_valid=True,
            results=[],
            field_confidences={},
        )
        mock_components.return_value = (mock_processor, mock_extractor, mock_validator)

        image_bytes = _make_test_image_bytes()
        response = client.post(
            "/extract",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["page_count"] == 1

    @patch("src.api.app._get_components")
    def test_extract_with_document_type(
        self, mock_components: MagicMock, client: TestClient
    ) -> None:
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            fields=[], raw_text="", overall_confidence=0.0
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(
            all_valid=True, results=[], field_confidences={}
        )
        mock_components.return_value = (mock_processor, mock_extractor, mock_validator)

        image_bytes = _make_test_image_bytes()
        response = client.post(
            "/extract?document_type=receipt",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["document_type"] == "receipt"

    def test_extract_unsupported_file_type(self, client: TestClient) -> None:
        response = client.post(
            "/extract",
            files={"file": ("test.txt", b"plain text", "text/plain")},
        )
        assert response.status_code == 400

    @patch("src.api.app._get_components")
    def test_extract_processing_error(
        self, mock_components: MagicMock, client: TestClient
    ) -> None:
        mock_processor = MagicMock()
        mock_processor.process.side_effect = RuntimeError("OCR failed")
        mock_components.return_value = (mock_processor, MagicMock(), MagicMock())

        image_bytes = _make_test_image_bytes()
        response = client.post(
            "/extract",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 500

    @patch("src.api.app._get_components")
    def test_extract_response_schema(
        self, mock_components: MagicMock, client: TestClient
    ) -> None:
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            fields=[], raw_text="", overall_confidence=0.0
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(
            all_valid=True, results=[], field_confidences={}
        )
        mock_components.return_value = (mock_processor, mock_extractor, mock_validator)

        image_bytes = _make_test_image_bytes()
        response = client.post(
            "/extract",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        data = response.json()
        assert "document_id" in data
        assert "fields" in data
        assert "raw_text" in data
        assert "overall_confidence" in data
        assert "processing_time_ms" in data


class TestBatchExtractEndpoint:
    """Tests for the /extract/batch endpoint."""

    @patch("src.api.app._get_components")
    def test_batch_extract(
        self, mock_components: MagicMock, client: TestClient
    ) -> None:
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            fields=[], raw_text="", overall_confidence=0.0
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(
            all_valid=True, results=[], field_confidences={}
        )
        mock_components.return_value = (mock_processor, mock_extractor, mock_validator)

        image_bytes = _make_test_image_bytes()
        response = client.post(
            "/extract/batch",
            files=[
                ("files", ("doc1.png", image_bytes, "image/png")),
                ("files", ("doc2.png", image_bytes, "image/png")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0

    @patch("src.api.app._get_components")
    def test_batch_with_failure(
        self, mock_components: MagicMock, client: TestClient
    ) -> None:
        mock_processor = MagicMock()
        mock_processor.process.side_effect = [
            _make_mock_doc_result(),
            RuntimeError("Failed"),
        ]
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            fields=[], raw_text="", overall_confidence=0.0
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(
            all_valid=True, results=[], field_confidences={}
        )
        mock_components.return_value = (mock_processor, mock_extractor, mock_validator)

        image_bytes = _make_test_image_bytes()
        response = client.post(
            "/extract/batch",
            files=[
                ("files", ("doc1.png", image_bytes, "image/png")),
                ("files", ("doc2.png", image_bytes, "image/png")),
            ],
        )
        data = response.json()
        assert data["successful"] == 1
        assert data["failed"] == 1
