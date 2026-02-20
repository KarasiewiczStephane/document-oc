"""Tests for PDF handling and document processing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.ocr.document_processor import DocumentProcessor, DocumentResult, PageResult
from src.ocr.pdf_handler import PDFHandler
from src.ocr.tesseract_engine import BoundingBox, OCRResult, OCRWord
from src.utils.config import AppConfig


def _mock_pil_image(width: int = 300, height: int = 200) -> Image.Image:
    """Create a mock PIL image."""
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


class TestPDFHandler:
    """Tests for the PDFHandler class."""

    def test_init_default_dpi(self) -> None:
        handler = PDFHandler()
        assert handler.dpi == 300

    def test_init_custom_dpi(self) -> None:
        handler = PDFHandler(dpi=150)
        assert handler.dpi == 150

    @patch("src.ocr.pdf_handler.convert_from_path")
    def test_pdf_to_images_from_path(self, mock_convert: MagicMock) -> None:
        mock_convert.return_value = [_mock_pil_image(), _mock_pil_image()]
        handler = PDFHandler(dpi=200)

        with patch.object(Path, "exists", return_value=True):
            images = handler.pdf_to_images(Path("/fake/doc.pdf"))

        assert len(images) == 2
        assert all(isinstance(img, np.ndarray) for img in images)
        mock_convert.assert_called_once_with("/fake/doc.pdf", dpi=200)

    @patch("src.ocr.pdf_handler.convert_from_bytes")
    def test_pdf_to_images_from_bytes(self, mock_convert: MagicMock) -> None:
        mock_convert.return_value = [_mock_pil_image()]
        handler = PDFHandler()

        images = handler.pdf_to_images(b"%PDF-1.4 fake content")

        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)

    def test_pdf_to_images_file_not_found(self) -> None:
        handler = PDFHandler()
        with pytest.raises(FileNotFoundError):
            handler.pdf_to_images(Path("/nonexistent/file.pdf"))

    @patch("src.ocr.pdf_handler.convert_from_path")
    def test_pdf_to_images_generator(self, mock_convert: MagicMock) -> None:
        mock_convert.return_value = [_mock_pil_image(), _mock_pil_image()]
        handler = PDFHandler()

        pages = list(handler.pdf_to_images_generator(Path("/fake/doc.pdf")))
        assert len(pages) == 2

    @patch("src.ocr.pdf_handler.pdfinfo_from_path")
    def test_get_page_count(self, mock_info: MagicMock) -> None:
        mock_info.return_value = {"Pages": 5}
        handler = PDFHandler()

        count = handler.get_page_count(Path("/fake/doc.pdf"))
        assert count == 5


class TestDocumentProcessor:
    """Tests for the DocumentProcessor class."""

    def _mock_ocr_result(self) -> OCRResult:
        return OCRResult(
            text="Hello World",
            words=[
                OCRWord(
                    text="Hello",
                    bbox=BoundingBox(10, 10, 50, 20),
                    confidence=0.9,
                    block_num=1,
                    line_num=1,
                    word_num=1,
                ),
            ],
            language="eng",
            confidence=0.9,
        )

    @patch("src.ocr.document_processor.TesseractEngine")
    @patch("src.ocr.document_processor.PDFHandler")
    def test_process_single_image(
        self,
        mock_pdf_cls: MagicMock,
        mock_ocr_cls: MagicMock,
    ) -> None:
        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.extract_text.return_value = self._mock_ocr_result()

        config = AppConfig()
        processor = DocumentProcessor(config)

        image_bytes = np.zeros((200, 300, 3), dtype=np.uint8)
        img = Image.fromarray(image_bytes)
        buf = __import__("io").BytesIO()
        img.save(buf, format="PNG")

        result = processor.process(buf.getvalue(), "test.png")

        assert isinstance(result, DocumentResult)
        assert result.page_count == 1
        assert len(result.pages) == 1
        assert result.source_file == "test.png"
        assert isinstance(result.pages[0], PageResult)

    @patch("src.ocr.document_processor.TesseractEngine")
    @patch("src.ocr.document_processor.PDFHandler")
    def test_process_pdf_bytes(
        self,
        mock_pdf_cls: MagicMock,
        mock_ocr_cls: MagicMock,
    ) -> None:
        mock_pdf = mock_pdf_cls.return_value
        mock_pdf.pdf_to_images.return_value = [
            np.zeros((200, 300, 3), dtype=np.uint8),
            np.zeros((200, 300, 3), dtype=np.uint8),
        ]
        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.extract_text.return_value = self._mock_ocr_result()

        config = AppConfig()
        processor = DocumentProcessor(config)

        result = processor.process(b"%PDF-1.4 content", "test.pdf")
        assert result.page_count == 2

    @patch("src.ocr.document_processor.TesseractEngine")
    @patch("src.ocr.document_processor.PDFHandler")
    def test_process_pdf_file_path(
        self,
        mock_pdf_cls: MagicMock,
        mock_ocr_cls: MagicMock,
    ) -> None:
        mock_pdf = mock_pdf_cls.return_value
        mock_pdf.pdf_to_images.return_value = [np.zeros((200, 300, 3), dtype=np.uint8)]
        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.extract_text.return_value = self._mock_ocr_result()

        config = AppConfig()
        processor = DocumentProcessor(config)

        result = processor.process(Path("/fake/doc.pdf"), "doc.pdf")
        assert result.page_count == 1
        mock_pdf.pdf_to_images.assert_called_once()

    @patch("src.ocr.document_processor.TesseractEngine")
    @patch("src.ocr.document_processor.PDFHandler")
    def test_combined_text_with_page_breaks(
        self,
        mock_pdf_cls: MagicMock,
        mock_ocr_cls: MagicMock,
    ) -> None:
        result1 = OCRResult(
            text="Page 1 text", words=[], language="eng", confidence=0.9
        )
        result2 = OCRResult(
            text="Page 2 text", words=[], language="eng", confidence=0.8
        )
        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.extract_text.side_effect = [result1, result2]

        mock_pdf = mock_pdf_cls.return_value
        mock_pdf.pdf_to_images.return_value = [
            np.zeros((200, 300, 3), dtype=np.uint8),
            np.zeros((200, 300, 3), dtype=np.uint8),
        ]

        config = AppConfig()
        processor = DocumentProcessor(config)
        result = processor.process(b"%PDF-1.4 content", "multi.pdf")

        assert "Page 1 text" in result.combined_text
        assert "Page 2 text" in result.combined_text
        assert "--- Page Break ---" in result.combined_text

    @patch("src.ocr.document_processor.TesseractEngine")
    @patch("src.ocr.document_processor.PDFHandler")
    def test_process_image_file_path(
        self,
        mock_pdf_cls: MagicMock,
        mock_ocr_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.extract_text.return_value = self._mock_ocr_result()

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8))
        img.save(str(img_path))

        config = AppConfig()
        processor = DocumentProcessor(config)
        result = processor.process(img_path, "test.png")

        assert result.page_count == 1
