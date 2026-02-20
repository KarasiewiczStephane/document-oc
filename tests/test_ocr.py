"""Tests for OCR engine and layout analysis."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ocr.layout_analyzer import LayoutAnalyzer
from src.ocr.tesseract_engine import BoundingBox, OCRResult, OCRWord, TesseractEngine


def _make_ocr_word(
    text: str = "hello",
    x: int = 10,
    y: int = 10,
    width: int = 50,
    height: int = 20,
    confidence: float = 0.9,
    block_num: int = 1,
    line_num: int = 1,
    word_num: int = 1,
) -> OCRWord:
    """Create a test OCRWord with defaults."""
    return OCRWord(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
        confidence=confidence,
        block_num=block_num,
        line_num=line_num,
        word_num=word_num,
    )


def _mock_tesseract_data() -> dict:
    """Create mock pytesseract output data."""
    return {
        "text": ["", "Hello", "World", "", "Test"],
        "conf": [-1, 95, 88, -1, 72],
        "left": [0, 10, 70, 0, 10],
        "top": [0, 10, 10, 0, 50],
        "width": [0, 50, 50, 0, 40],
        "height": [0, 20, 20, 0, 20],
        "block_num": [0, 1, 1, 0, 2],
        "line_num": [0, 1, 1, 0, 1],
        "word_num": [0, 1, 2, 0, 1],
    }


class TestBoundingBox:
    """Tests for the BoundingBox data class."""

    def test_creation(self) -> None:
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50


class TestOCRWord:
    """Tests for the OCRWord data class."""

    def test_creation(self) -> None:
        word = _make_ocr_word(text="test", confidence=0.85)
        assert word.text == "test"
        assert word.confidence == 0.85


class TestTesseractEngine:
    """Tests for the TesseractEngine class (mocked)."""

    @patch("src.ocr.tesseract_engine.pytesseract")
    def test_extract_text(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_string.return_value = "Hello World\nTest"
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data()
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine(default_lang="eng")
        image = np.zeros((100, 200), dtype=np.uint8)
        result = engine.extract_text(image, psm=3)

        assert isinstance(result, OCRResult)
        assert len(result.words) == 3
        assert result.words[0].text == "Hello"
        assert result.words[1].text == "World"
        assert result.words[2].text == "Test"
        assert result.language == "eng"
        assert 0.0 <= result.confidence <= 1.0

    @patch("src.ocr.tesseract_engine.pytesseract")
    def test_extract_text_empty_image(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_string.return_value = ""
        mock_pytesseract.image_to_data.return_value = {
            "text": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "block_num": [],
            "line_num": [],
            "word_num": [],
        }
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine()
        image = np.zeros((100, 200), dtype=np.uint8)
        result = engine.extract_text(image)

        assert result.text == ""
        assert len(result.words) == 0
        assert result.confidence == 0.0

    @patch("src.ocr.tesseract_engine.pytesseract")
    def test_extract_text_custom_lang(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_string.return_value = "Bonjour"
        mock_pytesseract.image_to_data.return_value = {
            "text": ["Bonjour"],
            "conf": [90],
            "left": [10],
            "top": [10],
            "width": [80],
            "height": [20],
            "block_num": [1],
            "line_num": [1],
            "word_num": [1],
        }
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine(default_lang="fra")
        image = np.zeros((100, 200), dtype=np.uint8)
        result = engine.extract_text(image, lang="fra")

        assert result.language == "fra"
        assert result.words[0].text == "Bonjour"

    @patch("src.ocr.tesseract_engine.pytesseract")
    def test_detect_language(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_osd.return_value = {"script": "Latin"}
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine()
        image = np.zeros((100, 200), dtype=np.uint8)
        result = engine.detect_language(image)
        assert result == "Latin"

    @patch("src.ocr.tesseract_engine.pytesseract")
    def test_detect_language_fallback(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.TesseractError = Exception
        mock_pytesseract.image_to_osd.side_effect = Exception("OSD failed")
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine(default_lang="eng")
        image = np.zeros((100, 200), dtype=np.uint8)
        result = engine.detect_language(image)
        assert result == "eng"

    def test_custom_tesseract_cmd(self) -> None:
        with patch("src.ocr.tesseract_engine.pytesseract") as mock_pt:
            TesseractEngine(tesseract_cmd="/usr/bin/tesseract")
            assert mock_pt.pytesseract.tesseract_cmd == "/usr/bin/tesseract"

    @patch("src.ocr.tesseract_engine.pytesseract")
    def test_confidence_scores_normalized(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_string.return_value = "Word"
        mock_pytesseract.image_to_data.return_value = {
            "text": ["Word"],
            "conf": [85],
            "left": [10],
            "top": [10],
            "width": [50],
            "height": [20],
            "block_num": [1],
            "line_num": [1],
            "word_num": [1],
        }
        mock_pytesseract.Output.DICT = "dict"

        engine = TesseractEngine()
        result = engine.extract_text(np.zeros((100, 200), dtype=np.uint8))
        assert result.words[0].confidence == 0.85
        assert 0.0 <= result.confidence <= 1.0


class TestLayoutAnalyzer:
    """Tests for the layout analysis component."""

    def test_empty_words(self) -> None:
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze([], image_height=1000)
        assert blocks == []

    def test_single_block_header(self) -> None:
        words = [
            _make_ocr_word(text="Title", x=10, y=10, block_num=1),
            _make_ocr_word(text="Company", x=70, y=10, block_num=1),
        ]
        analyzer = LayoutAnalyzer(header_ratio=0.15)
        blocks = analyzer.analyze(words, image_height=1000)

        assert len(blocks) == 1
        assert blocks[0].block_type == "header"
        assert len(blocks[0].words) == 2

    def test_paragraph_block(self) -> None:
        words = [
            _make_ocr_word(text="Content", x=10, y=300, block_num=1),
            _make_ocr_word(text="here", x=70, y=300, block_num=1),
        ]
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(words, image_height=1000)

        assert len(blocks) == 1
        assert blocks[0].block_type == "paragraph"

    def test_multiple_blocks(self) -> None:
        words = [
            _make_ocr_word(text="Header", x=10, y=10, block_num=1),
            _make_ocr_word(text="Body", x=10, y=300, block_num=2),
        ]
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(words, image_height=1000)
        assert len(blocks) == 2

    def test_table_detection(self) -> None:
        words = [
            _make_ocr_word(text="A", x=10, y=300, width=30, block_num=1),
            _make_ocr_word(text="B", x=50, y=300, width=30, block_num=1),
            _make_ocr_word(text="C", x=90, y=300, width=30, block_num=1),
            _make_ocr_word(text="D", x=130, y=300, width=30, block_num=1),
        ]
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(words, image_height=1000)
        assert blocks[0].block_type == "table"

    def test_table_detection_insufficient_words(self) -> None:
        words = [
            _make_ocr_word(text="A", x=10, y=300, block_num=1),
            _make_ocr_word(text="B", x=50, y=300, block_num=1),
        ]
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(words, image_height=1000)
        assert blocks[0].block_type == "paragraph"

    def test_block_bbox_calculation(self) -> None:
        words = [
            _make_ocr_word(x=10, y=20, width=50, height=20, block_num=1),
            _make_ocr_word(x=70, y=25, width=60, height=20, block_num=1),
        ]
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(words, image_height=1000)
        bbox = blocks[0].bbox
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 120  # 130 - 10
        assert bbox.height == 25  # 45 - 20

    def test_block_confidence_average(self) -> None:
        words = [
            _make_ocr_word(confidence=0.8, block_num=1),
            _make_ocr_word(confidence=0.6, block_num=1),
        ]
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(words, image_height=1000)
        assert blocks[0].confidence == pytest.approx(0.7)

    def test_custom_header_ratio(self) -> None:
        words = [_make_ocr_word(text="Top", x=10, y=90, block_num=1)]
        analyzer = LayoutAnalyzer(header_ratio=0.1)
        blocks = analyzer.analyze(words, image_height=1000)
        assert blocks[0].block_type == "header"

        analyzer2 = LayoutAnalyzer(header_ratio=0.05)
        blocks2 = analyzer2.analyze(words, image_height=1000)
        assert blocks2[0].block_type == "paragraph"
