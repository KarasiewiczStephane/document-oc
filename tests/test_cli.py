"""Tests for the batch processing CLI and CSV export."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.cli import (
    _find_documents,
    _print_summary,
    _write_csv,
    extract_single,
    main,
    process_folder,
)
from src.extraction.hybrid import ExtractionResult, HybridField
from src.ocr.document_processor import DocumentResult, PageResult
from src.ocr.tesseract_engine import BoundingBox, OCRResult, OCRWord
from src.preprocessing.pipeline import QualityMetrics
from src.validation.rules_engine import ValidationReport, ValidationResult


def _make_test_image(path: Path) -> None:
    """Create a minimal test PNG image at the given path."""
    img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    img.save(path, format="PNG")


def _make_mock_doc_result(filename: str = "test.png") -> DocumentResult:
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
        source_file=filename,
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


def _make_mock_extraction() -> ExtractionResult:
    """Create a mock ExtractionResult for testing."""
    return ExtractionResult(
        fields=[
            HybridField(
                field_name="date",
                value="01/15/2024",
                confidence=0.9,
                source="rule",
            ),
            HybridField(
                field_name="total_amount",
                value="500.00",
                confidence=0.85,
                source="rule",
            ),
        ],
        raw_text="Invoice #001\nTotal: $500.00",
        overall_confidence=0.875,
    )


def _make_mock_validation() -> ValidationReport:
    """Create a mock ValidationReport for testing."""
    return ValidationReport(
        all_valid=True,
        results=[
            ValidationResult("date", True, "Valid date", "date_format", 0.1),
        ],
        field_confidences={"date": 0.9, "total_amount": 0.85},
    )


class TestFindDocuments:
    """Tests for document discovery."""

    def test_find_png_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc1.png").touch()
        (tmp_path / "doc2.png").touch()
        (tmp_path / "readme.txt").touch()
        files = _find_documents(tmp_path)
        assert len(files) == 2
        assert all(f.suffix == ".png" for f in files)

    def test_find_mixed_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "doc.png").touch()
        (tmp_path / "doc.jpg").touch()
        (tmp_path / "doc.pdf").touch()
        (tmp_path / "doc.tiff").touch()
        files = _find_documents(tmp_path)
        assert len(files) == 4

    def test_find_no_documents(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").touch()
        files = _find_documents(tmp_path)
        assert len(files) == 0

    def test_find_uppercase_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "DOC.PNG").touch()
        files = _find_documents(tmp_path)
        assert len(files) == 1


class TestWriteCsv:
    """Tests for CSV writing."""

    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        results = [
            {
                "filename": "test.png",
                "status": "success",
                "date": "01/15/2024",
            }
        ]
        output = tmp_path / "results.csv"
        _write_csv(results, output)
        assert output.exists()

    def test_write_csv_content(self, tmp_path: Path) -> None:
        results = [
            {
                "filename": "test.png",
                "status": "success",
                "error": None,
                "date": "01/15/2024",
                "total_amount": "500.00",
            }
        ]
        output = tmp_path / "results.csv"
        _write_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["filename"] == "test.png"
        assert rows[0]["date"] == "01/15/2024"

    def test_write_csv_empty_results(self, tmp_path: Path) -> None:
        output = tmp_path / "results.csv"
        _write_csv([], output)
        assert not output.exists()

    def test_write_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = tmp_path / "subdir" / "results.csv"
        results = [{"filename": "test.png", "status": "success"}]
        _write_csv(results, output)
        assert output.exists()

    def test_csv_meta_columns_first(self, tmp_path: Path) -> None:
        results = [
            {
                "filename": "test.png",
                "status": "success",
                "error": None,
                "vendor": "ACME",
                "date": "2024-01-01",
            }
        ]
        output = tmp_path / "results.csv"
        _write_csv(results, output)

        with open(output) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers[0] == "filename"
        assert headers[1] == "status"


class TestPrintSummary:
    """Tests for summary printing."""

    def test_print_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        summary = {"total": 5, "successful": 4, "failed": 1}
        _print_summary(summary, Path("results.csv"))
        captured = capsys.readouterr()
        assert "Total:      5" in captured.out
        assert "Successful: 4" in captured.out
        assert "Failed:     1" in captured.out
        assert "results.csv" in captured.out


class TestProcessFolder:
    """Tests for batch folder processing."""

    @patch("src.cli.RulesEngine")
    @patch("src.cli.HybridExtractor")
    @patch("src.cli.DocumentProcessor")
    @patch("src.cli.load_config")
    def test_process_folder_success(
        self,
        mock_config: MagicMock,
        mock_processor_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        mock_validator_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract.return_value = _make_mock_extraction()
        mock_validator = mock_validator_cls.return_value
        mock_validator.validate.return_value = _make_mock_validation()

        (tmp_path / "doc1.png").touch()
        (tmp_path / "doc2.png").touch()
        output_csv = tmp_path / "output.csv"

        summary = process_folder(tmp_path, output_csv)
        assert summary["total"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert output_csv.exists()

    @patch("src.cli.RulesEngine")
    @patch("src.cli.HybridExtractor")
    @patch("src.cli.DocumentProcessor")
    @patch("src.cli.load_config")
    def test_process_folder_with_failure(
        self,
        mock_config: MagicMock,
        mock_processor_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        mock_validator_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.side_effect = [
            _make_mock_doc_result(),
            RuntimeError("OCR failed"),
        ]
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract.return_value = _make_mock_extraction()
        mock_validator = mock_validator_cls.return_value
        mock_validator.validate.return_value = _make_mock_validation()

        (tmp_path / "doc1.png").touch()
        (tmp_path / "doc2.png").touch()
        output_csv = tmp_path / "output.csv"

        summary = process_folder(tmp_path, output_csv)
        assert summary["successful"] == 1
        assert summary["failed"] == 1

    @patch("src.cli.load_config")
    def test_process_folder_empty(self, mock_config: MagicMock, tmp_path: Path) -> None:
        mock_config.return_value = MagicMock()
        output_csv = tmp_path / "output.csv"
        summary = process_folder(tmp_path, output_csv)
        assert summary["total"] == 0

    @patch("src.cli.RulesEngine")
    @patch("src.cli.HybridExtractor")
    @patch("src.cli.DocumentProcessor")
    @patch("src.cli.load_config")
    def test_process_folder_verbose(
        self,
        mock_config: MagicMock,
        mock_processor_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        mock_validator_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract.return_value = _make_mock_extraction()
        mock_validator = mock_validator_cls.return_value
        mock_validator.validate.return_value = _make_mock_validation()

        (tmp_path / "doc1.png").touch()
        output_csv = tmp_path / "output.csv"

        process_folder(tmp_path, output_csv, verbose=True)
        captured = capsys.readouterr()
        assert "Processing [1/1]" in captured.out


class TestExtractSingle:
    """Tests for single file extraction."""

    @patch("src.cli.HybridExtractor")
    @patch("src.cli.DocumentProcessor")
    @patch("src.cli.load_config")
    def test_extract_single_returns_fields(
        self,
        mock_config: MagicMock,
        mock_processor_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract.return_value = _make_mock_extraction()

        doc_path = tmp_path / "test.png"
        doc_path.touch()

        result = extract_single(doc_path)
        assert result["filename"] == "test.png"
        assert "fields" in result
        assert "raw_text" in result
        assert "date" in result["fields"]

    @patch("src.cli.HybridExtractor")
    @patch("src.cli.DocumentProcessor")
    @patch("src.cli.load_config")
    def test_extract_single_no_ml(
        self,
        mock_config: MagicMock,
        mock_processor_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock()
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = _make_mock_doc_result()
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract.return_value = _make_mock_extraction()

        doc_path = tmp_path / "test.png"
        doc_path.touch()

        result = extract_single(doc_path, use_ml=False)
        assert result["filename"] == "test.png"


class TestCLIMain:
    """Tests for the CLI argument parser and main entry point."""

    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_batch_nonexistent_directory(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["batch", "/nonexistent/path"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not a directory" in captured.err

    def test_extract_nonexistent_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["extract", "/nonexistent/file.png"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    @patch("src.cli.process_folder")
    def test_batch_command(self, mock_pf: MagicMock, tmp_path: Path) -> None:
        mock_pf.return_value = {"total": 1, "successful": 1, "failed": 0}
        main(["batch", str(tmp_path)])
        mock_pf.assert_called_once()

    @patch("src.cli.process_folder")
    def test_batch_with_options(self, mock_pf: MagicMock, tmp_path: Path) -> None:
        mock_pf.return_value = {"total": 1, "successful": 1, "failed": 0}
        output = tmp_path / "out.csv"
        main(
            [
                "batch",
                str(tmp_path),
                "-o",
                str(output),
                "-t",
                "receipt",
                "--no-ml",
                "-v",
            ]
        )
        mock_pf.assert_called_once_with(tmp_path, output, "receipt", False, True)

    @patch("src.cli.extract_single")
    def test_extract_command(
        self,
        mock_extract: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_extract.return_value = {
            "filename": "test.png",
            "fields": {},
            "raw_text": "",
        }
        doc = tmp_path / "test.png"
        doc.touch()
        main(["extract", str(doc)])
        captured = capsys.readouterr()
        assert "test.png" in captured.out

    @patch("src.cli.extract_single")
    def test_extract_to_output_file(
        self,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_extract.return_value = {
            "filename": "test.png",
            "fields": {"date": {"value": "2024-01-01", "confidence": 0.9}},
            "raw_text": "test",
        }
        doc = tmp_path / "test.png"
        doc.touch()
        output = tmp_path / "result.json"
        main(["extract", str(doc), "-o", str(output)])
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["filename"] == "test.png"
