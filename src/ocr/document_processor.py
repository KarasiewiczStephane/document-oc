"""Unified document processing pipeline.

Combines PDF handling, image preprocessing, OCR, and layout analysis
into a single processing interface for any document format.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.pipeline import PreprocessingPipeline, QualityMetrics
from src.utils.config import AppConfig
from src.utils.logger import get_logger

from .layout_analyzer import LayoutAnalyzer, TextBlock
from .pdf_handler import PDFHandler
from .tesseract_engine import OCRResult, TesseractEngine

logger = get_logger(__name__)


@dataclass
class PageResult:
    """OCR and analysis results for a single document page."""

    page_number: int
    ocr_result: OCRResult
    text_blocks: list[TextBlock]
    quality_metrics: QualityMetrics


@dataclass
class DocumentResult:
    """Complete processing results for a document."""

    source_file: str
    page_count: int
    pages: list[PageResult]
    combined_text: str


class DocumentProcessor:
    """End-to-end document processing pipeline.

    Handles loading documents (images or PDFs), preprocessing,
    OCR extraction, and layout analysis.

    Args:
        config: Application configuration object.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pdf_handler = PDFHandler(dpi=config.ocr.pdf_dpi)
        self.preprocessing = PreprocessingPipeline(config.preprocessing)
        self.ocr_engine = TesseractEngine(
            tesseract_cmd=config.ocr.tesseract_cmd,
            default_lang=config.ocr.default_lang,
        )
        self.layout_analyzer = LayoutAnalyzer()

    def process(
        self, source: Path | bytes, filename: str = "document"
    ) -> DocumentResult:
        """Process a document from file path or bytes.

        Args:
            source: Path to a document file, or raw file bytes.
            filename: Display name for the source document.

        Returns:
            Complete document processing results.
        """
        logger.info("Processing document: %s", filename)
        images = self._load_images(source)
        pages: list[PageResult] = []

        for i, image in enumerate(images):
            processed, metrics = self.preprocessing.process(image)
            ocr_result = self.ocr_engine.extract_text(
                processed,
                psm=self.config.ocr.psm,
            )
            blocks = self.layout_analyzer.analyze(ocr_result.words, image.shape[0])
            pages.append(
                PageResult(
                    page_number=i + 1,
                    ocr_result=ocr_result,
                    text_blocks=blocks,
                    quality_metrics=metrics,
                )
            )

        combined_text = "\n\n--- Page Break ---\n\n".join(
            p.ocr_result.text for p in pages
        )

        logger.info(
            "Processed %d pages from %s",
            len(pages),
            filename,
        )
        return DocumentResult(
            source_file=filename,
            page_count=len(pages),
            pages=pages,
            combined_text=combined_text,
        )

    def _load_images(self, source: Path | bytes) -> list[np.ndarray]:
        """Load document images from a file path or bytes.

        Supports PDF files and common image formats (PNG, JPEG, TIFF).

        Args:
            source: Path or raw bytes of the document.

        Returns:
            List of images as numpy arrays.
        """
        if isinstance(source, bytes):
            if source[:4] == b"%PDF":
                return self.pdf_handler.pdf_to_images(source)
            img = Image.open(io.BytesIO(source))
            return [np.array(img)]

        path = Path(source)
        if path.suffix.lower() == ".pdf":
            return self.pdf_handler.pdf_to_images(path)

        img = Image.open(path)
        return [np.array(img)]
