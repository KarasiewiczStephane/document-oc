"""PDF to image conversion for multi-page document processing.

Converts PDF documents to numpy arrays for subsequent OCR processing,
supporting both file paths and raw bytes input.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFHandler:
    """Handles PDF to image conversion for OCR processing.

    Args:
        dpi: Resolution for PDF rendering. Higher values produce
            better OCR results but use more memory.
    """

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def pdf_to_images(self, pdf_source: Path | bytes) -> list[np.ndarray]:
        """Convert a PDF to a list of images.

        Args:
            pdf_source: Path to a PDF file or raw PDF bytes.

        Returns:
            List of images as numpy arrays (RGB format).

        Raises:
            FileNotFoundError: If a path is given and the file does not exist.
            RuntimeError: If PDF conversion fails.
        """
        try:
            if isinstance(pdf_source, str | Path):
                path = Path(pdf_source)
                if not path.exists():
                    raise FileNotFoundError(f"PDF file not found: {path}")
                pil_images = convert_from_path(str(path), dpi=self.dpi)
            else:
                pil_images = convert_from_bytes(pdf_source, dpi=self.dpi)

            images = [np.array(img) for img in pil_images]
            logger.info("Converted PDF to %d images at %d DPI", len(images), self.dpi)
            return images

        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"PDF conversion failed: {exc}") from exc

    def pdf_to_images_generator(self, pdf_path: Path) -> Iterator[np.ndarray]:
        """Convert a PDF to images one page at a time.

        Memory-efficient generator for processing large PDFs without
        loading all pages into memory simultaneously.

        Args:
            pdf_path: Path to the PDF file.

        Yields:
            Individual page images as numpy arrays.
        """
        for pil_image in convert_from_path(str(pdf_path), dpi=self.dpi):
            yield np.array(pil_image)

    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF without converting.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Number of pages in the PDF.
        """
        info = pdfinfo_from_path(str(pdf_path))
        count = info["Pages"]
        logger.debug("PDF %s has %d pages", pdf_path, count)
        return count
