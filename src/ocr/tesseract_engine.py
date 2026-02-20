"""Tesseract OCR engine wrapper with word-level extraction.

Provides OCR text extraction with bounding boxes, confidence scores,
and configurable page segmentation modes.
"""

from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for a detected element."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class OCRWord:
    """A single word extracted by OCR with position and confidence."""

    text: str
    bbox: BoundingBox
    confidence: float
    block_num: int
    line_num: int
    word_num: int


@dataclass
class OCRResult:
    """Complete OCR result for a document page."""

    text: str
    words: list[OCRWord]
    language: str
    confidence: float


class TesseractEngine:
    """Wrapper around Tesseract OCR for document text extraction.

    Args:
        tesseract_cmd: Path to the Tesseract executable.
            If ``None``, uses the system default.
        default_lang: Default OCR language code.
    """

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        default_lang: str = "eng",
    ) -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.default_lang = default_lang

    def detect_language(self, image: np.ndarray) -> str:
        """Detect the script/language of text in an image.

        Args:
            image: Input image as a numpy array.

        Returns:
            Detected script name, or the default language on failure.
        """
        try:
            pil_image = Image.fromarray(image)
            osd = pytesseract.image_to_osd(
                pil_image, output_type=pytesseract.Output.DICT
            )
            script = osd.get("script", self.default_lang)
            logger.debug("Detected script: %s", script)
            return script
        except pytesseract.TesseractError as exc:
            logger.warning("Language detection failed: %s", exc)
            return self.default_lang

    def extract_text(
        self,
        image: np.ndarray,
        lang: str | None = None,
        psm: int = 3,
    ) -> OCRResult:
        """Extract text from an image with word-level bounding boxes.

        Args:
            image: Input image as a numpy array.
            lang: OCR language code. Defaults to the engine default.
            psm: Tesseract page segmentation mode.

        Returns:
            OCRResult containing full text, word details, and confidence.
        """
        lang = lang or self.default_lang
        config = f"--psm {psm}"

        pil_image = Image.fromarray(image)
        text = pytesseract.image_to_string(pil_image, lang=lang, config=config)

        data = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[OCRWord] = []
        total_conf = 0.0
        word_count = 0

        for i in range(len(data["text"])):
            conf = data["conf"][i]
            word_text = data["text"][i].strip()

            if conf > 0 and word_text:
                words.append(
                    OCRWord(
                        text=word_text,
                        bbox=BoundingBox(
                            x=data["left"][i],
                            y=data["top"][i],
                            width=data["width"][i],
                            height=data["height"][i],
                        ),
                        confidence=conf / 100.0,
                        block_num=data["block_num"][i],
                        line_num=data["line_num"][i],
                        word_num=data["word_num"][i],
                    )
                )
                total_conf += conf
                word_count += 1

        avg_conf = (total_conf / word_count / 100.0) if word_count > 0 else 0.0

        logger.info(
            "OCR extracted %d words with average confidence %.2f",
            word_count,
            avg_conf,
        )
        return OCRResult(
            text=text,
            words=words,
            language=lang,
            confidence=avg_conf,
        )
