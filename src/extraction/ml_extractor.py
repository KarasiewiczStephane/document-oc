"""ML-based structured extraction using LayoutLM transformers.

Provides transformer-based document understanding for intelligent
field extraction using spatial and textual features.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

from src.utils.logger import get_logger

logger = get_logger(__name__)


LABEL_MAP: dict[int, str] = {
    0: "O",
    1: "B-DATE",
    2: "I-DATE",
    3: "B-VENDOR",
    4: "I-VENDOR",
    5: "B-TOTAL",
    6: "I-TOTAL",
    7: "B-ITEM",
    8: "I-ITEM",
    9: "B-TAX",
    10: "I-TAX",
}


@dataclass
class MLExtractedField:
    """A field extracted by the ML model."""

    field_name: str
    value: str
    confidence: float
    bbox: tuple[int, int, int, int]


class LayoutLMExtractor:
    """LayoutLM-based document field extraction.

    Uses a pre-trained LayoutLM model for token classification to
    identify and extract structured fields from document images.

    Args:
        model_name: Hugging Face model identifier.
        device: Torch device (``"cuda"`` or ``"cpu"``). Auto-detected if ``None``.
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlm-base-uncased",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        logger.info("Loading LayoutLM model: %s on %s", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def extract(
        self,
        image: np.ndarray,
        words: list[str],
        boxes: list[tuple[int, int, int, int]],
    ) -> list[MLExtractedField]:
        """Extract fields from a document image using LayoutLM.

        Args:
            image: Document image as a numpy array.
            words: OCR-detected words.
            boxes: Bounding boxes normalized to 0-1000 range,
                as ``(x1, y1, x2, y2)`` tuples.

        Returns:
            List of extracted fields with confidence scores.
        """
        if not words:
            return []

        pil_image = Image.fromarray(image)

        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        confidences = (
            torch.softmax(outputs.logits, dim=-1).max(-1).values.squeeze().tolist()
        )

        if isinstance(predictions, int):
            predictions = [predictions]
            confidences = [confidences]

        return self._aggregate_fields(words, boxes, predictions, confidences)

    def _aggregate_fields(
        self,
        words: list[str],
        boxes: list[tuple[int, int, int, int]],
        predictions: list[int],
        confidences: list[float],
    ) -> list[MLExtractedField]:
        """Aggregate BIO-tagged tokens into complete fields.

        Args:
            words: OCR words.
            boxes: Word bounding boxes.
            predictions: Predicted label indices per token.
            confidences: Confidence scores per token.

        Returns:
            Aggregated field extractions.
        """
        fields: list[MLExtractedField] = []
        current_field: str | None = None
        current_words: list[str] = []
        current_conf: list[float] = []

        for word, pred, conf in zip(words, predictions, confidences, strict=False):
            label = LABEL_MAP.get(pred, "O")

            if label.startswith("B-"):
                if current_field:
                    fields.append(
                        self._create_field(current_field, current_words, current_conf)
                    )
                current_field = label[2:]
                current_words = [word]
                current_conf = [conf]
            elif label.startswith("I-") and current_field == label[2:]:
                current_words.append(word)
                current_conf.append(conf)
            else:
                if current_field:
                    fields.append(
                        self._create_field(current_field, current_words, current_conf)
                    )
                    current_field = None
                    current_words = []
                    current_conf = []

        if current_field:
            fields.append(
                self._create_field(current_field, current_words, current_conf)
            )

        logger.info("ML extraction found %d fields", len(fields))
        return fields

    def _create_field(
        self,
        field_name: str,
        words: list[str],
        confidences: list[float],
    ) -> MLExtractedField:
        """Create an MLExtractedField from accumulated tokens.

        Args:
            field_name: Semantic field name (e.g., DATE, VENDOR).
            words: Tokens belonging to this field.
            confidences: Per-token confidence scores.

        Returns:
            Aggregated extraction result.
        """
        return MLExtractedField(
            field_name=field_name.lower(),
            value=" ".join(words),
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            bbox=(0, 0, 0, 0),
        )
