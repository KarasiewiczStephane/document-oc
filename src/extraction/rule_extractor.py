"""Rule-based field extraction using regex patterns.

Extracts dates, amounts, emails, phone numbers, invoice numbers,
and vendor names from OCR text using configurable regular expressions.
"""

import re
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedField:
    """A field value extracted by a regex rule."""

    field_name: str
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    extraction_method: str


# Pattern definitions: (regex, base_confidence, optional_flags)
_DATE_PATTERNS: list[tuple[str, float, int]] = [
    (r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", 0.9, 0),
    (r"\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b", 0.9, 0),
    (
        r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+(\d{2,4})\b",
        0.85,
        re.IGNORECASE,
    ),
    (
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+(\d{1,2}),?\s+(\d{2,4})\b",
        0.85,
        re.IGNORECASE,
    ),
]

_AMOUNT_PATTERNS: list[tuple[str, float, int]] = [
    (r"\$\s*([\d,]+\.\d{2})\b", 0.95, 0),
    (r"\b([\d,]+\.\d{2})\s*(?:USD|EUR|GBP)\b", 0.9, 0),
    (r"(?:Total|Amount|Sum|Due)[:\s]*\$?\s*([\d,]+\.\d{2})", 0.85, re.IGNORECASE),
]

_EMAIL_PATTERNS: list[tuple[str, float, int]] = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", 0.95, 0),
]

_PHONE_PATTERNS: list[tuple[str, float, int]] = [
    (r"\b(?:\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b", 0.9, 0),
]

_INVOICE_PATTERNS: list[tuple[str, float, int]] = [
    (r"(?:Invoice|Inv)[\s#:]*([A-Z0-9\-]+)", 0.9, re.IGNORECASE),
]

_VENDOR_PATTERNS: list[tuple[str, float, int]] = [
    (
        r"^([A-Z][A-Za-z\s&,\.]+(?:Inc|LLC|Ltd|Corp|Company|Co)?\.?)\s*$",
        0.7,
        re.MULTILINE,
    ),
]

_TOTAL_PATTERNS: list[tuple[str, float]] = [
    (
        r"(?:Grand\s*Total|Total\s*Due|Amount\s*Due|Balance\s*Due)"
        r"[:\s]*\$?\s*([\d,]+\.\d{2})",
        0.95,
    ),
    (r"(?:Total)[:\s]*\$?\s*([\d,]+\.\d{2})", 0.85),
]


class RuleExtractor:
    """Regex-based field extractor for structured document data.

    Matches configurable patterns against OCR text to identify
    dates, amounts, emails, phone numbers, and other fields.
    """

    def __init__(self) -> None:
        self.patterns: dict[str, list[tuple[str, float, int]]] = {
            "date": _DATE_PATTERNS,
            "amount": _AMOUNT_PATTERNS,
            "email": _EMAIL_PATTERNS,
            "phone": _PHONE_PATTERNS,
            "invoice_number": _INVOICE_PATTERNS,
            "vendor_name": _VENDOR_PATTERNS,
        }

    def extract(
        self, text: str, fields: list[str] | None = None
    ) -> list[ExtractedField]:
        """Extract fields from text using regex patterns.

        Args:
            text: OCR text to search.
            fields: Specific field names to extract. If ``None``, extracts all.

        Returns:
            List of extracted field results with confidence scores.
        """
        results: list[ExtractedField] = []
        target_fields = fields or list(self.patterns.keys())

        for field_name in target_fields:
            if field_name not in self.patterns:
                continue
            for pattern_def in self.patterns[field_name]:
                pattern, confidence, flags = pattern_def

                for match in re.finditer(pattern, text, flags):
                    value = match.group(1) if match.groups() else match.group(0)
                    results.append(
                        ExtractedField(
                            field_name=field_name,
                            value=value.strip(),
                            confidence=confidence,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            extraction_method="regex",
                        )
                    )

        logger.info("Rule extraction found %d fields", len(results))
        return results

    def extract_total_amount(self, text: str) -> ExtractedField | None:
        """Extract the most likely total amount from invoice/receipt text.

        Uses specific total-oriented patterns with higher confidence
        than generic amount patterns.

        Args:
            text: OCR text to search.

        Returns:
            Extracted total amount field, or ``None`` if not found.
        """
        for pattern, confidence in _TOTAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")
                logger.debug("Found total amount: %s", value)
                return ExtractedField(
                    field_name="total_amount",
                    value=value,
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    extraction_method="regex_total",
                )
        return None
