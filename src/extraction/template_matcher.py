"""Template matching for known document formats.

Identifies document types by matching text against configurable
template identifiers and provides field extraction mappings.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateMatch:
    """Result of matching a document against known templates."""

    template_name: str
    confidence: float
    field_mappings: dict[str, str]


class TemplateMatcher:
    """Matches OCR text against document templates defined in YAML.

    Args:
        templates_path: Path to the YAML file defining templates.
    """

    def __init__(self, templates_path: Path = Path("configs/templates.yaml")) -> None:
        self.templates = self._load_templates(templates_path)

    def _load_templates(self, path: Path) -> dict:
        """Load template definitions from a YAML file.

        Args:
            path: Path to the templates YAML file.

        Returns:
            Dictionary of template definitions.
        """
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                return data or {}
        logger.debug("No templates file at %s, using empty templates", path)
        return {}

    def match_template(self, text: str) -> TemplateMatch | None:
        """Find the best matching template for the given text.

        Args:
            text: OCR text from the document.

        Returns:
            Best matching template, or ``None`` if no match exceeds
            the minimum confidence threshold.
        """
        best_match: TemplateMatch | None = None
        best_score = 0.0

        for name, template in self.templates.items():
            score = self._calculate_match_score(text, template)
            min_conf = template.get("min_confidence", 0.5)
            if score > best_score and score > min_conf:
                best_score = score
                best_match = TemplateMatch(
                    template_name=name,
                    confidence=score,
                    field_mappings=template.get("fields", {}),
                )

        if best_match:
            logger.info(
                "Matched template '%s' (confidence=%.2f)",
                best_match.template_name,
                best_match.confidence,
            )
        return best_match

    def _calculate_match_score(self, text: str, template: dict) -> float:
        """Calculate how well the text matches a template's identifiers.

        Args:
            text: OCR text to evaluate.
            template: Template definition with an ``identifiers`` list.

        Returns:
            Match score between 0.0 and 1.0.
        """
        identifiers = template.get("identifiers", [])
        if not identifiers:
            return 0.0
        matches = sum(
            1 for ident in identifiers if re.search(ident, text, re.IGNORECASE)
        )
        return matches / len(identifiers)
