"""Layout analysis for document structure detection.

Classifies OCR text blocks into semantic regions such as headers,
paragraphs, and tables based on position and arrangement.
"""

from dataclasses import dataclass

from src.utils.logger import get_logger

from .tesseract_engine import BoundingBox, OCRWord

logger = get_logger(__name__)


@dataclass
class TextBlock:
    """A classified region of text in a document."""

    block_type: str
    bbox: BoundingBox
    words: list[OCRWord]
    confidence: float


class LayoutAnalyzer:
    """Analyzes OCR word positions to identify document structure.

    Classifies text blocks into headers, paragraphs, and tables
    based on their position within the document image.

    Args:
        header_ratio: Fraction of image height considered the header region.
    """

    def __init__(self, header_ratio: float = 0.15) -> None:
        self.header_ratio = header_ratio

    def analyze(self, words: list[OCRWord], image_height: int) -> list[TextBlock]:
        """Analyze OCR words and group them into classified text blocks.

        Args:
            words: List of OCR word detections.
            image_height: Height of the source image in pixels.

        Returns:
            List of classified text blocks.
        """
        if not words:
            return []

        block_groups: dict[int, list[OCRWord]] = {}
        for word in words:
            if word.block_num not in block_groups:
                block_groups[word.block_num] = []
            block_groups[word.block_num].append(word)

        text_blocks: list[TextBlock] = []
        for block_words in block_groups.values():
            bbox = self._calculate_block_bbox(block_words)
            block_type = self._classify_block(block_words, bbox, image_height)
            avg_conf = sum(w.confidence for w in block_words) / len(block_words)
            text_blocks.append(
                TextBlock(
                    block_type=block_type,
                    bbox=bbox,
                    words=block_words,
                    confidence=avg_conf,
                )
            )

        logger.info("Detected %d text blocks", len(text_blocks))
        return text_blocks

    def _calculate_block_bbox(self, words: list[OCRWord]) -> BoundingBox:
        """Calculate the bounding box that encloses all words in a block.

        Args:
            words: Words belonging to the same block.

        Returns:
            Enclosing bounding box.
        """
        x_min = min(w.bbox.x for w in words)
        y_min = min(w.bbox.y for w in words)
        x_max = max(w.bbox.x + w.bbox.width for w in words)
        y_max = max(w.bbox.y + w.bbox.height for w in words)
        return BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)

    def _classify_block(
        self,
        words: list[OCRWord],
        bbox: BoundingBox,
        image_height: int,
    ) -> str:
        """Classify a text block by its position and word arrangement.

        Args:
            words: Words in the block.
            bbox: Block bounding box.
            image_height: Source image height.

        Returns:
            Block type: ``"header"``, ``"table"``, or ``"paragraph"``.
        """
        if bbox.y < image_height * self.header_ratio:
            return "header"
        if self._is_table_like(words):
            return "table"
        return "paragraph"

    def _is_table_like(self, words: list[OCRWord]) -> bool:
        """Heuristic check for table-like word arrangement.

        Looks for grid-like alignment by examining x-position spacing
        consistency.

        Args:
            words: Words to analyze.

        Returns:
            True if the arrangement resembles a table.
        """
        if len(words) < 4:
            return False

        x_positions = sorted({w.bbox.x for w in words})
        if len(x_positions) >= 3:
            gaps = [
                x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)
            ]
            min_gap = min(gaps)
            if min_gap > 0 and max(gaps) / min_gap < 2:
                return True
        return False
