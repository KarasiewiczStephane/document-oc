"""Accuracy benchmarking system for document extraction evaluation.

Compares extraction predictions against labeled ground truth sets
and computes precision, recall, F1 score, and accuracy metrics.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FieldMetrics:
    """Precision, recall, F1, and accuracy metrics for a single field.

    Args:
        field_name: Name of the extracted field being measured.
    """

    field_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    exact_matches: int = 0
    total: int = 0

    @property
    def precision(self) -> float:
        """Fraction of predicted values that are correct."""
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        """Fraction of expected values that were correctly predicted."""
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Fraction of exact string matches."""
        if self.total == 0:
            return 0.0
        return self.exact_matches / self.total


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results across all documents and fields.

    Args:
        total_documents: Number of documents in ground truth.
        successful_documents: Number of documents with predictions.
        overall_accuracy: Mean field-level accuracy.
        overall_f1: Mean field-level F1 score.
        field_metrics: Per-field metric details.
        avg_processing_time_ms: Average processing time in milliseconds.
        errors: List of error messages encountered.
    """

    total_documents: int
    successful_documents: int
    overall_accuracy: float
    overall_f1: float
    field_metrics: dict[str, FieldMetrics]
    avg_processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class Evaluator:
    """Evaluates extraction predictions against ground truth labels.

    Supports exact matching and fuzzy matching for numerical values.

    Args:
        fuzzy_threshold: Tolerance for numerical fuzzy matching.
    """

    def __init__(self, fuzzy_threshold: float = 0.01) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    def evaluate(
        self,
        predictions: dict[str, dict[str, str]],
        ground_truth: dict[str, dict[str, str]],
    ) -> BenchmarkResult:
        """Compare predictions against ground truth and compute metrics.

        Args:
            predictions: Mapping of filename to extracted field values.
            ground_truth: Mapping of filename to expected field values.

        Returns:
            Aggregated benchmark results with per-field metrics.
        """
        field_metrics: dict[str, FieldMetrics] = {}
        errors: list[str] = []
        missing_count = 0

        for filename, expected in ground_truth.items():
            if filename not in predictions:
                errors.append(f"Missing prediction for {filename}")
                missing_count += 1
                for field_name in expected:
                    if field_name not in field_metrics:
                        field_metrics[field_name] = FieldMetrics(field_name)
                    field_metrics[field_name].total += 1
                    field_metrics[field_name].false_negatives += 1
                continue

            predicted = predictions[filename]

            for field_name, expected_value in expected.items():
                if field_name not in field_metrics:
                    field_metrics[field_name] = FieldMetrics(field_name)
                metrics = field_metrics[field_name]
                metrics.total += 1

                if field_name in predicted:
                    pred_value = str(predicted[field_name]).strip().lower()
                    exp_value = str(expected_value).strip().lower()

                    if pred_value == exp_value:
                        metrics.true_positives += 1
                        metrics.exact_matches += 1
                    elif self._fuzzy_match(pred_value, exp_value):
                        metrics.true_positives += 1
                    else:
                        metrics.false_positives += 1
                else:
                    metrics.false_negatives += 1

        all_f1 = [m.f1 for m in field_metrics.values() if m.total > 0]
        all_acc = [m.accuracy for m in field_metrics.values() if m.total > 0]

        return BenchmarkResult(
            total_documents=len(ground_truth),
            successful_documents=len(ground_truth) - missing_count,
            overall_accuracy=sum(all_acc) / len(all_acc) if all_acc else 0.0,
            overall_f1=sum(all_f1) / len(all_f1) if all_f1 else 0.0,
            field_metrics=field_metrics,
            errors=errors,
        )

    def _fuzzy_match(self, pred: str, expected: str) -> bool:
        """Check if two values match after normalizing amounts and numbers.

        Args:
            pred: Predicted value (lowercased, stripped).
            expected: Expected value (lowercased, stripped).

        Returns:
            True if values are considered equivalent.
        """
        pred_clean = pred.replace(",", "").replace("$", "").replace(" ", "")
        exp_clean = expected.replace(",", "").replace("$", "").replace(" ", "")

        if pred_clean == exp_clean:
            return True

        try:
            pred_num = float(pred_clean)
            exp_num = float(exp_clean)
            return abs(pred_num - exp_num) < self.fuzzy_threshold
        except ValueError:
            return False

    def generate_report(
        self, result: BenchmarkResult, output_path: Path | None = None
    ) -> str:
        """Generate a human-readable benchmark report.

        Args:
            result: Benchmark results to format.
            output_path: Optional path to write the report file.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "BENCHMARK REPORT",
            "=" * 60,
            f"Total Documents:      {result.total_documents}",
            f"Successful:           {result.successful_documents}",
            f"Overall Accuracy:     {result.overall_accuracy:.2%}",
            f"Overall F1 Score:     {result.overall_f1:.3f}",
            f"Avg Processing Time:  {result.avg_processing_time_ms:.0f}ms",
            "",
            "Field-Level Metrics:",
            "-" * 60,
            f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}",
            "-" * 60,
        ]

        for name, metrics in sorted(result.field_metrics.items()):
            lines.append(
                f"{name:<20} {metrics.precision:>10.2%} {metrics.recall:>10.2%} "
                f"{metrics.f1:>10.3f} {metrics.accuracy:>10.2%}"
            )

        target_met = result.overall_accuracy >= 0.9
        lines.extend(
            [
                "-" * 60,
                "",
                f"Target: >90% accuracy - {'PASSED' if target_met else 'FAILED'}",
                "=" * 60,
            ]
        )

        if result.errors:
            lines.append("")
            lines.append("Errors:")
            for error in result.errors:
                lines.append(f"  - {error}")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info("Report written to %s", output_path)

        return report


def load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    """Load ground truth labels from a JSON or CSV file.

    JSON format: ``{"filename": {"field": "value", ...}, ...}``
    CSV format: rows with a ``filename`` column and field value columns.

    Args:
        path: Path to the ground truth file.

    Returns:
        Mapping of filename to field-value pairs.

    Raises:
        ValueError: If the file format is not supported.
    """
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    if path.suffix == ".csv":
        gt: dict[str, dict[str, str]] = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.pop("filename")
                gt[filename] = {k: v for k, v in row.items() if v}
        return gt

    raise ValueError(f"Unsupported ground truth format: {path.suffix}")
