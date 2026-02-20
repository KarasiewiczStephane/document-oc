"""Tests for the accuracy benchmarking system."""

import json
from pathlib import Path

import pytest

from src.benchmark.evaluator import (
    BenchmarkResult,
    Evaluator,
    FieldMetrics,
    load_ground_truth,
)


class TestFieldMetrics:
    """Tests for the FieldMetrics data class."""

    def test_precision_perfect(self) -> None:
        m = FieldMetrics("date", true_positives=5, false_positives=0)
        assert m.precision == 1.0

    def test_precision_partial(self) -> None:
        m = FieldMetrics("date", true_positives=3, false_positives=2)
        assert m.precision == 0.6

    def test_precision_zero_denom(self) -> None:
        m = FieldMetrics("date", true_positives=0, false_positives=0)
        assert m.precision == 0.0

    def test_recall_perfect(self) -> None:
        m = FieldMetrics("date", true_positives=5, false_negatives=0)
        assert m.recall == 1.0

    def test_recall_partial(self) -> None:
        m = FieldMetrics("date", true_positives=3, false_negatives=2)
        assert m.recall == 0.6

    def test_recall_zero_denom(self) -> None:
        m = FieldMetrics("date", true_positives=0, false_negatives=0)
        assert m.recall == 0.0

    def test_f1_perfect(self) -> None:
        m = FieldMetrics("date", true_positives=5, false_positives=0, false_negatives=0)
        assert m.f1 == 1.0

    def test_f1_zero(self) -> None:
        m = FieldMetrics("date", true_positives=0, false_positives=0, false_negatives=0)
        assert m.f1 == 0.0

    def test_accuracy_perfect(self) -> None:
        m = FieldMetrics("date", exact_matches=5, total=5)
        assert m.accuracy == 1.0

    def test_accuracy_partial(self) -> None:
        m = FieldMetrics("date", exact_matches=3, total=5)
        assert m.accuracy == 0.6

    def test_accuracy_zero_total(self) -> None:
        m = FieldMetrics("date", exact_matches=0, total=0)
        assert m.accuracy == 0.0


class TestEvaluator:
    """Tests for the Evaluator class."""

    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_perfect_predictions(self) -> None:
        gt = {"doc1.png": {"date": "01/15/2024", "total": "500.00"}}
        pred = {"doc1.png": {"date": "01/15/2024", "total": "500.00"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.overall_accuracy == 1.0
        assert result.overall_f1 == 1.0
        assert result.total_documents == 1
        assert result.successful_documents == 1

    def test_missing_prediction(self) -> None:
        gt = {"doc1.png": {"date": "01/15/2024"}}
        pred: dict[str, dict[str, str]] = {}
        result = self.evaluator.evaluate(pred, gt)
        assert result.successful_documents == 0
        assert len(result.errors) == 1
        assert "Missing" in result.errors[0]

    def test_missing_field(self) -> None:
        gt = {"doc1.png": {"date": "01/15/2024", "total": "500.00"}}
        pred = {"doc1.png": {"date": "01/15/2024"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["total"].false_negatives == 1

    def test_wrong_value(self) -> None:
        gt = {"doc1.png": {"date": "01/15/2024"}}
        pred = {"doc1.png": {"date": "02/20/2024"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["date"].false_positives == 1

    def test_case_insensitive_match(self) -> None:
        gt = {"doc1.png": {"vendor": "ACME Corp"}}
        pred = {"doc1.png": {"vendor": "acme corp"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["vendor"].true_positives == 1

    def test_fuzzy_amount_match(self) -> None:
        gt = {"doc1.png": {"total": "$1,234.56"}}
        pred = {"doc1.png": {"total": "1234.56"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["total"].true_positives == 1

    def test_fuzzy_numeric_tolerance(self) -> None:
        gt = {"doc1.png": {"total": "100.00"}}
        pred = {"doc1.png": {"total": "100.005"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["total"].true_positives == 1

    def test_no_fuzzy_match_beyond_threshold(self) -> None:
        gt = {"doc1.png": {"total": "100.00"}}
        pred = {"doc1.png": {"total": "100.05"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["total"].false_positives == 1

    def test_multiple_documents(self) -> None:
        gt = {
            "doc1.png": {"date": "01/15/2024"},
            "doc2.png": {"date": "02/20/2024"},
        }
        pred = {
            "doc1.png": {"date": "01/15/2024"},
            "doc2.png": {"date": "02/20/2024"},
        }
        result = self.evaluator.evaluate(pred, gt)
        assert result.total_documents == 2
        assert result.overall_accuracy == 1.0

    def test_empty_ground_truth(self) -> None:
        result = self.evaluator.evaluate({}, {})
        assert result.total_documents == 0
        assert result.overall_accuracy == 0.0
        assert result.overall_f1 == 0.0

    def test_non_numeric_no_fuzzy_match(self) -> None:
        gt = {"doc1.png": {"vendor": "ACME"}}
        pred = {"doc1.png": {"vendor": "ACM"}}
        result = self.evaluator.evaluate(pred, gt)
        assert result.field_metrics["vendor"].false_positives == 1


class TestFuzzyMatch:
    """Tests for the fuzzy matching logic."""

    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_dollar_sign_removed(self) -> None:
        assert self.evaluator._fuzzy_match("$100.00", "100.00")

    def test_comma_removed(self) -> None:
        assert self.evaluator._fuzzy_match("1,000.00", "1000.00")

    def test_space_removed(self) -> None:
        assert self.evaluator._fuzzy_match("100 .00", "100.00")

    def test_exact_after_cleanup(self) -> None:
        assert self.evaluator._fuzzy_match("$1,234.56", "$1,234.56")

    def test_numeric_within_threshold(self) -> None:
        assert self.evaluator._fuzzy_match("100.005", "100.00")

    def test_numeric_beyond_threshold(self) -> None:
        assert not self.evaluator._fuzzy_match("100.02", "100.00")

    def test_non_numeric_no_match(self) -> None:
        assert not self.evaluator._fuzzy_match("abc", "xyz")


class TestGenerateReport:
    """Tests for report generation."""

    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_report_contains_header(self) -> None:
        result = BenchmarkResult(
            total_documents=1,
            successful_documents=1,
            overall_accuracy=0.95,
            overall_f1=0.9,
            field_metrics={
                "date": FieldMetrics(
                    "date",
                    true_positives=1,
                    total=1,
                    exact_matches=1,
                )
            },
        )
        report = self.evaluator.generate_report(result)
        assert "BENCHMARK REPORT" in report
        assert "95.00%" in report

    def test_report_target_passed(self) -> None:
        result = BenchmarkResult(
            total_documents=1,
            successful_documents=1,
            overall_accuracy=0.95,
            overall_f1=0.9,
            field_metrics={},
        )
        report = self.evaluator.generate_report(result)
        assert "PASSED" in report

    def test_report_target_failed(self) -> None:
        result = BenchmarkResult(
            total_documents=1,
            successful_documents=1,
            overall_accuracy=0.5,
            overall_f1=0.4,
            field_metrics={},
        )
        report = self.evaluator.generate_report(result)
        assert "FAILED" in report

    def test_report_shows_errors(self) -> None:
        result = BenchmarkResult(
            total_documents=1,
            successful_documents=0,
            overall_accuracy=0.0,
            overall_f1=0.0,
            field_metrics={},
            errors=["Missing prediction for doc1.png"],
        )
        report = self.evaluator.generate_report(result)
        assert "Errors:" in report
        assert "Missing prediction" in report

    def test_report_writes_to_file(self, tmp_path: Path) -> None:
        result = BenchmarkResult(
            total_documents=1,
            successful_documents=1,
            overall_accuracy=1.0,
            overall_f1=1.0,
            field_metrics={},
        )
        output = tmp_path / "report.txt"
        self.evaluator.generate_report(result, output)
        assert output.exists()
        assert "BENCHMARK REPORT" in output.read_text()

    def test_report_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = BenchmarkResult(
            total_documents=0,
            successful_documents=0,
            overall_accuracy=0.0,
            overall_f1=0.0,
            field_metrics={},
        )
        output = tmp_path / "sub" / "report.txt"
        self.evaluator.generate_report(result, output)
        assert output.exists()


class TestLoadGroundTruth:
    """Tests for ground truth file loading."""

    def test_load_json(self, tmp_path: Path) -> None:
        gt_data = {
            "doc1.png": {"date": "01/15/2024", "total": "500.00"},
            "doc2.png": {"date": "02/20/2024"},
        }
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(gt_data))

        loaded = load_ground_truth(gt_file)
        assert loaded == gt_data

    def test_load_csv(self, tmp_path: Path) -> None:
        csv_content = (
            "filename,date,total\ndoc1.png,01/15/2024,500.00\ndoc2.png,02/20/2024,\n"
        )
        gt_file = tmp_path / "gt.csv"
        gt_file.write_text(csv_content)

        loaded = load_ground_truth(gt_file)
        assert loaded["doc1.png"]["date"] == "01/15/2024"
        assert loaded["doc1.png"]["total"] == "500.00"
        assert "total" not in loaded["doc2.png"]

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        gt_file = tmp_path / "gt.yaml"
        gt_file.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            load_ground_truth(gt_file)
