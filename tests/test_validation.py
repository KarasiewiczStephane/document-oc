"""Tests for the validation rules engine."""

from pathlib import Path

import yaml

from src.validation.rules_engine import RulesEngine, ValidationReport, ValidationResult


class TestValidationResult:
    """Tests for the ValidationResult data class."""

    def test_creation(self) -> None:
        result = ValidationResult("date", True, "Valid", "date_format", 0.1)
        assert result.field_name == "date"
        assert result.is_valid is True
        assert result.confidence_adjustment == 0.1


class TestRulesEngine:
    """Tests for the RulesEngine class."""

    def setup_method(self) -> None:
        self.engine = RulesEngine()

    def test_validate_date_valid_formats(self) -> None:
        valid_dates = [
            "01/15/2024",
            "2024-01-15",
            "January 15, 2024",
            "Jan 15, 2024",
            "15 January 2024",
        ]
        for date in valid_dates:
            result = self.engine._validate_date("date", date, {})
            assert result.is_valid, f"Expected valid: {date}"

    def test_validate_date_invalid(self) -> None:
        result = self.engine._validate_date("date", "not-a-date", {})
        assert result.is_valid is False
        assert result.confidence_adjustment < 0

    def test_validate_date_none(self) -> None:
        result = self.engine._validate_date("date", None, {})
        assert result.is_valid is True

    def test_validate_positive_amount_valid(self) -> None:
        result = self.engine._validate_positive_amount("total", "1234.56", {})
        assert result.is_valid is True

    def test_validate_positive_amount_with_comma(self) -> None:
        result = self.engine._validate_positive_amount("total", "$1,234.56", {})
        assert result.is_valid is True

    def test_validate_positive_amount_zero(self) -> None:
        result = self.engine._validate_positive_amount("total", "0", {})
        assert result.is_valid is False

    def test_validate_positive_amount_negative(self) -> None:
        result = self.engine._validate_positive_amount("total", "-100", {})
        assert result.is_valid is False

    def test_validate_positive_amount_invalid(self) -> None:
        result = self.engine._validate_positive_amount("total", "abc", {})
        assert result.is_valid is False

    def test_validate_positive_amount_none(self) -> None:
        result = self.engine._validate_positive_amount("total", None, {})
        assert result.is_valid is True

    def test_validate_amount_range_valid(self) -> None:
        rule = {"min": 10, "max": 1000}
        result = self.engine._validate_amount_range("total", "500", rule)
        assert result.is_valid is True

    def test_validate_amount_range_below(self) -> None:
        rule = {"min": 100, "max": 1000}
        result = self.engine._validate_amount_range("total", "50", rule)
        assert result.is_valid is False

    def test_validate_amount_range_above(self) -> None:
        rule = {"min": 0, "max": 100}
        result = self.engine._validate_amount_range("total", "500", rule)
        assert result.is_valid is False

    def test_validate_amount_range_none(self) -> None:
        result = self.engine._validate_amount_range("total", None, {})
        assert result.is_valid is True

    def test_validate_amount_range_invalid(self) -> None:
        result = self.engine._validate_amount_range("total", "not_num", {})
        assert result.is_valid is False

    def test_validate_required_present(self) -> None:
        result = self.engine._validate_required("vendor", "ACME Corp", {})
        assert result.is_valid is True

    def test_validate_required_missing(self) -> None:
        result = self.engine._validate_required("vendor", None, {})
        assert result.is_valid is False
        assert result.confidence_adjustment == -0.5

    def test_validate_required_empty_string(self) -> None:
        result = self.engine._validate_required("vendor", "  ", {})
        assert result.is_valid is False

    def test_validate_email_valid(self) -> None:
        result = self.engine._validate_email("email", "user@example.com", {})
        assert result.is_valid is True

    def test_validate_email_invalid(self) -> None:
        result = self.engine._validate_email("email", "not-an-email", {})
        assert result.is_valid is False

    def test_validate_email_none(self) -> None:
        result = self.engine._validate_email("email", None, {})
        assert result.is_valid is True

    def test_validate_phone_valid(self) -> None:
        valid_phones = ["5551234567", "+15551234567", "(555) 123-4567"]
        for phone in valid_phones:
            result = self.engine._validate_phone("phone", phone, {})
            assert result.is_valid, f"Expected valid: {phone}"

    def test_validate_phone_invalid(self) -> None:
        result = self.engine._validate_phone("phone", "123", {})
        assert result.is_valid is False

    def test_validate_phone_none(self) -> None:
        result = self.engine._validate_phone("phone", None, {})
        assert result.is_valid is True

    def test_validate_regex_match(self) -> None:
        rule = {"pattern": r"^INV-\d+$"}
        result = self.engine._validate_regex("invoice", "INV-001", rule)
        assert result.is_valid is True

    def test_validate_regex_no_match(self) -> None:
        rule = {"pattern": r"^INV-\d+$"}
        result = self.engine._validate_regex("invoice", "ABC-001", rule)
        assert result.is_valid is False

    def test_validate_regex_none(self) -> None:
        result = self.engine._validate_regex("invoice", None, {})
        assert result.is_valid is True


class TestValidationReport:
    """Tests for full validation report generation."""

    def setup_method(self) -> None:
        self.engine = RulesEngine()

    def test_validate_invoice_all_valid(self) -> None:
        fields = {
            "date": "01/15/2024",
            "total_amount": "500.00",
            "vendor": "ACME Corp",
        }
        report = self.engine.validate(fields, "invoice")
        assert isinstance(report, ValidationReport)
        assert report.all_valid is True

    def test_validate_invoice_missing_required(self) -> None:
        fields = {"date": "01/15/2024", "total_amount": "500.00"}
        report = self.engine.validate(fields, "invoice")
        assert report.all_valid is False
        failed = [r for r in report.results if not r.is_valid]
        assert any(r.field_name == "vendor" for r in failed)

    def test_validate_receipt(self) -> None:
        fields = {"date": "2024-01-15", "total_amount": "25.99", "vendor": "Store"}
        report = self.engine.validate(fields, "receipt")
        assert report.all_valid is True

    def test_validate_unknown_doc_type_empty_rules(self) -> None:
        fields = {"date": "2024-01-15"}
        report = self.engine.validate(fields, "unknown_type")
        assert report.all_valid is True
        assert len(report.results) == 0

    def test_confidence_adjustment(self) -> None:
        fields = {"date": "01/15/2024", "total_amount": "500.00", "vendor": "ACME"}
        confidences = {"date": 0.8, "total_amount": 0.9, "vendor": 0.7}
        report = self.engine.validate(fields, "invoice", confidences)

        assert report.field_confidences["date"] > 0.8
        assert report.field_confidences["total_amount"] > 0.9

    def test_confidence_clamped_to_bounds(self) -> None:
        fields = {"date": "01/15/2024"}
        confidences = {"date": 0.99}
        report = self.engine.validate(fields, "invoice", confidences)
        assert 0.0 <= report.field_confidences["date"] <= 1.0

    def test_unknown_rule_type_warns(self, tmp_path: Path) -> None:
        rules = {
            "invoice": {
                "date": [{"type": "nonexistent_rule"}],
            }
        }
        rules_file = tmp_path / "rules.yaml"
        with open(rules_file, "w") as f:
            yaml.dump(rules, f)

        engine = RulesEngine(rules_path=rules_file)
        report = engine.validate({"date": "2024-01-15"}, "invoice")
        assert len(report.warnings) > 0


class TestCrossValidation:
    """Tests for cross-field validation."""

    def setup_method(self) -> None:
        self.engine = RulesEngine()

    def test_line_items_match_total(self) -> None:
        fields = {
            "total_amount": "100.00",
            "line_items": [
                {"amount": "60.00"},
                {"amount": "40.00"},
            ],
        }
        results = self.engine._cross_validate(fields)
        assert len(results) == 1
        assert results[0].is_valid is True

    def test_line_items_within_tolerance(self) -> None:
        fields = {
            "total_amount": "100.00",
            "line_items": [
                {"amount": "60.00"},
                {"amount": "38.00"},
            ],
        }
        results = self.engine._cross_validate(fields)
        assert len(results) == 1
        assert results[0].is_valid is True

    def test_line_items_mismatch(self) -> None:
        fields = {
            "total_amount": "100.00",
            "line_items": [
                {"amount": "60.00"},
                {"amount": "10.00"},
            ],
        }
        results = self.engine._cross_validate(fields)
        assert len(results) == 1
        assert results[0].is_valid is False

    def test_no_cross_validation_without_both_fields(self) -> None:
        fields = {"total_amount": "100.00"}
        results = self.engine._cross_validate(fields)
        assert len(results) == 0

    def test_cross_validation_handles_invalid_amounts(self) -> None:
        fields = {
            "total_amount": "not_a_number",
            "line_items": [{"amount": "50.00"}],
        }
        results = self.engine._cross_validate(fields)
        assert len(results) == 0


class TestRulesEngineConfig:
    """Tests for rules engine configuration loading."""

    def test_load_from_yaml(self) -> None:
        engine = RulesEngine()
        assert "invoice" in engine.rules

    def test_load_missing_file_uses_defaults(self) -> None:
        engine = RulesEngine(rules_path=Path("/nonexistent/rules.yaml"))
        assert "invoice" in engine.rules

    def test_custom_rules(self, tmp_path: Path) -> None:
        custom_rules = {
            "custom_doc": {
                "field_a": [{"type": "required"}],
            }
        }
        rules_file = tmp_path / "rules.yaml"
        with open(rules_file, "w") as f:
            yaml.dump(custom_rules, f)

        engine = RulesEngine(rules_path=rules_file)
        assert "custom_doc" in engine.rules
