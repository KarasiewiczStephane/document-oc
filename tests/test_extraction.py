"""Tests for rule-based field extraction and template matching."""

from pathlib import Path

import pytest
import yaml

from src.extraction.rule_extractor import ExtractedField, RuleExtractor
from src.extraction.template_matcher import TemplateMatch, TemplateMatcher


class TestRuleExtractor:
    """Tests for the RuleExtractor class."""

    def setup_method(self) -> None:
        self.extractor = RuleExtractor()

    def test_extract_date_slash_format(self) -> None:
        results = self.extractor.extract("Invoice date: 01/15/2024", ["date"])
        assert any(r.field_name == "date" for r in results)

    def test_extract_date_dash_format(self) -> None:
        results = self.extractor.extract("Date: 2024-01-15", ["date"])
        assert any(r.field_name == "date" for r in results)

    def test_extract_date_month_name(self) -> None:
        results = self.extractor.extract("Date: January 15, 2024", ["date"])
        assert any(r.field_name == "date" for r in results)

    def test_extract_date_day_month_format(self) -> None:
        results = self.extractor.extract("15 Jan 2024", ["date"])
        assert any(r.field_name == "date" for r in results)

    def test_extract_amount_dollar_sign(self) -> None:
        results = self.extractor.extract("Total: $1,234.56", ["amount"])
        amounts = [r for r in results if r.field_name == "amount"]
        assert len(amounts) >= 1
        assert any("1,234.56" in r.value or "1234.56" in r.value for r in amounts)

    def test_extract_amount_currency_code(self) -> None:
        results = self.extractor.extract("Amount: 500.00 USD", ["amount"])
        amounts = [r for r in results if r.field_name == "amount"]
        assert len(amounts) >= 1

    def test_extract_amount_with_label(self) -> None:
        results = self.extractor.extract("Total Due: 99.99", ["amount"])
        amounts = [r for r in results if r.field_name == "amount"]
        assert len(amounts) >= 1

    def test_extract_email(self) -> None:
        results = self.extractor.extract("Contact: user@example.com", ["email"])
        emails = [r for r in results if r.field_name == "email"]
        assert len(emails) == 1
        assert emails[0].value == "user@example.com"

    def test_extract_phone(self) -> None:
        results = self.extractor.extract("Phone: (555) 123-4567", ["phone"])
        phones = [r for r in results if r.field_name == "phone"]
        assert len(phones) >= 1

    def test_extract_invoice_number(self) -> None:
        results = self.extractor.extract("Invoice #INV-2024-001", ["invoice_number"])
        invoices = [r for r in results if r.field_name == "invoice_number"]
        assert len(invoices) >= 1
        assert "INV-2024-001" in invoices[0].value

    def test_extract_all_fields(self) -> None:
        text = (
            "ACME Corp\n"
            "Invoice #INV-001\n"
            "Date: 01/15/2024\n"
            "Email: billing@acme.com\n"
            "Phone: 555-123-4567\n"
            "Total: $1,500.00"
        )
        results = self.extractor.extract(text)
        field_names = {r.field_name for r in results}
        assert "date" in field_names
        assert "amount" in field_names
        assert "email" in field_names
        assert "invoice_number" in field_names

    def test_extract_no_matches(self) -> None:
        results = self.extractor.extract("just plain text", ["date"])
        assert len(results) == 0

    def test_extract_unknown_field_ignored(self) -> None:
        results = self.extractor.extract("test", ["nonexistent_field"])
        assert len(results) == 0

    def test_extracted_field_structure(self) -> None:
        results = self.extractor.extract("$99.99", ["amount"])
        assert len(results) >= 1
        field = results[0]
        assert isinstance(field, ExtractedField)
        assert field.extraction_method == "regex"
        assert 0 < field.confidence <= 1.0
        assert field.start_pos >= 0
        assert field.end_pos > field.start_pos

    def test_extract_multiple_dates(self) -> None:
        text = "Start: 01/01/2024 End: 12/31/2024"
        results = self.extractor.extract(text, ["date"])
        dates = [r for r in results if r.field_name == "date"]
        assert len(dates) >= 2


class TestExtractTotalAmount:
    """Tests for the total amount extraction method."""

    def setup_method(self) -> None:
        self.extractor = RuleExtractor()

    def test_grand_total(self) -> None:
        result = self.extractor.extract_total_amount("Grand Total: $1,234.56")
        assert result is not None
        assert result.value == "1234.56"
        assert result.confidence == 0.95

    def test_total_due(self) -> None:
        result = self.extractor.extract_total_amount("Total Due: 500.00")
        assert result is not None
        assert result.value == "500.00"

    def test_simple_total(self) -> None:
        result = self.extractor.extract_total_amount("Total: 99.99")
        assert result is not None
        assert result.value == "99.99"

    def test_no_total(self) -> None:
        result = self.extractor.extract_total_amount("Just some text here")
        assert result is None

    def test_total_field_name(self) -> None:
        result = self.extractor.extract_total_amount("Total: $50.00")
        assert result is not None
        assert result.field_name == "total_amount"
        assert result.extraction_method == "regex_total"


class TestTemplateMatcher:
    """Tests for the template matching component."""

    def test_load_empty_templates(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        matcher = TemplateMatcher(templates_path=empty_file)
        assert matcher.templates == {}

    def test_load_missing_file(self) -> None:
        matcher = TemplateMatcher(templates_path=Path("/nonexistent/templates.yaml"))
        assert matcher.templates == {}

    def test_match_invoice_template(self, tmp_path: Path) -> None:
        templates = {
            "invoice": {
                "identifiers": ["Invoice", "Bill To", "Total"],
                "min_confidence": 0.5,
                "fields": {"total": r"Total:\s*\$?([\d.]+)"},
            }
        }
        tpl_file = tmp_path / "templates.yaml"
        with open(tpl_file, "w") as f:
            yaml.dump(templates, f)

        matcher = TemplateMatcher(templates_path=tpl_file)
        text = "Invoice #001\nBill To: John\nTotal: $100.00"
        result = matcher.match_template(text)

        assert result is not None
        assert isinstance(result, TemplateMatch)
        assert result.template_name == "invoice"
        assert result.confidence == 1.0

    def test_no_match_below_threshold(self, tmp_path: Path) -> None:
        templates = {
            "receipt": {
                "identifiers": ["Receipt", "Thank you", "Subtotal", "Tax"],
                "min_confidence": 0.75,
                "fields": {},
            }
        }
        tpl_file = tmp_path / "templates.yaml"
        with open(tpl_file, "w") as f:
            yaml.dump(templates, f)

        matcher = TemplateMatcher(templates_path=tpl_file)
        result = matcher.match_template("Some random text with Receipt")
        assert result is None

    def test_best_match_wins(self, tmp_path: Path) -> None:
        templates = {
            "invoice": {
                "identifiers": ["Invoice"],
                "min_confidence": 0.5,
                "fields": {},
            },
            "receipt": {
                "identifiers": ["Receipt", "Thank you"],
                "min_confidence": 0.5,
                "fields": {},
            },
        }
        tpl_file = tmp_path / "templates.yaml"
        with open(tpl_file, "w") as f:
            yaml.dump(templates, f)

        matcher = TemplateMatcher(templates_path=tpl_file)
        text = "Receipt\nThank you for shopping"
        result = matcher.match_template(text)

        assert result is not None
        assert result.template_name == "receipt"
        assert result.confidence == 1.0

    def test_match_score_calculation(self, tmp_path: Path) -> None:
        templates = {
            "test": {
                "identifiers": ["word1", "word2", "word3", "word4"],
                "min_confidence": 0.3,
                "fields": {},
            }
        }
        tpl_file = tmp_path / "templates.yaml"
        with open(tpl_file, "w") as f:
            yaml.dump(templates, f)

        matcher = TemplateMatcher(templates_path=tpl_file)
        result = matcher.match_template("word1 word2")
        assert result is not None
        assert result.confidence == pytest.approx(0.5)

    def test_empty_identifiers(self, tmp_path: Path) -> None:
        templates = {
            "empty": {
                "identifiers": [],
                "min_confidence": 0.5,
                "fields": {},
            }
        }
        tpl_file = tmp_path / "templates.yaml"
        with open(tpl_file, "w") as f:
            yaml.dump(templates, f)

        matcher = TemplateMatcher(templates_path=tpl_file)
        result = matcher.match_template("anything")
        assert result is None
