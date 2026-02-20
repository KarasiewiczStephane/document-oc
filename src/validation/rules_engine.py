"""Configurable validation rules engine for extracted document fields.

Validates dates, amounts, emails, phone numbers, and required fields
with confidence adjustments and cross-field validation.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


DATE_FORMATS: list[str] = [
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
]


@dataclass
class ValidationResult:
    """Result of a single field validation check."""

    field_name: str
    is_valid: bool
    message: str
    rule_name: str
    confidence_adjustment: float = 0.0


@dataclass
class ValidationReport:
    """Aggregated validation report for a document."""

    all_valid: bool
    results: list[ValidationResult]
    warnings: list[str] = field(default_factory=list)
    field_confidences: dict[str, float] = field(default_factory=dict)


class RulesEngine:
    """Configurable validation rules engine.

    Applies field-level and cross-field validation rules loaded from
    a YAML configuration file, with confidence score adjustments.

    Args:
        rules_path: Path to the validation rules YAML file.
    """

    def __init__(
        self, rules_path: Path = Path("configs/validation_rules.yaml")
    ) -> None:
        self.rules = self._load_rules(rules_path)
        self._validators: dict[str, Any] = {
            "date_format": self._validate_date,
            "positive_amount": self._validate_positive_amount,
            "amount_range": self._validate_amount_range,
            "required": self._validate_required,
            "regex": self._validate_regex,
            "email": self._validate_email,
            "phone": self._validate_phone,
        }

    def _load_rules(self, path: Path) -> dict:
        """Load validation rules from YAML file.

        Args:
            path: Path to the rules file.

        Returns:
            Dictionary of document-type-specific rules.
        """
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                if data:
                    logger.info("Loaded validation rules from %s", path)
                    return data
        logger.debug("Using default validation rules")
        return self._default_rules()

    def _default_rules(self) -> dict:
        """Provide sensible default rules when no config is available.

        Returns:
            Default rules dictionary.
        """
        return {
            "invoice": {
                "date": [{"type": "date_format"}, {"type": "required"}],
                "total_amount": [{"type": "positive_amount"}, {"type": "required"}],
                "vendor": [{"type": "required"}],
            },
            "receipt": {
                "date": [{"type": "date_format"}],
                "total_amount": [{"type": "positive_amount"}],
            },
        }

    def validate(
        self,
        fields: dict[str, Any],
        document_type: str = "invoice",
        field_confidences: dict[str, float] | None = None,
    ) -> ValidationReport:
        """Validate extracted fields against document-type rules.

        Args:
            fields: Extracted field name-value pairs.
            document_type: Type of document for rule selection.
            field_confidences: Initial confidence scores per field.

        Returns:
            Validation report with results and adjusted confidences.
        """
        results: list[ValidationResult] = []
        warnings: list[str] = []
        adjusted = dict(field_confidences or {})

        doc_rules = self.rules.get(document_type, {})

        for field_name, rules in doc_rules.items():
            value = fields.get(field_name)

            for rule in rules:
                rule_type = rule.get("type")
                validator = self._validators.get(rule_type)

                if not validator:
                    warnings.append(f"Unknown rule type: {rule_type}")
                    continue

                result = validator(field_name, value, rule)
                results.append(result)

                if field_name in adjusted:
                    adjusted[field_name] += result.confidence_adjustment
                    adjusted[field_name] = max(0.0, min(1.0, adjusted[field_name]))

        cross_results = self._cross_validate(fields)
        results.extend(cross_results)

        all_valid = all(r.is_valid for r in results)
        logger.info(
            "Validation for %s: %s (%d checks)",
            document_type,
            "PASSED" if all_valid else "FAILED",
            len(results),
        )

        return ValidationReport(
            all_valid=all_valid,
            results=results,
            warnings=warnings,
            field_confidences=adjusted,
        )

    def _validate_date(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Check if a value matches any supported date format."""
        if value is None:
            return ValidationResult(
                field_name, True, "No value to validate", "date_format"
            )

        for fmt in DATE_FORMATS:
            try:
                datetime.strptime(str(value), fmt)
                return ValidationResult(
                    field_name, True, f"Valid date format: {fmt}", "date_format", 0.1
                )
            except ValueError:
                continue

        return ValidationResult(
            field_name, False, f"Invalid date format: {value}", "date_format", -0.2
        )

    def _validate_positive_amount(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Check if a value is a positive monetary amount."""
        if value is None:
            return ValidationResult(
                field_name, True, "No value to validate", "positive_amount"
            )

        try:
            amount = Decimal(str(value).replace(",", "").replace("$", ""))
            if amount > 0:
                return ValidationResult(
                    field_name,
                    True,
                    f"Valid positive amount: {amount}",
                    "positive_amount",
                    0.1,
                )
            return ValidationResult(
                field_name,
                False,
                f"Amount must be positive: {amount}",
                "positive_amount",
                -0.2,
            )
        except InvalidOperation:
            return ValidationResult(
                field_name,
                False,
                f"Invalid amount format: {value}",
                "positive_amount",
                -0.3,
            )

    def _validate_amount_range(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Check if an amount falls within a specified range."""
        if value is None:
            return ValidationResult(
                field_name, True, "No value to validate", "amount_range"
            )

        try:
            amount = Decimal(str(value).replace(",", "").replace("$", ""))
            min_val = Decimal(str(rule.get("min", 0)))
            max_val = Decimal(str(rule.get("max", 1000000)))

            if min_val <= amount <= max_val:
                return ValidationResult(
                    field_name, True, "Amount in valid range", "amount_range", 0.05
                )
            return ValidationResult(
                field_name,
                False,
                f"Amount {amount} outside range [{min_val}, {max_val}]",
                "amount_range",
                -0.15,
            )
        except (InvalidOperation, TypeError):
            return ValidationResult(
                field_name, False, f"Invalid amount: {value}", "amount_range", -0.2
            )

    def _validate_required(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Check if a required field is present and non-empty."""
        if value is not None and str(value).strip():
            return ValidationResult(
                field_name, True, "Required field present", "required", 0.0
            )
        return ValidationResult(
            field_name,
            False,
            f"Required field missing: {field_name}",
            "required",
            -0.5,
        )

    def _validate_regex(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Validate a field value against a custom regex pattern."""
        if value is None:
            return ValidationResult(field_name, True, "No value to validate", "regex")

        pattern = rule.get("pattern", "")
        if re.match(pattern, str(value)):
            return ValidationResult(field_name, True, "Matches pattern", "regex", 0.05)
        return ValidationResult(
            field_name,
            False,
            f"Does not match pattern: {pattern}",
            "regex",
            -0.1,
        )

    def _validate_email(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Validate email format."""
        if value is None:
            return ValidationResult(field_name, True, "No value to validate", "email")

        pattern = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
        if re.match(pattern, str(value)):
            return ValidationResult(
                field_name, True, "Valid email format", "email", 0.1
            )
        return ValidationResult(
            field_name, False, f"Invalid email: {value}", "email", -0.2
        )

    def _validate_phone(
        self, field_name: str, value: Any, rule: dict
    ) -> ValidationResult:
        """Validate phone number format."""
        if value is None:
            return ValidationResult(field_name, True, "No value to validate", "phone")

        cleaned = re.sub(r"[\s\-\(\)\.]", "", str(value))
        if re.match(r"^\+?1?\d{10,11}$", cleaned):
            return ValidationResult(
                field_name, True, "Valid phone format", "phone", 0.1
            )
        return ValidationResult(
            field_name, False, f"Invalid phone: {value}", "phone", -0.2
        )

    def _cross_validate(self, fields: dict[str, Any]) -> list[ValidationResult]:
        """Run cross-field validation checks.

        Currently checks that line items sum matches the total amount.

        Args:
            fields: All extracted field values.

        Returns:
            List of cross-field validation results.
        """
        results: list[ValidationResult] = []

        if "line_items" in fields and "total_amount" in fields:
            try:
                items = fields["line_items"]
                if isinstance(items, list):
                    items_sum = sum(
                        Decimal(
                            str(item.get("amount", 0)).replace(",", "").replace("$", "")
                        )
                        for item in items
                    )
                    total = Decimal(
                        str(fields["total_amount"]).replace(",", "").replace("$", "")
                    )
                    tolerance = total * Decimal("0.05")
                    if abs(items_sum - total) <= tolerance:
                        results.append(
                            ValidationResult(
                                "line_items_total",
                                True,
                                "Line items sum matches total",
                                "cross_field",
                                0.1,
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                "line_items_total",
                                False,
                                f"Line items sum ({items_sum}) doesn't match total ({total})",
                                "cross_field",
                                -0.15,
                            )
                        )
            except (InvalidOperation, TypeError, AttributeError):
                pass

        return results
