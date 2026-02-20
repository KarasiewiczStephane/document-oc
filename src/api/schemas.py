"""Pydantic request/response schemas for the FastAPI endpoints."""

from enum import StrEnum

from pydantic import BaseModel


class DocumentType(StrEnum):
    """Supported document types for extraction."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    FORM = "form"
    AUTO = "auto"


class ExtractedFieldResponse(BaseModel):
    """Response schema for a single extracted field."""

    field_name: str
    value: str
    confidence: float
    source: str
    is_valid: bool = True
    validation_message: str | None = None


class ValidationResultResponse(BaseModel):
    """Response schema for a validation check result."""

    field_name: str
    is_valid: bool
    message: str
    rule_name: str


class ExtractionResponse(BaseModel):
    """Response schema for a document extraction request."""

    success: bool
    document_id: str
    document_type: str
    fields: list[ExtractedFieldResponse]
    raw_text: str
    overall_confidence: float
    validation: list[ValidationResultResponse]
    processing_time_ms: float
    page_count: int = 1


class BatchItemResponse(BaseModel):
    """Response schema for a single item in a batch extraction."""

    filename: str
    result: ExtractionResponse | None = None
    error: str | None = None


class BatchExtractionResponse(BaseModel):
    """Response schema for batch extraction of multiple documents."""

    success: bool
    total_documents: int
    successful: int
    failed: int
    results: list[BatchItemResponse]


class TemplateInfo(BaseModel):
    """Information about a supported document template."""

    name: str
    description: str
    supported_fields: list[str]


class TemplatesResponse(BaseModel):
    """Response schema listing available document templates."""

    templates: list[TemplateInfo]


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str
    version: str
    tesseract_available: bool
    gpu_available: bool
