"""FastAPI application for the Document Intelligence OCR API.

Provides REST endpoints for document extraction, batch processing,
template listing, and health checks.
"""

import shutil
import time
import uuid
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.extraction.hybrid import HybridExtractor
from src.ocr.document_processor import DocumentProcessor
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.validation.rules_engine import RulesEngine

from .schemas import (
    BatchExtractionResponse,
    BatchItemResponse,
    DocumentType,
    ExtractedFieldResponse,
    ExtractionResponse,
    HealthResponse,
    TemplateInfo,
    TemplatesResponse,
    ValidationResultResponse,
)

logger = get_logger(__name__)

app = FastAPI(
    title="Document Intelligence OCR API",
    description="Extract structured data from invoices, receipts, and forms",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_components() -> tuple[DocumentProcessor, HybridExtractor, RulesEngine]:
    """Initialize and return shared processing components.

    Returns:
        Tuple of (document_processor, hybrid_extractor, rules_engine).
    """
    config = load_config()
    doc_processor = DocumentProcessor(config)
    hybrid_extractor = HybridExtractor(config.extraction)
    rules_engine = RulesEngine()
    return doc_processor, hybrid_extractor, rules_engine


_ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "application/pdf",
    "application/octet-stream",
}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        tesseract_available=shutil.which("tesseract") is not None,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_document(
    file: Annotated[UploadFile, File(...)],
    document_type: Annotated[DocumentType, Query()] = DocumentType.AUTO,
    use_ml: Annotated[bool, Query()] = True,
) -> ExtractionResponse:
    """Extract structured fields from an uploaded document.

    Args:
        file: Uploaded document file (PNG, JPEG, TIFF, or PDF).
        document_type: Document type for validation rules.
        use_ml: Whether to use ML-based extraction.

    Returns:
        Extraction results with fields, confidence, and validation.
    """
    start_time = time.time()

    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}",
        )

    try:
        doc_processor, hybrid_extractor, rules_engine = _get_components()
        content = await file.read()
        doc_result = doc_processor.process(content, file.filename or "document")

        all_fields = []
        for page in doc_result.pages:
            extraction = hybrid_extractor.extract(
                image=None,
                ocr_text=page.ocr_result.text,
                ocr_words=page.ocr_result.words if page.ocr_result.words else None,
                ocr_confidence=page.ocr_result.confidence,
                use_ml=use_ml,
            )
            all_fields.extend(extraction.fields)

        doc_type_str = (
            document_type.value if document_type != DocumentType.AUTO else "invoice"
        )
        field_dict = {f.field_name: f.value for f in all_fields}
        field_confidences = {f.field_name: f.confidence for f in all_fields}
        validation = rules_engine.validate(field_dict, doc_type_str, field_confidences)

        fields_response = [
            ExtractedFieldResponse(
                field_name=f.field_name,
                value=f.value,
                confidence=validation.field_confidences.get(f.field_name, f.confidence),
                source=f.source,
                is_valid=all(
                    r.is_valid
                    for r in validation.results
                    if r.field_name == f.field_name
                ),
                validation_message=next(
                    (
                        r.message
                        for r in validation.results
                        if r.field_name == f.field_name and not r.is_valid
                    ),
                    None,
                ),
            )
            for f in all_fields
        ]

        validation_response = [
            ValidationResultResponse(
                field_name=r.field_name,
                is_valid=r.is_valid,
                message=r.message,
                rule_name=r.rule_name,
            )
            for r in validation.results
        ]

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResponse(
            success=True,
            document_id=str(uuid.uuid4()),
            document_type=doc_type_str,
            fields=fields_response,
            raw_text=doc_result.combined_text,
            overall_confidence=(
                sum(f.confidence for f in all_fields) / len(all_fields)
                if all_fields
                else 0.0
            ),
            validation=validation_response,
            processing_time_ms=processing_time,
            page_count=doc_result.page_count,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_batch(
    files: Annotated[list[UploadFile], File(...)],
) -> BatchExtractionResponse:
    """Extract structured fields from multiple uploaded documents.

    Args:
        files: List of uploaded document files.

    Returns:
        Batch extraction results with per-file outcomes.
    """
    results: list[BatchItemResponse] = []
    successful = 0

    for file in files:
        try:
            result = await extract_document(file)
            results.append(
                BatchItemResponse(filename=file.filename or "unknown", result=result)
            )
            successful += 1
        except HTTPException as exc:
            results.append(
                BatchItemResponse(filename=file.filename or "unknown", error=exc.detail)
            )
        except Exception as exc:
            results.append(
                BatchItemResponse(filename=file.filename or "unknown", error=str(exc))
            )

    return BatchExtractionResponse(
        success=successful > 0,
        total_documents=len(files),
        successful=successful,
        failed=len(files) - successful,
        results=results,
    )


@app.get("/templates", response_model=TemplatesResponse)
async def list_templates() -> TemplatesResponse:
    """List available document extraction templates."""
    return TemplatesResponse(
        templates=[
            TemplateInfo(
                name="invoice",
                description="Standard invoice extraction",
                supported_fields=[
                    "date",
                    "vendor",
                    "total_amount",
                    "line_items",
                    "tax",
                ],
            ),
            TemplateInfo(
                name="receipt",
                description="Receipt/POS extraction",
                supported_fields=["date", "vendor", "total_amount", "items"],
            ),
            TemplateInfo(
                name="form",
                description="Generic form extraction",
                supported_fields=["all_text_fields"],
            ),
        ]
    )
