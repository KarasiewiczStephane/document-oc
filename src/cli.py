"""Command-line interface for batch document processing and CSV export.

Provides subcommands for processing folders of documents with extraction
and exporting structured results to CSV.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from src.extraction.hybrid import HybridExtractor
from src.ocr.document_processor import DocumentProcessor
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging
from src.validation.rules_engine import RulesEngine

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.pdf")
_META_COLUMNS = [
    "filename",
    "status",
    "page_count",
    "processing_time_s",
    "overall_confidence",
    "validation_passed",
    "error",
]


def _find_documents(input_dir: Path) -> list[Path]:
    """Find all supported document files in a directory.

    Args:
        input_dir: Directory to scan for documents.

    Returns:
        Sorted list of document file paths.
    """
    files: list[Path] = []
    for ext in _SUPPORTED_EXTENSIONS:
        files.extend(input_dir.glob(ext))
        files.extend(input_dir.glob(ext.upper()))
    return sorted(set(files))


def process_folder(
    input_dir: Path,
    output_csv: Path,
    document_type: str = "invoice",
    use_ml: bool = True,
    verbose: bool = False,
) -> dict[str, int]:
    """Process all documents in a folder and export results to CSV.

    Args:
        input_dir: Directory containing document files.
        output_csv: Path for the output CSV file.
        document_type: Document type for validation rules.
        use_ml: Whether to use ML-based extraction.
        verbose: Whether to print per-file progress.

    Returns:
        Summary dict with total, successful, and failed counts.
    """
    config = load_config()
    processor = DocumentProcessor(config)
    extractor = HybridExtractor(config.extraction)
    validator = RulesEngine()

    files = _find_documents(input_dir)
    if not files:
        logger.warning("No documents found in %s", input_dir)
        return {"total": 0, "successful": 0, "failed": 0}

    logger.info("Found %d documents to process", len(files))

    results: list[dict[str, object]] = []
    successful = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        if verbose:
            print(f"Processing [{i}/{len(files)}]: {file_path.name}")

        start_time = time.time()
        try:
            result = _process_single_file(
                file_path, processor, extractor, validator, document_type, use_ml
            )
            result["processing_time_s"] = round(time.time() - start_time, 2)
            results.append(result)
            successful += 1
        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path.name, exc)
            results.append(
                {
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            failed += 1

    _write_csv(results, output_csv)
    logger.info("Results written to %s", output_csv)

    summary = {"total": len(files), "successful": successful, "failed": failed}
    _print_summary(summary, output_csv)
    return summary


def _process_single_file(
    file_path: Path,
    processor: DocumentProcessor,
    extractor: HybridExtractor,
    validator: RulesEngine,
    document_type: str,
    use_ml: bool,
) -> dict[str, object]:
    """Process a single document file through the full pipeline.

    Args:
        file_path: Path to the document file.
        processor: Document processor instance.
        extractor: Hybrid extractor instance.
        validator: Rules engine instance.
        document_type: Document type for validation.
        use_ml: Whether to use ML extraction.

    Returns:
        Dictionary of extraction results.
    """
    doc_result = processor.process(file_path, file_path.name)

    all_fields: dict[str, str] = {}
    all_confidences: dict[str, float] = {}

    for page in doc_result.pages:
        extraction = extractor.extract(
            image=None,
            ocr_text=page.ocr_result.text,
            ocr_words=page.ocr_result.words if page.ocr_result.words else None,
            ocr_confidence=page.ocr_result.confidence,
            use_ml=use_ml,
        )
        for field in extraction.fields:
            all_fields[field.field_name] = field.value
            all_confidences[field.field_name] = field.confidence

    validation = validator.validate(all_fields, document_type, all_confidences)

    overall_confidence = (
        round(sum(all_confidences.values()) / len(all_confidences), 3)
        if all_confidences
        else 0.0
    )

    result: dict[str, object] = {
        "filename": file_path.name,
        "status": "success",
        "page_count": doc_result.page_count,
        "overall_confidence": overall_confidence,
        "validation_passed": validation.all_valid,
        "error": None,
    }
    result.update(all_fields)
    return result


def _write_csv(results: list[dict[str, object]], output_path: Path) -> None:
    """Write extraction results to a CSV file.

    Args:
        results: List of result dictionaries.
        output_path: Path for the output CSV file.
    """
    if not results:
        return

    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.keys())

    field_columns = sorted(all_keys - set(_META_COLUMNS))
    columns = [c for c in _META_COLUMNS if c in all_keys] + field_columns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _print_summary(summary: dict[str, int], output_csv: Path) -> None:
    """Print batch processing summary to stdout.

    Args:
        summary: Counts of total, successful, and failed documents.
        output_csv: Path to the output CSV.
    """
    print(f"\n{'=' * 50}")
    print("Batch Processing Complete")
    print(f"{'=' * 50}")
    print(f"Total:      {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed:     {summary['failed']}")
    print(f"Output:     {output_csv}")


def extract_single(
    file_path: Path,
    document_type: str = "invoice",
    use_ml: bool = True,
) -> dict[str, object]:
    """Process a single document and return structured results.

    Args:
        file_path: Path to the document file.
        document_type: Document type for validation.
        use_ml: Whether to use ML extraction.

    Returns:
        Dictionary with filename, fields, and raw_text.
    """
    config = load_config()
    processor = DocumentProcessor(config)
    extractor = HybridExtractor(config.extraction)

    doc_result = processor.process(file_path, file_path.name)

    fields: dict[str, dict[str, object]] = {}
    for page in doc_result.pages:
        extraction = extractor.extract(
            image=None,
            ocr_text=page.ocr_result.text,
            ocr_words=page.ocr_result.words if page.ocr_result.words else None,
            ocr_confidence=page.ocr_result.confidence,
            use_ml=use_ml,
        )
        for f in extraction.fields:
            fields[f.field_name] = {
                "value": f.value,
                "confidence": f.confidence,
            }

    return {
        "filename": file_path.name,
        "fields": fields,
        "raw_text": doc_result.combined_text,
    }


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate command.

    Args:
        argv: Command-line arguments (defaults to sys.argv).
    """
    parser = argparse.ArgumentParser(
        description="Document OCR Batch Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    batch_parser = subparsers.add_parser("batch", help="Process a folder of documents")
    batch_parser.add_argument(
        "input_dir", type=Path, help="Input directory with documents"
    )
    batch_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV file (default: results.csv)",
    )
    batch_parser.add_argument(
        "-t",
        "--type",
        choices=["invoice", "receipt", "form"],
        default="invoice",
        dest="doc_type",
        help="Document type (default: invoice)",
    )
    batch_parser.add_argument(
        "--no-ml", action="store_true", help="Disable ML extraction"
    )
    batch_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    single_parser = subparsers.add_parser("extract", help="Process a single document")
    single_parser.add_argument("file", type=Path, help="Document file to process")
    single_parser.add_argument(
        "-t",
        "--type",
        choices=["invoice", "receipt", "form"],
        default="invoice",
        dest="doc_type",
        help="Document type (default: invoice)",
    )
    single_parser.add_argument(
        "--no-ml", action="store_true", help="Disable ML extraction"
    )
    single_parser.add_argument("-o", "--output", type=Path, help="Output JSON file")

    args = parser.parse_args(argv)

    setup_logging()

    if args.command == "batch":
        if not args.input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        process_folder(
            args.input_dir,
            args.output,
            args.doc_type,
            not args.no_ml,
            args.verbose,
        )
    elif args.command == "extract":
        if not args.file.exists():
            print(f"Error: {args.file} does not exist", file=sys.stderr)
            sys.exit(1)
        result = extract_single(args.file, args.doc_type, not args.no_ml)
        output_str = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_str)
            print(f"Output written to {args.output}")
        else:
            print(output_str)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
