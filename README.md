# Document Intelligence OCR System

> Extract structured data from invoices, receipts, and forms using Tesseract OCR and transformer-based ML models.

## Features

- **Image Preprocessing**: Deskew, denoise, binarize, and CLAHE contrast enhancement
- **OCR Engine**: Tesseract with layout analysis and multi-language support
- **ML Extraction**: LayoutLM transformer for intelligent field extraction
- **Rule-Based Extraction**: Regex patterns for dates, amounts, emails, phones, and invoice numbers
- **Hybrid Pipeline**: Merges ML and rule-based results with weighted confidence scoring
- **Validation Engine**: Configurable YAML rules with cross-field checks and confidence adjustments
- **REST API**: FastAPI endpoints for single and batch document processing
- **Dashboard**: Streamlit UI for OCR result visualization with bounding box overlays
- **CLI**: Batch folder processing with CSV export and accuracy benchmarking
- **Docker**: Production-ready container with Tesseract and all dependencies

## Quick Start

### 1. Install

```bash
# Prerequisites: Python 3.11+, Tesseract OCR
sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils  # Ubuntu
brew install tesseract poppler                                    # macOS

# Install Python dependencies
make install
```

### 2. Launch the Dashboard

The dashboard ships with built-in demo invoice data -- no external data or preparation is needed.

```bash
make dashboard
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Toggle "Use demo invoice data" to explore OCR extraction results, confidence scores, and bounding box visualization. You can also upload your own document images (PNG, JPG, TIFF).

### 3. Run the API Server

```bash
make run
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Using Docker (Alternative)

```bash
docker compose up -d
curl -X POST -F "file=@invoice.pdf" http://localhost:8000/extract
```

## API Endpoints

| Endpoint         | Method | Description                        |
|------------------|--------|------------------------------------|
| `/extract`       | POST   | Extract fields from single document|
| `/extract/batch` | POST   | Process multiple documents         |
| `/templates`     | GET    | List supported document types      |
| `/health`        | GET    | Health check with system info      |

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

### Example Request

```bash
curl -X POST "http://localhost:8000/extract" \
  -H "accept: application/json" \
  -F "file=@invoice.pdf" \
  -F "document_type=invoice"
```

### Example Response

```json
{
  "success": true,
  "document_id": "a1b2c3d4-...",
  "document_type": "invoice",
  "fields": [
    {"field_name": "vendor", "value": "ACME Corp", "confidence": 0.95, "source": "rule"},
    {"field_name": "date", "value": "2024-01-15", "confidence": 0.92, "source": "rule"},
    {"field_name": "total_amount", "value": "1234.56", "confidence": 0.98, "source": "hybrid"}
  ],
  "overall_confidence": 0.95,
  "validation": [
    {"field_name": "date", "is_valid": true, "message": "Valid date format", "rule_name": "date_format"}
  ],
  "processing_time_ms": 1250,
  "page_count": 1
}
```

## CLI Usage

```bash
# Extract from a single document
python -m src.cli extract invoice.png -t invoice -o result.json

# Batch process a folder
python -m src.cli batch ./documents -o results.csv -t invoice -v

# Run accuracy benchmark against labeled data
python -m src.cli benchmark ./documents ground_truth.json -o report.txt

# Disable ML extraction (rule-based only)
python -m src.cli batch ./documents --no-ml -o results.csv
```

## Configuration

Edit `configs/config.yaml`:

```yaml
preprocessing:
  deskew_enabled: true
  denoise_enabled: true
  denoise_method: bilateral
  binarize_enabled: true

ocr:
  default_lang: eng
  psm: 3
  pdf_dpi: 300

extraction:
  use_ml: true
  model_name: microsoft/layoutlm-base-uncased
  ml_weight: 0.6
  rule_weight: 0.4
  confidence_threshold: 0.5
```

Validation rules in `configs/validation_rules.yaml`:

```yaml
invoice:
  date:
    - type: date_format
    - type: required
  total_amount:
    - type: positive_amount
    - type: required
  vendor:
    - type: required
```

## Project Structure

```
document-oc/
├── src/
│   ├── preprocessing/    # Deskew, denoise, binarize, CLAHE
│   ├── ocr/              # Tesseract engine, layout analysis, PDF handling
│   ├── extraction/       # Rule-based, ML (LayoutLM), and hybrid pipeline
│   ├── validation/       # Configurable rules engine with cross-field checks
│   ├── benchmark/        # Accuracy evaluation with precision/recall/F1
│   ├── api/              # FastAPI REST endpoints and schemas
│   ├── dashboard/        # Streamlit UI with OCR result visualization
│   ├── cli.py            # Batch processing CLI with CSV export
│   └── utils/            # Configuration and structured logging
├── configs/              # YAML configuration files
├── tests/                # pytest test suite
├── data/                 # Sample documents (place your own here)
├── .github/workflows/    # CI pipeline (lint, test, Docker build)
├── Dockerfile            # Multi-stage build with Tesseract
├── docker-compose.yml    # Production deployment config
├── Makefile              # Development commands
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Ruff, pytest, and coverage config
```

## Development

```bash
make install      # Install dependencies
make test         # Run tests with coverage
make lint         # Run ruff linter and formatter
make run          # Start API server on port 8000
make dashboard    # Start Streamlit dashboard on port 8501
make docker       # Build Docker image
make docker-up    # Start with docker compose
make docker-down  # Stop containers
make clean        # Remove __pycache__
```

## License

MIT
