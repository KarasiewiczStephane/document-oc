"""Application entry point for the Document OCR API server."""

import uvicorn

from src.api.app import app
from src.utils.config import load_config
from src.utils.logger import setup_logging


def main() -> None:
    """Start the FastAPI application server."""
    config = load_config()
    setup_logging(config.log_level)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
