"""FastAPI application for Dorothy - Newspaper of Averages."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import structlog

from src.config import config
from src.storage import OpenSearchClient

logger = structlog.get_logger(__name__)

# Template and static directories
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def get_os_client() -> OpenSearchClient:
    """Get OpenSearch client with config."""
    auth_kwargs = {}
    if config.opensearch.username and config.opensearch.password:
        auth_kwargs["username"] = config.opensearch.username
        auth_kwargs["password"] = config.opensearch.password

    return OpenSearchClient(
        host=config.opensearch.host,
        port=config.opensearch.port,
        use_ssl=config.opensearch.use_ssl,
        **auth_kwargs,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Dorothy",
        description="Newspaper of Averages - Balanced news synthesis",
        version="0.1.0",
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Store templates in app state for route access
    app.state.templates = templates

    # Register routes
    from src.web.routes import register_routes
    register_routes(app)

    return app


# Create the app instance
app = create_app()
