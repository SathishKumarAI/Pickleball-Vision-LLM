"""Flask application factory for the Pickleball Vision LLM service.

`create_app()` is the entry point used by ``src.main``. It builds a minimal,
runnable Flask app exposing health/info routes. Vision and LLM blueprints are
registered here as they come online (see docs/PLAN.md, Phase 3).
"""

from flask import Flask, jsonify


def create_app() -> Flask:
    """Build and configure the Flask application.

    Returns:
        Configured :class:`flask.Flask` instance.
    """
    app = Flask(__name__)

    @app.get("/health")
    def health():
        """Liveness probe."""
        return jsonify(status="healthy")

    @app.get("/")
    def index():
        """Service metadata."""
        return jsonify(
            service="pickleball-vision-llm",
            version="0.1.0",
            endpoints=["/health", "/", "/analyze", "/analyze/video"],
        )

    from src.api.blueprints.analyze import bp as analyze_bp
    app.register_blueprint(analyze_bp)

    return app
