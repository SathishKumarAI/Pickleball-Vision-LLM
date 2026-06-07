"""Flask application factory for the Pickleball Vision LLM service.

`create_app()` is the entry point used by ``src.main``. It builds a minimal,
runnable Flask app exposing health/info routes. Vision and LLM blueprints are
registered here as they come online (see docs/PLAN.md, Phase 3).
"""

import os

from flask import Flask, jsonify


def create_app() -> Flask:
    """Build and configure the Flask application.

    Returns:
        Configured :class:`flask.Flask` instance.
    """
    app = Flask(__name__)
    # Signing key for auth bearer tokens — override via APP_SECRET in prod.
    app.config["SECRET_KEY"] = os.getenv("APP_SECRET", "dev-insecure-change-me")

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
            endpoints=["/health", "/", "/analyze", "/analyze/video",
                       "/auth/register", "/auth/login", "/auth/me",
                       "/jobs/video", "/jobs/<id>", "/jobs/<id>/result", "/jobs/<id>/video"],
        )

    from src.api.blueprints.analyze import bp as analyze_bp
    from src.api.blueprints.auth import bp as auth_bp
    from src.api.blueprints.jobs import bp as jobs_bp
    app.register_blueprint(analyze_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(jobs_bp)

    return app
