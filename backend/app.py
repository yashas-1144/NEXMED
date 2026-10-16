"""
app.py — Flask application factory and entry point for BioMedAI.

Responsibilities
----------------
- Creates and configures the Flask app
- Registers the API blueprint (routes.py)
- Serves the frontend HTML at GET /
- Serves all project-root static assets (GLB files, bins/) via Flask
- Loads the MobileViT model at startup (before first request)
- Configures Flask-CORS

Run (from project root)
-----------------------
    python -m backend.app
        OR
    python backend/app.py

Both work because the project root is added to sys.path below.
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `import backend.*` resolves
# correctly when this file is run directly from any working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, send_from_directory
from flask_cors import CORS

from backend.config import DEBUG, MAX_CONTENT_LENGTH, PORT, PROJECT_ROOT
from backend.routes import api_bp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Create and configure the Flask application."""

    # Flask's static_folder is set to the project root so that all assets
    # (GLB models, bins/*.glb, etc.) are reachable via their existing paths.
    app = Flask(
        __name__,
        static_folder=str(PROJECT_ROOT),
        static_url_path="",   # serve static files at root URL (e.g. /ExteriorView.glb)
    )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["PROPAGATE_EXCEPTIONS"] = False

    # ------------------------------------------------------------------
    # CORS — allow all origins (tighten in production as needed)
    # ------------------------------------------------------------------
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ------------------------------------------------------------------
    # Register blueprints
    # ------------------------------------------------------------------
    app.register_blueprint(api_bp)

    # ------------------------------------------------------------------
    # Frontend route — serve the HTML at /
    # ------------------------------------------------------------------
    @app.get("/")
    def index():
        """Serve the BioMedAI standalone frontend."""
        return send_from_directory(
            str(PROJECT_ROOT),
            "BioMedAI_Standalone_3D_BinTwin.html",
        )

    # ------------------------------------------------------------------
    # Generic error handlers — always return JSON
    # ------------------------------------------------------------------
    from flask import jsonify

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request."}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"error": "File too large. Maximum allowed size is 16 MB."}), 413

    @app.errorhandler(415)
    def unsupported_media_type(e):
        return jsonify({"error": "Unsupported file type."}), 415

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "An internal server error occurred."}), 500

    return app


# ---------------------------------------------------------------------------
# Model pre-loading — called before the app starts serving requests
# ---------------------------------------------------------------------------

def preload_model() -> None:
    """Load the MobileViT model into memory before Flask starts serving."""
    logger.info("=" * 60)
    logger.info("BioMedAI — MobileViT Medical Waste Classifier")
    logger.info("=" * 60)

    # Lazy import — only triggers torch/transformers when server actually starts
    from backend.model_loader import load_model

    try:
        bundle = load_model()
        logger.info(
            "Model ready.  Classes: %s",
            list(bundle.id2label.values()),
        )
        logger.info("Device: %s", bundle.device)
    except FileNotFoundError as exc:
        logger.error("CHECKPOINT NOT FOUND: %s", exc)
        logger.error(
            "Set the MODEL_PATH environment variable or place the checkpoint "
            "at: %s",
            PROJECT_ROOT / "mobilevit" / "mobilevit",
        )
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("MODEL LOAD FAILED: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preload_model()

    app = create_app()

    logger.info("Starting Flask server on http://0.0.0.0:%d", PORT)
    logger.info("Frontend: http://localhost:%d/", PORT)
    logger.info("Health:   http://localhost:%d/api/health", PORT)
    logger.info("Predict:  http://localhost:%d/api/predict  [POST]", PORT)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG,
        use_reloader=False,  # reloader would load the model twice
    )
