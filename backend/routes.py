"""
routes.py — Flask Blueprint defining the REST API for BioMedAI.

Endpoints
---------
GET  /api/health    — liveness probe
POST /api/predict   — multipart image upload → MobileViT prediction
"""

import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from backend.utils import (
    cleanup_upload,
    error_response,
    open_and_validate_image,
    save_upload,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@api_bp.get("/health")
def health():
    """Liveness probe — confirms the server is up and the model is loaded."""
    from backend.model_loader import is_model_loaded  # lazy — model loads at startup
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": is_model_loaded(),
        }
    ), 200


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------

@api_bp.post("/predict")
def predict_endpoint():
    """
    Accept a multipart image upload and return a MobileViT prediction.

    Request
    -------
    Content-Type: multipart/form-data
    Field name  : "image"

    Response (200)
    --------------
    {
      "prediction":    "Syringe",
      "confidence":    97.43,
      "probabilities": {
          "Cotton_Bandage": 0.12,
          "Glove":          0.31,
          "IV_Bottle":      0.74,
          "Mask":           0.95,
          "Pill_Strip":     0.45,
          "Syringe":        97.43
      }
    }

    Error responses always use {"error": "<message>"} with an appropriate
    HTTP status code.  Stack traces are never exposed.
    """
    # ------------------------------------------------------------------
    # 1. Validate request contains an image field
    # ------------------------------------------------------------------
    if "image" not in request.files:
        return error_response("No image field found in the request.", 400)

    file = request.files["image"]

    if file.filename == "" or file.filename is None:
        return error_response("No file was selected.", 400)

    # ------------------------------------------------------------------
    # 2. Save upload to disk
    # ------------------------------------------------------------------
    upload_path: Path | None = None
    try:
        upload_path = save_upload(file)
    except ValueError as exc:
        return error_response(str(exc), 415)
    except Exception as exc:
        logger.exception("Unexpected error saving upload.")
        return error_response("Failed to save the uploaded file.", 500)

    # ------------------------------------------------------------------
    # 3. Validate and open image
    # ------------------------------------------------------------------
    try:
        image = open_and_validate_image(upload_path)
    except ValueError as exc:
        cleanup_upload(upload_path)
        return error_response(str(exc), 422)
    except Exception as exc:
        cleanup_upload(upload_path)
        logger.exception("Unexpected error opening image.")
        return error_response("Could not open the image file.", 500)

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    try:
        from backend.predictor import predict  # lazy import — model must load first
        result = predict(image)
    except FileNotFoundError as exc:
        cleanup_upload(upload_path)
        logger.error("Checkpoint missing: %s", exc)
        return error_response(
            "Model checkpoint not found. Please verify the MODEL_PATH setting.", 503
        )
    except RuntimeError as exc:
        cleanup_upload(upload_path)
        logger.exception("Model inference failed.")
        return error_response("Model inference failed. Please try again.", 500)
    except Exception as exc:
        cleanup_upload(upload_path)
        logger.exception("Unexpected inference error.")
        return error_response("An unexpected error occurred during inference.", 500)
    finally:
        # Always clean up the temporary upload
        if upload_path:
            cleanup_upload(upload_path)

    # ------------------------------------------------------------------
    # 5. Return prediction
    # ------------------------------------------------------------------
    return jsonify(result), 200
