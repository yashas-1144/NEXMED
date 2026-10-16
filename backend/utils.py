"""
utils.py — Shared utility helpers for the BioMedAI Flask backend.
"""

import logging
import os
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def save_upload(file_storage) -> Path:
    """
    Save a Werkzeug FileStorage object to the uploads directory.

    Returns
    -------
    Path
        Absolute path to the saved file.

    Raises
    ------
    ValueError
        If the filename is empty or has a disallowed extension.
    """
    filename = file_storage.filename or ""
    if not filename:
        raise ValueError("No filename provided.")

    if not allowed_file(filename):
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "unknown"
        raise ValueError(
            f"File type '.{ext}' is not supported. "
            f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # Generate a unique filename to prevent collisions
    ext = filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_FOLDER / unique_name

    file_storage.save(str(save_path))
    logger.debug("Upload saved: %s", save_path)
    return save_path


def cleanup_upload(path: Path) -> None:
    """Delete a temporary upload file, ignoring errors if it is already gone."""
    try:
        if path and path.exists():
            os.remove(path)
            logger.debug("Cleaned up upload: %s", path)
    except OSError as exc:
        logger.warning("Could not delete upload file %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def open_and_validate_image(path: Path) -> Image.Image:
    """
    Open and validate an image file.

    Returns
    -------
    PIL.Image.Image

    Raises
    ------
    ValueError
        If the file cannot be identified as a valid image.
    """
    try:
        img = Image.open(str(path))
        img.verify()          # checks file integrity without fully decoding
        img = Image.open(str(path))  # re-open after verify() closes the file
        return img
    except UnidentifiedImageError:
        raise ValueError("The uploaded file is not a recognisable image.")
    except Exception as exc:
        raise ValueError(f"The image file appears to be corrupted: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON error helper
# ---------------------------------------------------------------------------

def error_response(message: str, status: int = 400) -> tuple:
    """Build a clean JSON error response tuple for Flask."""
    from flask import jsonify  # local import to avoid circular at module level
    return jsonify({"error": message}), status
