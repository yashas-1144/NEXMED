"""
config.py — Central configuration for the BioMedAI Flask backend.

Reads environment variables from a .env file (if present) and exposes them
as module-level constants.  All configurable values live here; nothing else
in the backend should hard-code paths or port numbers.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow python-dotenv to load a .env file sitting beside this module.
# We import it optionally so the server still works if the package is absent.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    # Look for .env in the backend directory first, then the project root.
    _backend_dir = Path(__file__).resolve().parent
    _dotenv_path = _backend_dir / ".env"
    if not _dotenv_path.exists():
        _dotenv_path = _backend_dir.parent / ".env"
    load_dotenv(dotenv_path=_dotenv_path, override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on real environment variables

# ---------------------------------------------------------------------------
# Project root is the directory that contains this backend/ folder.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# MobileViT checkpoint directory.
#
# Priority:
#   1. MODEL_PATH environment variable (absolute or relative to project root)
#   2. Auto-detected default: <project_root>/mobilevit/mobilevit
# ---------------------------------------------------------------------------
_model_path_env: str = os.environ.get("MODEL_PATH", "")

if _model_path_env:
    _candidate = Path(_model_path_env)
    if not _candidate.is_absolute():
        _candidate = PROJECT_ROOT / _candidate
    MODEL_PATH: Path = _candidate.resolve()
else:
    MODEL_PATH = (PROJECT_ROOT / "mobilevit" / "mobilevit").resolve()

# ---------------------------------------------------------------------------
# Flask server settings
# ---------------------------------------------------------------------------
PORT: int = int(os.environ.get("PORT", 5000))
DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Upload settings
# ---------------------------------------------------------------------------
UPLOAD_FOLDER: Path = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}

# Maximum upload size: 16 MB
MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: Path = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
