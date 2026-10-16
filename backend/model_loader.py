"""
model_loader.py — Loads the MobileViT checkpoint exactly once at startup.

Loading strategy follows the training script (train_mobilevit.py):
  - AutoImageProcessor.from_pretrained(checkpoint_dir)
  - MobileViTForImageClassification.from_pretrained(checkpoint_dir)
  - CUDA if available, else CPU
  - model.eval()
  - id2label and label2id read directly from model.config

The module exposes a single public function `load_model()` which returns a
named tuple.  Repeated calls return the cached singleton.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import torch
from transformers import AutoImageProcessor, MobileViTForImageClassification

logger = logging.getLogger(__name__)


class ModelBundle(NamedTuple):
    model: MobileViTForImageClassification
    processor: AutoImageProcessor
    id2label: dict[int, str]
    label2id: dict[str, int]
    device: torch.device


# Module-level singleton — populated on first call to load_model()
_bundle: ModelBundle | None = None


def load_model(checkpoint_dir: str | Path | None = None) -> ModelBundle:
    """
    Load (or return the cached) MobileViT model + processor.

    Parameters
    ----------
    checkpoint_dir:
        Path to the saved checkpoint directory.  If None, the path is read
        from config.MODEL_PATH.

    Returns
    -------
    ModelBundle
        Named tuple with (model, processor, id2label, label2id, device).

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory does not exist or is missing required files.
    RuntimeError
        If the model or processor cannot be loaded.
    """
    global _bundle

    if _bundle is not None:
        return _bundle

    # ------------------------------------------------------------------
    # Resolve checkpoint path
    # ------------------------------------------------------------------
    if checkpoint_dir is None:
        from backend.config import MODEL_PATH  # lazy import to avoid circular
        checkpoint_dir = MODEL_PATH

    checkpoint_dir = Path(checkpoint_dir).resolve()

    logger.info("Loading MobileViT checkpoint from: %s", checkpoint_dir)

    # Validate required files
    _assert_checkpoint_valid(checkpoint_dir)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # Load processor (AutoImageProcessor reads preprocessor_config.json)
    # ------------------------------------------------------------------
    try:
        processor = AutoImageProcessor.from_pretrained(str(checkpoint_dir))
        logger.info("AutoImageProcessor loaded successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load AutoImageProcessor from {checkpoint_dir}: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    try:
        model = MobileViTForImageClassification.from_pretrained(str(checkpoint_dir))
        model = model.to(device)
        model.eval()
        logger.info("MobileViTForImageClassification loaded successfully.")
        logger.info("Number of labels: %d", model.config.num_labels)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MobileViTForImageClassification from {checkpoint_dir}: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Read id2label / label2id directly from model config
    # ------------------------------------------------------------------
    # model.config.id2label keys are strings from JSON — convert to int
    id2label: dict[int, str] = {
        int(k): v for k, v in model.config.id2label.items()
    }
    label2id: dict[str, int] = {
        v: int(k) for k, v in model.config.id2label.items()
    }

    logger.info("Classes: %s", list(id2label.values()))

    _bundle = ModelBundle(
        model=model,
        processor=processor,
        id2label=id2label,
        label2id=label2id,
        device=device,
    )
    return _bundle


def is_model_loaded() -> bool:
    """Return True if the model singleton has been initialised."""
    return _bundle is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_checkpoint_valid(checkpoint_dir: Path) -> None:
    """Raise FileNotFoundError if required checkpoint files are missing."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    required = ["config.json", "preprocessor_config.json"]
    # Accept either safetensors or pytorch bin
    weight_files = ["model.safetensors", "pytorch_model.bin"]

    for name in required:
        if not (checkpoint_dir / name).exists():
            raise FileNotFoundError(
                f"Missing required checkpoint file: {checkpoint_dir / name}"
            )

    has_weights = any((checkpoint_dir / w).exists() for w in weight_files)
    if not has_weights:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            f"Expected one of: {weight_files}"
        )
