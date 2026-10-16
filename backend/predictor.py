"""
predictor.py — Exact inference pipeline matching train_mobilevit.py.

Pipeline (mirrors the training script's MobileViTDataset.__getitem__):
  1. Accept a PIL Image
  2. Convert to RGB
  3. processor(images=image, return_tensors="pt")   — AutoImageProcessor
  4. pixel_values = inputs["pixel_values"]
  5. pixel_values.to(device)
  6. with torch.no_grad(): outputs = model(pixel_values=pixel_values)
  7. logits = outputs.logits
  8. probs  = softmax(logits, dim=1)
  9. pred_idx = argmax(probs)
  10. prediction = id2label[pred_idx]
  11. Return structured prediction dict
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def predict(image: Image.Image) -> dict[str, Any]:
    """
    Run MobileViT inference on a single PIL Image.

    Parameters
    ----------
    image:
        PIL Image object (any mode — will be converted to RGB internally).

    Returns
    -------
    dict with keys:
        prediction   (str)   — class name of the top prediction
        confidence   (float) — confidence percentage (0–100), 2 decimal places
        probabilities (dict) — {class_name: percentage} for all classes
    """
    from backend.model_loader import load_model  # lazy import

    bundle = load_model()
    model = bundle.model
    processor = bundle.processor
    id2label = bundle.id2label
    device = bundle.device

    # ------------------------------------------------------------------
    # Step 1 & 2 — Ensure RGB
    # ------------------------------------------------------------------
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ------------------------------------------------------------------
    # Step 3 & 4 — Preprocess with AutoImageProcessor
    # Exactly as in training: processor(images=image, return_tensors="pt")
    # ------------------------------------------------------------------
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # shape: (1, 3, H, W)

    # ------------------------------------------------------------------
    # Step 5 — Move to device
    # ------------------------------------------------------------------
    pixel_values = pixel_values.to(device)

    # ------------------------------------------------------------------
    # Step 6 & 7 — Inference
    # ------------------------------------------------------------------
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # shape: (1, num_classes)

    # ------------------------------------------------------------------
    # Step 8 — Softmax probabilities
    # ------------------------------------------------------------------
    probs = F.softmax(logits, dim=1)  # shape: (1, num_classes)
    probs_list = probs[0].cpu().tolist()  # list of floats, one per class

    # ------------------------------------------------------------------
    # Step 9 & 10 — Top prediction
    # ------------------------------------------------------------------
    pred_idx = int(torch.argmax(probs[0]).item())
    prediction = id2label[pred_idx]
    confidence = round(probs_list[pred_idx] * 100, 2)

    # ------------------------------------------------------------------
    # Step 11 — Build full probabilities dict (all classes)
    # ------------------------------------------------------------------
    probabilities: dict[str, float] = {
        id2label[i]: round(prob * 100, 2)
        for i, prob in enumerate(probs_list)
    }

    logger.info(
        "Prediction: %s (%.2f%%) | Device: %s",
        prediction,
        confidence,
        device,
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }
