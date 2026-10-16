# BioMedAI — Smart Biomedical Waste Control Center

AI-powered biomedical waste classification system combining a fine-tuned **MobileViT-XXS** image classifier with a 3D digital-twin dashboard.

> **Linux only.** This guide covers Linux exclusively.

---

## Architecture

```
BioMedAI_BinTwin_Package/
├── BioMedAI_Standalone_3D_BinTwin.html   # Frontend (served by Flask at /)
├── mobilevit/
│   └── mobilevit/                        # MobileViT checkpoint (DO NOT MODIFY)
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
├── backend/
│   ├── app.py                            # Flask entry point
│   ├── routes.py                         # API endpoints
│   ├── model_loader.py                   # Checkpoint loader (singleton)
│   ├── predictor.py                      # Inference pipeline
│   ├── config.py                         # Configuration
│   ├── utils.py                          # Helpers
│   ├── requirements.txt
│   ├── .env.example
│   ├── uploads/                          # Temporary upload storage
│   └── logs/
├── bins/                                 # 3D bin GLB models
├── ExteriorView.glb
├── SectionView.glb
├── ExplodedView.glb
└── train_mobilevit.py                    # Training reference (do not run)
```

## Model

| Property | Value |
|---|---|
| Architecture | MobileViT-XXS (`apple/mobilevit-xx-small`) |
| Task | Image Classification |
| Classes | Cotton_Bandage, Glove, IV_Bottle, Mask, Pill_Strip, Syringe |
| Input | 256×256 RGB, normalized (0.5, 0.5, 0.5) |
| Checkpoint | `mobilevit/mobilevit/` |
| Test accuracy | 84.3% |

---

## Prerequisites

- Python **3.10** or newer
- pip
- (Optional) CUDA-capable GPU with CUDA 11.8 or 12.1

Verify Python version:
```bash
python3 --version
```

---

## Setup

### 1. Clone / extract the project

```bash
# If cloning from git:
git clone <repo-url>
cd BioMedAI_BinTwin_Package
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

**CPU (default):**
```bash
pip install -r backend/requirements.txt
```

**GPU — CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r backend/requirements.txt
```

**GPU — CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt
```

### 4. Configure environment (optional)

```bash
cp backend/.env.example backend/.env
# Edit backend/.env if you need a different port or MODEL_PATH
```

The server auto-detects the checkpoint at `mobilevit/mobilevit/` — no configuration needed unless you move it.

---

## Running the Server

```bash
# Always run from the project root directory
python backend/app.py
```

Expected output:
```
INFO  BioMedAI — MobileViT Medical Waste Classifier
INFO  Model ready.  Classes: ['Cotton_Bandage', 'Glove', 'IV_Bottle', 'Mask', 'Pill_Strip', 'Syringe']
INFO  Device: cpu
INFO  Starting Flask server on http://0.0.0.0:5000
INFO  Frontend: http://localhost:5000/
INFO  Health:   http://localhost:5000/api/health
INFO  Predict:  http://localhost:5000/api/predict  [POST]
```

Open **http://localhost:5000** in your browser.

---

## API Reference

### `GET /api/health`

Liveness probe.

```bash
curl http://localhost:5000/api/health
```

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### `POST /api/predict`

Classify a biomedical waste image.

```bash
curl -X POST \
  -F "image=@/path/to/syringe.jpg" \
  http://localhost:5000/api/predict
```

**Success response (200):**
```json
{
  "prediction": "Syringe",
  "confidence": 97.43,
  "probabilities": {
    "Cotton_Bandage": 0.12,
    "Glove": 0.31,
    "IV_Bottle": 0.74,
    "Mask": 0.95,
    "Pill_Strip": 0.45,
    "Syringe": 97.43
  }
}
```

**Error response:**
```json
{
  "error": "No image field found in the request."
}
```

---

## Using the Frontend

1. Start the Flask server (above).
2. Navigate to **http://localhost:5000**.
3. Click **"Live Monitoring"** in the sidebar.
4. Scroll to the **AI Image Classification** panel.
5. Click **SELECT IMAGE** or drag-and-drop a medical waste image.
6. Click **CLASSIFY**.
7. Results appear in:
   - The **AI Detection Results** panel (right column): detected object, category, confidence, destination bin.
   - The **CLASS PROBABILITIES** bars (full distribution).
   - The **Detection Timeline** (timestamped entry appended).

---

## Supported Image Formats

`PNG` · `JPG` · `JPEG` · `WEBP` · `BMP` · `GIF` · `TIFF`

Maximum file size: **16 MB**

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `BACKEND OFFLINE` badge in UI | Flask server is not running. Run `python backend/app.py`. |
| `FileNotFoundError: Checkpoint directory does not exist` | Checkpoint is missing or MODEL_PATH is wrong. Verify `mobilevit/mobilevit/` exists. |
| CUDA out-of-memory | CPU fallback is automatic — no action needed. |
| Port already in use | Set `PORT=5001` in `backend/.env`. |
| 3D models not loading | Must access via `http://localhost:5000`, not `file://`. |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Flask server port |
| `MODEL_PATH` | `mobilevit/mobilevit` | Path to checkpoint directory |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

---

## Notes

- **Never open the HTML file directly** (`file://`) — 3D models and API calls require HTTP.
- The model is loaded **once at startup** and kept in memory for all subsequent requests.
- Uploaded images are deleted immediately after inference.
- The training script (`train_mobilevit.py`) is included for reference only. **Do not run it.**