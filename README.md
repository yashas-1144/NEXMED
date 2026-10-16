# NEXMED — Biomedical Waste Segregation & 3D Digital Twin

NEXMED is a prototype for biomedical-waste identification and segregation. It combines a fine-tuned **MobileViT-XXS image classifier**, a Flask API, and an interactive 3D digital-twin interface with bin and robot models.

The application lets a user submit an image, view the model's predicted class and confidence, and see a suggested waste-bin destination in the interface.

> **Prototype / safety notice:** This project is for demonstration and educational use. Model predictions are not a substitute for institutional biomedical-waste procedures, trained personnel, or applicable regulations. Do not use the classifier as the sole basis for real-world waste handling.

## Features

- **Image classification:** predicts one of six trained classes: cotton bandage, glove, IV bottle, mask, pill strip, or syringe.
- **Prediction details:** returns the top class, confidence percentage, and scores for all six classes.
- **3D digital twin:** interactive GLB models for the robot views and waste bins.
- **Flask backend:** serves the frontend and exposes health-check and prediction endpoints.
- **Temporary-upload cleanup:** removes uploaded image files after inference.

## Demo workflow

1. Start the Flask server.
2. Open the local web interface in a browser.
3. Go to **Live Monitoring** and find the **AI Image Classification** panel.
4. Select or drag-and-drop a supported image, then choose **CLASSIFY**.
5. Review the predicted class, confidence, class-score distribution, and suggested bin destination.

The frontend must be opened through the Flask server (http://localhost:5000 by default), not directly as a file URL, because the app loads API endpoints and 3D assets over HTTP.

## Architecture

~~~text
NEXMED/
├── BioMedAI_Standalone_3D_BinTwin.html  # Web interface and 3D viewer
├── ExteriorView.glb                    # 3D model asset
├── SectionView.glb                      # 3D model asset
├── ExplodedView.glb                     # 3D model asset
├── bins/
│   ├── blackbin.glb
│   ├── bluebin.glb
│   ├── redbin.glb
│   └── yellowbin.glb
├── mobilevit/
│   └── mobilevit/                       # Fine-tuned model checkpoint
│       ├── config.json
│       ├── preprocessor_config.json
│       └── model.safetensors
└── backend/
    ├── app.py                           # Flask application / server entry point
    ├── routes.py                        # HTTP API endpoints
    ├── model_loader.py                  # Loads and caches the model
    ├── predictor.py                     # Image preprocessing and inference
    ├── config.py                        # Environment and path configuration
    ├── utils.py                         # Image validation and upload cleanup
    ├── requirements.txt
    ├── .env.example
    ├── logs/
    └── uploads/
~~~

### How inference works

1. The client sends an image as multipart form data to POST /api/predict, using the field name **image**.
2. The backend checks the filename, extension, upload size, and whether the file can be decoded as an image.
3. The image is converted to RGB and processed with the checkpoint's Hugging Face image processor.
4. MobileViT runs inference on the available device (CUDA when available, otherwise CPU).
5. The backend applies softmax to the model logits and returns the predicted label, confidence, and class scores as percentages.
6. The temporary uploaded file is cleaned up after inference.

## Model details

| Property | Details |
|---|---|
| Architecture | MobileViT-XXS (apple/mobilevit-xx-small base architecture) |
| Task | Six-class image classification |
| Classes | Cotton_Bandage, Glove, IV_Bottle, Mask, Pill_Strip, Syringe |
| Checkpoint directory | mobilevit/mobilevit/ |
| Reported test accuracy | Approximately **84.3%** |

The accuracy above is the metric recorded for this project's test run, not an independently verified benchmark. Actual performance can vary with image quality, lighting, object appearance, and how closely new images resemble the training data. A high confidence score is not a guarantee that a prediction is correct.

### Prototype bin mapping

The interface associates predicted classes with suggested bin destinations:

| Predicted class | Suggested destination |
|---|---|
| Cotton_Bandage | Yellow bin |
| Glove | Yellow bin |
| Mask | Yellow bin |
| IV_Bottle | Blue bin |
| Pill_Strip | Blue bin |
| Syringe | Red bin |

These are the prototype's configured mappings. Always follow the applicable local biomedical-waste segregation rules and institutional procedures in real settings.

## Requirements

- Python **3.10 or newer**
- pip
- Internet access for installing Python dependencies
- A CPU is sufficient to try the application; a compatible NVIDIA GPU can accelerate inference when a matching PyTorch/CUDA setup is installed
- A modern browser with WebGL support for the 3D interface

## Installation

### 1. Clone the repository

~~~bash
git clone https://github.com/yashas-1144/NEXMED.git
cd NEXMED
~~~

### 2. Create and activate a virtual environment

Linux / macOS:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

Windows PowerShell:

~~~powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

### 3. Install dependencies

~~~bash
python -m pip install -r backend/requirements.txt
~~~

PyTorch installation options depend on your operating system and whether you want CPU or GPU inference. If you need CUDA, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a compatible command for your system, then verify that torch.cuda.is_available() returns True.

### 4. Configure environment (optional)

The default configuration uses the checkpoint at mobilevit/mobilevit/ and port 5000. To override settings, copy the example file:

~~~bash
cp backend/.env.example backend/.env
~~~

Edit backend/.env if needed. Supported variables:

| Variable | Default | Purpose |
|---|---|---|
| PORT | 5000 | Flask listening port |
| MODEL_PATH | mobilevit/mobilevit | Checkpoint path; relative paths resolve from the repository root |
| FLASK_DEBUG | false | Enables Flask debug mode when set to true |

Keep local secrets and machine-specific environment files out of version control.

## Run the application

Run the command from the repository root with the virtual environment activated:

~~~bash
python backend/app.py
~~~

On startup, the backend loads the model into memory. Once it starts successfully, open:

- **Web app:** [http://localhost:5000](http://localhost:5000)
- **Health endpoint:** [http://localhost:5000/api/health](http://localhost:5000/api/health)

To use a different port, set PORT in backend/.env before starting the server.

## API reference

### GET /api/health

Checks that the server is responding and reports whether the model is loaded.

~~~bash
curl http://localhost:5000/api/health
~~~

Example response:

~~~json
{
  "status": "healthy",
  "model_loaded": true
}
~~~

### POST /api/predict

Uploads an image and returns a prediction.

~~~bash
curl -X POST \
  -F "image=@/path/to/image.jpg" \
  http://localhost:5000/api/predict
~~~

Example response (illustrative values only):

~~~json
{
  "prediction": "Syringe",
  "confidence": 72.41,
  "probabilities": {
    "Cotton_Bandage": 2.13,
    "Glove": 3.47,
    "IV_Bottle": 5.28,
    "Mask": 8.16,
    "Pill_Strip": 8.55,
    "Syringe": 72.41
  }
}
~~~

All values in probabilities are percentages and should total approximately 100 (minor rounding differences are possible).

Common errors include:

| HTTP status | Meaning |
|---|---|
| 400 | Missing image field or no file selected |
| 413 | Upload exceeds the 16 MB limit |
| 415 | Unsupported file extension |
| 422 | Image file could not be decoded or is corrupted |
| 503 | Model checkpoint is unavailable |

Supported extensions configured by the backend: PNG, JPG, JPEG, GIF, BMP, WEBP, and TIFF. Maximum upload size: **16 MB**.

## Troubleshooting

| Problem | What to check |
|---|---|
| BACKEND OFFLINE in the interface | Confirm the server is running and open the app at the configured localhost port. |
| Checkpoint directory or weights not found | Confirm mobilevit/mobilevit/ contains config.json, preprocessor_config.json, and model.safetensors. |
| Model fails to load | Confirm the virtual environment is active and dependencies installed successfully. Review the server terminal output. |
| CUDA is unavailable | The backend uses CPU when CUDA is not available; install a compatible PyTorch build if GPU inference is needed. |
| Port already in use | Change PORT in backend/.env, then restart the server. |
| 3D assets do not load | Access the interface through Flask over HTTP rather than opening the HTML file directly. |

## Limitations and responsible use

- This is a prototype, not a certified medical or waste-management device.
- The classifier is limited to its six trained classes; unfamiliar objects may still receive a confident but incorrect label.
- Test accuracy alone does not establish safety, robustness, or performance across hospitals and real-world environments.
- Bin recommendations must be checked against current local regulations and institutional policies.
- The Flask development server and permissive CORS configuration are intended for local prototyping, not direct production deployment. Production use would require deployment hardening, access controls, monitoring, and additional validation.

## License

No license is specified in this repository at present. Unless a license is added, others should not assume they have permission to reuse, distribute, or modify the project.
