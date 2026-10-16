# NEXMED System Architecture

This document describes the intended software pipeline and the conceptual robotics workflow for SIH26115, **Design and Develop a Smart Mobile Medical-Waste Collection and Segregation System**.

## 1. AI inference pipeline

NEXMED's described AI pipeline is sequential: a trained YOLO11s object detector finds an object, the corresponding bounding-box crop is extracted, and the crop is sent to a trained MobileViT classifier.

```text
Input image
    |
    v
Image validation / preprocessing
    |
    v
Trained YOLO11s detector
    |
    v
Object bounding box(es)
    |
    v
Crop each detected object from the image
    |
    v
Trained MobileViT classifier
    |
    v
Predicted class + class scores
    |
    v
Prototype waste-to-bin mapping
    |
    v
Web UI / 3D bin visualization
```

### Model responsibilities

- **YOLO11s:** Detects waste objects and returns their locations as bounding boxes.
- **Crop extraction:** Uses a detection box to isolate the detected item region before classification.
- **MobileViT:** Classifies the crop into one of the project's supported labels: `Cotton_Bandage`, `Glove`, `IV_Bottle`, `Mask`, `Pill_Strip`, or `Syringe`.
- **UI:** Presents the prediction and configured bin destination.

This description records the workflow supplied by the project team. For reproducibility, the repository should include the YOLO11s weights, crop/preprocessing implementation, model configuration, and an evaluation procedure. The currently published backend files include a MobileViT inference path; the complete sequential detector-to-classifier path and YOLO weights should be checked into the repository before claiming that a fresh clone can reproduce both stages.

## 2. Prototype waste-to-bin mapping

| MobileViT class | Configured destination |
|---|---|
| `Cotton_Bandage` | Yellow bin |
| `Glove` | Yellow bin |
| `Mask` | Yellow bin |
| `IV_Bottle` | Blue bin |
| `Pill_Strip` | Blue bin |
| `Syringe` | Red bin |

This is the mapping configured for the prototype. It is not a substitute for local biomedical-waste rules or institutional policy.

## 3. Conceptual robotics workflow

The proposed physical system is described as:

```text
Waste input
  -> Singulation / separate items
  -> Camera capture
  -> YOLO11s detection
  -> Crop and MobileViT classification
  -> Confidence / safety decision
  -> Mechanical orientation and routing
  -> Designated waste bin
  -> Disposal-event / compliance log
```

The concept includes a mobile unit, sensing, a feeder/receiver mechanism, servo or DC motor actuation, designated bins, safety checks, and digital logging. The presentation also discusses UV-C surface disinfection and fail-safe handling as design ideas. These are proposed elements, not verified capabilities of a physically assembled machine.

## 4. Implementation status

- **AI models:** The team reports training both YOLO11s and MobileViT and using them in the sequential application pipeline.
- **Web software:** A Flask backend and browser-based 3D interface are present in the repository.
- **3D assets:** GLB assets for system views and bins are present.
- **Robotics:** Conceptual sketch / Autodesk Fusion 360 model only; no physical robotics prototype was built.
- **CAD source:** The original Fusion 360 project file is not currently available.
- **Reproducibility note:** The published repository should be reviewed to ensure detector weights and the full sequential inference implementation are committed. Do not commit secrets, virtual environments, or unrelated local files.

## 5. Safety and validation

NEXMED is a prototype, not a certified medical or waste-management device. Low-confidence, ambiguous, or unsupported objects require a safe fallback policy in any real implementation. Predictions and bin suggestions must not override trained staff, institutional procedures, or applicable regulations. A real deployment would need representative clinical validation, mechanical and electrical safety testing, infection-control review, security hardening, and regulatory assessment.
