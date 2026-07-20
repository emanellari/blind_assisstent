# VisionGuide AI

An accessible object-detection prototype designed to help visually impaired users understand the visual content of an uploaded image.

The application uses a custom **YOLOv5** checkpoint to detect objects, generates an annotated result, summarizes the detections in text, and can read that summary aloud using browser speech synthesis.

## Features

- Custom YOLOv5 object detection
- Secure image validation and upload-size limits
- Annotated detection output
- Human-readable object counts
- Browser-based spoken summary
- Temporary input-file cleanup
- Responsive Flask interface

## Project structure

```text
.
├── app.py
├── best.pt
├── templates/
│   └── index.html
├── static/
│   └── results/
├── uploads/
├── requirements.txt
├── Procfile
└── .env.example
```

## Run locally

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

> The first model load may require internet access because PyTorch Hub downloads the YOLOv5 repository. The custom weights are loaded from `best.pt`.

## Configuration

Optional environment variables:

```text
FLASK_SECRET_KEY=your-secret
MODEL_PATH=./best.pt
DETECTION_CONFIDENCE=0.25
PORT=5000
```

## Accessibility

The interface uses clear contrast, descriptive labels, keyboard-accessible controls, meaningful alternative text, and an optional spoken detection summary.

## Portfolio note

This is an academic prototype. Detection quality depends on the classes and examples used to train the custom checkpoint.
