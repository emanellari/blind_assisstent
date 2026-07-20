from __future__ import annotations

import os
import uuid
from pathlib import Path

import torch
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "best.pt"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = 10

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "development-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@torch.inference_mode()
def load_model():
    """Load the custom YOLOv5 model once when the process starts."""
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(MODEL_PATH),
        force_reload=False,
    )
    model.conf = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    return model


try:
    model = load_model()
    model_error = None
except Exception as exc:  # Keep the UI available with a clear setup message.
    model = None
    model_error = str(exc)


def summarize_detections(results) -> tuple[list[dict], str]:
    detections = results.pandas().xyxy[0]
    if detections.empty:
        return [], "No objects were detected in the uploaded image."

    grouped = (
        detections.groupby("name")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    items = [
        {"label": str(label), "count": int(count)}
        for label, count in grouped.items()
    ]

    spoken_parts = [
        f"{count} {label}{'' if count == 1 else 's'}"
        for label, count in grouped.items()
    ]
    summary = "Detected " + ", ".join(spoken_parts) + "."
    return items, summary


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", model_error=model_error)

    if model is None:
        flash("The detection model could not be loaded. Check the setup instructions.", "error")
        return redirect(url_for("index"))

    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        flash("Choose an image before starting detection.", "error")
        return redirect(url_for("index"))

    if not allowed_file(uploaded.filename):
        flash("Supported formats: PNG, JPG, JPEG, and WEBP.", "error")
        return redirect(url_for("index"))

    suffix = Path(secure_filename(uploaded.filename)).suffix.lower() or ".jpg"
    request_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{request_id}{suffix}"
    result_name = f"{request_id}_detected.jpg"
    result_path = RESULT_DIR / result_name

    try:
        uploaded.save(input_path)
        with Image.open(input_path) as image:
            image.verify()

        results = model(str(input_path))
        rendered = results.render()[0]
        Image.fromarray(rendered).save(result_path, format="JPEG", quality=92)
        detections, summary = summarize_detections(results)

        return render_template(
            "index.html",
            model_error=model_error,
            result_img=url_for("static", filename=f"results/{result_name}"),
            detections=detections,
            summary=summary,
        )
    except (UnidentifiedImageError, OSError):
        flash("The uploaded file is not a valid image.", "error")
        return redirect(url_for("index"))
    except Exception as exc:
        app.logger.exception("Detection failed")
        flash(f"Detection failed: {exc}", "error")
        return redirect(url_for("index"))
    finally:
        input_path.unlink(missing_ok=True)


@app.errorhandler(413)
def file_too_large(_error):
    flash(f"The image is too large. Maximum size: {MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
