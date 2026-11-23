"""Optional development-only inference bridge for a trained five-class model.

Run: SKKU_MODEL_PATH=/path/to/best.pt uvicorn server:app --host 127.0.0.1 --port 8000
Do not expose to the public Internet without authentication, rate limiting and TLS.
"""
from __future__ import annotations
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from skku1398_ml import CLASS_NAMES, predictions_to_json

MODEL_PATH = os.getenv("SKKU_MODEL_PATH", "best.pt")
if not Path(MODEL_PATH).is_file():
    raise RuntimeError(f"Missing trained model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
app = FastAPI(title="SKKU 1398 Fitness equipment development bridge")
MAX_BYTES = 8 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names, "model_file": Path(MODEL_PATH).name}


@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = 0.25):
    if not (0.01 <= confidence <= 0.95):
        raise HTTPException(400, "confidence must be in [0.01, 0.95]")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "JPEG, PNG or WebP only")
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 8 MiB")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            if rgb.width * rgb.height > 24_000_000:
                raise HTTPException(413, "Image dimensions too large")
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode image")
    result = model.predict(source=np.asarray(rgb)[..., ::-1].copy(), conf=confidence, verbose=False)[0]
    response = predictions_to_json(result, threshold=confidence)
    response["model_scope"] = "public_proxy_requires_1398_photo_validation"
    return response
