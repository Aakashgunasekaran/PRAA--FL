"""FastAPI backend: ``POST /api/predict`` classifies an uploaded image; ``/`` serves the frontend."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.inference_service import (CLASS_CODES, CLASS_NAMES, MAX_UPLOAD_BYTES, Classifier, ImageError,
                                   load_model)

log = logging.getLogger("praa.app")
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = os.environ.get("PRAA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_model(device)
    app.state.classifier = Classifier(loaded)
    log.info("model ready: mode=%s rule=%s device=%s checkpoint=%s", loaded.mode, loaded.rule, device,
             loaded.checkpoint)
    for note in loaded.notes:
        log.warning(note)
    yield


app = FastAPI(title="PRAA-FL image classifier", version="1.0", lifespan=lifespan)
app.add_middleware(  # lets a separately hosted frontend (e.g. a dev server on another port) call the API
    CORSMiddleware,
    allow_origins=os.environ.get("PRAA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    loaded = app.state.classifier.loaded
    return {
        "status": "ok",
        "mode": loaded.mode,
        "rule": loaded.rule,
        "description": loaded.description,
        "checkpoint": loaded.checkpoint,
        "device": loaded.device,
        "num_classes": len(CLASS_NAMES),
        "classes_without_prototype": loaded.missing_classes,
        "notes": loaded.notes,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.get("/api/classes")
def classes():
    return [{"class_id": c, "code": CLASS_CODES[c], "name": CLASS_NAMES[c]} for c in sorted(CLASS_NAMES)]


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)   # never buffer more than the limit + 1 byte
    try:
        return app.state.classifier.predict(data)
    except ImageError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    except FloatingPointError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
