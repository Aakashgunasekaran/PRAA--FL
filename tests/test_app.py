"""Tests for the web app: preprocessing, model/prototype loading, and the HTTP API (offline, no dataset)."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import transforms

from app import inference_service as svc
from app.server import app
from models.classifier import FLAMEModel

ROOT = Path(__file__).resolve().parent.parent


def png_bytes(size=(64, 48), mode="RGB", color=(180, 60, 120), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture()
def demo_client(monkeypatch):
    """App running in DEMO mode regardless of any real checkpoint present in the repo."""
    monkeypatch.delenv("PRAA_CHECKPOINT", raising=False)
    monkeypatch.delenv("PRAA_PROTOTYPES", raising=False)
    monkeypatch.setattr(svc, "CHECKPOINT_CANDIDATES", [])
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------- preprocessing
def test_preprocessing_matches_training_transform():
    rng = np.random.RandomState(0)
    small = Image.fromarray(rng.randint(0, 256, (28, 28, 3), dtype=np.uint8))
    reference = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])(small)
    assert torch.allclose(svc.to_tensor(small)[0], reference, atol=1e-6)


def test_non_square_image_is_centre_cropped_to_28():
    wide = Image.new("RGB", (200, 50), (255, 0, 0))
    wide.paste((0, 0, 255), (0, 0, 75, 50))                  # left third blue, rest red; the crop sees the centre only
    small = svc.to_model_image(wide)
    assert small.size == (28, 28)
    assert np.asarray(small)[14, 14].tolist() == [255, 0, 0]


def test_transparency_is_flattened_on_white():
    rgba = Image.open(io.BytesIO(png_bytes(mode="RGBA", color=(0, 0, 0, 0))))
    assert svc.decode_image(png_bytes(mode="RGBA", color=(0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
    assert rgba.mode == "RGBA"


def test_decode_rejects_bad_input():
    with pytest.raises(svc.ImageError) as empty:
        svc.decode_image(b"")
    assert empty.value.status == 400
    with pytest.raises(svc.ImageError) as junk:
        svc.decode_image(b"this is not an image")
    assert junk.value.status == 415
    with pytest.raises(svc.ImageError) as big:
        svc.decode_image(b"0" * (svc.MAX_UPLOAD_BYTES + 1))
    assert big.value.status == 413


# ---------------------------------------------------------------- API
def test_health_reports_demo_mode(demo_client):
    body = demo_client.get("/api/health").json()
    assert body["status"] == "ok" and body["mode"] == "demo" and body["num_classes"] == 9
    assert body["notes"], "demo mode must explain itself"


def test_classes_endpoint(demo_client):
    classes = demo_client.get("/api/classes").json()
    assert [c["code"] for c in classes] == ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


@pytest.mark.parametrize("fmt,mode,size", [("PNG", "RGB", (64, 48)), ("JPEG", "RGB", (300, 300)),
                                           ("PNG", "L", (28, 28)), ("PNG", "RGBA", (10, 40)),
                                           ("BMP", "RGB", (500, 200))])
def test_predict_returns_valid_distribution(demo_client, fmt, mode, size):
    color = 128 if mode == "L" else ((10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30))
    r = demo_client.post("/api/predict", files={"file": ("x", png_bytes(size, mode, color, fmt), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [c["probability"] for c in body["probabilities"]]
    assert len(probs) == 9 and abs(sum(probs) - 1) < 1e-4
    assert body["predicted"]["class_id"] == int(np.argmax(probs))
    assert body["top_k"][0]["class_id"] == body["predicted"]["class_id"]
    assert body["demo"] is True and body["model_input_png"].startswith("data:image/png;base64,")
    assert body["input"]["original_size"] == list(size)


def test_predict_is_deterministic(demo_client):
    data = png_bytes((64, 64))
    a = demo_client.post("/api/predict", files={"file": ("a.png", data, "image/png")}).json()
    b = demo_client.post("/api/predict", files={"file": ("a.png", data, "image/png")}).json()
    assert a["probabilities"] == b["probabilities"]


def test_predict_rejects_non_image_and_missing_file(demo_client):
    r = demo_client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and "image" in r.json()["detail"].lower()
    assert demo_client.post("/api/predict").status_code == 422
    r = demo_client.post("/api/predict", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code == 400


def test_frontend_is_served(demo_client):
    page = demo_client.get("/")
    assert page.status_code == 200 and "Tissue patch classifier" in page.text
    assert "Disease-related tissue signal" in page.text
    assert "cannot rule disease in or out" in page.text
    assert demo_client.get("/static/app.js").status_code == 200
    assert demo_client.get("/static/style.css").status_code == 200


# ---------------------------------------------------------------- model / prototype loading
def tiny_checkpoint(directory: Path, method="flame", with_prototypes=True) -> Path:
    config = {"method": method, "embedding_dim": 32, "num_classes": 9, "encoder": "small_cnn", "feature_dim": 64}
    torch.manual_seed(1)
    model = FLAMEModel(feature_dim=64, embedding_dim=32, num_classes=9, encoder="small_cnn")
    path = directory / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "history": [], "config": config}, path)
    if with_prototypes:
        torch.save({"prototypes": {c: torch.randn(32) for c in range(9)}}, directory / "global_prototypes.pt")
    return path


def test_trained_checkpoint_with_prototypes(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAA_CHECKPOINT", str(tiny_checkpoint(tmp_path)))
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["mode"] == "trained" and health["rule"] == "prototype"
        body = client.post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")}).json()
        assert body["demo"] is False and body["nearest_prototype_sq_distance"] >= 0


def test_fedavg_checkpoint_uses_classifier_head(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAA_CHECKPOINT", str(tiny_checkpoint(tmp_path, method="fedavg", with_prototypes=False)))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["rule"] == "classifier_head"
        body = client.post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")}).json()
        assert body["rule"] == "classifier_head" and body["nearest_prototype_sq_distance"] is None


def test_missing_prototypes_fail_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAA_CHECKPOINT", str(tiny_checkpoint(tmp_path, with_prototypes=False)))
    with pytest.raises(FileNotFoundError, match="export_global_prototypes"):
        svc.load_model()


def test_missing_checkpoint_env_fails_loudly(monkeypatch):
    monkeypatch.setenv("PRAA_CHECKPOINT", "/nonexistent/final_model.pt")
    with pytest.raises(FileNotFoundError):
        svc.load_model()


def test_preserved_phase4_prototypes_aggregate():
    directory = ROOT / "results/phase4/prototypes"
    protos = svc.prototypes_from_phase4(directory)
    assert sorted(protos) == list(range(9))
    assert all(v.shape == (128,) and torch.isfinite(v).all() for v in protos.values())
