"""Model loading and single-image inference for the web app.

Inference rule (identical to ``evaluation/inference.py`` and the FLAME paper, Eq. 6/10):

    P(y = c | x) = softmax_c( -|| g(f(x)) - P_c ||^2 )

with GLOBAL prototypes P_c aggregated from the clients' prototypes (``federated.praa.aggregate_prototypes``).
A FedAvg checkpoint (``config["method"] == "fedavg"``) uses its supervised classifier head instead.

Where the model and prototypes come from (first match wins)
-----------------------------------------------------------
1. ``PRAA_CHECKPOINT`` env var (path to a ``final_model.pt``)
2. ``results/runs/praa/final_model.pt``, ``results/runs/flame/final_model.pt``,
   ``results/runs/fedavg/final_model.pt``
3. ``results/phase3/final_model.pt``      (the preserved original FLAME model, 200 rounds)
4. nothing found -> DEMO mode (untrained random model; predictions are NOT meaningful)

Prototypes for a checkpoint:
* ``PRAA_PROTOTYPES`` env var, or ``global_prototypes.pt`` next to the checkpoint
  (create it with ``python scripts/export_global_prototypes.py --run-dir <run dir>``);
* for the preserved Phase 3 model only: aggregated from ``results/phase4/prototypes/client_*.pt``.
"""

import io
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from evaluation.inference import prototype_logits, softmax_probabilities
from federated.praa import aggregate_prototypes
from models.classifier import FLAMEModel, model_kwargs_from_config

ROOT = Path(__file__).resolve().parent.parent

# PathMNIST label set (medmnist.INFO["pathmnist"]["label"]), kept here so the app does not need medmnist.
CLASS_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}
CLASS_CODES = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC", 5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 28                      # PathMNIST native resolution
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard (PIL raises above 2x this value)

CHECKPOINT_CANDIDATES = [
    "results/runs/praa/final_model.pt",
    "results/runs/flame/final_model.pt",
    "results/runs/fedavg/final_model.pt",
    "results/phase3/final_model.pt",
]


class ImageError(ValueError):
    """The uploaded bytes are not a usable image (maps to HTTP 400/413/415)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class LoadedModel:
    model: FLAMEModel
    prototypes: Dict[int, torch.Tensor]
    rule: str                      # "prototype" | "classifier_head"
    mode: str                      # "trained" | "demo"
    checkpoint: Optional[str]
    description: str
    device: str = "cpu"
    notes: List[str] = field(default_factory=list)

    @property
    def missing_classes(self) -> List[int]:
        return [c for c in range(NUM_CLASSES) if c not in self.prototypes] if self.rule == "prototype" else []


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise ImageError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageError(f"The image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", status=413)
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()                                   # cheap integrity check
        image = Image.open(io.BytesIO(data))             # verify() invalidates the object -> reopen
        image = ImageOps.exif_transpose(image)
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, ValueError) as exc:
        raise ImageError("Could not read that file as an image. Use a PNG, JPG, BMP, TIFF or WEBP file.",
                         status=415) from exc
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)  # flatten transparency onto white
    return image.convert("RGB")


def to_model_image(image: Image.Image) -> Image.Image:
    """Centre-crop to a square and resize to 28x28 (PathMNIST's native size)."""
    return ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.BICUBIC, centering=(0.5, 0.5))


def to_tensor(small: Image.Image) -> torch.Tensor:
    """[1, 3, 28, 28] float tensor, same transform as ``datasets/preprocess.py``: ToTensor + Normalize(0.5, 0.5)."""
    array = np.asarray(small, dtype=np.float32) / 255.0           # H, W, C in [0, 1]
    tensor = torch.from_numpy(array).permute(2, 0, 1)              # C, H, W
    return ((tensor - 0.5) / 0.5).unsqueeze(0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a dict checkpoint, got {type(state).__name__}")
    key = "model_state_dict" if "model_state_dict" in state else "global_model_state"
    if key not in state:
        raise ValueError(f"{path}: no 'model_state_dict' / 'global_model_state' key (found {sorted(state)})")
    state["_weights_key"] = key
    return state


def prototypes_from_phase4(directory: Path) -> Dict[int, torch.Tensor]:
    """FLAME-rule global prototypes (sample-weighted class means) from the preserved Phase 4 client files."""
    files = sorted(directory.glob("client_*.pt"))
    if not files:
        raise FileNotFoundError(f"no client_*.pt prototype files in {directory}")
    protos: Dict[int, Dict[int, torch.Tensor]] = {}
    counts: Dict[int, Dict[int, int]] = {}
    for file in files:
        record = torch.load(file, map_location="cpu", weights_only=True)
        cid = int(record["client_id"])
        class_ids = [int(c) for c in record["class_ids"].tolist()]
        vectors, sizes = record["prototype_vectors"], [int(n) for n in record["sample_counts"].tolist()]
        protos[cid] = {c: vectors[i].float() for i, c in enumerate(class_ids)}
        counts[cid] = {c: sizes[i] for i, c in enumerate(class_ids)}
    global_prototypes, _ = aggregate_prototypes(protos, counts, None)
    return global_prototypes


def prototypes_from_file(path: Path) -> Dict[int, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    raw = data.get("prototypes", data) if isinstance(data, dict) else data
    out = {int(c): torch.as_tensor(v).float() for c, v in raw.items()}
    if not out:
        raise ValueError(f"{path}: no prototypes found")
    return out


def _resolve_checkpoint() -> Optional[Path]:
    env = os.environ.get("PRAA_CHECKPOINT")
    if env:
        path = Path(env).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"PRAA_CHECKPOINT points to a missing file: {path}")
        return path
    for candidate in CHECKPOINT_CANDIDATES:
        if (ROOT / candidate).is_file():
            return ROOT / candidate
    return None


def _demo_model() -> LoadedModel:
    """Untrained tiny model + prototypes from SYNTHETIC data, only so the app can start without a checkpoint."""
    from datasets.synthetic import SyntheticPathMNIST

    torch.manual_seed(0)
    model = FLAMEModel(feature_dim=64, embedding_dim=32, num_classes=NUM_CLASSES, encoder="small_cnn").eval()
    data = SyntheticPathMNIST("train", transform=None, size=180)
    images = torch.stack([torch.from_numpy(data.imgs[i]).permute(2, 0, 1).float() / 255.0 for i in range(len(data))])
    images = (images - 0.5) / 0.5
    labels = torch.from_numpy(data.labels.reshape(-1))
    with torch.no_grad():
        _, emb = model(images)
    prototypes = {c: emb[labels == c].mean(dim=0) for c in range(NUM_CLASSES)}
    return LoadedModel(
        model=model, prototypes=prototypes, rule="prototype", mode="demo", checkpoint=None,
        description="DEMO model: random weights, synthetic prototypes",
        notes=["No trained checkpoint was found, so the app is running with an untrained demo model. "
               "Its predictions are arbitrary. Put final_model.pt at results/phase3/final_model.pt "
               "(or set PRAA_CHECKPOINT) and restart."])


def load_model(device: str = "cpu") -> LoadedModel:
    path = _resolve_checkpoint()
    if path is None:
        return _demo_model()

    state = _load_state(path)
    config = state.get("config")
    method = (config or {}).get("method", "flame")
    kwargs = model_kwargs_from_config(config) if config else {}      # Phase 3 file: paper defaults
    model = FLAMEModel(**kwargs)
    model.load_state_dict(state[state["_weights_key"]], strict=True)
    model.to(device).eval()
    num_classes = int(model.num_classes)
    if num_classes != NUM_CLASSES:
        raise ValueError(f"checkpoint has {num_classes} classes; this app is built for PathMNIST ({NUM_CLASSES})")

    if method == "fedavg":
        return LoadedModel(model=model, prototypes={}, rule="classifier_head", mode="trained",
                           checkpoint=str(path), description="FedAvg classifier head", device=device)

    env_protos = os.environ.get("PRAA_PROTOTYPES")
    sibling = path.with_name("global_prototypes.pt")
    notes: List[str] = []
    if env_protos:
        prototypes, origin = prototypes_from_file(Path(env_protos).expanduser()), f"PRAA_PROTOTYPES={env_protos}"
    elif sibling.is_file():
        prototypes, origin = prototypes_from_file(sibling), str(sibling)
    elif path.resolve() == (ROOT / "results/phase3/final_model.pt").resolve():
        prototypes = prototypes_from_phase4(ROOT / "results/phase4/prototypes")
        origin = "results/phase4/prototypes (FLAME rule, sample-weighted)"
    else:
        raise FileNotFoundError(
            f"No global prototypes for {path}. Run:\n"
            f"  python scripts/export_global_prototypes.py --run-dir {path.parent}\n"
            f"or set PRAA_PROTOTYPES to a global_prototypes.pt file.")
    dim = model.embedding_dim
    bad = [c for c, v in prototypes.items() if v.numel() != dim]
    if bad:
        raise ValueError(f"prototype dimension mismatch for classes {bad}: model embedding is {dim}-d")
    if len(prototypes) < NUM_CLASSES:
        notes.append("Classes without a prototype can never be predicted: "
                     f"{[CLASS_NAMES[c] for c in range(NUM_CLASSES) if c not in prototypes]}.")
    return LoadedModel(model=model, prototypes=prototypes, rule="prototype", mode="trained",
                       checkpoint=str(path), device=device, notes=notes,
                       description=f"{method.upper()} model, nearest global prototype ({origin})")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class Classifier:
    def __init__(self, loaded: LoadedModel) -> None:
        self.loaded = loaded
        self._lock = threading.Lock()

    def predict(self, data: bytes, top_k: int = 3) -> dict:
        started = time.perf_counter()
        original = decode_image(data)
        small = to_model_image(original)
        x = to_tensor(small)
        loaded = self.loaded
        with self._lock, torch.no_grad():
            if loaded.rule == "classifier_head":
                logits = loaded.model.logits(x.to(loaded.device)).float().cpu()
                nearest = None
            else:
                _, emb = loaded.model(x.to(loaded.device))
                emb = emb.float().cpu()
                if not torch.isfinite(emb).all():
                    raise FloatingPointError("model produced a non-finite embedding for this image")
                logits, _ = prototype_logits(emb, loaded.prototypes, NUM_CLASSES)
                nearest = float((-logits[0][torch.isfinite(logits[0])]).min())
        probs = softmax_probabilities(logits)[0].tolist()
        order = sorted(range(NUM_CLASSES), key=lambda c: probs[c], reverse=True)
        entry = lambda c: {"class_id": c, "code": CLASS_CODES[c], "name": CLASS_NAMES[c], "probability": probs[c]}
        return {
            "predicted": entry(order[0]),
            "top_k": [entry(c) for c in order[:max(1, min(top_k, NUM_CLASSES))]],
            "probabilities": [entry(c) for c in range(NUM_CLASSES)],
            "rule": loaded.rule,
            "mode": loaded.mode,
            "demo": loaded.mode == "demo",
            "nearest_prototype_sq_distance": nearest,
            "input": {"original_size": list(original.size), "model_size": [IMAGE_SIZE, IMAGE_SIZE]},
            "model_input_png": _png_data_uri(small),
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def _png_data_uri(small: Image.Image, scale: int = 8) -> str:
    import base64
    big = small.resize((small.width * scale, small.height * scale), Image.NEAREST)
    buffer = io.BytesIO()
    big.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
