"""Phase 4 prototype records: construction, validation, and persistence.

This module has no dependency on the PathMNIST loaders, so it can be unit-tested
with small fixtures. Only prototype vectors, sample counts, and provenance
metadata are exported; raw images never leave the simulated client.

Record layout (one file per client, ``client_<id>.pt``, loadable with
``torch.load(..., weights_only=True)``):

    schema_version        str    "praa-fl.phase4.prototypes/1"
    client_id             int
    class_ids             LongTensor [K]      sorted classes present on the client
    prototype_vectors     FloatTensor [K, D]  row i is the mean embedding of class_ids[i]
    sample_counts         LongTensor [K]      samples contributing to each prototype
    embedding_dim         int (D)
    num_classes           int                 total classes in the task
    absent_class_ids      list[int]           classes with no samples (no prototype exists)
    num_local_samples     int                 size of the client's Phase 2 partition
    num_samples_used      int                 samples that contributed (label policy applied)
    label_policy          str
    provenance            dict                source model path/hash/round, partition hash
    config                dict                seed, label fraction, batch size, mode, ...
    prototypes            {class_id: Tensor}  legacy view used by Phase 5
    metadata              {class_id: dict}    legacy per-class metadata used by Phase 5
    samples_processed     int                 legacy alias of num_samples_used
    representation        str

PRM/PRS/PRAA phases should read records with ``load_client_prototype_file`` and,
when a dense layout is needed, ``to_dense`` (which returns a presence mask
instead of fabricating zero prototypes).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from training.prototypes import extract_features, generate_prototypes

SCHEMA_VERSION = "praa-fl.phase4.prototypes/1"
REPRESENTATION = "FLAME projected embedding (model.proto(model.encoder(x)))"
LABEL_POLICIES = ("labeled_subset", "all_labels")


# ---------------------------------------------------------------------------
# Hashing and train-split guard
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_fingerprint(partition: Mapping[int, Sequence[int]]) -> str:
    """Order-insensitive hash of a client -> indices mapping."""
    canonical = {str(int(k)): sorted(int(i) for i in v) for k, v in sorted(partition.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def assert_train_split(dataset) -> None:
    """Refuse datasets that identify themselves as a non-train split.

    Follows ``torch.utils.data.Subset`` wrappers down to the base dataset; MedMNIST
    datasets expose ``split``. Fixtures without a ``split`` attribute are accepted.
    """
    base = dataset
    while hasattr(base, "dataset") and hasattr(base, "indices"):
        base = base.dataset
    split = getattr(base, "split", None)
    if split is not None and str(split) != "train":
        raise PermissionError(
            f"Phase 4 may only read the training split; dataset reports split={split!r}"
        )


# ---------------------------------------------------------------------------
# Which samples a client may use
# ---------------------------------------------------------------------------

def select_labeled_positions(
    num_samples: int, client_id: int, seed: int, label_fraction: float
) -> List[int]:
    """Positions (within a client's partition) of its labeled samples.

    Mirrors ``federated.client.Client`` exactly (same generator seed, permutation,
    and labeled count) so Phase 4 uses the very labels Phase 3 trained with.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1]")
    generator = torch.Generator().manual_seed(int(seed) + int(client_id))
    order = torch.randperm(num_samples, generator=generator).tolist()
    labeled_count = min(max(2, int(num_samples * label_fraction)), num_samples)
    return order[:labeled_count]


def apply_label_policy(
    dataset, client_id: int, label_policy: str, seed: int, label_fraction: float
):
    """Return the subset of a client's dataset that may contribute to prototypes."""
    if label_policy not in LABEL_POLICIES:
        raise ValueError(f"label_policy must be one of {LABEL_POLICIES}")
    if len(dataset) == 0:
        raise ValueError(f"Client {client_id} has an empty dataset")
    if label_policy == "all_labels":
        return dataset
    positions = select_labeled_positions(len(dataset), client_id, seed, label_fraction)
    return torch.utils.data.Subset(dataset, positions)


# ---------------------------------------------------------------------------
# Record construction / validation
# ---------------------------------------------------------------------------

def build_client_record(
    client_id: int,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    num_local_samples: int,
    label_policy: str,
    provenance: Mapping[str, object],
    config: Mapping[str, object],
) -> Dict[str, object]:
    labels = labels.view(-1).long()
    if labels.numel() == 0:
        raise ValueError(f"Client {client_id} produced no embeddings")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError(f"Client {client_id} has labels outside [0, {num_classes})")
    prototypes, counts = generate_prototypes(embeddings, labels)
    class_ids = sorted(prototypes)
    vectors = torch.stack([prototypes[c] for c in class_ids]).float().contiguous()
    count_tensor = torch.tensor([counts[c] for c in class_ids], dtype=torch.long)
    dim = int(vectors.size(1))
    record = {
        "schema_version": SCHEMA_VERSION,
        "client_id": int(client_id),
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "prototype_vectors": vectors,
        "sample_counts": count_tensor,
        "embedding_dim": dim,
        "num_classes": int(num_classes),
        "absent_class_ids": [c for c in range(num_classes) if c not in prototypes],
        "num_local_samples": int(num_local_samples),
        "num_samples_used": int(labels.numel()),
        "label_policy": label_policy,
        "provenance": dict(provenance),
        "config": dict(config),
        "prototypes": {c: vectors[i].clone() for i, c in enumerate(class_ids)},
        "metadata": {
            c: {
                "client_id": int(client_id),
                "class_id": int(c),
                "sample_count": int(counts[c]),
                "embedding_dimension": dim,
                "prototype_norm": float(vectors[i].norm().item()),
            }
            for i, c in enumerate(class_ids)
        },
        "samples_processed": int(labels.numel()),
        "representation": REPRESENTATION,
    }
    validate_record(record)
    return record


def validate_record(record: Mapping[str, object]) -> None:
    """Raise ValueError/FloatingPointError if a record is malformed."""
    required = (
        "schema_version", "client_id", "class_ids", "prototype_vectors", "sample_counts",
        "embedding_dim", "num_classes", "absent_class_ids", "provenance", "config",
        "prototypes", "metadata",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Prototype record missing keys: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema {record['schema_version']!r}")
    class_ids = record["class_ids"]
    vectors = record["prototype_vectors"]
    counts = record["sample_counts"]
    k = int(class_ids.numel())
    if k == 0:
        raise ValueError("A record must contain at least one prototype")
    if vectors.shape != (k, int(record["embedding_dim"])):
        raise ValueError("prototype_vectors shape does not match class_ids / embedding_dim")
    if counts.shape != (k,) or bool((counts <= 0).any()):
        raise ValueError("sample_counts must be positive and align with class_ids")
    ids = class_ids.tolist()
    if ids != sorted(set(ids)) or min(ids) < 0 or max(ids) >= int(record["num_classes"]):
        raise ValueError("class_ids must be unique, sorted, and within [0, num_classes)")
    if not torch.isfinite(vectors).all():
        raise FloatingPointError("Prototype vectors contain NaN or Inf")
    expected_absent = [c for c in range(int(record["num_classes"])) if c not in ids]
    if list(record["absent_class_ids"]) != expected_absent:
        raise ValueError("absent_class_ids inconsistent with class_ids")
    if sorted(int(c) for c in record["prototypes"]) != ids:
        raise ValueError("legacy prototypes dict inconsistent with class_ids")
    for i, class_id in enumerate(ids):
        if not torch.equal(record["prototypes"][class_id], vectors[i]):
            raise ValueError(f"legacy prototype for class {class_id} differs from matrix row")


def save_client_record(record: Mapping[str, object], directory) -> Path:
    """Atomically save ``client_<id>.pt`` under ``directory``."""
    validate_record(record)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"client_{int(record['client_id'])}.pt"
    temp_path = final_path.with_suffix(".pt.tmp")
    torch.save(dict(record), temp_path)
    os.replace(temp_path, final_path)
    return final_path


def load_client_prototype_file(path) -> Dict[str, object]:
    """Load and validate a Phase 4 record (tensors/primitives only)."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    validate_record(record)
    expected_id = int(Path(path).stem.split("_")[1])
    if int(record["client_id"]) != expected_id:
        raise ValueError(
            f"{path}: file name says client {expected_id}, record says {record['client_id']}"
        )
    return record


def to_dense(record: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ([num_classes, D] matrix, [num_classes] bool presence mask).

    Absent classes hold zeros *only as storage filler*; the mask must be used to
    ignore them. No prototype is ever fabricated.
    """
    num_classes = int(record["num_classes"])
    dense = torch.zeros(num_classes, int(record["embedding_dim"]))
    mask = torch.zeros(num_classes, dtype=torch.bool)
    for i, class_id in enumerate(record["class_ids"].tolist()):
        dense[class_id] = record["prototype_vectors"][i]
        mask[class_id] = True
    return dense, mask


# ---------------------------------------------------------------------------
# Per-client generation
# ---------------------------------------------------------------------------

def generate_client_record(
    model,
    dataset,
    client_id: int,
    *,
    num_classes: int,
    label_policy: str,
    seed: int,
    label_fraction: float,
    batch_size: int,
    device: str,
    provenance: Mapping[str, object],
    config: Mapping[str, object],
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Extract embeddings from one client's permitted samples and build its record."""
    assert_train_split(dataset)
    permitted = apply_label_policy(dataset, client_id, label_policy, seed, label_fraction)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        permitted = torch.utils.data.Subset(permitted, range(min(len(permitted), max_samples)))
    loader = torch.utils.data.DataLoader(permitted, batch_size=batch_size, shuffle=False)
    embeddings, labels = extract_features(model, loader, device=device)
    return build_client_record(
        client_id, embeddings, labels, num_classes, len(dataset), label_policy,
        provenance, config,
    )


# ---------------------------------------------------------------------------
# Phase 3 artifact validation
# ---------------------------------------------------------------------------

def _tensors_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def validate_phase3_artifacts(
    phase3_dir,
    expected: Mapping[str, object],
    require_rounds: Optional[int] = None,
) -> Dict[str, object]:
    """Validate the frozen Phase 3 outputs and return provenance for Phase 4.

    Checks: files exist, config matches fixed project settings, history is complete
    (``require_rounds`` rounds, consecutive), weights are finite, and, when the
    matching ``checkpoint_round_<N>.pt`` is present, that its global weights are
    identical to ``final_model.pt``. Nothing is rewritten.
    """
    phase3_dir = Path(phase3_dir)
    model_path = phase3_dir / "final_model.pt"
    config_path = phase3_dir / "config.json"
    for path in (model_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 3 artifact: {path}")
    config = json.loads(config_path.read_text())
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase 3 config differs from fixed project config: {mismatches}")
    # Trusted local artifact written by Phase 3 (history holds plain Python objects).
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    history = checkpoint["history"]
    rounds = [int(item["round"]) for item in history]
    if rounds != list(range(1, len(rounds) + 1)):
        raise ValueError("Phase 3 history rounds are not consecutive from 1")
    if require_rounds is not None and len(rounds) != require_rounds:
        raise ValueError(f"Expected {require_rounds} completed rounds, found {len(rounds)}")
    if int(config["num_rounds"]) != len(rounds):
        raise ValueError("config.json num_rounds does not match history length")
    bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise FloatingPointError(f"Non-finite weights in final model: {bad[:3]}")
    crosscheck = {"performed": False}
    checkpoint_path = phase3_dir / "checkpoints" / f"checkpoint_round_{rounds[-1]}.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        crosscheck = {
            "performed": True,
            "file": checkpoint_path.name,
            "round_matches": int(saved["round"]) == rounds[-1],
            "weights_identical": _tensors_equal(saved["global_model_state"], state),
        }
        if not (crosscheck["round_matches"] and crosscheck["weights_identical"]):
            raise ValueError(f"{checkpoint_path.name} disagrees with final_model.pt")
    return {
        "state_dict": state,
        "checkpoint_keys": sorted(checkpoint.keys()),
        "config": config,
        "provenance": {
            "source_kind": "final_global_model",
            "source_model_file": model_path.name,
            "source_model_sha256": sha256_file(model_path),
            "source_round": int(rounds[-1]),
            "source_config_sha256": sha256_file(config_path),
            "checkpoint_crosscheck": crosscheck,
        },
    }
