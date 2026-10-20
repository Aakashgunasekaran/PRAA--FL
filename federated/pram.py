"""Prototype Reliability Assessment Module (PRAM) - Phase 5.

PRAM measures three per-(client, class) signals and keeps "cannot be computed"
strictly separate from "computed and low":

    status "valid"       -> `value` is the normalized score in [0, 1]
    status "degenerate"  -> a number exists (`raw`) but is uninformative by
                            construction; `value` is None and Phase 6 must not use it
    status "unavailable" -> required evidence is missing; `value` and `raw` are None

Formulas (formula version ``pram/1``; the repository's documented definitions, see
docs/PHASE_5_PRAM.md; weights/PRS combination belong to Phase 6 and are NOT here):

    validation   nearest-prototype recall on the shared PathMNIST validation split.
                 Validation embeddings come from the global Phase 3 model; candidate
                 classes are the classes the client holds; squared-Euclidean distance;
                 ties go to the lowest class id.
    consistency  cosine(client prototype, reference prototype) mapped to [0, 1] via
                 (cos + 1) / 2. Reference = unweighted mean of all clients' prototypes
                 for the same class (self-inclusive, as documented).
    stability    mean over consecutive attested snapshots of the same client/class of
                 cosine similarity mapped to [0, 1].

No test data is read; this module never touches datasets.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from federated.prs import _finite_vector, cosine_score

FORMULA_VERSION = "pram/1"
SCHEMA_VERSION = "praa-fl.phase5.pram/1"
VALID, DEGENERATE, UNAVAILABLE = "valid", "degenerate", "unavailable"
STATUSES = (VALID, DEGENERATE, UNAVAILABLE)
COMPONENTS = ("validation", "consistency", "stability")
ATTESTED_SNAPSHOT_KIND = "client_local_model"

FORMULAS = {
    "validation": "per-class recall of nearest-prototype classification (squared Euclidean, "
                  "candidates = client's classes) on the shared validation split using the "
                  "global model's embeddings",
    "consistency": "(cos(client_prototype, reference_prototype) + 1) / 2; reference = unweighted "
                   "mean of all clients' same-class prototypes (self-inclusive)",
    "stability": "mean over consecutive attested client-local snapshots of (cos + 1) / 2",
}


def component(status: str, value=None, raw=None, reason: Optional[str] = None, **details) -> Dict:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if status == VALID and value is None:
        raise ValueError("a valid component needs a value")
    if status != VALID and value is not None:
        raise ValueError("only valid components carry a value")
    if status == UNAVAILABLE and raw is not None:
        raise ValueError("an unavailable component cannot carry a raw measurement")
    return {
        "status": status,
        "value": None if value is None else float(value),
        "raw": None if raw is None else float(raw),
        "reason": reason,
        "details": details,
    }


# ---------------------------------------------------------------------------
# A. Validation performance
# ---------------------------------------------------------------------------

def validation_components(
    embeddings: torch.Tensor, labels: torch.Tensor, prototypes: Mapping[int, torch.Tensor]
) -> Dict[int, Dict]:
    """Per-class nearest-prototype recall for one client's prototypes."""
    if not prototypes:
        return {}
    class_ids = sorted(int(c) for c in prototypes)
    matrix = torch.stack([_finite_vector(prototypes[c], f"class {c} prototype") for c in class_ids])
    if embeddings.ndim != 2 or embeddings.size(1) != matrix.size(1):
        raise ValueError("validation embedding dimension does not match prototypes")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("validation embeddings contain NaN or Inf")
    labels = labels.view(-1)
    if labels.numel() != embeddings.size(0):
        raise ValueError("embeddings and labels must have the same length")
    distances = ((embeddings.float()[:, None, :] - matrix[None, :, :]) ** 2).sum(dim=2)
    predicted = torch.tensor(class_ids)[distances.argmin(dim=1)]
    k = len(class_ids)
    out = {}
    for class_id in class_ids:
        mask = labels == class_id
        n = int(mask.sum())
        common = dict(num_eval_samples=n, num_candidate_classes=k, chance_level=1.0 / k)
        if n == 0:
            out[class_id] = component(UNAVAILABLE, reason="no_validation_samples_for_class", **common)
            continue
        recall = float((predicted[mask] == class_id).sum()) / n
        if k == 1:
            out[class_id] = component(
                DEGENERATE, raw=recall, reason="single_candidate_class_recall_is_1_by_construction", **common)
        else:
            out[class_id] = component(VALID, value=recall, raw=recall, **common)
    return out


# ---------------------------------------------------------------------------
# B. Prototype consistency
# ---------------------------------------------------------------------------

def build_reference_prototypes(records: Mapping[int, Mapping]) -> Dict[int, Dict]:
    """Unweighted per-class mean of all clients' prototypes (documented reference)."""
    dims = {int(r["embedding_dim"]) for r in records.values()}
    if len(dims) != 1:
        raise ValueError(f"Phase 4 records mix embedding dimensions: {sorted(dims)}")
    by_class: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for client_id, record in sorted(records.items()):
        for class_id, vector in record["prototypes"].items():
            by_class.setdefault(int(class_id), []).append(
                (int(client_id), _finite_vector(vector, f"client {client_id} class {class_id}")))
    reference = {}
    for class_id, items in sorted(by_class.items()):
        vectors = torch.stack([v for _, v in items])
        reference[class_id] = {
            "vector": vectors.double().mean(dim=0).float(),
            "contributors": [c for c, _ in items],
            "_by_client": {c: v for c, v in items},
        }
    return reference


def consistency_component(
    client_id: int, class_id: int, vector: torch.Tensor, reference: Mapping[int, Dict]
) -> Dict:
    vector = _finite_vector(vector, f"client {client_id} class {class_id}")
    if class_id not in reference:
        return component(UNAVAILABLE, reason="no_reference_for_class")
    ref = reference[class_id]
    if ref["vector"].shape != vector.shape:
        raise ValueError(f"prototype dimension {tuple(vector.shape)} != reference {tuple(ref['vector'].shape)}")
    contributors = ref["contributors"]
    details = dict(num_reference_contributors=len(contributors))
    try:
        score = cosine_score(vector, ref["vector"])
    except ValueError:
        return component(UNAVAILABLE, reason="zero_norm_prototype_or_reference", **details)
    if len(contributors) < 2:
        return component(DEGENERATE, raw=score,
                         reason="reference_built_only_from_this_client", **details)
    # Diagnostic only (not the documented score): reference excluding this client.
    others = [c for c in contributors if c != client_id]
    loo = None
    if others and client_id in ref["_by_client"]:
        stacked = torch.stack([ref["_by_client"][c] for c in others])
        try:
            loo = cosine_score(vector, stacked.double().mean(0).float())
        except ValueError:
            loo = None
    return component(VALID, value=score, raw=score, **details, leave_one_out_diagnostic=loo)


# ---------------------------------------------------------------------------
# C. Training stability (requires attested client-local snapshots)
# ---------------------------------------------------------------------------

def discover_client_snapshots(phase4_dir, client_id: int):
    """Return (accepted [(round, prototypes)], rejected [(file, reason)]) for one client.

    A snapshot is accepted only if it carries matching client/round metadata AND
    provenance.source_kind == "client_local_model". Unattested files (for example
    copies derived from the global model) are rejected, never used.
    """
    root = Path(phase4_dir) / "snapshots"
    accepted, rejected = [], []
    if not root.exists():
        return accepted, rejected
    rounds_seen = set()
    for round_dir in sorted(root.glob("round_*"), key=lambda p: int(p.name.split("_")[1])):
        path = round_dir / f"client_{client_id}.pt"
        if not path.exists():
            continue
        round_id = int(round_dir.name.split("_")[1])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        reasons = []
        if int(payload.get("client_id", -1)) != client_id:
            reasons.append("client_id_mismatch")
        if int(payload.get("round", -1)) != round_id:
            reasons.append("round_mismatch")
        if (payload.get("provenance") or {}).get("source_kind") != ATTESTED_SNAPSHOT_KIND:
            reasons.append("unattested_source_kind")
        if not payload.get("prototypes"):
            reasons.append("no_prototypes")
        if round_id in rounds_seen:
            reasons.append("duplicate_round")
        if reasons:
            rejected.append((str(path), ",".join(reasons)))
            continue
        rounds_seen.add(round_id)
        accepted.append((round_id, {int(c): v for c, v in payload["prototypes"].items()}))
    return accepted, rejected


def stability_components(
    snapshots: Sequence[Tuple[int, Mapping[int, torch.Tensor]]], classes: Sequence[int]
) -> Dict[int, Dict]:
    """Per-class temporal stability; unavailable unless >= 2 attested snapshots hold the class."""
    ordered = sorted(snapshots, key=lambda item: item[0])
    out = {}
    for class_id in classes:
        holding = [(r, protos[class_id]) for r, protos in ordered if class_id in protos]
        if len(ordered) < 2:
            out[class_id] = component(UNAVAILABLE, reason="fewer_than_two_attested_client_snapshots",
                                      num_snapshots=len(ordered))
            continue
        if len(holding) < 2:
            out[class_id] = component(UNAVAILABLE, reason="class_present_in_fewer_than_two_snapshots",
                                      num_snapshots=len(ordered))
            continue
        scores = [cosine_score(a[1], b[1]) for a, b in zip(holding, holding[1:])]
        value = sum(scores) / len(scores)
        out[class_id] = component(
            VALID, value=value, raw=value, num_pairs=len(scores),
            rounds=[r for r, _ in holding], round_gaps=[b[0] - a[0] for a, b in zip(holding, holding[1:])])
    return out


def inspect_training_history(history: Sequence[Mapping], num_clients: int) -> Dict:
    """Describe what the Phase 3 history can and cannot say about client-level stability."""
    participation = Counter()
    mismatched = []
    metric_keys = set()
    for entry in history:
        selected = sorted(int(c) for c in entry["selected_clients"])
        recorded = sorted(int(m["client_id"]) for m in entry.get("client_metrics", []))
        if selected != recorded:
            mismatched.append(int(entry["round"]))
        for metrics in entry.get("client_metrics", []):
            metric_keys.update(k for k in metrics if k != "client_id")
        participation.update(selected)
    counts = [participation.get(c, 0) for c in range(num_clients)]
    return {
        "rounds_in_history": len(history),
        "final_round": int(history[-1]["round"]) if history else None,
        "participation_min_max_mean": [min(counts), max(counts), sum(counts) / len(counts)],
        "clients_never_selected": [c for c in range(num_clients) if counts[c] == 0],
        "rounds_with_selection_mismatch": mismatched,
        "per_round_client_metric_keys": sorted(metric_keys),
        "contains_client_prototype_or_model_snapshots": False,
        "note": "history holds scalar losses for participating clients only; non-participation "
                "is not instability and scalar losses are not the documented stability evidence",
    }


# ---------------------------------------------------------------------------
# Assembly, validation, persistence
# ---------------------------------------------------------------------------

def assess_client(
    client_id: int,
    record: Mapping,
    reference: Mapping[int, Dict],
    validation_embeddings: torch.Tensor,
    validation_labels: torch.Tensor,
    snapshots: Sequence[Tuple[int, Mapping[int, torch.Tensor]]],
    round_id: int,
) -> List[Dict]:
    prototypes = {int(c): v for c, v in record["prototypes"].items()}
    validation = validation_components(validation_embeddings, validation_labels, prototypes)
    stability = stability_components(snapshots, sorted(prototypes))
    rows = []
    for class_id in sorted(prototypes):
        consistency = consistency_component(client_id, class_id, prototypes[class_id], reference)
        parts = {"validation": validation[class_id], "consistency": consistency,
                 "stability": stability[class_id]}
        rows.append({
            "client_id": int(client_id),
            "class_id": int(class_id),
            "round_id": int(round_id),
            "sample_count": int(record["metadata"][class_id]["sample_count"]),
            "prototype_dimension": int(record["embedding_dim"]),
            **parts,
            "complete": all(p["status"] == VALID for p in parts.values()),
        })
    return rows


def summarize(rows: Sequence[Mapping]) -> Dict:
    summary = {name: {s: sum(r[name]["status"] == s for r in rows) for s in STATUSES}
               for name in COMPONENTS}
    reasons = {}
    for name in COMPONENTS:
        reasons[name] = dict(Counter(r[name]["reason"] for r in rows if r[name]["reason"]))
    return {"records": len(rows), "complete_records": sum(r["complete"] for r in rows),
            "status_counts": summary, "reason_counts": reasons}


def phase6_readiness(rows: Sequence[Mapping]) -> Dict:
    summary = summarize(rows)
    blocking = []
    for name in COMPONENTS:
        for reason, count in summary["reason_counts"][name].items():
            blocking.append(f"{name}: {reason} ({count} records)")
    complete = summary["complete_records"]
    return {
        "records_with_all_three_valid": complete,
        "records_total": len(rows),
        "prs_computable_for_all_records": bool(rows) and complete == len(rows),
        "prs_computable_for_some_records": complete > 0,
        "blocking_reasons": blocking,
        "rule": "Phase 6 may combine components only for records with complete == true; "
                "null values must never be replaced with 0, 1, or a neutral constant",
    }


def validate_output(document: Mapping) -> None:
    """Raise ValueError if a PRAM document violates the schema invariants."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PRAM schema version")
    seen = set()
    for row in document["records"]:
        key = (row["client_id"], row["class_id"])
        if key in seen:
            raise ValueError(f"duplicate record {key}")
        seen.add(key)
        for name in COMPONENTS:
            part = row[name]
            if part["status"] not in STATUSES:
                raise ValueError(f"{key} {name}: bad status {part['status']!r}")
            for field in ("value", "raw"):
                v = part[field]
                if v is not None and not (isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0):
                    raise ValueError(f"{key} {name}.{field} invalid: {v!r}")
            if (part["status"] == VALID) != (part["value"] is not None):
                raise ValueError(f"{key} {name}: value present iff status is valid")
            if part["status"] == UNAVAILABLE and part["raw"] is not None:
                raise ValueError(f"{key} {name}: unavailable component carries a raw value")
            if part["status"] != VALID and not part["reason"]:
                raise ValueError(f"{key} {name}: non-valid component needs a reason")
        if row["complete"] != all(row[n]["status"] == VALID for n in COMPONENTS):
            raise ValueError(f"{key}: complete flag inconsistent")
    # Derived sections must agree with the records they describe (Phase 6 relies on them).
    if "summary" in document and document["summary"] != summarize(document["records"]):
        raise ValueError("summary disagrees with the records")
    if "phase6_readiness" in document and document["phase6_readiness"] != phase6_readiness(document["records"]):
        raise ValueError("phase6_readiness disagrees with the records")


def write_json_atomic(document: Mapping, path, overwrite: bool = False) -> Path:
    """Validate and write JSON via a temp file; refuse to clobber existing output."""
    validate_output(document)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path


def load_pram_output(path) -> Dict:
    """Load and validate a PRAM document (the Phase 6 entry point)."""
    document = json.loads(Path(path).read_text())
    validate_output(document)
    return document
