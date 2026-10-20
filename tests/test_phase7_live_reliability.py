"""Live PRAM evidence, attested snapshots and stability during training."""

import pytest
import torch

from federated.pram import discover_client_snapshots
from federated.reliability import ReliabilityTracker, local_evidence
from models.classifier import FLAMEModel


def _evidence(shift=0.0, dim=6):
    torch.manual_seed(1)
    protos = {0: torch.ones(dim) + shift, 1: -torch.ones(dim) + shift}
    emb = torch.cat([torch.ones(5, dim), -torch.ones(5, dim)])
    return {"prototypes": protos, "counts": {0: 4, 1: 6}, "val_embeddings": emb}


LABELS = torch.tensor([0] * 5 + [1] * 5)


def test_first_round_has_no_stability_then_it_appears(tmp_path):
    tracker = ReliabilityTracker(tmp_path, stability_window=3)
    ev = {0: _evidence(), 1: _evidence(0.1)}
    rows1 = tracker.assess_round(1, ev, LABELS)
    assert all(r["stability"]["status"] == "unavailable" for r in rows1)
    rows2 = tracker.assess_round(2, ev, LABELS)
    assert all(r["stability"]["status"] == "valid" for r in rows2)
    assert all(r["complete"] for r in rows2)


def test_non_participation_is_not_instability(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    tracker.assess_round(1, {0: _evidence(), 1: _evidence()}, LABELS)
    tracker.assess_round(2, {1: _evidence()}, LABELS)          # client 0 skipped round 2
    rows = tracker.assess_round(3, {0: _evidence(), 1: _evidence()}, LABELS)
    c0 = [r for r in rows if r["client_id"] == 0][0]["stability"]
    assert c0["status"] == "valid" and c0["details"]["round_gaps"] == [2]
    assert c0["value"] == pytest.approx(1.0)  # identical prototypes -> cosine 1


def test_stability_detects_prototype_drift(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    stable, drifting = _evidence(), _evidence()
    tracker.assess_round(1, {0: stable, 1: drifting}, LABELS)
    drifting2 = _evidence()
    drifting2["prototypes"] = {0: torch.tensor([1.0, -1, 1, -1, 1, -1]), 1: torch.tensor([-1.0, 1, -1, 1, -1, 1])}
    rows = tracker.assess_round(2, {0: stable, 1: drifting2}, LABELS)
    s = {r["client_id"]: r["stability"]["value"] for r in rows if r["class_id"] == 0}
    assert s[0] > s[1]


def test_snapshots_on_disk_are_accepted_as_attested_by_phase5(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    for rnd in (1, 2, 3):
        tracker.assess_round(rnd, {0: _evidence(), 1: _evidence()}, LABELS)
    accepted, rejected = discover_client_snapshots(tmp_path, 0)
    assert [r for r, _ in accepted] == [1, 2, 3] and rejected == []


def test_resume_reloads_history_and_drops_newer_snapshots(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    for rnd in (1, 2, 3):
        tracker.assess_round(rnd, {0: _evidence()}, LABELS)
    fresh = ReliabilityTracker(tmp_path)
    assert fresh.load_from_disk(max_round=2, delete_newer=True) == 2
    assert [r for r, _ in fresh.history[0]] == [1, 2]
    assert not (tmp_path / "snapshots" / "round_3").exists()


def test_snapshot_rounds_must_increase(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    tracker.add_snapshot(0, 2, {0: torch.ones(3)}, {0: 1})
    with pytest.raises(ValueError):
        tracker.add_snapshot(0, 2, {0: torch.ones(3)}, {0: 1})


def test_unattested_snapshot_file_is_rejected_on_reload(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    tracker.assess_round(1, {0: _evidence()}, LABELS)
    path = tmp_path / "snapshots" / "round_1" / "client_0.pt"
    payload = torch.load(path, weights_only=True)
    payload["provenance"]["source_kind"] = "final_global_model"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        ReliabilityTracker(tmp_path).load_from_disk()


def test_stability_window_limits_history(tmp_path):
    tracker = ReliabilityTracker(None, stability_window=2)
    for rnd in range(1, 5):
        tracker.add_snapshot(0, rnd, {0: torch.ones(3) * rnd}, {0: 1})
    assert len(tracker._windowed(0)) == 2


def test_global_assessment_uses_all_clients_and_local_stability(tmp_path):
    tracker = ReliabilityTracker(tmp_path)
    for rnd in (1, 2):
        tracker.assess_round(rnd, {0: _evidence(), 1: _evidence()}, LABELS)
    protos = {0: _evidence()["prototypes"], 1: _evidence()["prototypes"], 2: _evidence()["prototypes"]}
    counts = {c: {0: 4, 1: 6} for c in protos}
    rows = tracker.assess_global_prototypes(2, protos, counts, _evidence()["val_embeddings"], LABELS)
    by = {(r["client_id"], r["class_id"]): r for r in rows}
    assert by[(0, 0)]["complete"] and not by[(2, 0)]["complete"]  # client 2 never trained: no stability
    assert by[(2, 0)]["stability"]["status"] == "unavailable"


def test_local_evidence_shapes_with_tiny_model():
    model = FLAMEModel(feature_dim=16, embedding_dim=8, num_classes=9, encoder="small_cnn")
    images = torch.randn(6, 3, 28, 28)
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    ds = torch.utils.data.TensorDataset(images, labels.view(-1, 1))
    ev = local_evidence(model, ds, torch.randn(10, 3, 28, 28))
    assert set(ev["prototypes"]) == {0, 1, 2} and ev["counts"] == {0: 2, 1: 3, 2: 1}
    assert ev["val_embeddings"].shape == (10, 8) and model.training  # mode restored
