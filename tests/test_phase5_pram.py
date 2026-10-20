"""PRAM tests. Synthetic fixtures only: no PathMNIST download, no Phase 3 rerun."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import run_phase4
import run_phase5
from federated import pram
from federated.prs import validation_scores_from_embeddings
from training.prototype_store import build_client_record, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def t(*values):
    return torch.tensor(values, dtype=torch.float32)


def record(client_id, protos):
    """Real Phase 4 record built from per-class prototype vectors (1 sample each)."""
    embeddings = torch.stack([v for v in protos.values()])
    labels = torch.tensor(list(protos))
    return build_client_record(client_id, embeddings, labels, 9, len(labels), "all_labels", {}, {})


# --------------------------------------------------------------------------
# 1. Validation performance
# --------------------------------------------------------------------------

def test_validation_recall_hand_example_and_matches_legacy_function():
    protos = {0: t(0.0, 0.0), 1: t(10.0, 0.0)}
    embeddings = torch.tensor([[1.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
    labels = torch.tensor([0, 0, 1])
    out = pram.validation_components(embeddings, labels, protos)
    assert out[0]["status"] == "valid" and out[0]["value"] == pytest.approx(0.5)
    assert out[1]["value"] == pytest.approx(1.0)
    assert out[0]["details"]["chance_level"] == 0.5 and out[0]["details"]["num_eval_samples"] == 2
    legacy = validation_scores_from_embeddings([embeddings], [labels], [0, 1], protos)
    assert legacy[0] == pytest.approx(out[0]["value"]) and legacy[1] == pytest.approx(out[1]["value"])


def test_single_candidate_class_is_degenerate_not_a_valid_score():
    out = pram.validation_components(torch.randn(5, 2), torch.zeros(5, dtype=torch.long), {0: t(0.0, 0.0)})
    assert out[0]["status"] == "degenerate" and out[0]["value"] is None and out[0]["raw"] == 1.0


def test_class_without_validation_samples_is_unavailable():
    out = pram.validation_components(torch.randn(4, 2), torch.zeros(4, dtype=torch.long),
                                     {0: t(0.0, 0.0), 3: t(1.0, 1.0)})
    assert out[3]["status"] == "unavailable" and out[3]["value"] is None and out[3]["raw"] is None


def test_validation_rejects_dimension_mismatch_and_nonfinite():
    with pytest.raises(ValueError):
        pram.validation_components(torch.randn(4, 3), torch.zeros(4, dtype=torch.long), {0: t(0.0, 0.0)})
    with pytest.raises(FloatingPointError):
        pram.validation_components(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, dtype=torch.long),
                                   {0: t(0.0, 0.0)})


# --------------------------------------------------------------------------
# 2-5. Prototype consistency
# --------------------------------------------------------------------------

def test_consistency_matches_hand_calculation():
    records = {0: record(0, {0: t(1.0, 0.0)}), 1: record(1, {0: t(0.0, 1.0)}), 2: record(2, {0: t(1.0, 0.0)})}
    reference = pram.build_reference_prototypes(records)
    assert torch.allclose(reference[0]["vector"], t(2 / 3, 1 / 3))
    out = pram.consistency_component(0, 0, t(1.0, 0.0), reference)
    assert out["status"] == "valid"
    assert out["value"] == pytest.approx((2 / np.sqrt(5) + 1) / 2)          # 0.947214
    assert out["details"]["num_reference_contributors"] == 3
    assert out["details"]["leave_one_out_diagnostic"] == pytest.approx((np.sqrt(0.5) + 1) / 2)  # 0.853553
    opposite = pram.consistency_component(1, 0, t(-2 / 3, -1 / 3), reference)
    assert opposite["status"] == "valid" and opposite["value"] == pytest.approx(0.0, abs=1e-6)


def test_single_contributor_reference_is_degenerate():
    records = {0: record(0, {0: t(1.0, 0.0), 5: t(0.0, 1.0)}), 1: record(1, {0: t(0.9, 0.1)})}
    reference = pram.build_reference_prototypes(records)
    out = pram.consistency_component(0, 5, t(0.0, 1.0), reference)
    assert out["status"] == "degenerate" and out["value"] is None and out["raw"] == pytest.approx(1.0)
    assert out["reason"] == "reference_built_only_from_this_client"
    assert reference[0]["contributors"] == [0, 1] and reference[5]["contributors"] == [0]


def test_missing_classes_are_explicit_in_rows():
    records = {0: record(0, {0: t(1.0, 0.0), 1: t(0.0, 1.0)}), 1: record(1, {1: t(0.0, 1.0)})}
    reference = pram.build_reference_prototypes(records)
    rows = pram.assess_client(1, records[1], reference, torch.randn(6, 2),
                              torch.tensor([0, 0, 1, 1, 1, 1]), [], 200)
    assert [r["class_id"] for r in rows] == [1]  # no row is invented for classes client 1 lacks


def test_dimension_mismatch_and_malformed_vectors_rejected():
    mixed = {0: record(0, {0: t(1.0, 0.0)}), 1: record(1, {0: t(1.0, 0.0, 0.0)})}
    with pytest.raises(ValueError):
        pram.build_reference_prototypes(mixed)
    reference = pram.build_reference_prototypes({0: record(0, {0: t(1.0, 0.0)}), 1: record(1, {0: t(0.0, 1.0)})})
    with pytest.raises(ValueError):
        pram.consistency_component(0, 0, t(1.0, 0.0, 0.0), reference)
    with pytest.raises(FloatingPointError):
        pram.consistency_component(0, 0, t(float("nan"), 0.0), reference)
    with pytest.raises(ValueError):
        pram.consistency_component(0, 0, torch.ones(2, 2), reference)
    zero = pram.consistency_component(0, 0, t(0.0, 0.0), reference)
    assert zero["status"] == "unavailable" and zero["value"] is None


# --------------------------------------------------------------------------
# 6-8. Stability, attribution, missing evidence vs low score
# --------------------------------------------------------------------------

def attested(client_id, round_id, protos, **overrides):
    payload = {"client_id": client_id, "round": round_id, "prototypes": protos,
               "provenance": {"source_kind": pram.ATTESTED_SNAPSHOT_KIND}}
    payload.update(overrides)
    return payload


def write_snapshot(root, round_id, client_id, payload):
    folder = root / "snapshots" / f"round_{round_id}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(payload, folder / f"client_{client_id}.pt")


def test_no_or_single_snapshot_means_unavailable_not_a_number():
    for snapshots in ([], [(10, {0: t(1.0, 0.0)})]):
        out = pram.stability_components(snapshots, [0])
        assert out[0]["status"] == "unavailable" and out[0]["value"] is None and out[0]["raw"] is None
        assert out[0]["reason"] == "fewer_than_two_attested_client_snapshots"


def test_stability_hand_calculation_and_round_gaps():
    snaps = [(5, {0: t(1.0, 0.0)}), (9, {0: t(0.0, 1.0)}), (20, {0: t(0.0, 1.0)})]
    out = pram.stability_components(snaps, [0])[0]
    assert out["status"] == "valid" and out["value"] == pytest.approx((0.5 + 1.0) / 2)
    assert out["details"]["round_gaps"] == [4, 11] and out["details"]["num_pairs"] == 2


def test_low_score_is_valid_while_missing_evidence_is_not():
    low = pram.stability_components([(1, {0: t(1.0, 0.0)}), (2, {0: t(-1.0, 0.0)})], [0])[0]
    missing = pram.stability_components([], [0])[0]
    assert low["status"] == "valid" and low["value"] == pytest.approx(0.0)
    assert missing["status"] == "unavailable" and missing["value"] is None
    with pytest.raises(ValueError):
        pram.component("unavailable", value=0.0)
    with pytest.raises(ValueError):
        pram.component("valid")


def test_class_absent_from_snapshots_is_unavailable():
    snaps = [(1, {0: t(1.0, 0.0)}), (2, {0: t(1.0, 0.0)})]
    assert pram.stability_components(snaps, [4])[4]["reason"] == "class_present_in_fewer_than_two_snapshots"


def test_snapshot_attribution_and_attestation(tmp_path):
    protos = {0: t(1.0, 0.0)}
    write_snapshot(tmp_path, 1, 3, attested(3, 1, protos))
    write_snapshot(tmp_path, 2, 3, attested(3, 2, protos))
    write_snapshot(tmp_path, 3, 3, attested(7, 3, protos))                    # wrong client
    write_snapshot(tmp_path, 4, 3, attested(3, 99, protos))                   # wrong round
    write_snapshot(tmp_path, 5, 3, {"client_id": 3, "round": 5, "prototypes": protos})  # legacy, unattested
    write_snapshot(tmp_path, 6, 3, attested(3, 6, protos, provenance={"source_kind": "global_final_model"}))
    accepted, rejected = pram.discover_client_snapshots(tmp_path, 3)
    assert [r for r, _ in accepted] == [1, 2]
    reasons = {Path(p).parent.name: why for p, why in rejected}
    assert reasons["round_3"] == "client_id_mismatch"
    assert reasons["round_4"] == "round_mismatch"
    assert reasons["round_5"] == "unattested_source_kind" and reasons["round_6"] == "unattested_source_kind"
    assert pram.discover_client_snapshots(tmp_path, 8) == ([], [])             # no other client's files used


def test_history_inspection_reports_participation_not_instability():
    history = [
        {"round": 1, "selected_clients": [0, 1], "client_metrics": [{"client_id": 0, "mae_loss": 1.0},
                                                                    {"client_id": 1, "mae_loss": 2.0}]},
        {"round": 2, "selected_clients": [1, 2], "client_metrics": [{"client_id": 1, "mae_loss": 1.0}]},
    ]
    info = pram.inspect_training_history(history, 4)
    assert info["rounds_in_history"] == 2 and info["clients_never_selected"] == [3]
    assert info["rounds_with_selection_mismatch"] == [2]
    assert info["contains_client_prototype_or_model_snapshots"] is False


# --------------------------------------------------------------------------
# 9-12. Pipeline fixtures: serialization, determinism, isolation, compatibility
# --------------------------------------------------------------------------

class ValDataset(TensorDataset):
    split = "val"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from models.classifier import FLAMEModel

    root = tmp_path_factory.mktemp("world")
    torch.manual_seed(0)
    state = {k: v.detach().clone() for k, v in FLAMEModel(embedding_dim=128, num_classes=9).state_dict().items()}
    history = [{"round": r, "selected_clients": [0, 1], "client_metrics": [
        {"client_id": 0, "mae_loss": 1.0}, {"client_id": 1, "mae_loss": 1.0}]} for r in range(1, 4)]
    p3 = root / "phase3"
    (p3 / "checkpoints").mkdir(parents=True)
    json.dump(dict(run_phase4.expected_phase3_config(), num_rounds=3), open(p3 / "config.json", "w"))
    torch.save({"model_state_dict": state, "history": history}, p3 / "final_model.pt")
    torch.save({"round": 3, "global_model_state": state, "history": history, "client_states": {}},
               p3 / "checkpoints" / "checkpoint_round_3.pt")
    rng = np.random.RandomState(0)
    class_sets = {0: [0, 1, 2], 1: [1, 2, 3], 2: [2, 3, 4]}
    clients = {}
    for k, classes in class_sets.items():
        n = 18
        clients[k] = TensorDataset(torch.tensor(rng.randn(n, 3, 28, 28), dtype=torch.float32),
                                   torch.tensor(np.resize(classes, n)).view(-1, 1))
    p4 = root / "phase4"
    run_phase4.run_phase4(p3, p4, clients, "partition-hash", mode="smoke", label_policy="all_labels")
    validation = ValDataset(torch.tensor(rng.randn(60, 3, 28, 28), dtype=torch.float32),
                            torch.tensor(np.resize(np.arange(5), 60)).view(-1, 1))
    return {"root": root, "p3": p3, "p4": p4, "validation": validation}


def run(world, out, **kwargs):
    return run_phase5.run_pram(world["p3"], kwargs.pop("p4", world["p4"]), out, world["validation"],
                               mode="smoke", **kwargs)


def test_pipeline_schema_status_and_phase6_readiness(world, tmp_path):
    doc = run(world, tmp_path / "out")
    assert doc["status"] == "PARTIAL_EVIDENCE_MISSING"
    assert len(doc["records"]) == 9 and doc["evidence"]["client_snapshot_files_accepted"] == 0
    assert all(r["stability"]["status"] == "unavailable" and r["stability"]["value"] is None
               for r in doc["records"])
    assert all(r["round_id"] == 3 for r in doc["records"])
    ready = doc["phase6_readiness"]
    assert ready["prs_computable_for_all_records"] is False and ready["records_with_all_three_valid"] == 0
    assert any("stability" in reason for reason in ready["blocking_reasons"])
    by_key = {(r["client_id"], r["class_id"]): r for r in doc["records"]}
    assert by_key[(0, 0)]["consistency"]["status"] == "degenerate"          # only client 0 holds class 0
    assert by_key[(1, 2)]["consistency"]["status"] == "valid"               # classes held by 3 clients
    assert by_key[(1, 2)]["validation"]["status"] == "valid"
    assert doc["provenance"]["phase3_model_sha256"] == sha256_file(world["p3"] / "final_model.pt")
    assert doc["provenance"]["test_set_accessed"] is False


def test_serialization_round_trip_atomic_and_overwrite_guard(world, tmp_path):
    out = tmp_path / "out"
    doc = run(world, out)
    path = out / "pram.json"
    assert not list(out.glob("*.tmp"))
    assert pram.load_pram_output(path) == json.loads(json.dumps(doc))
    with pytest.raises(FileExistsError):
        run(world, out)
    run(world, out, overwrite=True)


def test_results_are_deterministic(world, tmp_path):
    run(world, tmp_path / "a")
    run(world, tmp_path / "b")
    assert (tmp_path / "a" / "pram.json").read_text() == (tmp_path / "b" / "pram.json").read_text()


def test_phase4_inputs_are_read_only(world, tmp_path):
    files = sorted((world["p4"] / "prototypes").glob("*.pt")) + [world["p4"] / "summary.json"]
    before = [sha256_file(f) for f in files]
    run(world, tmp_path / "out")
    assert [sha256_file(f) for f in files] == before


def test_attested_snapshots_enable_stability_for_that_client_only(world, tmp_path):
    p4 = tmp_path / "p4"
    shutil.copytree(world["p4"], p4)
    base = torch.load(p4 / "prototypes" / "client_1.pt", weights_only=True)["prototypes"]
    write_snapshot(p4, 1, 1, attested(1, 1, base))
    write_snapshot(p4, 2, 1, attested(1, 2, base))
    doc = run(world, tmp_path / "out", p4=p4)
    stability = {(r["client_id"], r["class_id"]): r["stability"] for r in doc["records"]}
    assert all(stability[(1, c)]["status"] == "valid" and stability[(1, c)]["value"] == pytest.approx(1.0)
               for c in (1, 2, 3))
    assert all(stability[(0, c)]["status"] == "unavailable" for c in (0, 1, 2))
    assert doc["phase6_readiness"]["records_with_all_three_valid"] == 3


def test_unattested_snapshots_are_rejected_and_reported(world, tmp_path):
    p4 = tmp_path / "p4"
    shutil.copytree(world["p4"], p4)
    base = torch.load(p4 / "prototypes" / "client_1.pt", weights_only=True)["prototypes"]
    for r in (1, 2):
        write_snapshot(p4, r, 1, {"client_id": 1, "round": r, "prototypes": base})
    doc = run(world, tmp_path / "out", p4=p4)
    assert doc["evidence"]["client_snapshot_files_rejected"] == 2
    assert doc["evidence"]["client_snapshot_rejection_reasons"] == {"unattested_source_kind": 2}
    assert all(r["stability"]["status"] == "unavailable" for r in doc["records"])


def test_provenance_mismatch_and_legacy_files_rejected(world, tmp_path):
    p4 = tmp_path / "p4"
    shutil.copytree(world["p4"], p4)
    path = p4 / "prototypes" / "client_0.pt"
    payload = torch.load(path, weights_only=True)
    payload["provenance"]["source_model_sha256"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError):
        run(world, tmp_path / "o1", p4=p4)                                  # records disagree with each other
    for other in (1, 2):
        other_path = p4 / "prototypes" / f"client_{other}.pt"
        other_payload = torch.load(other_path, weights_only=True)
        other_payload["provenance"]["source_model_sha256"] = "0" * 64
        torch.save(other_payload, other_path)
    with pytest.raises(ValueError):
        run(world, tmp_path / "o2", p4=p4)                                  # consistent but not this Phase 3 model
    legacy = tmp_path / "legacy"
    (legacy / "prototypes").mkdir(parents=True)
    shutil.copy(world["p4"] / "summary.json", legacy / "summary.json")
    torch.save({"client_id": 0, "prototypes": {0: t(1.0, 0.0)}}, legacy / "prototypes" / "client_0.pt")
    with pytest.raises(ValueError):
        run(world, tmp_path / "o3", p4=legacy)


def test_final_mode_requires_complete_final_phase4(world, tmp_path):
    with pytest.raises(ValueError):
        run_phase5.run_pram(world["p3"], world["p4"], tmp_path / "out", world["validation"], mode="final")


def test_schema_validator_rejects_tampering(world, tmp_path):
    doc = run(world, tmp_path / "out")
    bad = copy.deepcopy(doc)
    bad["records"][0]["stability"]["value"] = 0.0                            # number on unavailable component
    with pytest.raises(ValueError):
        pram.validate_output(bad)
    bad = copy.deepcopy(doc)
    bad["records"][0]["validation"]["value"] = float("nan")
    with pytest.raises(ValueError):
        pram.validate_output(bad)
    bad = copy.deepcopy(doc)
    bad["records"][1]["complete"] = True
    with pytest.raises(ValueError):
        pram.validate_output(bad)
    bad = copy.deepcopy(doc)
    bad["summary"]["complete_records"] += 1
    with pytest.raises(ValueError):
        pram.validate_output(bad)
    bad = copy.deepcopy(doc)
    bad["phase6_readiness"]["prs_computable_for_all_records"] = True
    with pytest.raises(ValueError):
        pram.validate_output(bad)
    with pytest.raises(ValueError):
        pram.write_json_atomic(bad, tmp_path / "never.json")
    assert not (tmp_path / "never.json").exists()


def test_validation_split_guard_and_no_test_references():
    class Split(TensorDataset):
        pass

    data = Split(torch.zeros(3, 2), torch.zeros(3, 1))
    for bad in ("test", "train"):
        data.split = bad
        with pytest.raises(PermissionError):
            run_phase5.assert_validation_split(torch.utils.data.Subset(data, [0, 1]))
    data.split = "val"
    run_phase5.assert_validation_split(torch.utils.data.Subset(data, [0, 1]))
    for relative in ("run_phase5.py", "federated/pram.py"):
        text = (ROOT / relative).read_text()
        for forbidden in ("test_dataset", "test_loader", "split=\"test\""):
            assert forbidden not in text, f"{relative} references {forbidden}"


def test_legacy_prs_module_unchanged_and_importable():
    from federated.prs import PRSConfig, compute_prs, stability_scores  # noqa: F401
    assert stability_scores(None, [0]) is None


# --------------------------------------------------------------------------
# audit_phase5.py (read-only auditor) on fixtures
# --------------------------------------------------------------------------

def test_audit_script_passes_on_fixture_and_detects_tampering(world, tmp_path):
    import audit_phase5

    out = tmp_path / "out"
    run(world, out)
    clean = audit_phase5.audit(world["p3"], world["p4"], out, "smoke", [0, 1], world["validation"])
    assert not clean.failed, clean.failed
    labels = {r[1] for r in clean.results if r[0] == "WARN"}
    assert "stability unavailable for all records" in labels

    path = out / "pram.json"
    doc = json.loads(path.read_text())
    victim = next(r for r in doc["records"] if r["consistency"]["status"] == "valid")
    victim["consistency"]["value"] = victim["consistency"]["raw"] = 0.123    # plausible but wrong
    doc["records"][0]["validation"]["raw"] = doc["records"][0]["validation"]["value"] = 0.987
    path.write_text(json.dumps(doc))
    tampered = audit_phase5.audit(world["p3"], world["p4"], out, "smoke", [0], world["validation"])
    failed = {r[1] for r in tampered.failed}
    assert "consistency recomputed independently for all records" in failed
    assert any("validation recomputed" in name for name in failed)

    doc = json.loads((tmp_path / "out" / "pram.json").read_text())
    doc["provenance"]["phase3_model_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    assert "recorded Phase 3 model hash equals actual file" in {
        r[1] for r in audit_phase5.audit(world["p3"], world["p4"], out, "smoke").failed}
