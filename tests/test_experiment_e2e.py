"""Small end-to-end runs (synthetic PathMNIST-shaped data, tiny CNN, CPU): FedAvg, FLAME and
PRAA training, evaluation and comparison.  Marked slow-ish (about one minute)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from federated.experiment import run_experiment
from federated.experiment_config import load_config
from run_phase8 import build_comparison, evaluate_run, write_comparison

ROOT = Path(__file__).resolve().parent.parent
os.environ["PRAA_SYNTHETIC_DATA"] = "1"


def _cfg(method, rounds=3, **extra):
    overrides = {"method": method, "num_rounds": rounds, "checkpoint_every": 1, **extra}
    return load_config(str(ROOT / "configs" / "smoke_synthetic.json"), overrides)


def _quiet(*_a, **_k):
    pass


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    out = {}
    for method in ("praa", "flame", "fedavg"):
        out[method] = base / method
        run_experiment(_cfg(method), out[method], device="cpu", log=_quiet)
    return out


def test_runs_write_expected_artifacts(runs):
    for method, path in runs.items():
        assert (path / "final_model.pt").exists() and (path / "metrics.json").exists()
        assert json.loads((path / "config.json").read_text())["method"] == method
    assert (runs["praa"] / "reliability_log.jsonl").exists() and (runs["praa"] / "snapshots").exists()
    assert not (runs["fedavg"] / "snapshots").exists()  # FedAvg has no prototype evidence


def test_flame_run_uses_size_weights_and_praa_round1_equals_flame(runs):
    flame = json.loads((runs["flame"] / "metrics.json").read_text())
    praa = json.loads((runs["praa"] / "metrics.json").read_text())
    assert all(h["weights"]["used"] == "flame" for h in flame)
    assert all(h["weights"]["used"] == "praa" for h in praa)
    # round 1: no evidence -> PRAA weights are exactly the FLAME weights
    assert praa[0]["weights"]["num_assessed"] == 0 and praa[0]["weights"]["max_weight_shift"] == 0.0
    # PRAA and FLAME have identical round-1 client selection and local training
    assert flame[0]["selected_clients"] == praa[0]["selected_clients"]
    for a, b in zip(flame[0]["client_metrics"], praa[0]["client_metrics"]):
        assert a["mae_loss"] == pytest.approx(b["mae_loss"])
    total = sum(1 for _ in flame[0]["weights"]["flame"])
    assert sum(flame[0]["weights"]["flame"].values()) == pytest.approx(1.0) and total == 5


def test_praa_evidence_accumulates_and_weights_stay_valid(runs):
    praa = json.loads((runs["praa"] / "metrics.json").read_text())
    assert praa[-1]["validation"]["reliability"]["records_with_prs"] > 0
    for h in praa:
        assert sum(h["weights"]["praa"].values()) == pytest.approx(1.0)
        assert all(m > 0 for m in h["weights"]["multipliers"].values())


def test_history_has_both_inference_rules_and_scalar_metrics(runs):
    h = json.loads((runs["praa"] / "metrics.json").read_text())[-1]
    for rule in ("flame_rule", "praa_rule"):
        assert {"accuracy", "balanced_accuracy", "macro_f1", "auroc_macro_ovr", "nll"} <= set(h["validation"][rule])
    assert 0.0 <= h["validation_accuracy"] <= 1.0 and h["validation_loss"] > 0


def test_resume_reproduces_uninterrupted_run(runs, tmp_path):
    part = tmp_path / "resumed"
    run_experiment(_cfg("praa", rounds=2), part, device="cpu", log=_quiet)
    run_experiment(_cfg("praa", rounds=3), part, device="cpu", resume=True, log=_quiet)
    a = torch.load(runs["praa"] / "final_model.pt", weights_only=False)["model_state_dict"]
    b = torch.load(part / "final_model.pt", weights_only=False)["model_state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    ha = json.loads((runs["praa"] / "metrics.json").read_text())
    hb = json.loads((part / "metrics.json").read_text())
    assert [h["validation_accuracy"] for h in ha] == [h["validation_accuracy"] for h in hb]


def test_existing_finished_run_is_not_overwritten(runs):
    with pytest.raises(FileExistsError):
        run_experiment(_cfg("praa"), runs["praa"], device="cpu", log=_quiet)


def test_training_never_reads_the_test_split(tmp_path, monkeypatch):
    from datasets import preprocess

    class Poison:
        split = "test"
        labels = None

        def __len__(self):
            raise AssertionError("test split was accessed")

        def __getitem__(self, i):
            raise AssertionError("test split was accessed")

    monkeypatch.setattr(preprocess, "test_dataset", Poison())
    run_experiment(_cfg("praa", rounds=2), tmp_path / "poison", device="cpu", log=_quiet)


def test_evaluate_val_then_test_requires_final_flag_and_compare(runs, tmp_path):
    results = {}
    for method, path in runs.items():
        ev = evaluate_run(path, "val", final=False, device="cpu")
        assert ev["test_set_accessed"] is False
        (path / "eval_val.json").write_text(json.dumps(ev, allow_nan=False))
        results[method] = ev
    with pytest.raises(PermissionError):
        evaluate_run(runs["praa"], "test", final=False, device="cpu")
    test_ev = evaluate_run(runs["praa"], "test", final=True, device="cpu")
    assert test_ev["test_set_accessed"] is True and test_ev["split"] == "test"
    assert set(results["praa"]["rules"]) == {"flame_rule", "praa_rule"}
    assert set(results["fedavg"]["rules"]) == {"classifier_head"}
    histories = {m: json.loads((p / "metrics.json").read_text()) for m, p in runs.items()}
    doc = build_comparison(results, histories, "val")
    write_comparison(doc, tmp_path / "cmp", histories)
    for name in ("comparison.json", "comparison.md", "comparison.csv", "convergence.png"):
        assert (tmp_path / "cmp" / name).exists()
    # metrics cover accuracy, macro-F1, AUROC and convergence
    row = doc["rows"][0]
    assert all(row[k] is not None for k in ("accuracy", "macro_f1", "auroc_macro_ovr"))
    assert "praa" in doc["convergence_validation_accuracy"]


def test_cli_smoke_via_run_phase7(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "run_phase7.py"), "--smoke", "--rounds", "2",
         "--output-dir", str(tmp_path / "cli")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    assert (tmp_path / "cli" / "final_model.pt").exists()


def test_fresh_run_does_not_append_to_a_stale_reliability_log(tmp_path):
    out = tmp_path / "stale"
    out.mkdir()
    (out / "reliability_log.jsonl").write_text(json.dumps({"round": 99, "plan": {}}) + "\n")
    run_experiment(_cfg("praa", rounds=1), out, device="cpu", log=_quiet)
    lines = (out / "reliability_log.jsonl").read_text().splitlines()
    assert [json.loads(line)["round"] for line in lines] == [1]
