"""evaluate_original() on the REAL Phase 4 prototypes and PRAM evidence.

The real 199 MB Phase 3 model is not in the repository, so a randomly initialised model with
the identical architecture stands in: the numbers are meaningless, the code path (strict state
dict load, Phase 4 loading, FLAME rule, post-hoc PRAA rule, metrics, test guard) is exercised.
"""

import os
from pathlib import Path

import pytest
import torch

import training.prototype_store as store
from federated.praa import PRAAConfig
from models.classifier import FLAMEModel
from run_phase8 import evaluate_original

ROOT = Path(__file__).resolve().parent.parent
os.environ["PRAA_SYNTHETIC_DATA"] = "1"


@pytest.fixture(scope="module")
def fake_source():
    torch.manual_seed(0)
    model = FLAMEModel(embedding_dim=128, num_classes=9)
    return {"config": {"embedding_dim": 128, "num_classes": 9, "batch_size": 64},
            "state_dict": model.state_dict(),
            "provenance": {"source_model_sha256": "f" * 64, "source_round": 200}}


@pytest.fixture
def patched(monkeypatch, fake_source):
    monkeypatch.setattr(store, "validate_phase3_artifacts", lambda *a, **k: fake_source)


def _run(split, policy, final=False):
    return evaluate_original(ROOT / "results" / "phase3", ROOT / "results" / "phase4",
                             ROOT / "results" / "phase5" / "pram.json", split, final, "cpu",
                             PRAAConfig(missing_component_policy=policy))


def test_strict_posthoc_praa_is_identical_to_flame(patched):
    out = _run("val", "strict")
    assert out["praa_posthoc_identical_to_flame"] is True
    assert out["rules"]["flame_rule"]["accuracy"] == out["rules"]["praa_posthoc_rule"]["accuracy"]
    assert out["test_set_accessed"] is False and out["reliability"]["records_with_prs"] == 0
    assert out["rules"]["flame_rule"]["missing_prototype_classes"] == []  # all 9 classes have prototypes


def test_renormalize_extension_changes_prototypes_and_is_labelled(patched):
    out = _run("val", "renormalize")
    assert out["praa_posthoc_identical_to_flame"] is False
    assert "EXTENSION" in out["praa_posthoc_note"]
    assert out["reliability"]["records_with_prs"] == 115 and out["reliability"]["classes_reweighted"] > 0
    for rule in out["rules"].values():
        assert 0.0 <= rule["accuracy"] <= 1.0 and rule["num_samples"] == 500


def test_test_split_blocked_without_final(patched):
    with pytest.raises(PermissionError):
        _run("test", "strict", final=False)
    assert _run("test", "strict", final=True)["test_set_accessed"] is True
