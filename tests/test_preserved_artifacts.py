"""The original Phase 3-5 evidence is preserved byte-for-byte and still readable."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from federated.pram import load_pram_output
from models.classifier import FLAMEModel, model_kwargs_from_config
from run_phase6 import build_prs_document
from federated.praa import PRAAConfig
from training.prototype_store import load_client_prototype_file

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def test_preserved_files_unchanged():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_preserved.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_original_phase3_run_is_the_200_round_paper_config():
    cfg = json.loads((RES / "phase3" / "config.json").read_text())
    history = json.loads((RES / "phase3" / "metrics.json").read_text())
    assert cfg["num_rounds"] == 200 and cfg["num_clients"] == 50 and cfg["clients_per_round"] == 10
    assert cfg["mask_ratio"] == 0.8 and cfg["mae_weight"] == 0.7 and cfg["label_fraction"] == 0.01
    assert [h["round"] for h in history] == list(range(1, 201))


def test_phase4_prototypes_load_for_all_50_clients():
    files = sorted((RES / "phase4" / "prototypes").glob("client_*.pt"))
    assert len(files) == 50
    record = load_client_prototype_file(files[0])
    assert record["embedding_dim"] == 128 and record["provenance"]["source_round"] == 200


def test_phase5_pram_evidence_has_no_stability_and_phase6_strict_gives_no_prs():
    pram = load_pram_output(RES / "phase5" / "pram.json")
    assert pram["summary"]["complete_records"] == 0
    assert pram["summary"]["status_counts"]["stability"]["unavailable"] == 126
    doc = build_prs_document(pram, PRAAConfig())
    assert doc["status"] == "NO_PRS_COMPUTABLE"
    assert all(abs(c["praa_weight"] - c["flame_weight"]) < 1e-12 for c in doc["clients"].values())


def test_partial_extension_is_exploratory_and_labelled():
    pram = load_pram_output(RES / "phase5" / "pram.json")
    doc = build_prs_document(pram, PRAAConfig(missing_component_policy="renormalize"))
    assert doc["extension_notice"] and doc["summary"]["records_with_prs"] == 115
    assert doc["summary"]["clients_assessed"] == 39


def test_default_model_matches_preserved_checkpoint_layout():
    """The original checkpoint is ~199 MB (52.09M parameters); the default model must not change."""
    cfg = json.loads((RES / "phase3" / "config.json").read_text())
    assert model_kwargs_from_config(cfg) == {"embedding_dim": 128, "num_classes": 9}
    model = FLAMEModel(**model_kwargs_from_config(cfg))
    assert sum(p.numel() for p in model.parameters()) == 52_090_828
    keys = list(model.state_dict())
    assert keys[0] == "encoder.backbone.conv1.weight" and keys[-1] == "classifier.bias" and len(keys) == 332
