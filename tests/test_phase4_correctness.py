"""Phase 4 correctness tests. Fixture-only: no PathMNIST download, no Phase 3 rerun."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Subset, TensorDataset

import run_phase4
from training.prototype_store import (
    SCHEMA_VERSION,
    assert_train_split,
    build_client_record,
    generate_client_record,
    load_client_prototype_file,
    partition_fingerprint,
    save_client_record,
    select_labeled_positions,
    sha256_file,
    to_dense,
    validate_phase3_artifacts,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 9
PROVENANCE = {"source_kind": "fixture"}
CONFIG = {"mode": "test"}


class Echo(torch.nn.Module):
    """Stand-in 'FLAME' model whose embedding is the first two input values."""

    def forward(self, images):
        flat = images.flatten(1)[:, :2]
        return flat, flat


def make_dataset(rows, labels):
    return TensorDataset(
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long).view(-1, 1),
    )


def make_record(embeddings, labels, client_id=0, num_local=None):
    return build_client_record(
        client_id, torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels),
        NUM_CLASSES, num_local or len(labels), "all_labels", PROVENANCE, CONFIG,
    )


# --------------------------------------------------------------------------
# Dimensions, numerical validity, class-wise means
# --------------------------------------------------------------------------

def test_record_dimensions_and_validity():
    rng = np.random.RandomState(0)
    record = make_record(rng.randn(30, 128), rng.randint(0, 5, 30))
    k = len(record["class_ids"])
    assert record["prototype_vectors"].shape == (k, 128)
    assert record["embedding_dim"] == 128
    assert record["sample_counts"].sum().item() == 30
    assert torch.isfinite(record["prototype_vectors"]).all()
    assert record["schema_version"] == SCHEMA_VERSION


def test_prototypes_equal_numpy_class_means_and_counts():
    rng = np.random.RandomState(1)
    embeddings = rng.randn(200, 16).astype(np.float32)
    labels = rng.randint(0, 7, 200)
    record = make_record(embeddings, labels)
    for i, class_id in enumerate(record["class_ids"].tolist()):
        expected = embeddings[labels == class_id].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(record["prototype_vectors"][i].numpy(), expected, rtol=1e-5, atol=1e-6)
        assert record["sample_counts"][i].item() == int((labels == class_id).sum())


def test_non_finite_and_out_of_range_labels_rejected():
    with pytest.raises(FloatingPointError):
        make_record([[float("nan"), 0.0]], [0])
    with pytest.raises(ValueError):
        make_record([[1.0, 1.0]], [NUM_CLASSES])
    with pytest.raises(ValueError):
        make_record([[1.0, 1.0]], [-1])


# --------------------------------------------------------------------------
# Missing classes and small partitions
# --------------------------------------------------------------------------

def test_absent_classes_have_no_prototype_and_dense_mask_marks_them():
    record = make_record([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]], [2, 2, 5])
    assert record["class_ids"].tolist() == [2, 5]
    assert record["absent_class_ids"] == [0, 1, 3, 4, 6, 7, 8]
    dense, mask = to_dense(record)
    assert dense.shape == (NUM_CLASSES, 2)
    assert mask.tolist() == [c in (2, 5) for c in range(NUM_CLASSES)]
    assert torch.equal(dense[2], torch.tensor([2.0, 3.0]))


def test_single_sample_client_is_valid():
    record = make_record([[0.5, -0.5]], [4])
    assert record["class_ids"].tolist() == [4]
    assert record["sample_counts"].tolist() == [1]


def test_minimum_size_client_uses_at_least_two_labeled_samples():
    positions = select_labeled_positions(10, client_id=3, seed=42, label_fraction=0.01)
    assert len(positions) == 2 and len(set(positions)) == 2
    assert select_labeled_positions(1, 0, 42, 0.01) == [0]  # cannot exceed the partition


def test_empty_client_rejected():
    empty = TensorDataset(torch.empty(0, 3), torch.empty(0, 1, dtype=torch.long))
    with pytest.raises(ValueError):
        generate_client_record(
            Echo(), empty, 0, num_classes=NUM_CLASSES, label_policy="all_labels", seed=42,
            label_fraction=0.01, batch_size=4, device="cpu", provenance=PROVENANCE, config=CONFIG,
        )


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def _echo_dataset(n=40, seed=0):
    rng = np.random.RandomState(seed)
    return make_dataset(rng.randn(n, 3), rng.randint(0, 6, n))


def _generate(dataset, policy, client_id=7):
    return generate_client_record(
        Echo(), dataset, client_id, num_classes=NUM_CLASSES, label_policy=policy, seed=42,
        label_fraction=0.25, batch_size=8, device="cpu", provenance=PROVENANCE, config=CONFIG,
    )


@pytest.mark.parametrize("policy", ["labeled_subset", "all_labels"])
def test_generation_is_deterministic(policy):
    first, second = _generate(_echo_dataset(), policy), _generate(_echo_dataset(), policy)
    assert torch.equal(first["class_ids"], second["class_ids"])
    assert torch.equal(first["prototype_vectors"], second["prototype_vectors"])
    assert torch.equal(first["sample_counts"], second["sample_counts"])


def test_labeled_subset_policy_uses_fraction_and_differs_per_client():
    dataset = _echo_dataset(40)
    record = _generate(dataset, "labeled_subset")
    assert record["num_samples_used"] == 10 and record["num_local_samples"] == 40
    assert select_labeled_positions(40, 1, 42, 0.25) != select_labeled_positions(40, 2, 42, 0.25)
    assert _generate(dataset, "all_labels")["num_samples_used"] == 40


def test_labeled_subset_matches_phase3_client_exactly(monkeypatch):
    """Phase 4 must select the very labeled samples Phase 3's Client trained with."""
    fake_pre = types.ModuleType("datasets.preprocess")
    train = TensorDataset(torch.randn(200, 3, 28, 28), torch.randint(0, 9, (200, 1)))
    train.labels = np.random.RandomState(0).randint(0, 9, (200, 1))
    fake_pre.train_dataset = train
    package = types.ModuleType("datasets")
    package.__path__ = [str(ROOT / "datasets")]
    spec = importlib.util.spec_from_file_location("datasets.partition", ROOT / "datasets" / "partition.py")
    partition_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(partition_module)
    package.preprocess, package.partition = fake_pre, partition_module
    for name, module in (("datasets", package), ("datasets.preprocess", fake_pre),
                         ("datasets.partition", partition_module)):
        monkeypatch.setitem(sys.modules, name, module)
    for name in [n for n in sys.modules if n in ("federated.client", "federated.client_data")]:
        monkeypatch.delitem(sys.modules, name)
    from federated.client import Client

    base = _echo_dataset(137, seed=5)
    config = {"label_fraction": 0.01, "seed": 42, "batch_size": 64, "learning_rate": 0.001}
    for client_id in (0, 1, 17, 49):
        phase3_indices = Client(client_id, config, base_dataset=base).labeled_indices
        assert select_labeled_positions(137, client_id, 42, 0.01) == phase3_indices


# --------------------------------------------------------------------------
# Save / load round trip and record validation
# --------------------------------------------------------------------------

def test_save_load_round_trip_is_identical(tmp_path):
    record = make_record(np.random.RandomState(2).randn(25, 128), np.random.RandomState(3).randint(0, 4, 25), client_id=12)
    path = save_client_record(record, tmp_path)
    assert path.name == "client_12.pt" and not list(tmp_path.glob("*.tmp"))
    loaded = load_client_prototype_file(path)
    for key in ("class_ids", "prototype_vectors", "sample_counts"):
        assert torch.equal(loaded[key], record[key])
    for key in ("client_id", "embedding_dim", "num_classes", "absent_class_ids", "label_policy",
                "provenance", "config", "num_local_samples", "num_samples_used"):
        assert loaded[key] == record[key]
    for class_id, vector in record["prototypes"].items():
        assert torch.equal(loaded["prototypes"][class_id], vector)
    assert loaded["metadata"] == record["metadata"]


def test_filename_client_mismatch_and_corruption_rejected(tmp_path):
    record = make_record([[1.0, 2.0]], [1], client_id=3)
    path = save_client_record(record, tmp_path)
    renamed = path.rename(tmp_path / "client_4.pt")
    with pytest.raises(ValueError):
        load_client_prototype_file(renamed)
    broken = dict(record, sample_counts=torch.tensor([0]))
    with pytest.raises(ValueError):
        validate_record(broken)


# --------------------------------------------------------------------------
# Held-out test set isolation
# --------------------------------------------------------------------------

def test_non_train_split_is_rejected():
    class Split(TensorDataset):
        pass

    dataset = Split(torch.zeros(4, 2), torch.zeros(4, 1, dtype=torch.long))
    dataset.split = "test"
    with pytest.raises(PermissionError):
        assert_train_split(Subset(dataset, [0, 1]))
    dataset.split = "train"
    assert_train_split(Subset(dataset, [0, 1]))


def test_phase4_code_never_references_test_data():
    for relative in ("run_phase4.py", "training/prototype_store.py", "training/prototypes.py"):
        text = (ROOT / relative).read_text()
        for forbidden in ("test_dataset", "test_loader", "split=\"test\"", "split='test'"):
            assert forbidden not in text, f"{relative} references {forbidden}"


# --------------------------------------------------------------------------
# Phase 3 artifact compatibility, provenance, end-to-end smoke on fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phase3_dir(tmp_path_factory):
    from models.classifier import FLAMEModel

    root = tmp_path_factory.mktemp("phase3")
    torch.manual_seed(0)
    model = FLAMEModel(embedding_dim=128, num_classes=9)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"round": r, "selected_clients": [0], "validation_loss": 2.2} for r in range(1, 4)]
    config = dict(run_phase4.expected_phase3_config(), num_rounds=3)
    (root / "config.json").write_text(json.dumps(config))
    torch.save({"model_state_dict": state, "history": history}, root / "final_model.pt")
    (root / "checkpoints").mkdir()
    torch.save({"round": 3, "global_model_state": state, "history": history, "client_states": {}},
               root / "checkpoints" / "checkpoint_round_3.pt")
    return root


def _fixture_clients(n_clients=3, seed=0):
    rng = np.random.RandomState(seed)
    datasets = {}
    for client_id in range(n_clients):
        n = 12 + 5 * client_id
        classes = rng.choice(9, size=3, replace=False)
        datasets[client_id] = TensorDataset(
            torch.tensor(rng.randn(n, 3, 28, 28), dtype=torch.float32),
            torch.tensor(rng.choice(classes, n), dtype=torch.long).view(-1, 1),
        )
    return datasets


def test_validate_phase3_artifacts_accepts_real_layout(phase3_dir):
    result = validate_phase3_artifacts(phase3_dir, run_phase4.expected_phase3_config(), require_rounds=3)
    provenance = result["provenance"]
    assert provenance["source_round"] == 3
    assert provenance["source_model_sha256"] == sha256_file(phase3_dir / "final_model.pt")
    assert provenance["checkpoint_crosscheck"]["weights_identical"] is True


def test_validate_phase3_artifacts_rejects_problems(phase3_dir, tmp_path):
    expected = run_phase4.expected_phase3_config()
    with pytest.raises(ValueError):  # wrong number of rounds
        validate_phase3_artifacts(phase3_dir, expected, require_rounds=200)
    with pytest.raises(ValueError):  # config mismatch
        validate_phase3_artifacts(phase3_dir, dict(expected, seed=7))
    with pytest.raises(FileNotFoundError):
        validate_phase3_artifacts(tmp_path, expected)


def test_smoke_pipeline_provenance_and_compatibility(phase3_dir, tmp_path):
    clients = _fixture_clients()
    partition_hash = partition_fingerprint({k: range(len(v)) for k, v in clients.items()})
    out = tmp_path / "phase4_smoke"
    summary = run_phase4.run_phase4(
        phase3_dir, out, clients, partition_hash, mode="smoke", label_policy="all_labels",
        device="cpu", max_samples=None,
    )
    assert summary["status"] == "SMOKE_NOT_FINAL" and summary["final_experiment_ready"] is False
    assert summary["test_set_accessed"] is False
    saved = json.loads((out / "summary.json").read_text())
    assert saved["num_clients"] == 50  # Phase 5 reads this key
    model_hash = sha256_file(phase3_dir / "final_model.pt")
    for client_id, dataset in clients.items():
        record = load_client_prototype_file(out / "prototypes" / f"client_{client_id}.pt")
        assert record["client_id"] == client_id
        assert record["provenance"]["source_model_sha256"] == model_hash
        assert record["provenance"]["source_round"] == 3
        assert record["provenance"]["partition_sha256"] == partition_hash
        assert record["config"]["seed"] == 42 and record["config"]["label_fraction"] == 0.01
        assert record["embedding_dim"] == 128 and record["num_local_samples"] == len(dataset)
        # raw images/labels never stored: only tensors of prototype size plus primitives
        assert all(t.shape[-1] == 128 for t in record["prototypes"].values())
        # Phase 5 compatibility: legacy view loads under weights_only=True
        legacy = torch.load(out / "prototypes" / f"client_{client_id}.pt", map_location="cpu", weights_only=True)
        assert {int(c) for c in legacy["prototypes"]} == set(record["class_ids"].tolist())


def test_pipeline_matches_independent_class_means(phase3_dir, tmp_path):
    """End-to-end: saved prototypes equal class means computed independently from the model."""
    from models.classifier import FLAMEModel

    clients = _fixture_clients(2, seed=9)
    out = tmp_path / "check"
    run_phase4.run_phase4(phase3_dir, out, clients, "hash", mode="smoke", label_policy="all_labels")
    state = torch.load(phase3_dir / "final_model.pt", weights_only=False)["model_state_dict"]
    model = FLAMEModel(embedding_dim=128, num_classes=9)
    model.load_state_dict(state)
    model.eval()
    for client_id, dataset in clients.items():
        images, labels = dataset.tensors
        with torch.no_grad():
            embeddings = torch.cat([model(images[i:i + 64])[1] for i in range(0, len(images), 64)])
        record = load_client_prototype_file(out / "prototypes" / f"client_{client_id}.pt")
        for i, class_id in enumerate(record["class_ids"].tolist()):
            expected = embeddings[labels.view(-1) == class_id].double().mean(0)
            np.testing.assert_allclose(record["prototype_vectors"][i].numpy(), expected.numpy(), rtol=1e-4, atol=1e-5)


def test_final_mode_requires_all_clients_and_overwrite_is_guarded(phase3_dir, tmp_path):
    clients = _fixture_clients(2)
    with pytest.raises(ValueError):
        run_phase4.run_phase4(phase3_dir, tmp_path / "final", clients, "hash", mode="final")
    out = tmp_path / "smoke"
    run_phase4.run_phase4(phase3_dir, out, clients, "hash", mode="smoke", label_policy="all_labels")
    with pytest.raises(FileExistsError):
        run_phase4.run_phase4(phase3_dir, out, clients, "hash", mode="smoke", label_policy="all_labels")
    run_phase4.run_phase4(phase3_dir, out, clients, "hash", mode="smoke", label_policy="all_labels", overwrite=True)


def test_verify_partition_invariants():
    good = {k: list(range(k * 10, k * 10 + 10)) for k in range(50)}
    assert len(run_phase4.verify_partition(good, expected_total=500)) == 64
    with pytest.raises(ValueError):
        run_phase4.verify_partition({**good, 3: good[2]})
    with pytest.raises(ValueError):
        run_phase4.verify_partition({k: v for k, v in good.items() if k != 7})
    with pytest.raises(ValueError):
        run_phase4.verify_partition({**good, 3: good[3][:5]})


def test_legacy_load_client_state_still_available_for_phase5():
    assert callable(run_phase4.load_client_state)


# --------------------------------------------------------------------------
# Colab layout, honest labelling, overwrite safety, checkpoint structure
# --------------------------------------------------------------------------

import shutil

import audit_phase4


@pytest.fixture(scope="module")
def phase3_200(tmp_path_factory):
    """Frozen-Phase-3 lookalike: final_model.pt (model_state_dict + history of 200 rounds) + config.json."""
    from models.classifier import FLAMEModel

    root = tmp_path_factory.mktemp("phase3_200")
    torch.manual_seed(0)
    state = {k: v.detach().cpu().clone() for k, v in FLAMEModel(embedding_dim=128, num_classes=9).state_dict().items()}
    history = [{"round": r, "selected_clients": [r % 50], "client_metrics": [{"client_id": r % 50}]}
               for r in range(1, 201)]
    json.dump(dict(run_phase4.expected_phase3_config(), num_rounds=200), open(root / "config.json", "w"))
    torch.save({"model_state_dict": state, "history": history}, root / "final_model.pt")
    return root


def fifty_clients(seed=0):
    rng = np.random.RandomState(seed)
    return {c: TensorDataset(torch.tensor(rng.randn(16, 3, 28, 28), dtype=torch.float32),
                             torch.tensor(np.resize(sorted({c % 4, (c + 1) % 4}), 16), dtype=torch.long).view(-1, 1))
            for c in range(50)}


@pytest.fixture(scope="module")
def global_run(phase3_200, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase4_global")
    summary = run_phase4.run_phase4(phase3_200, out, fifty_clients(), "partition-hash",
                                    mode=run_phase4.MODE_GLOBAL, label_policy="all_labels")
    return {"out": out, "summary": summary, "p3": phase3_200}


def test_default_paths_follow_the_colab_layout():
    assert run_phase4.PHASE3_PATH == Path("/content/drive/MyDrive/phase3")
    assert run_phase4.CHECKPOINT_PATH == Path("/content/drive/MyDrive/phase3/final_model.pt")
    assert run_phase4.CONFIG_PATH == Path("/content/drive/MyDrive/phase3/config.json")
    assert run_phase4.OUTPUT_PATH == Path("/content/drive/MyDrive/phase4")
    assert run_phase4.SMOKE_OUTPUT_PATH not in (run_phase4.OUTPUT_PATH, run_phase4.OUTPUT_PATH / "prototypes")


def test_global_model_run_is_labelled_honestly(global_run):
    summary = global_run["summary"]
    assert summary["status"] == "GLOBAL_MODEL_PROTOTYPES_NOT_FINAL" and summary["mode"] == "global_model_prototypes"
    assert summary["final_experiment_ready"] is False
    assert summary["prototype_origin"] == "global_model_applied_to_client_partition"
    assert summary["local_model_states"] == "unavailable" and summary["temporal_stability"] == "unavailable"
    assert summary["clients_with_local_states"] == 0 and summary["clients_using_global_model"] == 50
    assert summary["checkpoint_top_level_keys"] == ["history", "model_state_dict"]
    assert summary["state_dict_check"]["strict_compatible"] is True
    assert summary["num_classes"] == 9 and summary["embedding_dimension"] == 128
    assert "not representations from client-locally trained models" in summary["prototype_origin_description"]
    assert "client_local" not in json.dumps(summary["source_provenance"]).replace("client_local_model_states", "")
    for client_id in (0, 17, 49):
        record = load_client_prototype_file(global_run["out"] / "prototypes" / f"client_{client_id}.pt")
        prov = record["provenance"]
        assert prov["source_kind"] == "final_global_model" and prov["source_round"] == 200
        assert prov["prototype_origin"] == summary["prototype_origin"]
        assert prov["client_local_model_states"] == "unavailable" and prov["temporal_stability_history"] == "unavailable"
        assert record["config"]["run_status"] == "GLOBAL_MODEL_PROTOTYPES_NOT_FINAL"
        assert record["prototype_vectors"].shape[1] == 128 and int(record["class_ids"].max()) < 9


def test_global_mode_needs_all_50_clients_and_no_cap(phase3_200, tmp_path):
    with pytest.raises(ValueError):
        run_phase4.run_phase4(phase3_200, tmp_path / "a", dict(list(fifty_clients().items())[:3]), "h")
    with pytest.raises(ValueError):
        run_phase4.run_phase4(phase3_200, tmp_path / "b", fifty_clients(), "h", max_samples=10)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_existing_results_are_protected_with_a_clear_message(phase3_dir, tmp_path):
    clients = _fixture_clients(2)
    out = tmp_path / "phase4"
    run_phase4.run_phase4(phase3_dir, out, clients, "h", mode="smoke", label_policy="all_labels")
    before = {p.name: sha256_file(p) for p in out.rglob("*.*")}
    with pytest.raises(FileExistsError, match="already contains results.*--overwrite"):
        run_phase4.run_phase4(phase3_dir, out, clients, "h", mode="smoke", label_policy="all_labels")
    assert {p.name: sha256_file(p) for p in out.rglob("*.*")} == before            # nothing touched
    only_summary = tmp_path / "only_summary"
    only_summary.mkdir()
    (only_summary / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="summary.json"):
        run_phase4.run_phase4(phase3_dir, only_summary, clients, "h", mode="smoke", label_policy="all_labels")
    shutil.copy(out / "prototypes" / "client_0.pt", out / "prototypes" / "client_77.pt")
    with pytest.raises(FileExistsError, match="outside this run"):
        run_phase4.run_phase4(phase3_dir, out, clients, "h", mode="smoke", label_policy="all_labels", overwrite=True)
    (out / "prototypes" / "client_77.pt").unlink()
    run_phase4.run_phase4(phase3_dir, out, clients, "h", mode="smoke", label_policy="all_labels", overwrite=True)


def test_nothing_is_written_if_any_client_fails(phase3_dir, tmp_path):
    clients = _fixture_clients(3)
    images, _ = clients[2].tensors
    clients[2] = TensorDataset(images, torch.full((len(images), 1), 12, dtype=torch.long))   # label outside 0..8
    out = tmp_path / "phase4"
    with pytest.raises(ValueError):
        run_phase4.run_phase4(phase3_dir, out, clients, "h", mode="smoke", label_policy="all_labels")
    assert not out.exists()


def test_main_checks_destination_before_loading_any_data(tmp_path):
    out = tmp_path / "phase4"
    (out / "prototypes").mkdir(parents=True)
    (out / "prototypes" / "client_0.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_phase4.main(["--output-dir", str(out)])           # would need PathMNIST otherwise
    with pytest.raises(SystemExit):
        run_phase4.main(["--temporal"])


def test_checkpoint_structure_check_and_strict_load(phase3_200, tmp_path, monkeypatch):
    state = torch.load(phase3_200 / "final_model.pt", weights_only=False)["model_state_dict"]
    assert run_phase4.check_state_dict_compat(state, 128, 9)["strict_compatible"] is True
    broken = {k: v for k, v in state.items() if k != "proto.net.2.weight"}
    broken["extra.weight"] = torch.zeros(1)
    broken["proto.net.0.weight"] = torch.zeros(3, 3)
    report = run_phase4.check_state_dict_compat(broken, 128, 9)
    assert report["strict_compatible"] is False and report["missing_keys"] == ["proto.net.2.weight"]
    assert report["unexpected_keys"] == ["extra.weight"] and report["shape_mismatches"][0]["key"] == "proto.net.0.weight"

    real = run_phase4.validate_phase3_artifacts

    def broken_source(*args, **kwargs):
        out = real(*args, **kwargs)
        out["state_dict"] = {k: v for k, v in out["state_dict"].items() if k != "proto.net.2.weight"}
        return out

    monkeypatch.setattr(run_phase4, "validate_phase3_artifacts", broken_source)
    with pytest.raises(ValueError, match="does not match FLAMEModel"):
        run_phase4.run_phase4(phase3_200, tmp_path / "out", fifty_clients(), "h")
    assert not (tmp_path / "out").exists()


def test_audit_accepts_honest_global_run_and_flags_dishonest_labels(global_run, tmp_path):
    def run_audit(directory):
        audit_phase4.RESULTS.clear()
        source = audit_phase4.audit_phase3(global_run["p3"], 200)
        audit_phase4.audit_phase4(directory, source, "global_model_prototypes", 50)
        return {r[1] for r in audit_phase4.RESULTS if r[0] == "FAIL"}, {r[1] for r in audit_phase4.RESULTS if r[0] == "PASS"}

    failed, passed = run_audit(global_run["out"])
    assert not failed, failed
    assert "state dict loads strictly into FLAMEModel" in passed and "9 classes and 128-d embeddings in config" in passed
    assert "run status is 'GLOBAL_MODEL_PROTOTYPES_NOT_FINAL'" in passed

    for mutate, expected_failure in (
        (lambda d: d.update(status="PASS"), "run status is 'GLOBAL_MODEL_PROTOTYPES_NOT_FINAL'"),
        (lambda d: d.update(final_experiment_ready=True), "not reported as a ready final experiment"),
        (lambda d: d.update(local_model_states="available"),
         "summary marks local model states and temporal stability unavailable"),
        (lambda d: d.update(prototype_origin="client_local_models"),
         "summary states prototypes come from the GLOBAL model on client partitions"),
    ):
        copy_dir = tmp_path / expected_failure[:12].replace(" ", "_").replace("'", "")
        shutil.copytree(global_run["out"], copy_dir)
        summary = json.loads((copy_dir / "summary.json").read_text())
        mutate(summary)
        (copy_dir / "summary.json").write_text(json.dumps(summary))
        failed, _ = run_audit(copy_dir)
        assert expected_failure in failed, (expected_failure, failed)

    tampered = tmp_path / "tampered_record"
    shutil.copytree(global_run["out"], tampered)
    path = tampered / "prototypes" / "client_3.pt"
    payload = torch.load(path, weights_only=True)
    payload["provenance"]["prototype_origin"] = "client_local_models"
    torch.save(payload, path)
    failed, _ = run_audit(tampered)
    assert "every record loads and validates (weights_only=True)" in failed
