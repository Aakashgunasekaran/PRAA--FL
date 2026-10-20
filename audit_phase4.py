"""READ-ONLY Phase 4 audit. Writes nothing, trains nothing, never touches test data.

    python audit_phase4.py                                   # Colab defaults, global-model run
    python audit_phase4.py --expect-mode smoke --phase4-dir /content/phase4_smoke
    python audit_phase4.py --recompute-clients 0,7,31       # needs PathMNIST (train split)

Defaults: --phase3-dir /content/drive/MyDrive/phase3, --phase4-dir /content/drive/MyDrive/phase4,
--expect-mode global_model_prototypes (status GLOBAL_MODEL_PROTOTYPES_NOT_FINAL).

Exit status is non-zero if any check FAILs. WARN lines are limitations, not failures.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from federated import flame_config as cfg
from run_phase4 import (
    MODE_FINAL,
    MODE_GLOBAL,
    MODE_SMOKE,
    OUTPUT_PATH,
    PHASE3_PATH,
    PROTOTYPE_ORIGIN,
    RUN_STATUS,
    UNAVAILABLE,
)
from training.prototype_store import (
    load_client_prototype_file,
    select_labeled_positions,
    sha256_file,
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def warn(name, detail):
    RESULTS.append(("WARN", name, detail))
    print(f"[WARN] {name} -- {detail}")


def audit_phase3(phase3_dir, rounds):
    from models.classifier import FLAMEModel

    phase3_dir = Path(phase3_dir)
    model_path, config_path = phase3_dir / "final_model.pt", phase3_dir / "config.json"
    if not check("final_model.pt and config.json exist", model_path.exists() and config_path.exists()):
        return None
    print("final_model.pt sha256:", sha256_file(model_path))
    config = json.loads(config_path.read_text())
    final = torch.load(model_path, map_location="cpu", weights_only=False)
    if not check("checkpoint has model_state_dict and history", {"model_state_dict", "history"} <= set(final),
                 str(sorted(final))):
        return None
    state, history = final["model_state_dict"], final["history"]
    check("history has expected valid rounds",
          [h["round"] for h in history] == list(range(1, rounds + 1)),
          f"{len(history)} entries, last round {history[-1]['round'] if history else None}")
    check("config.json num_rounds matches history", config.get("num_rounds") == len(history))
    reference = FLAMEModel(embedding_dim=config["embedding_dim"], num_classes=config["num_classes"]).state_dict()
    check("state-dict keys identical to FLAMEModel", set(state) == set(reference),
          f"{len(state)} tensors")
    check("state-dict shapes identical", all(state[k].shape == reference[k].shape for k in reference if k in state))
    try:
        FLAMEModel(embedding_dim=config["embedding_dim"], num_classes=config["num_classes"]).load_state_dict(
            state, strict=True)
        strict_ok, strict_detail = True, ""
    except Exception as error:  # noqa: BLE001
        strict_ok, strict_detail = False, str(error)[:200]
    check("state dict loads strictly into FLAMEModel", strict_ok, strict_detail)
    check("9 classes and 128-d embeddings in config",
          config.get("num_classes") == 9 and config.get("embedding_dim") == 128)
    check("all floating weights finite",
          all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()))
    check("prototype head output is 128-d", state["proto.net.2.weight"].shape[0] == 128)
    expected = {"num_clients": cfg.NUM_CLIENTS, "seed": cfg.RANDOM_SEED, "label_fraction": cfg.LABEL_FRACTION,
                "embedding_dim": cfg.EMBEDDING_DIM, "batch_size": cfg.BATCH_SIZE, "mask_ratio": cfg.MASK_RATIO}
    check("config matches fixed project settings", all(config.get(k) == v for k, v in expected.items()),
          str({k: config.get(k) for k in expected}))
    ck = phase3_dir / "checkpoints" / f"checkpoint_round_{rounds}.pt"
    if ck.exists():
        saved = torch.load(ck, map_location="cpu", weights_only=False)
        same = saved["global_model_state"].keys() == state.keys() and all(
            torch.equal(saved["global_model_state"][k], state[k]) for k in state)
        check(f"checkpoint_round_{rounds}.pt round and global weights match final_model.pt",
              int(saved["round"]) == rounds and same)
    else:
        warn("round checkpoint absent", f"{ck.name} not found; equivalence not checked")
    return {"state": state, "config": config, "sha": sha256_file(model_path), "dir": str(phase3_dir)}


def audit_phase4(phase4_dir, source, expect_mode, expect_clients):
    phase4_dir = Path(phase4_dir)
    summary_path = phase4_dir / "summary.json"
    if not check("summary.json exists", summary_path.exists(), str(summary_path)):
        return {}
    summary = json.loads(summary_path.read_text())
    files = sorted((phase4_dir / "prototypes").glob("client_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    check("prototype file count equals summary.clients_processed",
          len(files) == summary.get("clients_processed"), f"{len(files)} files")
    check(f"run mode is '{expect_mode}'", summary.get("mode") == expect_mode, str(summary.get("mode")))
    check("expected client coverage", len(files) == expect_clients, f"{len(files)} / {expect_clients}")
    check("summary says test set not accessed", summary.get("test_set_accessed") is False)
    if expect_mode in (MODE_GLOBAL, MODE_FINAL):
        check("full run has no sample cap", summary.get("max_samples_per_client") is None)
    check(f"run status is '{RUN_STATUS[expect_mode]}'", summary.get("status") == RUN_STATUS[expect_mode],
          str(summary.get("status")))
    if expect_mode != MODE_FINAL:
        check("not reported as a ready final experiment", summary.get("final_experiment_ready") is False)
    check("summary states prototypes come from the GLOBAL model on client partitions",
          summary.get("prototype_origin") == PROTOTYPE_ORIGIN)
    check("summary marks local model states and temporal stability unavailable",
          summary.get("local_model_states") == UNAVAILABLE and summary.get("temporal_stability") == UNAVAILABLE
          and summary.get("clients_with_local_states") == 0)
    check("all clients attributed to the global model",
          summary.get("clients_using_global_model") == summary.get("clients_processed"))
    check("summary reports a strict state-dict match", bool((summary.get("state_dict_check") or {}).get("strict_compatible")))
    check("summary has 9 classes and 128-d embeddings",
          summary.get("num_classes") == 9 and summary.get("embedding_dimension") == 128)
    if source and summary.get("phase3_dir") and Path(summary["phase3_dir"]) != Path(source["dir"]):
        warn("summary phase3_dir differs from audited directory", f"{summary['phase3_dir']} vs {source['dir']}")
    records, hashes, rounds_seen, parts = {}, set(), set(), set()
    problems, total_protos = [], 0
    for path in files:
        try:
            record = load_client_prototype_file(path)
        except Exception as error:  # noqa: BLE001 - report every bad file
            problems.append(f"{path.name}: {error}")
            continue
        client_id = int(record["client_id"])
        records[client_id] = record
        total_protos += len(record["class_ids"])
        prov = record["provenance"]
        hashes.add(prov["source_model_sha256"])
        rounds_seen.add(prov["source_round"])
        parts.add(prov["partition_sha256"])
        if record["config"]["mode"] != expect_mode:
            problems.append(f"{path.name}: record mode {record['config']['mode']}")
        if record["config"].get("run_status") != RUN_STATUS[expect_mode]:
            problems.append(f"{path.name}: record run_status {record['config'].get('run_status')}")
        if (prov.get("prototype_origin") != PROTOTYPE_ORIGIN
                or prov.get("client_local_model_states") != UNAVAILABLE
                or prov.get("temporal_stability_history") != UNAVAILABLE):
            problems.append(f"{path.name}: provenance does not state global-model origin / unavailable history")
        if record["num_classes"] != 9 or int(record["class_ids"].max()) >= 9 or int(record["class_ids"].min()) < 0:
            problems.append(f"{path.name}: class ids outside 0..8")
        if tuple(record["prototype_vectors"].shape) != (len(record["class_ids"]), 128):
            problems.append(f"{path.name}: prototype matrix shape {tuple(record['prototype_vectors'].shape)}")
        if int(record["sample_counts"].sum()) != record["num_samples_used"]:
            problems.append(f"{path.name}: counts do not sum to samples used")
        if record["label_policy"] == "labeled_subset":
            n = record["num_local_samples"]
            want = min(max(2, int(n * record["config"]["label_fraction"])), n)
            if record["config"].get("max_samples_per_client") is None and record["num_samples_used"] != want:
                problems.append(f"{path.name}: used {record['num_samples_used']} samples, policy implies {want}")
    check("every record loads and validates (weights_only=True)", not problems, "; ".join(problems[:5]))
    check("one source model hash across records", len(hashes) == 1)
    if source:
        check("recorded hash equals actual final_model.pt hash", hashes == {source["sha"]})
    check("one source round across records", len(rounds_seen) == 1, str(sorted(rounds_seen)))
    check("one partition hash across records", len(parts) == 1)
    check("summary total_prototypes equals files", summary.get("total_prototypes") == total_protos)
    check("summary classes_per_client equals files", all(
        summary["classes_per_client"].get(str(k)) == r["class_ids"].tolist() for k, r in records.items()))
    check("embedding dimension 128 everywhere", all(r["embedding_dim"] == 128 for r in records.values()))
    covered = sorted({c for r in records.values() for c in r["class_ids"].tolist()})
    print("classes covered:", covered, "| prototypes per client (min/max):",
          min((len(r["class_ids"]) for r in records.values()), default=0),
          max((len(r["class_ids"]) for r in records.values()), default=0))
    thin = sum(1 for r in records.values() for n in r["sample_counts"].tolist() if n < 5)
    if thin:
        warn("noisy prototypes", f"{thin} of {total_protos} prototypes are built from <5 samples")
    return records


def recompute_client(model, dataset, positions, saved, atol=1e-4):
    """Independently recompute one client's prototypes and compare with the saved record."""
    model.eval()
    subset = torch.utils.data.Subset(dataset, positions)
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False):
            embeddings.append(model(images)[1].double())
            labels.append(batch_labels.view(-1))
    embeddings, labels = torch.cat(embeddings), torch.cat(labels)
    if sorted(labels.unique().tolist()) != saved["class_ids"].tolist():
        return False, "class set differs"
    for i, class_id in enumerate(saved["class_ids"].tolist()):
        mask = labels == class_id
        if int(mask.sum()) != int(saved["sample_counts"][i]):
            return False, f"count differs for class {class_id}"
        diff = (embeddings[mask].mean(0).float() - saved["prototype_vectors"][i]).abs().max().item()
        if diff > atol:
            return False, f"class {class_id} max abs diff {diff:.2e}"
    return True, "ok"


def recompute_with_real_data(source, records, client_ids):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from federated.client import Client
        from federated.flame_data import get_flame_client_dataset
    from models.classifier import FLAMEModel

    config = source["config"]
    model = FLAMEModel(embedding_dim=config["embedding_dim"], num_classes=config["num_classes"])
    model.load_state_dict(source["state"])
    for client_id in client_ids:
        if client_id not in records:
            check(f"recompute client {client_id}", False, "no saved record")
            continue
        dataset = get_flame_client_dataset(client_id)
        saved = records[client_id]
        if saved["label_policy"] == "labeled_subset":
            # positions come from the ORIGINAL Phase 3 Client implementation, not Phase 4 code
            positions = Client(client_id, dict(config, learning_rate=config.get("learning_rate", 0.001)),
                               base_dataset=dataset).labeled_indices
            check(f"client {client_id}: Phase 4 labeled positions equal Phase 3 Client",
                  positions == select_labeled_positions(len(dataset), client_id, config["seed"], config["label_fraction"]))
        else:
            positions = list(range(len(dataset)))
        if saved["config"].get("max_samples_per_client"):
            positions = positions[: saved["config"]["max_samples_per_client"]]
        ok, detail = recompute_client(model, dataset, positions, saved)
        check(f"client {client_id}: independent recomputation matches saved prototypes", ok, detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3_PATH)
    parser.add_argument("--phase4-dir", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--expect-mode", choices=[MODE_GLOBAL, MODE_SMOKE, MODE_FINAL], default=MODE_GLOBAL)
    parser.add_argument("--expect-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    parser.add_argument("--recompute-clients", default="")
    args = parser.parse_args()
    expect_clients = args.expect_clients or (5 if args.expect_mode == MODE_SMOKE else cfg.NUM_CLIENTS)
    check("Phase 3 directory exists", args.phase3_dir.exists(), str(args.phase3_dir))
    check("Phase 4 directory exists", args.phase4_dir.exists(), str(args.phase4_dir))

    source = audit_phase3(args.phase3_dir, args.rounds)
    records = audit_phase4(args.phase4_dir, source, args.expect_mode, expect_clients)
    if args.recompute_clients and source and records:
        recompute_with_real_data(source, records, [int(c) for c in args.recompute_clients.split(",")])
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\nAUDIT: {sum(r[0] == 'PASS' for r in RESULTS)} passed, {len(failed)} failed, "
          f"{sum(r[0] == 'WARN' for r in RESULTS)} warnings")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
