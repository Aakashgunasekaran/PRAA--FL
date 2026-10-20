"""Phase 4 mentor demo: class-wise prototypes from the saved Phase 3 GLOBAL model.

What this produces (and what it does not)
-----------------------------------------
Phase 3 saved only the final global model (`final_model.pt`) and its history; it did
NOT save per-client local models or per-round client snapshots. Phase 4 therefore
applies the saved global model to each simulated client's Phase 2 partition and stores
the class-wise mean of the resulting 128-d projected embeddings. These are
"global-model prototypes of each client's data", NOT outputs of client-locally trained
models. Local model states and temporal stability are recorded as UNAVAILABLE; nothing
is synthesized. The run is labelled GLOBAL_MODEL_PROTOTYPES_NOT_FINAL.

Usage in Colab (from /content/PRAA--FL):

    python run_phase4.py                  # 50 clients -> /content/drive/MyDrive/phase4
    python run_phase4.py --smoke          # 5 clients, <=64 samples -> /content/phase4_smoke
    python run_phase4.py --overwrite      # only if you really want to replace results
    python run_phase4.py --label-policy all_labels   # default is the 1% labeled subset

Phase 3 is never re-run, nothing is retrained, and the held-out test split is never read.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch

from federated import flame_config as cfg
from training.prototype_store import (
    LABEL_POLICIES,
    REPRESENTATION,
    generate_client_record,
    partition_fingerprint,
    save_client_record,
    validate_phase3_artifacts,
)

DRIVE_ROOT = Path("/content/drive/MyDrive")
PHASE3_PATH = DRIVE_ROOT / "phase3"
CHECKPOINT_PATH = PHASE3_PATH / "final_model.pt"
CONFIG_PATH = PHASE3_PATH / "config.json"
CLIENT_STATE_PATH = PHASE3_PATH / "client_states"      # absent in the real Phase 3 output
OUTPUT_PATH = DRIVE_ROOT / "phase4"
SMOKE_OUTPUT_PATH = Path("/content/phase4_smoke")
SMOKE_CLIENTS = 5
SMOKE_MAX_SAMPLES = 64
EXPECTED_TRAIN_SAMPLES = 89_996
# Legacy per-round snapshot helper keeps its original 64-sample cap. It needs Phase 3
# client snapshots that do not exist, so the CLI no longer exposes it.
MAX_SAMPLES_PER_CLIENT = SMOKE_MAX_SAMPLES

MODE_GLOBAL = "global_model_prototypes"   # CLI default: all 50 clients, global model
MODE_SMOKE = "smoke"
MODE_FINAL = "final"                      # API/test compatibility only; the CLI never emits it
FULL_RUN_MODES = (MODE_GLOBAL, MODE_FINAL)
RUN_STATUS = {
    MODE_GLOBAL: "GLOBAL_MODEL_PROTOTYPES_NOT_FINAL",
    MODE_SMOKE: "SMOKE_NOT_FINAL",
    MODE_FINAL: "PASS",
}
PROTOTYPE_ORIGIN = "global_model_applied_to_client_partition"
ORIGIN_DESCRIPTION = (
    "Class-wise means of embeddings computed by the saved Phase 3 GLOBAL model on the samples "
    "permitted by the label policy from each simulated client's Phase 2 partition. They are not "
    "representations from client-locally trained models."
)
UNAVAILABLE = "unavailable"


def expected_phase3_config():
    """Fixed project settings the frozen Phase 3 run must match."""
    return {
        "num_clients": cfg.NUM_CLIENTS,
        "clients_per_round": cfg.CLIENTS_PER_ROUND,
        "local_epochs": cfg.LOCAL_EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "embedding_dim": cfg.EMBEDDING_DIM,
        "num_classes": 9,
        "mask_ratio": cfg.MASK_RATIO,
        "mae_weight": cfg.MAE_WEIGHT,
        "seed": cfg.RANDOM_SEED,
        "label_fraction": cfg.LABEL_FRACTION,
    }


def verify_partition(partition, expected_total=None, num_clients=cfg.NUM_CLIENTS,
                     min_samples=cfg.MIN_SAMPLES_PER_CLIENT):
    """Check the Phase 2 partition invariants Phase 4 relies on."""
    if sorted(partition) != list(range(num_clients)):
        raise ValueError(f"Partition must contain clients 0..{num_clients - 1}")
    assigned = [i for indices in partition.values() for i in indices]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Partition assigns a sample to more than one client")
    if expected_total is not None and len(assigned) != expected_total:
        raise ValueError(f"Partition covers {len(assigned)} samples, expected {expected_total}")
    small = [k for k, v in partition.items() if len(v) < min_samples]
    if small:
        raise ValueError(f"Clients below the {min_samples}-sample minimum: {small}")
    return partition_fingerprint(partition)


def check_state_dict_compat(state, embedding_dim, num_classes):
    """Compare a checkpoint's state dict with a fresh FLAMEModel (names and shapes)."""
    from models.classifier import FLAMEModel

    reference = FLAMEModel(embedding_dim=embedding_dim, num_classes=num_classes).state_dict()
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    shape_mismatches = [
        {"key": k, "checkpoint": list(state[k].shape), "model": list(reference[k].shape)}
        for k in sorted(reference) if k in state and state[k].shape != reference[k].shape
    ]
    return {
        "tensors_in_checkpoint": len(state),
        "tensors_expected_by_model": len(reference),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "strict_compatible": not (missing or unexpected or shape_mismatches),
    }


def check_destination(output_dir, client_ids, overwrite):
    """Fail clearly if the destination already holds Phase 4 results."""
    output_dir = Path(output_dir)
    prototype_dir = output_dir / "prototypes"
    existing = sorted(prototype_dir.glob("client_*.pt")) if prototype_dir.exists() else []
    found = [p.name for p in existing] + (["summary.json"] if (output_dir / "summary.json").exists() else [])
    if found and not overwrite:
        raise FileExistsError(
            f"Phase 4 destination {output_dir} already contains results "
            f"({len(existing)} prototype files{', summary.json' if (output_dir / 'summary.json').exists() else ''}). "
            "Nothing was written. Re-run with --overwrite to replace them, or choose a different --output-dir."
        )
    wanted = {int(c) for c in client_ids}
    stale = [p.name for p in existing if int(p.stem.split("_")[1]) not in wanted]
    if found and overwrite and stale:
        raise FileExistsError(
            f"{output_dir} holds prototype files for clients outside this run ({stale[:5]}). "
            "Refusing to mix runs; choose a new --output-dir."
        )
    return found


def load_client_state(client_id, checkpoint, allow_global_fallback):
    state_path = CLIENT_STATE_PATH / f"client_{client_id}.pt"
    if state_path.exists():
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in payload:
            saved_client_id = payload.get("client_id")
            if saved_client_id is not None and int(saved_client_id) != int(client_id):
                raise ValueError(
                    f"Client state mismatch: requested client {client_id}, "
                    f"file contains client {saved_client_id}"
                )
            return payload["model_state_dict"], False
        return payload, False
    if allow_global_fallback:
        return checkpoint["model_state_dict"], True
    raise FileNotFoundError(
        f"Missing local Phase 3 state for client {client_id}. "
        "Run the required Phase 3 experiment first."
    )


def generate_round_snapshots(config, device, max_samples_per_client=MAX_SAMPLES_PER_CLIENT):
    """Generate prototype files for each persisted Phase 3 round snapshot."""
    from federated.flame_data import PARTITION, get_flame_client_dataset
    from models.classifier import FLAMEModel
    from training.prototypes import generate_client_prototypes

    snapshot_root = CLIENT_STATE_PATH
    round_dirs = sorted(
        snapshot_root.glob("round_*"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    output_root = OUTPUT_PATH / "snapshots"
    generated_count = 0
    for round_dir in round_dirs:
        round_number = int(round_dir.name.split("_")[1])
        round_output = output_root / f"round_{round_number}"
        round_output.mkdir(parents=True, exist_ok=True)
        for state_path in sorted(round_dir.glob("client_*.pt")):
            client_id = int(state_path.stem.split("_")[1])
            payload = torch.load(state_path, map_location=device, weights_only=True)
            if int(payload["client_id"]) != client_id or int(payload["round"]) != round_number:
                raise ValueError(f"Invalid round/client metadata in {state_path}")
            model = FLAMEModel(
                embedding_dim=config["embedding_dim"], num_classes=config["num_classes"]
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
            result = generate_client_prototypes(
                model,
                {client_id: get_flame_client_dataset(client_id)},
                batch_size=config["batch_size"],
                device=device,
                max_samples_per_client=max_samples_per_client,
            )[client_id]
            torch.save(
                {
                    "round": round_number,
                    "client_id": client_id,
                    "prototypes": result["prototypes"],
                    "metadata": result["metadata"],
                    "samples_processed": result["samples_processed"],
                    "embedding_dimension": result["embedding_dimension"],
                    "representation": "FLAME projected embedding (model.proto(model.encoder(x)))",
                },
                round_output / f"client_{client_id}.pt",
            )
            generated_count += 1
            del model
    return generated_count, len(round_dirs)


def run_phase4(
    phase3_dir,
    output_dir,
    client_datasets,
    partition_hash,
    *,
    mode=MODE_GLOBAL,
    label_policy="labeled_subset",
    device="cpu",
    max_samples=None,
    overwrite=False,
    require_rounds=cfg.NUM_ROUNDS,
):
    """Generate and save global-model prototypes for the given clients; return the summary.

    ``client_datasets`` maps client id -> that client's Phase 2 training subset. Kept
    separate from data loading so it can be tested with small fixtures. All records are
    computed before anything is written, and ``summary.json`` is written last.
    """
    from models.classifier import FLAMEModel

    if mode not in (MODE_GLOBAL, MODE_SMOKE, MODE_FINAL):
        raise ValueError(f"mode must be one of {(MODE_GLOBAL, MODE_SMOKE, MODE_FINAL)}")
    if mode in FULL_RUN_MODES and (max_samples is not None or sorted(client_datasets) != list(range(cfg.NUM_CLIENTS))):
        raise ValueError("full runs require all 50 clients and no sample cap")
    check_destination(output_dir, client_datasets, overwrite)  # fail fast, before any computation

    source = validate_phase3_artifacts(
        phase3_dir, expected_phase3_config(),
        require_rounds=require_rounds if mode in FULL_RUN_MODES else None,
    )
    config = source["config"]
    compat = check_state_dict_compat(source["state_dict"], config["embedding_dim"], config["num_classes"])
    if not compat["strict_compatible"]:
        raise ValueError(
            "Checkpoint state dict does not match FLAMEModel: "
            f"missing={compat['missing_keys'][:5]} unexpected={compat['unexpected_keys'][:5]} "
            f"shape_mismatches={compat['shape_mismatches'][:3]}"
        )
    model = FLAMEModel(embedding_dim=config["embedding_dim"], num_classes=config["num_classes"])
    model.load_state_dict(source["state_dict"], strict=True)
    model.to(device).eval()

    provenance = dict(
        source["provenance"],
        partition_sha256=partition_hash,
        partition_source="Phase 2 Dirichlet partition rebuilt from flame_config",
        prototype_origin=PROTOTYPE_ORIGIN,
        prototype_origin_description=ORIGIN_DESCRIPTION,
        client_local_model_states=UNAVAILABLE,
        temporal_stability_history=UNAVAILABLE,
    )
    record_config = {
        "mode": mode,
        "run_status": RUN_STATUS[mode],
        "seed": int(config["seed"]),
        "label_fraction": float(config["label_fraction"]),
        "label_policy": label_policy,
        "batch_size": int(config["batch_size"]),
        "embedding_dim": int(config["embedding_dim"]),
        "num_classes": int(config["num_classes"]),
        "num_clients": int(config["num_clients"]),
        "dirichlet_beta": cfg.DIRICHLET_BETA,
        "min_samples_per_client": cfg.MIN_SAMPLES_PER_CLIENT,
        "max_samples_per_client": max_samples,
        "phase3_num_rounds": int(config["num_rounds"]),
    }

    records = {}
    for client_id in sorted(client_datasets):
        records[client_id] = generate_client_record(
            model, client_datasets[client_id], client_id,
            num_classes=int(config["num_classes"]), label_policy=label_policy,
            seed=int(config["seed"]), label_fraction=float(config["label_fraction"]),
            batch_size=int(config["batch_size"]), device=device,
            provenance=provenance, config=record_config, max_samples=max_samples,
        )

    check_destination(output_dir, client_datasets, overwrite)  # re-check right before writing
    prototype_dir = Path(output_dir) / "prototypes"
    for record in records.values():
        save_client_record(record, prototype_dir)

    classes_observed = sorted({c for r in records.values() for c in r["class_ids"].tolist()})
    summary = {
        "phase": 4,
        "status": RUN_STATUS[mode],
        "mode": mode,
        "dataset": "PathMNIST",
        "data_source": "Phase 3 FLAME final global model; Phase 2 client training partitions",
        "prototype_origin": PROTOTYPE_ORIGIN,
        "prototype_origin_description": ORIGIN_DESCRIPTION,
        "local_model_states": UNAVAILABLE,
        "temporal_stability": UNAVAILABLE,
        "clients_with_local_states": 0,
        "clients_using_global_model": len(records),
        "representation": REPRESENTATION,
        "label_policy": label_policy,
        "num_clients": int(config["num_clients"]),
        "clients_processed": len(records),
        "clients_with_prototypes": len(records),
        "embedding_dimension": int(config["embedding_dim"]),
        "num_classes": int(config["num_classes"]),
        "total_prototypes": sum(len(r["class_ids"]) for r in records.values()),
        "classes_observed": classes_observed,
        "samples_used_per_client": {str(k): r["num_samples_used"] for k, r in records.items()},
        "classes_per_client": {str(k): r["class_ids"].tolist() for k, r in records.items()},
        "max_samples_per_client": max_samples,
        "state_dict_check": compat,
        "checkpoint_top_level_keys": source["checkpoint_keys"],
        "phase3_dir": str(phase3_dir),
        "output_dir": str(output_dir),
        "source_provenance": provenance,
        "test_set_accessed": False,
        "final_experiment_ready": False if mode != MODE_FINAL else len(records) == int(config["num_clients"]),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_report(summary):
    line = "=" * 64
    print(line)
    print("PRAA-FL - PHASE 4: PROTOTYPES FROM THE SAVED GLOBAL MODEL")
    print(line)
    print(f"Mode: {summary['mode']}    Label policy: {summary['label_policy']}")
    print("Dataset: PathMNIST (training partitions only)")
    prov = summary["source_provenance"]
    print(f"Source model: {prov['source_model_file']} (round {prov['source_round']}, "
          f"sha256 {prov['source_model_sha256'][:12]}...)")
    check = summary["state_dict_check"]
    print(f"Checkpoint keys: {summary['checkpoint_top_level_keys']}; "
          f"{check['tensors_in_checkpoint']} tensors, strict load into FLAMEModel: {check['strict_compatible']}")
    print(f"Representation: {summary['representation']}")
    print(f"Embedding dimension: {summary['embedding_dimension']}   Classes: {summary['num_classes']}")
    print()
    for client_id, classes in summary["classes_per_client"].items():
        used = summary["samples_used_per_client"][client_id]
        print(f"Client {client_id}: {len(classes)} prototypes, classes {classes}, samples used {used}")
    print()
    print(f"Clients processed: {summary['clients_processed']} / {summary['num_clients']}")
    print(f"Total prototypes: {summary['total_prototypes']}   Classes represented: {summary['classes_observed']}")
    print("Prototype origin: GLOBAL model applied to each client's partition")
    print("  (not client-locally trained models)")
    print(f"Client local model states: {summary['local_model_states'].upper()}")
    print(f"Temporal stability history: {summary['temporal_stability'].upper()}")
    print("Test set accessed: False")
    print(f"Output: {summary['output_dir']}")
    print(line)
    print(f"PHASE 4 STATUS: {summary['status']}")
    if not summary["final_experiment_ready"]:
        print("This is NOT a completed final experiment.")
    print(line)


def main(args=None):
    parser = argparse.ArgumentParser(description="Generate Phase 4 prototypes from the saved global model")
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label-policy", choices=LABEL_POLICIES, default="labeled_subset")
    parser.add_argument("--smoke", action="store_true",
                        help=f"first {SMOKE_CLIENTS} clients, <= {SMOKE_MAX_SAMPLES} samples each")
    parser.add_argument("--overwrite", action="store_true", help="replace existing Phase 4 results")
    parser.add_argument("--temporal", action="store_true", help="unavailable (see error)")
    options = parser.parse_args(args)
    if options.temporal:
        parser.error("--temporal is unavailable: Phase 3 saved no per-client or per-round snapshots, "
                     "and global-model copies must not be presented as client history")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = options.output_dir or (SMOKE_OUTPUT_PATH if options.smoke else OUTPUT_PATH)
    check_destination(output_dir, range(SMOKE_CLIENTS if options.smoke else cfg.NUM_CLIENTS),
                      options.overwrite)  # fail before loading PathMNIST
    with contextlib.redirect_stdout(io.StringIO()):
        from datasets import preprocess
        from federated.flame_data import PARTITION, get_flame_client_dataset

    if len(preprocess.train_dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_TRAIN_SAMPLES} training samples, "
                         f"found {len(preprocess.train_dataset)}")
    partition_hash = verify_partition(PARTITION, expected_total=EXPECTED_TRAIN_SAMPLES)
    client_ids = sorted(PARTITION)[:SMOKE_CLIENTS] if options.smoke else sorted(PARTITION)
    client_datasets = {k: get_flame_client_dataset(k) for k in client_ids}

    summary = run_phase4(
        options.phase3_dir, output_dir, client_datasets, partition_hash,
        mode=MODE_SMOKE if options.smoke else MODE_GLOBAL,
        label_policy=options.label_policy, device=device,
        max_samples=SMOKE_MAX_SAMPLES if options.smoke else None,
        overwrite=options.overwrite,
    )
    print_report(summary)


if __name__ == "__main__":
    main()
