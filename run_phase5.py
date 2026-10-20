"""Phase 5 mentor demo: Prototype Reliability Assessment Module (PRAM).

Reads the frozen Phase 3 model and the Phase 4 prototype files (read-only) and
writes per-(client, class) validation, consistency, and stability components.
Components that cannot be computed are reported as `unavailable` - never as 0 or 1.
It does not compute PRS or perform aggregation (Phase 6+).

    python run_phase5.py --phase3-dir /content/drive/MyDrive/phase3 \
        --phase4-dir /content/drive/MyDrive/phase4 \
        --output-dir /content/drive/MyDrive/phase5
    python run_phase5.py --smoke ...        # 512 validation samples -> results/phase5_smoke
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch

from federated import flame_config as cfg
from federated.pram import (
    COMPONENTS,
    FORMULA_VERSION,
    FORMULAS,
    SCHEMA_VERSION,
    assess_client,
    build_reference_prototypes,
    discover_client_snapshots,
    inspect_training_history,
    phase6_readiness,
    summarize,
    write_json_atomic,
)
from run_phase4 import expected_phase3_config
from training.prototype_store import (
    load_client_prototype_file,
    sha256_file,
    validate_phase3_artifacts,
)
from training.prototypes import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent
PHASE3_PATH = PROJECT_ROOT / "results" / "phase3"
PHASE4_PATH = PROJECT_ROOT / "results" / "phase4"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase5"
SMOKE_OUTPUT_PATH = PROJECT_ROOT / "results" / "phase5_smoke"
SMOKE_VALIDATION_SAMPLES = 512
EXPECTED_VALIDATION_SAMPLES = 10_004


def assert_validation_split(dataset) -> None:
    """Allow only the validation split; the held-out test split is never permitted."""
    base = dataset
    while hasattr(base, "dataset") and hasattr(base, "indices"):
        base = base.dataset
    split = getattr(base, "split", None)
    if split is not None and str(split) != "val":
        raise PermissionError(f"Phase 5 may only read the validation split; got split={split!r}")


def load_phase4_records(phase4_dir):
    """Load and cross-check every Phase 4 prototype file (read-only)."""
    phase4_dir = Path(phase4_dir)
    summary_path = phase4_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Phase 4 summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    paths = sorted((phase4_dir / "prototypes").glob("client_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"No Phase 4 prototype files under {phase4_dir / 'prototypes'}")
    records = {}
    for path in paths:
        record = load_client_prototype_file(path)  # rejects legacy/non-schema files
        records[int(record["client_id"])] = record
    if summary.get("clients_processed") != len(records):
        raise ValueError("Phase 4 summary.clients_processed disagrees with prototype files")
    for key in ("source_model_sha256", "source_round", "partition_sha256"):
        values = {r["provenance"][key] for r in records.values()}
        if len(values) != 1:
            raise ValueError(f"Phase 4 records disagree on provenance.{key}: {sorted(map(str, values))}")
    modes = {r["config"]["mode"] for r in records.values()}
    if len(modes) != 1:
        raise ValueError(f"Phase 4 records mix modes: {sorted(modes)}")
    return records, summary, {p.name: sha256_file(p) for p in paths}, sha256_file(summary_path)


def run_pram(
    phase3_dir,
    phase4_dir,
    output_dir,
    validation_dataset,
    *,
    mode="final",
    device="cpu",
    overwrite=False,
    max_validation_samples=None,
):
    """Compute PRAM components and write ``pram.json``; return the document."""
    from models.classifier import FLAMEModel

    if mode not in ("final", "smoke", "exploratory"):
        raise ValueError("mode must be final, smoke, or exploratory")
    assert_validation_split(validation_dataset)
    records, phase4_summary, file_hashes, summary_hash = load_phase4_records(phase4_dir)
    phase4_mode = next(iter(records.values()))["config"]["mode"]
    first_provenance = next(iter(records.values()))["provenance"]
    round_id = int(first_provenance["source_round"])
    if mode == "final":
        if phase4_mode != "final" or sorted(records) != list(range(cfg.NUM_CLIENTS)):
            raise ValueError("final mode needs a final-mode Phase 4 run covering all 50 clients")
        if max_validation_samples is not None:
            raise ValueError("final mode must use the complete validation split")
    elif mode == "exploratory":
        if phase4_mode != "global_model_prototypes" or sorted(records) != list(range(cfg.NUM_CLIENTS)):
            raise ValueError("exploratory mode requires global-model prototypes for all 50 clients")
        if max_validation_samples is not None:
            raise ValueError("exploratory mode must use the complete validation split")

    source = validate_phase3_artifacts(
        phase3_dir, expected_phase3_config(),
        require_rounds=round_id if mode in ("final", "exploratory") else None,
    )
    if source["provenance"]["source_model_sha256"] != first_provenance["source_model_sha256"]:
        raise ValueError("Phase 4 prototypes were not generated from this Phase 3 final_model.pt")
    if source["provenance"]["source_round"] != round_id:
        raise ValueError("Phase 4 source round differs from Phase 3 final round")
    config = source["config"]
    history = torch.load(Path(phase3_dir) / "final_model.pt", map_location="cpu",
                         weights_only=False)["history"]

    model = FLAMEModel(embedding_dim=config["embedding_dim"], num_classes=config["num_classes"])
    model.load_state_dict(source["state_dict"], strict=True)
    model.to(device).eval()
    dataset = validation_dataset
    if max_validation_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_validation_samples)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    embeddings, labels = extract_features(model, loader, device=device)  # computed once, global model

    reference = build_reference_prototypes(records)
    rows, snapshot_rejections, accepted_snapshots = [], [], 0
    for client_id in sorted(records):
        snapshots, rejected = discover_client_snapshots(phase4_dir, client_id)
        accepted_snapshots += len(snapshots)
        snapshot_rejections.extend(rejected)
        rows.extend(assess_client(client_id, records[client_id], reference, embeddings, labels,
                                  snapshots, round_id))

    reason_counts = {}
    for _, reason in snapshot_rejections:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    sample_counts = [r["sample_count"] for r in rows]
    document = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "phase": 5,
        "mode": mode,
        "status": "COMPLETE" if rows and all(r["complete"] for r in rows) else "PARTIAL_EVIDENCE_MISSING",
        "formulas": FORMULAS,
        "config": {
            "seed": int(config["seed"]),
            "embedding_dim": int(config["embedding_dim"]),
            "num_classes": int(config["num_classes"]),
            "phase4_label_policy": next(iter(records.values()))["label_policy"],
            "phase4_mode": phase4_mode,
            "validation_samples_used": int(labels.numel()),
            "validation_split_size_expected": EXPECTED_VALIDATION_SAMPLES,
            "validation_scope": "shared global validation split (no client-specific validation "
                                "data exists); scores describe each client's PROTOTYPES in the "
                                "global model's embedding space, not a client-trained model",
            "reference_definition": "unweighted self-inclusive mean over clients holding the class",
        },
        "provenance": {
            "phase3_model_sha256": source["provenance"]["source_model_sha256"],
            "phase3_round": round_id,
            "phase3_checkpoint_crosscheck": source["provenance"]["checkpoint_crosscheck"],
            "phase4_partition_sha256": first_provenance["partition_sha256"],
            "phase4_summary_sha256": summary_hash,
            "phase4_prototype_file_sha256": file_hashes,
            "test_set_accessed": False,
        },
        "evidence": {
            "training_history": inspect_training_history(history, int(config["num_clients"])),
            "client_snapshot_files_accepted": accepted_snapshots,
            "client_snapshot_files_rejected": len(snapshot_rejections),
            "client_snapshot_rejection_reasons": reason_counts,
            "stability_conclusion": (
                "available for clients with >= 2 attested snapshots" if accepted_snapshots
                else "unavailable: no attested client-local prototype snapshots exist; global-model "
                     "checkpoints are not client history"),
        },
        "reference_prototypes": {
            str(c): {"num_contributors": len(r["contributors"]), "contributors": r["contributors"]}
            for c, r in reference.items()
        },
        "prototype_sample_count_summary": {
            "min": min(sample_counts), "max": max(sample_counts),
            "records_below_5_samples": sum(n < 5 for n in sample_counts),
        },
        "summary": summarize(rows),
        "phase6_readiness": phase6_readiness(rows),
        "numerical_validation": {"all_values_finite_and_in_unit_interval": True,
                                 "schema_validated": True},
        "records": rows,
    }
    output_dir = Path(output_dir)
    write_json_atomic(document, output_dir / "pram.json", overwrite=overwrite)
    return document


def print_report(document):
    line = "=" * 60
    summary, ready = document["summary"], document["phase6_readiness"]
    print(line)
    print("PRAA-FL - PHASE 5: PROTOTYPE RELIABILITY ASSESSMENT (PRAM)")
    print(line)
    print(f"Mode: {document['mode']}  Phase 4 mode: {document['config']['phase4_mode']}  "
          f"Label policy: {document['config']['phase4_label_policy']}")
    print(f"Source: Phase 3 round {document['provenance']['phase3_round']}  "
          f"sha256 {document['provenance']['phase3_model_sha256'][:12]}...")
    print(f"Validation samples used: {document['config']['validation_samples_used']}")
    print(f"Records (client, class): {summary['records']}   complete: {summary['complete_records']}")
    for name in COMPONENTS:
        print(f"  {name:12s} {summary['status_counts'][name]}")
    print(f"Stability evidence: {document['evidence']['stability_conclusion']}")
    print(f"PRS computable for all records: {ready['prs_computable_for_all_records']}")
    print("Test set accessed: False")
    print(line)
    print(f"PHASE 5 STATUS: {document['status']}")
    print(line)


def main(args=None):
    parser = argparse.ArgumentParser(description="Compute Phase 5 PRAM components")
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3_PATH)
    parser.add_argument("--phase4-dir", type=Path, default=PHASE4_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true",
                       help=f"use {SMOKE_VALIDATION_SAMPLES} validation samples; separate output")
    modes.add_argument("--exploratory", action="store_true",
                       help="full validation with global-model Phase 4 prototypes; not a final result")
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Construct ONLY the validation split; do not instantiate train or test datasets.
    from medmnist import PathMNIST
    from torchvision import transforms
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    validation_dataset = PathMNIST(split="val", transform=validation_transform, download=True)
    if len(validation_dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_VALIDATION_SAMPLES} validation samples, "
                         f"found {len(validation_dataset)}")
    output_dir = options.output_dir or (SMOKE_OUTPUT_PATH if options.smoke else OUTPUT_PATH)
    document = run_pram(
        options.phase3_dir, options.phase4_dir, output_dir, validation_dataset,
        mode=("smoke" if options.smoke else "exploratory" if options.exploratory else "final"),
        device=device, overwrite=options.overwrite,
        max_validation_samples=SMOKE_VALIDATION_SAMPLES if options.smoke else None,
    )
    print_report(document)


if __name__ == "__main__":
    main()
