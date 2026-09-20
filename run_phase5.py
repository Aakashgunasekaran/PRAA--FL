"""Phase 5 mentor runner: Prototype Reliability Assessment Module."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from federated.prs import PRSConfig, compute_prs, consistency_scores, stability_scores, validation_scores


ROOT = Path(__file__).resolve().parent
PHASE3 = ROOT / "results" / "phase3"
PHASE4 = ROOT / "results" / "phase4"
OUTPUT = ROOT / "results" / "phase5"


def main(max_validation_samples=None):
    phase4_summary = json.loads((PHASE4 / "summary.json").read_text())
    config = json.loads((PHASE3 / "config.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        from datasets import preprocess
        from models.classifier import FLAMEModel
        from run_phase4 import load_client_state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(PHASE3 / "final_model.pt", map_location=device, weights_only=False)
    model = FLAMEModel(
        embedding_dim=config["embedding_dim"], num_classes=config["num_classes"]
    ).to(device)
    validation_dataset = preprocess.val_dataset
    if max_validation_samples is not None:
        validation_dataset = torch.utils.data.Subset(
            validation_dataset,
            range(min(max_validation_samples, len(validation_dataset))),
        )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=False
    )

    prototype_dir = PHASE4 / "prototypes"
    temporal_root = PHASE4 / "snapshots"
    temporal_by_client = {}
    if temporal_root.exists():
        for round_dir in sorted(temporal_root.glob("round_*")):
            round_number = int(round_dir.name.split("_")[1])
            for snapshot_path in round_dir.glob("client_*.pt"):
                snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
                temporal_by_client.setdefault(int(snapshot["client_id"]), []).append(
                    (round_number, snapshot["prototypes"])
                )
    records = []
    local_count = 0
    fallback_count = 0
    for path in sorted(prototype_dir.glob("client_*.pt"), key=lambda item: int(item.stem.split("_")[1])):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        client_id = int(artifact["client_id"])
        prototypes = {int(class_id): value for class_id, value in artifact["prototypes"].items()}
        state, fallback = load_client_state(client_id, checkpoint, True)
        local_count += not fallback
        fallback_count += fallback
        model.load_state_dict(state)
        classes = sorted(prototypes)
        validation = validation_scores(
            model, validation_loader, classes, prototypes, device=device
        )
        reference = {}
        for other_path in prototype_dir.glob("client_*.pt"):
            other = torch.load(other_path, map_location="cpu", weights_only=True)
            for class_id, prototype in other["prototypes"].items():
                reference.setdefault(int(class_id), []).append(prototype)
        reference = {
            class_id: torch.stack(values).mean(dim=0)
            for class_id, values in reference.items()
        }
        consistency = consistency_scores(prototypes, reference)
        client_snapshots = temporal_by_client.get(client_id, [])
        client_snapshots.sort(key=lambda item: item[0])
        stability = stability_scores(
            [snapshot for _, snapshot in client_snapshots],
            classes,
        )
        prs = compute_prs(validation, stability, consistency)
        for class_id in classes:
            state_source = "GLOBAL_FALLBACK" if fallback else "LOCAL"
            if fallback:
                incomplete_reason = "missing_local_state"
            elif stability is None:
                incomplete_reason = "missing_temporal_snapshot"
            elif validation[class_id] is None:
                incomplete_reason = "missing_validation_information"
            elif prs[class_id] is None:
                incomplete_reason = "missing_reliability_component"
            else:
                incomplete_reason = None
            records.append({
                "client_id": client_id,
                "class_id": class_id,
                "state_source": state_source,
                "validation_score": validation[class_id],
                "prototype_stability": None,
                "prototype_consistency": consistency[class_id],
                "prs": prs[class_id],
                "prototype_dimension": int(artifact["metadata"][class_id]["embedding_dimension"]),
                "sample_count": int(artifact["metadata"][class_id]["sample_count"]),
                "stability_status": "unavailable_without_multiple_snapshots",
                "incomplete_reason": incomplete_reason,
            })
            if stability is not None:
                records[-1]["prototype_stability"] = stability[class_id]
                records[-1]["stability_status"] = "temporal_cosine_similarity"
                records[-1]["rounds"] = [round_number for round_number, _ in client_snapshots]
                records[-1]["prs"] = prs[class_id]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    total_clients = int(phase4_summary["num_clients"])
    complete_records = sum(record["prs"] is not None for record in records)
    incomplete_records = len(records) - complete_records
    if fallback_count:
        status = "INCOMPLETE_CLIENT_COVERAGE"
    elif incomplete_records:
        status = "INCOMPLETE_RECORDS"
    else:
        status = "COMPLETE"
    result = {
        "phase": 5,
        "status": status,
        "total_partition_clients": total_clients,
        "participating_clients": int(local_count),
        "non_participating_clients": int(total_clients - local_count),
        "clients_evaluated": len({record["client_id"] for record in records}),
        "records": records,
        "complete_records": complete_records,
        "incomplete_records": incomplete_records,
        "incomplete_reason_counts": {
            reason: sum(record["incomplete_reason"] == reason for record in records)
            for reason in sorted({
                record["incomplete_reason"]
                for record in records
                if record["incomplete_reason"] is not None
            })
        },
        "clients_with_local_states": int(local_count),
        "clients_using_global_fallback": int(fallback_count),
        "persisted_local_state_clients": sorted(
            client_id for client_id, snapshots in temporal_by_client.items()
            if snapshots
        ),
        "validation_data": "PathMNIST validation split only",
        "test_set_accessed": False,
        "stability_snapshots_available": any(
            len(snapshots) >= 2 for snapshots in temporal_by_client.values()
        ),
        "temporal_rounds": sorted({
            round_number
            for snapshots in temporal_by_client.values()
            for round_number, _ in snapshots
        }),
        "prs_formula": "weighted mean of validation, stability, and consistency; weights configurable",
        "prs_weights": {
            "validation": PRSConfig().validation_weight,
            "stability": PRSConfig().stability_weight,
            "consistency": PRSConfig().consistency_weight,
        },
    }
    (OUTPUT / "prs.json").write_text(json.dumps(result, indent=2))
    print("=" * 56)
    print("PHASE 5 — PRAM / PRS SANITY CHECK")
    print("=" * 56)
    print(f"Clients evaluated: {result['clients_evaluated']}")
    print(f"Records: {len(records)}")
    print(f"Local states: {local_count}")
    print(f"Global fallbacks: {fallback_count}")
    finite_stability = [
        record["prototype_stability"]
        for record in records
        if record["prototype_stability"] is not None
    ]
    print(
        "Prototype stability: "
        + ("FINITE" if finite_stability else "UNAVAILABLE")
    )
    print(f"Complete PRS records: {result['complete_records']}")
    print(f"Incomplete PRS records: {result['incomplete_records']}")
    print(f"Incomplete reasons: {result['incomplete_reason_counts']}")
    print("Test set accessed: False")
    print(f"PHASE 5 STATUS: {result['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()
    main(args.max_validation_samples)
