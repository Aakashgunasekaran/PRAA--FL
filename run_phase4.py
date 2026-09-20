"""Phase 4 mentor demo: client-specific prototype generation."""

import contextlib
import io
import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "phase3" / "final_model.pt"
CONFIG_PATH = PROJECT_ROOT / "results" / "phase3" / "config.json"
CLIENT_STATE_PATH = PROJECT_ROOT / "results" / "phase3" / "client_states"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase4"
MAX_SAMPLES_PER_CLIENT = 64


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


def main(allow_global_fallback=True, temporal=False):
    config = json.loads(CONFIG_PATH.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        from federated.flame_data import PARTITION, get_flame_client_dataset
        from models.classifier import FLAMEModel
        from training.prototypes import generate_client_prototypes

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = FLAMEModel(
        embedding_dim=config["embedding_dim"], num_classes=config["num_classes"]
    ).to(device)

    client_datasets = {
        client_id: get_flame_client_dataset(client_id)
        for client_id in sorted(PARTITION)
    }
    generated = {}
    clients_with_local_states = 0
    clients_using_global_fallback = 0
    missing_clients = []
    for client_id, dataset in client_datasets.items():
        try:
            state, used_fallback = load_client_state(
                client_id, checkpoint, allow_global_fallback
            )
        except FileNotFoundError:
            missing_clients.append(client_id)
            continue
        clients_with_local_states += not used_fallback
        clients_using_global_fallback += used_fallback
        model.load_state_dict(state)
        result = generate_client_prototypes(
            model, {client_id: dataset}, batch_size=config["batch_size"],
            device=device, max_samples_per_client=MAX_SAMPLES_PER_CLIENT,
        )[client_id]
        for metadata in result["metadata"].values():
            if metadata["client_id"] != client_id:
                raise ValueError(f"Prototype metadata mismatch for client {client_id}")
        generated[client_id] = result

    if missing_clients:
        raise RuntimeError(
            f"Missing local Phase 3 states for clients: {missing_clients}. "
            "Run the required Phase 3 experiment first."
        )

    prototype_dir = OUTPUT_PATH / "prototypes"
    prototype_dir.mkdir(parents=True, exist_ok=True)
    classes_observed = set()
    total_prototypes = 0
    embedding_dimensions = set()
    for client_id, result in generated.items():
        prototypes = result["prototypes"]
        metadata = result["metadata"]
        classes_observed.update(prototypes)
        total_prototypes += len(prototypes)
        embedding_dimensions.add(result["embedding_dimension"])
        torch.save(
            {
                "client_id": client_id,
                "prototypes": prototypes,
                "metadata": metadata,
                "samples_processed": result["samples_processed"],
                "representation": "FLAME projected embedding (model.proto(model.encoder(x)))",
            },
            prototype_dir / f"client_{client_id}.pt",
        )

    summary = {
        "phase": 4,
        "status": "PASS" if not allow_global_fallback else "SMOKE_PASS",
        "dataset": "PathMNIST",
        "data_source": (
            "Phase 3 FLAME local representations"
            if not allow_global_fallback
            else "Phase 3 FLAME local representations with explicit smoke fallback"
        ),
        "num_clients": len(PARTITION),
        "embedding_dimension": next(iter(embedding_dimensions)),
        "clients_processed": len(generated),
        "clients_with_local_states": int(clients_with_local_states),
        "clients_using_global_fallback": int(clients_using_global_fallback),
        "clients_with_prototypes": sum(
            bool(item["prototypes"]) for item in generated.values()
        ),
        "total_prototypes": total_prototypes,
        "classes_observed": sorted(classes_observed),
        "max_samples_per_client": MAX_SAMPLES_PER_CLIENT,
        "test_set_accessed": False,
    }
    summary["final_experiment_ready"] = (
        summary["clients_processed"] == summary["num_clients"]
        and summary["clients_with_local_states"] == summary["num_clients"]
        and summary["clients_using_global_fallback"] == 0
        and not summary["test_set_accessed"]
        and all(
            torch.isfinite(prototype).all()
            for result in generated.values()
            for prototype in result["prototypes"].values()
        )
    )
    (OUTPUT_PATH / "summary.json").write_text(json.dumps(summary, indent=2))
    temporal_files = 0
    temporal_rounds = 0
    if temporal:
        temporal_files, temporal_rounds = generate_round_snapshots(
            config, device, max_samples_per_client=MAX_SAMPLES_PER_CLIENT
        )
        summary["temporal_snapshot_files"] = temporal_files
        summary["temporal_rounds"] = temporal_rounds
        (OUTPUT_PATH / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 60)
    print("PRAA-FL - PHASE 4: PROTOTYPE GENERATION")
    print("=" * 60)
    print()
    print("Dataset: PathMNIST")
    print("Data source: Phase 3 FLAME representations")
    print()
    print(f"Phase 2 clients: {len(PARTITION)}")
    print(f"Clients processed: {summary['clients_processed']}")
    print(f"Clients with local states: {summary['clients_with_local_states']}")
    print(f"Clients using global fallback: {summary['clients_using_global_fallback']}")
    print("Representation: FLAME projected embedding (encoder -> projection head)")
    print(f"Embedding dimension: {summary['embedding_dimension']}")
    print()
    print("Prototype generation:")
    for client_id, result in generated.items():
        print(f"Client {client_id}:")
        print(f"    classes: {sorted(result['prototypes'])}")
        print(f"    prototypes generated: {len(result['prototypes'])}")
    print()
    print(f"Total clients processed: {summary['clients_processed']}")
    print(f"Total prototypes generated: {summary['total_prototypes']}")
    print(f"Classes represented: {summary['classes_observed']}")
    print()
    print("Test set accessed: False")
    print(
        "FINAL EXPERIMENT READINESS: "
        + ("READY" if summary["final_experiment_ready"] else "NOT READY")
    )
    print("=" * 60)
    print(f"PHASE 4 STATUS: {summary['status']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 4 prototypes")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require an actual local Phase 3 state for every client",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Generate round-scoped prototypes from Phase 3 round snapshots",
    )
    args = parser.parse_args()
    main(allow_global_fallback=not args.final, temporal=args.temporal)
