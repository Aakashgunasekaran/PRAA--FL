"""Mentor-facing Phase 2 demonstration using the canonical Phase 2 partition API."""

import contextlib
import io
import json
from pathlib import Path


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        from datasets import preprocess
        from federated import client_data
        from federated.flame_data import PARTITION
        from federated.flame_config import (
            DIRICHLET_BETA,
            MIN_SAMPLES_PER_CLIENT,
            NUM_CLIENTS,
            RANDOM_SEED,
        )

    verification = client_data.verify_partition() if NUM_CLIENTS == client_data.NUM_CLIENTS else {
        "sum_counts": sum(map(len, PARTITION.values())) == len(preprocess.train_dataset),
        "unique_assigned_count": len({i for values in PARTITION.values() for i in values}) == len(preprocess.train_dataset),
        "no_missing": {i for values in PARTITION.values() for i in values} == set(range(len(preprocess.train_dataset))),
        "indices_in_range": all(0 <= i < len(preprocess.train_dataset) for values in PARTITION.values() for i in values),
        "no_duplicates": sum(map(len, PARTITION.values())) == len({i for values in PARTITION.values() for i in values}),
    }
    same_seed = PARTITION == client_data.build_partition(NUM_CLIENTS, DIRICHLET_BETA, RANDOM_SEED, MIN_SAMPLES_PER_CLIENT)
    summary = {
        "num_clients": NUM_CLIENTS,
        "beta": DIRICHLET_BETA,
        "minimum_samples": MIN_SAMPLES_PER_CLIENT,
        "seed": RANDOM_SEED,
        "counts": {str(k): len(v) for k, v in PARTITION.items()},
        "verification": verification,
        "same_seed": same_seed,
    }
    output = Path("results/phase2")
    output.mkdir(parents=True, exist_ok=True)
    (output / "partition.json").write_text(json.dumps(PARTITION, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 60)
    print("PRAA-FL - PHASE 2: FEDERATED CLIENT PARTITIONING")
    print("=" * 60)
    print("\nConfiguration\n" + "-" * 60)
    print("Total Clients       :", NUM_CLIENTS)
    print("Partition Strategy  : Dirichlet")
    print("Beta                :", DIRICHLET_BETA)
    print("Minimum Samples     :", MIN_SAMPLES_PER_CLIENT)
    print("Random Seed         :", RANDOM_SEED)
    print("\nClient Partition\n" + "-" * 60)
    for client_id, count in summary["counts"].items():
        print(f"Client {client_id:>2} : {count} samples")
    print("\nPartition Verification\n" + "-" * 60)
    for key, value in (
        ("Completeness", verification["sum_counts"]),
        ("Duplicate Indices", verification["no_duplicates"]),
        ("Missing Indices", verification["no_missing"]),
        ("Valid Indices", verification["indices_in_range"]),
        ("Seed Reproducibility", same_seed),
    ):
        print(f"{key:<20}: {'PASS' if value else 'FAIL'}")
    print("\n" + "=" * 60)
    print("PHASE 2 STATUS: PASS" if all(verification.values()) and same_seed else "PHASE 2 STATUS: FAIL")
    print("=" * 60)


if __name__ == "__main__":
    main()
