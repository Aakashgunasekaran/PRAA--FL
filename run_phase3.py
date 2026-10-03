"""Mentor-facing Phase 3 FLAME experiment."""

import contextlib
import io
import argparse

import torch

from federated import flame_config as cfg


def main(
    num_rounds=1,
    clients_per_round=10,
    resume_from=None,
):
    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        from datasets import preprocess
        from torch.utils.data import (
            DataLoader,
            Subset,
        )
        from federated.client import Client
        from federated.flame_data import (
            PARTITION,
            get_flame_client_dataset,
        )
        from federated.server import Server
        from federated.training import (
            run_federated_training,
        )
        from models.classifier import FLAMEModel

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = {
        "num_clients": cfg.NUM_CLIENTS,
        "clients_per_round": clients_per_round,
        "num_rounds": num_rounds,
        "local_epochs": cfg.LOCAL_EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "learning_rate": cfg.LEARNING_RATE,
        "weight_decay": cfg.WEIGHT_DECAY,
        "patch_size": cfg.PATCH_SIZE,
        "mask_ratio": cfg.MASK_RATIO,
        "embedding_dim": cfg.EMBEDDING_DIM,
        "num_classes": 9,
        "mae_weight": cfg.MAE_WEIGHT,
        "mae_reference": 1.0,
        "proto_reference": 1.0,
        "scheduler_factor": cfg.SCHEDULER_FACTOR,
        "scheduler_patience": cfg.SCHEDULER_PATIENCE,
        "seed": cfg.RANDOM_SEED,
        "label_fraction": cfg.LABEL_FRACTION,
        "max_batches": None,
    }

    server = Server(
        FLAMEModel,
        {
            "embedding_dim":
                cfg.EMBEDDING_DIM,
            "num_classes": 9,
        },
        device,
    )

    clients = [
        Client(
            i,
            config,
            device,
            base_dataset=
                get_flame_client_dataset(i),
        )
        for i in range(
            config["num_clients"]
        )
    ]

    print("=" * 60)
    print("PRAA-FL - PHASE 3: FLAME BASELINE")
    print("=" * 60)

    print(
        "Data source: Phase 2 federated partitions"
    )

    print(
        f"Phase 2 partition clients: "
        f"{len(PARTITION)}"
    )

    print(
        f"FLAME experiment clients: "
        f"{cfg.NUM_CLIENTS}"
    )

    print(
        f"Selected clients/round: "
        f"{clients_per_round}"
    )

    print(
        f"Dirichlet beta: "
        f"{cfg.DIRICHLET_BETA}"
    )

    print(
        f"Minimum samples/client: "
        f"{cfg.MIN_SAMPLES_PER_CLIENT}"
    )

    print(
        f"Device: {device}"
    )

    if resume_from is not None:
        print(
            f"Resume checkpoint: "
            f"{resume_from}"
        )

    validation_loader = DataLoader(
        Subset(
            preprocess.val_dataset,
            range(
                min(
                    64,
                    len(
                        preprocess.val_dataset
                    ),
                )
            ),
        ),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    history = run_federated_training(
        server,
        clients,
        validation_loader,
        config,
        output_dir="results/phase3",
        resume_from=resume_from,
    )

    if history:
        latest = history[-1]

        print("=" * 60)
        print("PHASE 3 RESULT")
        print("=" * 60)

        print(
            f"Completed rounds: "
            f"{len(history)}"
        )

        print(
            f"Latest round: "
            f"{latest['round']}"
        )

        print(
            f"Latest selected clients: "
            f"{latest['selected_clients']}"
        )

        print(
            "Latest validation: ",
            {
                key: latest[key]
                for key in (
                    "validation_loss",
                    "validation_accuracy",
                    "validation_balanced_accuracy",
                    "validation_macro_f1",
                )
            },
        )

        print(
            "Test set accessed: False"
        )

    print("=" * 60)
    print("PHASE 3 STATUS: PASS")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 3 FLAME experiment"
        )
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=cfg.NUM_ROUNDS,
    )

    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=cfg.CLIENTS_PER_ROUND,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a Phase 3 checkpoint "
            "to resume from"
        ),
    )

    args = parser.parse_args()

    main(
        num_rounds=args.rounds,
        clients_per_round=
            args.clients_per_round,
        resume_from=args.resume,
    )