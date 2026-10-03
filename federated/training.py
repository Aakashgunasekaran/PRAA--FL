"""Reusable Algorithm 1 loop with validation-driven scheduling
and checkpoint/resume support.
"""

from pathlib import Path
import json
import torch

from federated.sampling import select_clients


def _get_client_runtime_state(client):
    """Capture client state that persists across federated rounds."""
    return {
        "learning_rate": float(client._learning_rate),
        "scheduler_state": client._scheduler_state,
        "pending_validation_loss": client._pending_validation_loss,
    }


def _restore_client_runtime_state(client, state):
    """Restore client state required to continue training."""
    client._learning_rate = float(state["learning_rate"])
    client._scheduler_state = state["scheduler_state"]
    client._pending_validation_loss = state[
        "pending_validation_loss"
    ]


def _save_checkpoint(
    checkpoint_path,
    round_number,
    server,
    clients,
    history,
):
    """Save the complete state required to resume training."""

    checkpoint = {
        "round": int(round_number),
        "global_model_state": server.get_global_state(),
        "history": history,
        "client_states": {
            str(client.client_id): _get_client_runtime_state(client)
            for client in clients
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def _load_checkpoint(
    checkpoint_path,
    server,
    clients,
):
    """Restore global model, history, and client runtime state."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    server.set_global_state(
        checkpoint["global_model_state"]
    )

    saved_clients = checkpoint["client_states"]

    for client in clients:
        client_id = str(client.client_id)

        if client_id in saved_clients:
            _restore_client_runtime_state(
                client,
                saved_clients[client_id],
            )

    return (
        int(checkpoint["round"]),
        checkpoint["history"],
    )


def run_federated_training(
    server,
    clients,
    val_loader,
    config,
    output_dir=None,
    resume_from=None,
):
    history = []

    checkpoint_dir = None

    if output_dir:
        output_path = Path(output_dir)

        checkpoint_dir = (
            output_path / "checkpoints"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    start_round = 1

    # ---------------------------------------------------------
    # Resume from checkpoint
    # ---------------------------------------------------------
    if resume_from is not None:

        checkpoint_path = Path(resume_from)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        completed_round, history = _load_checkpoint(
            checkpoint_path,
            server,
            clients,
        )

        start_round = completed_round + 1

        print("=" * 60)
        print("RESUMING FEDERATED TRAINING")
        print("=" * 60)
        print(
            f"Checkpoint: {checkpoint_path}"
        )
        print(
            f"Completed rounds: {completed_round}"
        )
        print(
            f"Starting from round: {start_round}"
        )
        print("=" * 60)

    # ---------------------------------------------------------
    # Federated training loop
    # ---------------------------------------------------------
    for round_number in range(
        start_round,
        config["num_rounds"] + 1,
    ):

        selected = select_clients(
            config["num_clients"],
            config["clients_per_round"],
            config["seed"],
            round_number,
        )

        states = []
        sizes = []
        client_metrics = []

        # Global model BEFORE local client training
        before = server.get_global_state()

        # -----------------------------------------------------
        # Client local training
        # -----------------------------------------------------
        for client_id in selected:

            state, size, metrics = clients[
                client_id
            ].client_update(
                before
            )

            states.append(state)
            sizes.append(size)

            client_metrics.append(
                {
                    "client_id": int(client_id),
                    **metrics,
                }
            )

        # -----------------------------------------------------
        # Server aggregation
        # -----------------------------------------------------
        aggregated_state = server.aggregate(
            states,
            sizes,
        )[0]

        server.set_global_state(
            aggregated_state
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        validation = server.evaluate(
            val_loader
        )

        # -----------------------------------------------------
        # Scheduler update
        # -----------------------------------------------------
        for client_id in selected:

            clients[
                client_id
            ].step_scheduler(
                validation[
                    "validation_loss"
                ]
            )

        # -----------------------------------------------------
        # Record completed round
        # -----------------------------------------------------
        history.append(
            {
                "round": int(round_number),
                "selected_clients": selected,
                "client_metrics": client_metrics,
                **validation,
                "learning_rate": clients[
                    selected[0]
                ].learning_rate,
            }
        )

        # -----------------------------------------------------
        # Save checkpoint AFTER the entire round is complete
        # -----------------------------------------------------
        if checkpoint_dir is not None:

            checkpoint_path = (
                checkpoint_dir
                / f"checkpoint_round_{round_number}.pt"
            )

            _save_checkpoint(
                checkpoint_path,
                round_number,
                server,
                clients,
                history,
            )

            print(
                f"[Checkpoint] Round "
                f"{round_number} saved."
            )

    # ---------------------------------------------------------
    # Final artifacts
    # ---------------------------------------------------------
    if output_dir:

        path = Path(output_dir)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save configuration
        (path / "config.json").write_text(
            json.dumps(
                config,
                indent=2,
            )
        )

        # Save complete metrics history
        (path / "metrics.json").write_text(
            json.dumps(
                history,
                indent=2,
            )
        )

        # Save final global model
        torch.save(
            {
                "model_state_dict":
                    server.get_global_state(),
                "history": history,
            },
            path / "final_model.pt",
        )

    return history