"""Reusable Algorithm 1 loop with validation-driven scheduling."""

from pathlib import Path
import json
import torch
from federated.sampling import select_clients


def run_federated_training(server, clients, val_loader, config, output_dir=None):
    history = []
    client_state_dir = None
    if output_dir:
        client_state_dir = Path(output_dir) / "client_states"
        client_state_dir.mkdir(parents=True, exist_ok=True)
    for round_number in range(1, config["num_rounds"] + 1):
        selected = select_clients(
            config["num_clients"], config["clients_per_round"],
            config["seed"], round_number
        )
        states, sizes, client_metrics = [], [], []
        before = server.get_global_state()
        for client_id in selected:
            state, size, metrics = clients[client_id].client_update(before)
            states.append(state)
            sizes.append(size)
            client_metrics.append({"client_id": client_id, **metrics})
            if client_state_dir is not None:
                payload = {
                    "client_id": int(client_id),
                    "round": int(round_number),
                    "model_state_dict": state,
                }
                torch.save(payload, client_state_dir / f"client_{client_id}.pt")
                round_dir = client_state_dir / f"round_{round_number}"
                round_dir.mkdir(exist_ok=True)
                torch.save(payload, round_dir / f"client_{client_id}.pt")
        server.set_global_state(server.aggregate(states, sizes)[0])
        validation = server.evaluate(val_loader)
        for client_id in selected:
            clients[client_id].step_scheduler(validation["validation_loss"])
        history.append({
            "round": round_number,
            "selected_clients": selected,
            "client_metrics": client_metrics,
            **validation,
            "learning_rate": clients[selected[0]].learning_rate,
        })
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config, indent=2))
        (path / "metrics.json").write_text(json.dumps(history, indent=2))
        torch.save({"model_state_dict": server.get_global_state(), "history": history}, path / "final_model.pt")
    return history
