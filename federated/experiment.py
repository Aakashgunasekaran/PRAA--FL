"""Unified federated experiment runner: FedAvg, FLAME (matched) and PRAA.

One round loop, three aggregation rules, identical data / client selection / local
training code (``federated.client.Client`` is the Phase 3 FLAME client, reused as is):

    fedavg  supervised cross-entropy on labelled samples, size-weighted average
    flame   MAE + prototypical loss (Phase 3), size-weighted average  nu_r = n_r / sum n
    praa    flame + reliability-aware weights  w_r ~ nu_r * m_r  (Phases 5-7)

The test split is never touched here; validation drives the LR scheduler and all
development decisions.  Per-round validation is a seeded subset (``val_samples``) of the
validation split.  The original Phase 3 run is not re-executed or modified.
"""

import contextlib
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from evaluation.inference import (classifier_logits_images, embed_images, prototype_logits,
                                  softmax_probabilities)
from evaluation.metrics import compute_metrics
from federated.client import Client
from federated.experiment_config import validate_config
from federated.fedavg_baseline import SupervisedClient
from federated.flame import average_state_dicts, check_no_nan_inf
from federated.praa import (PRAAConfig, aggregate_model_states, aggregate_prototypes,
                            class_multipliers, compute_record_prs, round_aggregation_plan)
from federated.reliability import ReliabilityTracker, local_evidence
from federated.sampling import select_clients
from federated.server import Server
from models.classifier import FLAMEModel, model_kwargs_from_config
from training.prototypes import generate_prototypes

SCALAR_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "auroc_macro_ovr", "nll")


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

class World:
    """Data, clients and cached tensors shared by training and evaluation."""

    def __init__(self, config: Dict, device: str, split: str = "val"):
        self.config, self.device = config, device
        with contextlib.redirect_stdout(io.StringIO()):
            from datasets import preprocess
            from federated.client_data import build_partition, get_client_dataset
        self.preprocess = preprocess
        self.partition = build_partition(
            num_clients=config["num_clients"], alpha=config["dirichlet_beta"],
            seed=config["seed"], min_samples_per_client=config["min_samples_per_client"])
        client_cls = SupervisedClient if config["method"] == "fedavg" else Client
        self.clients = [
            client_cls(i, config, device, base_dataset=get_client_dataset(i, partition=self.partition))
            for i in range(config["num_clients"])
        ]
        self.label_cache = self._build_label_cache()
        self.val_images, self.val_labels = self.validation_subset(config["val_samples"])

    def _build_label_cache(self) -> Dict[int, tuple]:
        cache = {}
        for client in self.clients:
            loader = torch.utils.data.DataLoader(client.labeled_dataset, batch_size=256, shuffle=False)
            images, labels = [], []
            for x, y in loader:
                images.append(x)
                labels.append(y.view(-1))
            cache[client.client_id] = (torch.cat(images), torch.cat(labels))
        return cache

    def validation_subset(self, size: int):
        dataset = self.preprocess.val_dataset
        assert_split(dataset, "val")
        n = min(int(size), len(dataset))
        idx = np.random.RandomState(self.config["seed"]).permutation(len(dataset))[:n]
        return load_tensors(dataset, sorted(idx.tolist()))


def assert_split(dataset, expected: str) -> None:
    base = dataset
    while hasattr(base, "dataset") and hasattr(base, "indices"):
        base = base.dataset
    split = getattr(base, "split", None)
    if split is not None and str(split) != expected:
        raise PermissionError(f"expected the {expected!r} split but dataset reports {split!r}")


def load_tensors(dataset, indices) -> tuple:
    images, labels = [], []
    for i in indices:
        x, y = dataset[int(i)]
        images.append(x)
        labels.append(int(np.asarray(y).reshape(-1)[0]))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compact(metrics: Dict) -> Dict:
    return {k: metrics[k] for k in SCALAR_KEYS}


def client_prototypes_with_model(model, label_cache, device) -> tuple:
    """Prototypes of every client from ONE model (the Phase 4 procedure), single forward pass."""
    ids = sorted(label_cache)
    sizes = [label_cache[c][0].size(0) for c in ids]
    embeddings = embed_images(model, torch.cat([label_cache[c][0] for c in ids]), device=device)
    protos, counts, start = {}, {}, 0
    for cid, size in zip(ids, sizes):
        p, n = generate_prototypes(embeddings[start:start + size], label_cache[cid][1])
        protos[cid], counts[cid] = p, n
        start += size
    return protos, counts


def evaluate_prototype_family(model, label_cache, reliability_images, reliability_labels,
                              eval_images, eval_labels, tracker: ReliabilityTracker,
                              praa_cfg: PRAAConfig, round_id: int, device: str,
                              num_classes: int, full: bool = False) -> Dict:
    """Evaluate FLAME's rule and PRAA's rule on ``eval_*`` with the SAME global model.

    Reliability (PRAM -> PRS -> multipliers) always uses ``reliability_*`` = validation data;
    ``eval_*`` may be the test split in the final evaluation.
    """
    protos, counts = client_prototypes_with_model(model, label_cache, device)
    gp_flame, _ = aggregate_prototypes(protos, counts, None)
    rel_emb = embed_images(model, reliability_images, device=device)
    rows = tracker.assess_global_prototypes(round_id, protos, counts, rel_emb, reliability_labels)
    prs = compute_record_prs(rows, praa_cfg)
    multipliers, info = class_multipliers(prs, praa_cfg)
    gp_praa, w_praa = aggregate_prototypes(protos, counts, multipliers)
    eval_emb = rel_emb if eval_images is reliability_images else embed_images(model, eval_images, device=device)
    out = {}
    for name, gp in (("flame_rule", gp_flame), ("praa_rule", gp_praa)):
        logits, missing = prototype_logits(eval_emb, gp, num_classes)
        metrics = compute_metrics(eval_labels.numpy(), softmax_probabilities(logits).numpy(), num_classes)
        metrics["missing_prototype_classes"] = missing
        out[name] = metrics if full else {**compact(metrics), "missing_prototype_classes": missing}
    scored = [r for r in prs if r["prs"] is not None]
    out["reliability"] = {
        "records": len(prs), "records_with_prs": len(scored),
        "records_unavailable": len(prs) - len(scored),
        "classes_reweighted": int(sum(1 for i in info.values() if i["applied"])),
        "mean_prs": float(np.mean([r["prs"] for r in scored])) if scored else None,
        "prs_status_counts": {s: sum(r["prs_status"] == s for r in prs)
                              for s in ("complete", "partial", "unavailable")},
    }
    if full:
        out["pram_rows"], out["prs_records"] = rows, prs
        out["praa_prototype_weights"] = {str(c): {str(k): v for k, v in w.items()} for c, w in w_praa.items()}
    return out


def evaluate_classifier_head(model, images, labels, device, num_classes, full=False) -> Dict:
    logits = classifier_logits_images(model, images, device=device)
    metrics = compute_metrics(labels.numpy(), softmax_probabilities(logits).numpy(), num_classes)
    return metrics if full else compact(metrics)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, round_number, server, clients, history, config):
    torch.save({
        "round": int(round_number),
        "global_model_state": server.get_global_state(),
        "history": history,
        "config": config,
        "client_states": {str(c.client_id): {
            "learning_rate": float(c._learning_rate),
            "scheduler_state": c._scheduler_state,
            "pending_validation_loss": c._pending_validation_loss,
            # shuffling RNG of the client's loaders (shared generator) -> exact resume
            "loader_generator_state": c.labeled_loader.generator.get_state()} for c in clients},
    }, path)


def _prune_checkpoints(directory: Path, keep: int = 2):
    files = sorted(directory.glob("checkpoint_round_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in files[:-keep]:
        old.unlink()


def _latest_checkpoint(directory: Path) -> Optional[Path]:
    files = sorted(directory.glob("checkpoint_round_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return files[-1] if files else None


def _truncate_jsonl(path: Path, max_round: int):
    if not path.exists():
        return
    kept = [line for line in path.read_text().splitlines() if line and json.loads(line)["round"] <= max_round]
    path.write_text("".join(line + "\n" for line in kept))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_experiment(config: Dict, output_dir, device: Optional[str] = None, resume: bool = False,
                   overwrite: bool = False, log=print, world: Optional[World] = None) -> List[Dict]:
    validate_config(config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    method = config["method"]
    praa_cfg = PRAAConfig.from_dict(config["praa"])
    out = Path(output_dir)
    ckpt_dir = out / "checkpoints"
    if (out / "final_model.pt").exists() and not overwrite and not resume:
        raise FileExistsError(f"{out} already holds a finished run; use overwrite or a new directory")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    world = world or World(config, device)
    server = Server(FLAMEModel, model_kwargs_from_config(config), device)
    if config.get("init_from"):
        init = torch.load(config["init_from"], map_location="cpu", weights_only=False)
        server.set_global_state(init["model_state_dict"])  # strict load; shape mismatch raises
        log(f"[init] warm start from {config['init_from']}")
    tracker = ReliabilityTracker(out, praa_cfg.stability_window) if method != "fedavg" else None
    pram_images = world.val_images[:config["pram_val_samples"]]
    pram_labels = world.val_labels[:config["pram_val_samples"]]

    history: List[Dict] = []
    start_round = 1
    if resume:
        latest = _latest_checkpoint(ckpt_dir)
        if latest is None:
            log("[resume] no checkpoint found; starting from round 1")
        else:
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            server.set_global_state(ckpt["global_model_state"])
            history = ckpt["history"]
            for client in world.clients:
                state = ckpt["client_states"].get(str(client.client_id))
                if state:
                    client._learning_rate = float(state["learning_rate"])
                    client._scheduler_state = state["scheduler_state"]
                    client._pending_validation_loss = state["pending_validation_loss"]
                    if state.get("loader_generator_state") is not None:
                        client.labeled_loader.generator.set_state(state["loader_generator_state"])
            start_round = int(ckpt["round"]) + 1
            if tracker is not None:
                tracker.load_from_disk(max_round=int(ckpt["round"]), delete_newer=True)
            _truncate_jsonl(out / "reliability_log.jsonl", int(ckpt["round"]))
            log(f"[resume] continuing from round {start_round} ({latest.name})")
    elif tracker is not None and (out / "snapshots").exists():
        raise FileExistsError(f"{out / 'snapshots'} exists; use resume or a clean directory")
    if not resume or start_round == 1:
        (out / "reliability_log.jsonl").unlink(missing_ok=True)  # a fresh run never appends to a stale log
    (out / "config.json").write_text(json.dumps(config, indent=2))

    def hook(client):
        return local_evidence(client.model, client.labeled_dataset, pram_images, device=device)

    for round_number in range(start_round, config["num_rounds"] + 1):
        t0 = time.time()
        selected = select_clients(config["num_clients"], config["clients_per_round"],
                                  config["seed"], round_number + config["round_offset"])
        before = server.get_global_state()
        states, sizes, metrics_rows, evidence = [], {}, [], {}
        for cid in selected:
            state, size, stats = world.clients[cid].client_update(
                before, after_update=hook if method != "fedavg" else None)
            if method != "fedavg":
                evidence[cid] = stats.pop("hook_result")
            states.append(state)
            sizes[cid] = size
            metrics_rows.append({"client_id": int(cid), **stats})

        entry = {"round": int(round_number), "selected_clients": selected, "client_metrics": metrics_rows}
        if method == "fedavg":
            total = sum(sizes.values())
            weights = [sizes[c] / total for c in selected]
            new_state = average_state_dicts(states, weights)
            nan, inf = check_no_nan_inf(new_state)
            if nan or inf:
                raise FloatingPointError("aggregated model contains NaN or Inf")
            entry["weights"] = {"used": "fedavg", "weights": dict(zip(map(str, selected), weights))}
        else:
            rows = tracker.assess_round(round_number, evidence, pram_labels)
            plan = round_aggregation_plan(sizes, rows, praa_cfg)
            use_praa = method == "praa" and praa_cfg.apply_to_model
            chosen = plan["praa_weights"] if use_praa else plan["flame_weights"]
            new_state = aggregate_model_states(states, [chosen[c] for c in selected])
            entry["weights"] = {
                "used": "praa" if use_praa else "flame",
                "flame": {str(c): plan["flame_weights"][c] for c in selected},
                "praa": {str(c): plan["praa_weights"][c] for c in selected},
                "multipliers": {str(c): plan["multipliers"][c] for c in selected},
                "reliability": {str(c): plan["reliability"][c]["reliability"] for c in selected},
                "num_assessed": plan["multiplier_info"]["num_assessed"],
                "reweighting_applied": plan["multiplier_info"]["applied"],
                "max_weight_shift": plan["max_weight_shift"],
            }
            with open(out / "reliability_log.jsonl", "a") as handle:
                handle.write(json.dumps({"round": int(round_number), "plan": {
                    k: plan[k] for k in ("flame_weights", "praa_weights", "multipliers",
                                         "multiplier_info", "reliability", "prs_records")}}) + "\n")
        server.set_global_state(new_state)

        evaluate_now = (round_number % config["eval_every"] == 0) or round_number == config["num_rounds"]
        if evaluate_now:
            if method == "fedavg":
                val = evaluate_classifier_head(server.model, world.val_images, world.val_labels, device,
                                               config["num_classes"])
                entry["validation"] = {"classifier_head": val}
                signal = val["nll"]
                entry.update(validation_loss=signal, validation_accuracy=val["accuracy"],
                             validation_macro_f1=val["macro_f1"],
                             validation_auroc=val["auroc_macro_ovr"])
            else:
                fam = evaluate_prototype_family(
                    server.model, world.label_cache, world.val_images, world.val_labels,
                    world.val_images, world.val_labels, tracker, praa_cfg, round_number, device,
                    config["num_classes"])
                entry["validation"] = fam
                own = fam["praa_rule"] if (method == "praa" and praa_cfg.apply_to_prototypes) else fam["flame_rule"]
                signal = own["nll"]
                entry.update(validation_loss=signal, validation_accuracy=own["accuracy"],
                             validation_macro_f1=own["macro_f1"], validation_auroc=own["auroc_macro_ovr"])
            if config["scheduler_signal"] == "classifier_head":
                signal = evaluate_classifier_head(server.model, world.val_images, world.val_labels,
                                                  device, config["num_classes"])["nll"]
                entry["scheduler_signal_value"] = signal
            for cid in selected:
                world.clients[cid].step_scheduler(signal)
        entry["learning_rate"] = world.clients[selected[0]].learning_rate
        entry["elapsed_sec"] = round(time.time() - t0, 3)
        history.append(entry)

        if round_number % config["checkpoint_every"] == 0 or round_number == config["num_rounds"]:
            _save_checkpoint(ckpt_dir / f"checkpoint_round_{round_number}.pt", round_number, server,
                             world.clients, history, config)
            _prune_checkpoints(ckpt_dir)
        va = entry.get("validation_accuracy")
        log(f"[{method}] round {round_number}/{config['num_rounds']} "
            f"val_acc={'n/a' if va is None else f'{va:.4f}'} "
            f"lr={entry['learning_rate']:.2e} {entry['elapsed_sec']:.1f}s")

    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    torch.save({"model_state_dict": server.get_global_state(), "history": history, "config": config},
               out / "final_model.pt")
    return history
