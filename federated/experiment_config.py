"""Experiment configuration for the FedAvg / FLAME / PRAA runs.

The first block is the paper / Phase 3 configuration and is taken from
``federated.flame_config`` unchanged.  The second block holds the additions of Phases 5-8.
"""

import copy
import json
from pathlib import Path
from typing import Dict, Optional

from federated import flame_config as cfg
from federated.praa import PRAAConfig

METHODS = ("fedavg", "flame", "praa")
SCHEDULER_SIGNALS = ("method_val_loss", "classifier_head")


def default_config() -> Dict:
    return {
        # --- paper / Phase 3 (unchanged) -------------------------------------------------
        "num_clients": cfg.NUM_CLIENTS,
        "clients_per_round": cfg.CLIENTS_PER_ROUND,
        "num_rounds": cfg.NUM_ROUNDS,
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
        "dirichlet_beta": cfg.DIRICHLET_BETA,
        "min_samples_per_client": cfg.MIN_SAMPLES_PER_CLIENT,
        # --- additions (Phases 5-8) ------------------------------------------------------
        "method": "praa",
        "encoder": "resnet50",          # "small_cnn" only for tests / CPU smoke runs
        "feature_dim": None,            # None -> 2048 (ResNet-50)
        "val_samples": 2000,            # seeded subset of the validation split used each round
        "pram_val_samples": 1000,       # prefix of that subset used for per-client local PRAM
        "eval_every": 1,
        "checkpoint_every": 5,
        # "method_val_loss": scheduler driven by the method's own validation NLL/CE.
        # "classifier_head": original Phase 3 behaviour (loss of the UNTRAINED classifier head).
        "scheduler_signal": "method_val_loss",
        # Warm start from a saved global model (e.g. the PRESERVED Phase 3 final_model.pt) so
        # FLAME / PRAA aggregation can be compared for K further rounds without redoing the
        # original 200.  ``round_offset`` shifts the client-selection seed (round + offset),
        # e.g. 200 continues the original selection sequence.
        "init_from": None,
        "round_offset": 0,
        "praa": PRAAConfig().to_dict(),
    }


def validate_config(config: Dict) -> Dict:
    if config["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if config["scheduler_signal"] not in SCHEDULER_SIGNALS:
        raise ValueError(f"scheduler_signal must be one of {SCHEDULER_SIGNALS}")
    if not 0 < config["clients_per_round"] <= config["num_clients"]:
        raise ValueError("clients_per_round must be in (0, num_clients]")
    if config["num_rounds"] < 1:
        raise ValueError("num_rounds must be >= 1")
    if config["eval_every"] < 1 or config["checkpoint_every"] < 1:
        raise ValueError("eval_every and checkpoint_every must be >= 1")
    if config["round_offset"] < 0:
        raise ValueError("round_offset must be >= 0")
    if config["pram_val_samples"] > config["val_samples"]:
        raise ValueError("pram_val_samples must be <= val_samples")
    PRAAConfig.from_dict(config["praa"])  # raises on invalid PRAA settings
    return config


def load_config(path: Optional[str] = None, overrides: Optional[Dict] = None) -> Dict:
    config = default_config()
    if path:
        loaded = json.loads(Path(path).read_text())
        loaded = {k: v for k, v in loaded.items() if not k.startswith("_")}
        praa = {**config["praa"], **loaded.pop("praa", {})}
        config.update(loaded)
        config["praa"] = praa
    for key, value in (overrides or {}).items():
        if key.startswith("praa."):
            config["praa"][key.split(".", 1)[1]] = value
        else:
            if key not in config:
                raise KeyError(f"unknown config key {key!r}")
            config[key] = value
    return validate_config(copy.deepcopy(config))


def parse_overrides(items) -> Dict:
    """['num_rounds=3', 'praa.tau=2'] -> {'num_rounds': 3, 'praa.tau': 2} (JSON values)."""
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out
