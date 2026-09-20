from typing import Dict, List
import torch
from federated.flame import average_state_dicts, check_no_nan_inf
from training.validation import evaluate_classifier


class Server:
    def __init__(self, model_cls, model_kwargs=None, device="cpu"):
        self.device = device
        self.model = model_cls(**(model_kwargs or {})).to(device)

    def get_global_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_global_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def aggregate(self, client_states, client_sizes):
        if not client_states or len(client_states) != len(client_sizes):
            raise ValueError("client states and sizes must be non-empty and aligned")
        total = sum(client_sizes)
        if total <= 0:
            raise ValueError("client sample total must be positive")
        weights = [size / total for size in client_sizes]
        state = average_state_dicts(client_states, weights)
        nan, inf = check_no_nan_inf(state)
        if nan or inf:
            raise FloatingPointError("aggregated model contains NaN or Inf")
        return state, nan, inf

    def evaluate(self, val_loader):
        return evaluate_classifier(self.model, val_loader, self.device)
