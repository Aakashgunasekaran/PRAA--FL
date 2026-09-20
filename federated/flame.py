"""
FLAME server-side aggregation helpers.
This module isolates the FLAME aggregation procedure. For the baseline
implementation we implement the aggregation as weighted averaging of
client model parameters using client sample counts (ν_r = n_r / sum n_r),
as described in the paper (Algorithm 1 lines 7 and description).

If future FLAME-specific server-side weighting is specified in the paper,
this module is the place to adapt the aggregation logic.
"""

from collections import OrderedDict
import torch


def average_state_dicts(state_dicts, weights=None):
    """Weighted average of a list of state_dicts. If weights is None, use
    uniform averaging.

    Args:
        state_dicts: list of state_dict (mapping -> tensor)
        weights: list of floats (same length) summing to >0
    Returns:
        averaged state_dict
    """
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided")
    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    total = sum(weights)
    weights = [w / total for w in weights]

    avg = OrderedDict()
    for k in state_dicts[0].keys():
        values = [state_dict[k] for state_dict in state_dicts]
        reference = values[0]
        if reference.is_floating_point() or reference.is_complex():
            result = sum(value.to(dtype=reference.dtype) * weight for value, weight in zip(values, weights))
        elif reference.dtype == torch.bool:
            votes = sum(value.to(dtype=torch.float32) * weight for value, weight in zip(values, weights))
            result = votes >= 0.5
        else:
            total_value = sum(value.to(dtype=torch.float32) * weight for value, weight in zip(values, weights))
            result = total_value.round().to(dtype=reference.dtype)
        avg[k] = result
    return avg


def check_no_nan_inf(state_dict):
    nan = 0
    inf = 0
    for k, v in state_dict.items():
        t = v.detach().cpu()
        nan += int((t != t).any())
        inf += int((t == float('inf')).any() or (t == float('-inf')).any())
    return nan, inf
