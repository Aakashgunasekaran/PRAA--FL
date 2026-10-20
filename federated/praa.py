"""Phase 6 (PRS -> client contribution) and Phase 7 (PRAA aggregation weights).

This module is pure PyTorch/Python (no datasets, no I/O) so every rule can be unit
tested.  It consumes PRAM records (``federated.pram``) and produces weights.

Definitions
-----------
Phase 6 - Prototype Reliability Score (PRS), per (client r, class c):

    PRS_{r,c} = (w_v * V_{r,c} + w_s * S_{r,c} + w_c * C_{r,c}) / (w_v + w_s + w_c)

with V = validation, S = stability, C = consistency (documented in
``docs/PHASE_5_PROTOTYPE_RELIABILITY_SCORE.md``; default weights 1/3 each).
STRICT policy (default, the documented rule): PRS exists only for records whose three
components are all ``valid``.  A missing/degenerate component is never replaced by 0,
1 or any constant, so PRS is ``None`` for that record.

Client reliability (project decision, see docs/PHASE_6_PRS.md):

    R_r = sum_c n_{r,c} * PRS_{r,c} / sum_c n_{r,c}       over classes with a PRS

(``client_reduce="mean"`` gives the unweighted mean instead).  A client with no PRS
record is ``unassessed``.

Phase 7 - Prototype Reliability-Aware Adaptive Aggregation (PRAA):

    m_r      = (R_r ** tau) / mean_{r' assessed}(R_{r'} ** tau)    clipped to [floor, cap]
    m_r      = 1                                    for unassessed clients, and for all
                                                    clients if fewer than ``min_assessed``
                                                    clients were assessed
    w_r      = nu_r * m_r / sum_j nu_j * m_j        nu_r = n_r / sum n  (FLAME weight)

and, for global prototypes (class level, same rule with PRS_{r,c} and holders of c):

    P_c = sum_r a_{r,c} p_{r,c},   a_{r,c} = n_{r,c} * m_{r,c} / sum_j n_{j,c} * m_{j,c}

``tau = 0`` or "no evidence" reproduces FLAME exactly.  "No evidence => FLAME weight" is
the documented missing-evidence rule at the *aggregation* level; PRS itself is never
imputed.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from federated.flame import average_state_dicts, check_no_nan_inf
from federated.pram import COMPONENTS, VALID
from federated.prs import PRSConfig

MISSING_POLICIES = ("strict", "renormalize")
CLIENT_REDUCTIONS = ("sample_weighted", "mean")


@dataclass(frozen=True)
class PRAAConfig:
    prs: PRSConfig = field(default_factory=PRSConfig)
    tau: float = 1.0                    # reliability exponent; 0 == FLAME weights
    weight_floor: float = 0.05          # lower clip of the relative multiplier
    weight_cap: float = 5.0             # upper clip of the relative multiplier
    min_assessed: int = 2               # fewer assessed clients => no reweighting
    client_reduce: str = "sample_weighted"
    # "strict" = documented rule.  "renormalize" = EXTENSION (not in the docs): average only
    # the valid components (needs >= 2).  Never the default; for explicitly labelled ablations.
    missing_component_policy: str = "strict"
    stability_window: Optional[int] = 3  # most recent snapshots used for stability (None = all)
    apply_to_model: bool = True         # reliability-aware parameter aggregation
    apply_to_prototypes: bool = True    # reliability-aware global-prototype aggregation

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("tau must be a finite number >= 0")
        if not 0 < self.weight_floor <= 1 <= self.weight_cap:
            raise ValueError("need 0 < weight_floor <= 1 <= weight_cap")
        if self.min_assessed < 1:
            raise ValueError("min_assessed must be >= 1")
        if self.client_reduce not in CLIENT_REDUCTIONS:
            raise ValueError(f"client_reduce must be one of {CLIENT_REDUCTIONS}")
        if self.missing_component_policy not in MISSING_POLICIES:
            raise ValueError(f"missing_component_policy must be one of {MISSING_POLICIES}")
        if self.stability_window is not None and self.stability_window < 2:
            raise ValueError("stability_window must be None or >= 2")

    def to_dict(self) -> Dict:
        out = asdict(self)
        return out

    @staticmethod
    def from_dict(data: Mapping) -> "PRAAConfig":
        data = dict(data)
        prs = data.pop("prs", None)
        prs_cfg = PRSConfig(**prs) if isinstance(prs, Mapping) else (prs or PRSConfig())
        return PRAAConfig(prs=prs_cfg, **data)


_PRS_WEIGHT_ATTR = {
    "validation": "validation_weight",
    "stability": "stability_weight",
    "consistency": "consistency_weight",
}


# ---------------------------------------------------------------------------
# Phase 6: PRS per record
# ---------------------------------------------------------------------------

def record_prs(row: Mapping, config: PRAAConfig) -> Dict:
    """PRS of one PRAM record.  Returns a dict with ``prs`` (float or None)."""
    prs_cfg = config.prs
    statuses = {name: row[name]["status"] for name in COMPONENTS}
    base = {
        "client_id": int(row["client_id"]),
        "class_id": int(row["class_id"]),
        "sample_count": int(row["sample_count"]),
        "component_status": statuses,
    }
    valid = [name for name in COMPONENTS if statuses[name] == VALID]
    if len(valid) == len(COMPONENTS):
        used, status, reason = list(COMPONENTS), "complete", None
    elif config.missing_component_policy == "renormalize" and len(valid) >= 2:
        used, status, reason = valid, "partial", "renormalized_over_valid_components"
    else:
        missing = {n: row[n]["reason"] for n in COMPONENTS if statuses[n] != VALID}
        return {**base, "prs": None, "prs_status": "unavailable", "used_components": [],
                "reason": f"components_not_valid:{missing}"}
    weights = {name: getattr(prs_cfg, _PRS_WEIGHT_ATTR[name]) for name in used}
    total = sum(weights.values())
    if total <= 0:
        return {**base, "prs": None, "prs_status": "unavailable", "used_components": used,
                "reason": "zero_total_weight_over_used_components"}
    score = sum(weights[name] * float(row[name]["value"]) for name in used) / total
    if not (math.isfinite(score) and 0.0 <= score <= 1.0 + 1e-9):
        raise ValueError(f"invalid PRS {score} for client {row['client_id']} class {row['class_id']}")
    return {**base, "prs": min(max(score, 0.0), 1.0), "prs_status": status,
            "used_components": used, "reason": reason}


def compute_record_prs(rows: Sequence[Mapping], config: PRAAConfig = PRAAConfig()) -> List[Dict]:
    return [record_prs(row, config) for row in rows]


# ---------------------------------------------------------------------------
# Phase 6: client reliability / contribution
# ---------------------------------------------------------------------------

def client_reliability(prs_records: Sequence[Mapping], client_ids: Sequence[int],
                       config: PRAAConfig = PRAAConfig()) -> Dict[int, Dict]:
    """Reduce per-class PRS to one reliability per client (None when unassessed)."""
    by_client: Dict[int, List[Mapping]] = {int(c): [] for c in client_ids}
    for rec in prs_records:
        by_client.setdefault(int(rec["client_id"]), []).append(rec)
    out = {}
    for cid, recs in sorted(by_client.items()):
        scored = [r for r in recs if r["prs"] is not None]
        if not scored:
            out[cid] = {"reliability": None, "status": "unassessed",
                        "num_records": len(recs), "num_scored": 0,
                        "reason": "no_record_with_a_valid_prs" if recs else "client_has_no_prototype_records"}
            continue
        if config.client_reduce == "sample_weighted":
            weights = [max(int(r["sample_count"]), 0) for r in scored]
            if sum(weights) == 0:
                weights = [1] * len(scored)
        else:
            weights = [1] * len(scored)
        value = sum(w * r["prs"] for w, r in zip(weights, scored)) / sum(weights)
        out[cid] = {"reliability": float(value), "status": "assessed",
                    "num_records": len(recs), "num_scored": len(scored), "reason": None}
    return out


# ---------------------------------------------------------------------------
# Phase 7: relative multipliers and aggregation weights
# ---------------------------------------------------------------------------

def relative_multipliers(reliabilities: Mapping[object, Optional[float]],
                         config: PRAAConfig = PRAAConfig()) -> Tuple[Dict, Dict]:
    """Map reliabilities (None = unassessed) to relative multipliers with mean 1 over assessed.

    Returns (multipliers, info).  Unassessed keys get exactly 1.0; if fewer than
    ``min_assessed`` keys are assessed, or tau == 0, every key gets 1.0.
    """
    assessed = {k: float(v) for k, v in reliabilities.items() if v is not None}
    for k, v in assessed.items():
        if not (math.isfinite(v) and 0.0 <= v <= 1.0 + 1e-9):
            raise ValueError(f"reliability for {k!r} must be in [0, 1], got {v}")
    info = {"num_assessed": len(assessed), "num_total": len(reliabilities), "applied": False,
            "reason": None}
    neutral = {k: 1.0 for k in reliabilities}
    if config.tau == 0:
        info["reason"] = "tau_is_zero"
        return neutral, info
    if len(assessed) < config.min_assessed:
        info["reason"] = "fewer_than_min_assessed"
        return neutral, info
    # R == 0 would give weight 0; the floor below keeps every client's influence positive.
    powered = {k: (max(v, 0.0) ** config.tau) for k, v in assessed.items()}
    mean = sum(powered.values()) / len(powered)
    if mean <= 0:
        info["reason"] = "all_assessed_reliabilities_are_zero"
        return neutral, info
    multipliers = dict(neutral)
    for k, p in powered.items():
        multipliers[k] = min(max(p / mean, config.weight_floor), config.weight_cap)
    info["applied"] = True
    return multipliers, info


def flame_weights(sizes: Mapping[int, int]) -> Dict[int, float]:
    """FLAME / FedAvg weights nu_r = n_r / sum n."""
    if not sizes:
        raise ValueError("sizes must be non-empty")
    total = float(sum(sizes.values()))
    if total <= 0 or any(s <= 0 for s in sizes.values()):
        raise ValueError("client sample sizes must be positive")
    return {int(c): s / total for c, s in sizes.items()}


def praa_client_weights(sizes: Mapping[int, int], multipliers: Mapping[int, float]) -> Dict[int, float]:
    """w_r = nu_r * m_r / sum_j nu_j * m_j (normalised, strictly positive)."""
    base = flame_weights(sizes)
    raw = {c: base[c] * float(multipliers.get(c, 1.0)) for c in base}
    if any((not math.isfinite(v)) or v <= 0 for v in raw.values()):
        raise ValueError("PRAA weights must be finite and strictly positive")
    total = sum(raw.values())
    return {c: v / total for c, v in raw.items()}


def round_aggregation_plan(
    sizes: Mapping[int, int],
    pram_rows: Sequence[Mapping],
    config: PRAAConfig = PRAAConfig(),
) -> Dict:
    """Everything the server needs for one round, with a full audit trail.

    ``pram_rows`` are PRAM records of the clients that participated this round.
    """
    client_ids = sorted(int(c) for c in sizes)
    prs = compute_record_prs(pram_rows, config)
    reliability = client_reliability(prs, client_ids, config)
    multipliers, info = relative_multipliers(
        {c: reliability[c]["reliability"] for c in client_ids}, config)
    base = flame_weights(sizes)
    weights = praa_client_weights(sizes, multipliers)
    return {
        "client_ids": client_ids,
        "flame_weights": base,
        "praa_weights": weights,
        "multipliers": multipliers,
        "multiplier_info": info,
        "reliability": reliability,
        "prs_records": prs,
        "max_weight_shift": max(abs(weights[c] - base[c]) for c in client_ids),
    }


def aggregate_model_states(states: Sequence[Mapping], weights: Sequence[float]):
    """Weighted parameter average (shared with FLAME's ``average_state_dicts``)."""
    if len(states) != len(weights) or not states:
        raise ValueError("states and weights must be aligned and non-empty")
    state = average_state_dicts(list(states), list(weights))
    nan, inf = check_no_nan_inf(state)
    if nan or inf:
        raise FloatingPointError("aggregated model contains NaN or Inf")
    return state


# ---------------------------------------------------------------------------
# Phase 7: global prototype aggregation
# ---------------------------------------------------------------------------

def class_multipliers(prs_records: Sequence[Mapping], config: PRAAConfig = PRAAConfig()) -> Tuple[Dict, Dict]:
    """Per-(client, class) multipliers: relative PRS among the holders of each class."""
    by_class: Dict[int, Dict[int, Optional[float]]] = {}
    for rec in prs_records:
        by_class.setdefault(int(rec["class_id"]), {})[int(rec["client_id"])] = rec["prs"]
    out: Dict[Tuple[int, int], float] = {}
    info = {}
    for class_id, holders in sorted(by_class.items()):
        mult, class_info = relative_multipliers(holders, config)
        info[class_id] = class_info
        for client_id, m in mult.items():
            out[(client_id, class_id)] = m
    return out, info


def aggregate_prototypes(
    client_prototypes: Mapping[int, Mapping[int, torch.Tensor]],
    client_counts: Mapping[int, Mapping[int, int]],
    multipliers: Optional[Mapping[Tuple[int, int], float]] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[int, float]]]:
    """Global prototypes P_c = sum_r a_{r,c} p_{r,c}.

    With ``multipliers=None`` the weights are n_{r,c}/sum n (FLAME: the pooled class mean
    of the clients' labelled support embeddings).  Returns (prototypes, weights) where
    ``weights[c][client_id]`` is a_{r,c} for auditing.
    """
    holders: Dict[int, List[Tuple[int, torch.Tensor, int]]] = {}
    for cid, protos in client_prototypes.items():
        for class_id, vector in protos.items():
            count = int(client_counts[cid][class_id])
            if count <= 0:
                raise ValueError(f"client {cid} class {class_id}: non-positive sample count")
            if not torch.isfinite(vector).all():
                raise FloatingPointError(f"client {cid} class {class_id}: non-finite prototype")
            holders.setdefault(int(class_id), []).append((int(cid), vector.detach().float().cpu(), count))
    prototypes, weights = {}, {}
    for class_id, items in sorted(holders.items()):
        raw = []
        for cid, _, count in items:
            m = 1.0 if multipliers is None else float(multipliers.get((cid, class_id), 1.0))
            if not math.isfinite(m) or m <= 0:
                raise ValueError("prototype multipliers must be finite and positive")
            raw.append(count * m)
        total = sum(raw)
        stacked = torch.stack([v for _, v, _ in items]).double()
        w = torch.tensor([r / total for r in raw], dtype=torch.float64)
        prototypes[class_id] = (stacked * w[:, None]).sum(dim=0).float()
        weights[class_id] = {cid: float(wi) for (cid, _, _), wi in zip(items, w.tolist())}
    return prototypes, weights
