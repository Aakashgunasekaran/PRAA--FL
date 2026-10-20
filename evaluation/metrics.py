"""Phase 8 metrics: accuracy, balanced accuracy, macro-F1, macro AUROC (one-vs-rest),
negative log-likelihood, and convergence summaries.  Pure numpy/sklearn, unit tested."""

import math
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)


def compute_metrics(y_true, probabilities, num_classes: int) -> Dict:
    """All headline metrics from true labels and class probabilities [N, num_classes].

    AUROC is the macro average of one-vs-rest AUROCs over the classes that have BOTH
    positive and negative samples in ``y_true``; the classes used are reported
    (``auroc_classes_used``) so a missing class is visible, never zero-filled.
    """
    y = np.asarray(y_true).reshape(-1).astype(int)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != y.shape[0] or p.shape[1] != num_classes:
        raise ValueError(f"probabilities must have shape [{y.shape[0]}, {num_classes}], got {p.shape}")
    if y.size == 0:
        raise ValueError("empty evaluation set")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or Inf")
    if (p < -1e-9).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probabilities must be non-negative and sum to 1 per row")
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError("labels out of range")
    pred = p.argmax(axis=1)
    labels = list(range(num_classes))
    per_class_auc, used = {}, []
    for c in labels:
        positives = (y == c)
        if positives.any() and (~positives).any():
            per_class_auc[c] = float(roc_auc_score(positives.astype(int), p[:, c]))
            used.append(c)
        else:
            per_class_auc[c] = None
    auroc = float(np.mean([per_class_auc[c] for c in used])) if used else None
    nll = float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], 1e-12, 1.0))))
    return {
        "num_samples": int(y.size),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "auroc_macro_ovr": auroc,
        "auroc_per_class": per_class_auc,
        "auroc_classes_used": used,
        "nll": nll,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "class_support": [int((y == c).sum()) for c in labels],
    }


def rounds_to_target(values: Sequence[Optional[float]], target: float, rounds: Optional[Sequence[int]] = None):
    """First round whose value >= target, else None."""
    for i, v in enumerate(values):
        if v is not None and v >= target:
            return int(rounds[i]) if rounds is not None else i + 1
    return None


def convergence_summary(values: Sequence[Optional[float]], rounds: Optional[Sequence[int]] = None,
                        targets: Sequence[float] = (0.5, 0.6, 0.7, 0.8), last_k: int = 10) -> Dict:
    """Convergence descriptors of a per-round curve (e.g. validation accuracy)."""
    pairs = [(rounds[i] if rounds is not None else i + 1, v) for i, v in enumerate(values) if v is not None]
    if not pairs:
        raise ValueError("no recorded values")
    rs = [r for r, _ in pairs]
    vs = np.array([v for _, v in pairs], dtype=float)
    best_i = int(vs.argmax())
    return {
        "num_points": len(pairs),
        "final": float(vs[-1]),
        "final_mean_last_k": float(vs[-last_k:].mean()),
        "last_k": int(min(last_k, len(vs))),
        "best": float(vs[best_i]),
        "best_round": int(rs[best_i]),
        "mean_over_rounds": float(vs.mean()),  # normalised area under the curve
        "rounds_to_target": {str(t): rounds_to_target(vs.tolist(), t, rs) for t in targets},
        "round_to_95pct_of_best": rounds_to_target(vs.tolist(), 0.95 * float(vs.max()), rs),
    }
