"""Phase 8: metrics, inference rules, split guard."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from evaluation.inference import prototype_logits, softmax_probabilities
from evaluation.metrics import compute_metrics, convergence_summary, rounds_to_target
from run_phase8 import build_comparison, load_eval_split


def _data(n=120, k=4, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, k, n)
    p = rng.dirichlet(np.ones(k), n)
    return y, p


def test_metrics_match_sklearn_reference():
    y, p = _data()
    m = compute_metrics(y, p, 4)
    pred = p.argmax(1)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["auroc_macro_ovr"] == pytest.approx(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    assert m["nll"] == pytest.approx(float(-np.mean(np.log(p[np.arange(len(y)), y]))))
    assert sum(map(sum, m["confusion_matrix"])) == len(y)


def test_metrics_perfect_and_chance_extremes():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y]
    m = compute_metrics(y, perfect, 3)
    assert m["accuracy"] == m["macro_f1"] == m["auroc_macro_ovr"] == 1.0
    uniform = np.full((6, 3), 1 / 3)
    assert compute_metrics(y, uniform, 3)["auroc_macro_ovr"] == pytest.approx(0.5)


def test_auroc_reports_classes_without_both_labels_instead_of_zero_filling():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.tile([0.3, 0.3, 0.4], (6, 1))
    p[y == 0, 0], p[y == 1, 1] = 0.5, 0.5
    p /= p.sum(1, keepdims=True)
    m = compute_metrics(y, p, 3)
    assert m["auroc_classes_used"] == [0, 1] and m["auroc_per_class"][2] is None


@pytest.mark.parametrize("bad", ["nan", "rowsum", "shape", "label"])
def test_metrics_reject_invalid_input(bad):
    y, p = _data(10, 3)
    if bad == "nan":
        p[0, 0] = np.nan
    elif bad == "rowsum":
        p = p * 2
    elif bad == "shape":
        p = p[:, :2]
    else:
        y = y + 5
    with pytest.raises(ValueError):
        compute_metrics(y, p, 3)


def test_prototype_logits_and_probabilities():
    emb = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    protos = {0: torch.zeros(2), 2: torch.full((2,), 10.0)}
    logits, missing = prototype_logits(emb, protos, 3)
    assert missing == [1] and torch.isinf(logits[:, 1]).all()
    probs = softmax_probabilities(logits)
    assert probs[:, 1].sum() == 0 and torch.allclose(probs.sum(1), torch.ones(2))
    assert probs[0].argmax() == 0 and probs[1].argmax() == 2
    with pytest.raises(ValueError):
        prototype_logits(emb, {0: torch.zeros(3)}, 3)
    with pytest.raises(ValueError):
        prototype_logits(emb, {}, 3)


def test_convergence_summary_and_targets():
    curve = [0.1, 0.4, 0.55, 0.7, 0.68, 0.8]
    s = convergence_summary(curve, targets=(0.5, 0.7, 0.9))
    assert s["rounds_to_target"] == {"0.5": 3, "0.7": 4, "0.9": None}
    assert s["best"] == 0.8 and s["best_round"] == 6 and s["final"] == 0.8
    assert s["mean_over_rounds"] == pytest.approx(np.mean(curve))
    assert rounds_to_target([None, 0.2, 0.6], 0.5) == 3
    with pytest.raises(ValueError):
        convergence_summary([None, None])


class _Split:
    def __init__(self, name):
        self.split = name
        self.labels = np.zeros((3, 1), dtype=int)

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return torch.zeros(3, 2, 2), np.array([0])


class _Pre:
    val_dataset, test_dataset = _Split("val"), _Split("test")


def test_test_split_requires_explicit_final_flag():
    with pytest.raises(PermissionError):
        load_eval_split(_Pre, "test", final=False)
    images, labels = load_eval_split(_Pre, "test", final=True)
    assert images.shape[0] == 3
    assert load_eval_split(_Pre, "val", final=False)[0].shape[0] == 3


def test_validation_split_mislabelled_as_test_is_refused():
    class Pre:
        val_dataset, test_dataset = _Split("test"), _Split("test")
    with pytest.raises(PermissionError):
        load_eval_split(Pre, "val", final=False)


def _eval(name, split="val", acc=0.8):
    m = {"accuracy": acc, "balanced_accuracy": acc, "macro_f1": acc, "auroc_macro_ovr": 0.9}
    return {"split": split, "primary_rule": "r", "test_set_accessed": split == "test",
            "primary": m, "rules": {"r": m, "other": {**m, "accuracy": acc - 0.1}}}


def test_build_comparison_rows_and_convergence_json_safe():
    hist = {"A": [{"round": i + 1, "validation_accuracy": 0.1 * (i + 1)} for i in range(8)]}
    doc = build_comparison({"A": _eval("A"), "B": _eval("B", acc=0.7)}, hist, "val")
    assert [r["name"] for r in doc["rows"] if not r.get("ablation")] == ["A", "B"]
    assert "A" in doc["convergence_validation_accuracy"]
    json.dumps(doc, allow_nan=False)
