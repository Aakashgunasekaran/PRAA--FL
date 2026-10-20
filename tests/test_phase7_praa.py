"""Phase 7: reliability-aware weights and aggregation."""

import copy

import pytest
import torch

from federated.praa import (PRAAConfig, aggregate_model_states, aggregate_prototypes, class_multipliers,
                            compute_record_prs, flame_weights, praa_client_weights, relative_multipliers,
                            round_aggregation_plan)
from federated.server import Server
from tests.helpers import row


def test_multipliers_mean_one_over_assessed_and_unassessed_neutral():
    mult, info = relative_multipliers({0: 0.9, 1: 0.5, 2: None, 3: 0.7}, PRAAConfig())
    assessed = [mult[0], mult[1], mult[3]]
    assert sum(assessed) / 3 == pytest.approx(1.0)
    assert mult[2] == 1.0 and info["applied"] and info["num_assessed"] == 3
    assert mult[0] > mult[3] > mult[1]


def test_no_reweighting_when_too_few_assessed_or_tau_zero():
    mult, info = relative_multipliers({0: 0.9, 1: None, 2: None}, PRAAConfig(min_assessed=2))
    assert set(mult.values()) == {1.0} and info["reason"] == "fewer_than_min_assessed"
    mult, info = relative_multipliers({0: 0.9, 1: 0.1}, PRAAConfig(tau=0))
    assert set(mult.values()) == {1.0} and info["reason"] == "tau_is_zero"


def test_floor_and_cap_keep_every_client_positive():
    mult, _ = relative_multipliers({0: 1.0, 1: 0.0, 2: 1.0}, PRAAConfig(tau=1.0, weight_floor=0.2, weight_cap=1.5))
    assert mult[1] == pytest.approx(0.2) and max(mult.values()) <= 1.5
    w = praa_client_weights({0: 10, 1: 10, 2: 10}, mult)
    assert all(v > 0 for v in w.values()) and sum(w.values()) == pytest.approx(1.0)


def test_tau_increases_contrast_monotonically():
    rel = {0: 0.9, 1: 0.5}
    gaps = []
    for tau in (0.5, 1, 2, 4):
        m, _ = relative_multipliers(rel, PRAAConfig(tau=tau))
        gaps.append(m[0] - m[1])
    assert gaps == sorted(gaps)


def test_reliability_outside_unit_interval_rejected():
    with pytest.raises(ValueError):
        relative_multipliers({0: 1.3, 1: 0.5}, PRAAConfig())


def test_flame_weights_are_sample_proportional():
    w = flame_weights({3: 100, 5: 300})
    assert w == {3: 0.25, 5: 0.75}
    with pytest.raises(ValueError):
        flame_weights({1: 0})


def test_praa_weights_reduce_unreliable_client_and_sum_to_one():
    sizes = {0: 100, 1: 100, 2: 100}
    mult = {0: 1.2, 1: 0.4, 2: 1.0}
    base, w = flame_weights(sizes), praa_client_weights(sizes, mult)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w[1] < base[1] and w[0] > base[0]


def test_praa_equals_flame_when_multipliers_are_one():
    sizes = {0: 7, 1: 13, 2: 80}
    assert praa_client_weights(sizes, {0: 1.0, 1: 1.0, 2: 1.0}) == pytest.approx(flame_weights(sizes))


def _state(value):
    return {"w": torch.full((2, 2), float(value)), "n": torch.tensor(3, dtype=torch.long)}


def test_aggregate_model_states_matches_manual_average_and_flame_server():
    states = [_state(1.0), _state(3.0), _state(5.0)]
    sizes = [10, 20, 70]
    server = Server.__new__(Server)  # only aggregate() is needed
    expected, _, _ = Server.aggregate(server, states, sizes)
    got = aggregate_model_states(states, [s / sum(sizes) for s in sizes])
    assert torch.allclose(got["w"], expected["w"]) and got["w"][0, 0].item() == pytest.approx(0.1 + 0.6 + 3.5)
    assert got["n"].item() == 3


def test_aggregate_model_states_rejects_nan_and_misaligned():
    bad = _state(1.0)
    bad["w"][0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        aggregate_model_states([bad, _state(1.0)], [0.5, 0.5])
    with pytest.raises(ValueError):
        aggregate_model_states([_state(1.0)], [0.5, 0.5])


def test_prototype_aggregation_without_multipliers_is_pooled_class_mean():
    torch.manual_seed(0)
    a, b = torch.randn(3, 4), torch.randn(5, 4)  # embeddings of two clients, same class
    protos = {0: {1: a.mean(0)}, 1: {1: b.mean(0)}}
    counts = {0: {1: 3}, 1: {1: 5}}
    gp, w = aggregate_prototypes(protos, counts, None)
    assert torch.allclose(gp[1], torch.cat([a, b]).mean(0), atol=1e-6)
    assert w[1][0] == pytest.approx(3 / 8) and sum(w[1].values()) == pytest.approx(1.0)


def test_prototype_aggregation_moves_toward_reliable_client():
    protos = {0: {2: torch.zeros(4)}, 1: {2: torch.ones(4) * 10}}  # client 1 is the noisy one
    counts = {0: {2: 10}, 1: {2: 10}}
    flame, _ = aggregate_prototypes(protos, counts, None)
    praa, w = aggregate_prototypes(protos, counts, {(0, 2): 1.5, (1, 2): 0.5})
    assert flame[2][0].item() == pytest.approx(5.0)
    assert praa[2][0].item() == pytest.approx(2.5)
    assert w[2][1] < w[2][0]


def test_prototype_aggregation_validates_inputs():
    with pytest.raises(ValueError):
        aggregate_prototypes({0: {1: torch.zeros(2)}}, {0: {1: 0}})
    with pytest.raises(FloatingPointError):
        aggregate_prototypes({0: {1: torch.tensor([float("inf"), 0.0])}}, {0: {1: 3}})
    with pytest.raises(ValueError):
        aggregate_prototypes({0: {1: torch.zeros(2)}}, {0: {1: 3}}, {(0, 1): -1.0})


def test_class_multipliers_single_holder_is_neutral():
    prs = compute_record_prs([row(0, 1, 0.9, 0.9, 0.9), row(0, 2, 0.5, 0.5, 0.5), row(1, 2, 0.9, 0.9, 0.9)],
                             PRAAConfig())
    mult, info = class_multipliers(prs, PRAAConfig())
    assert mult[(0, 1)] == 1.0 and info[1]["reason"] == "fewer_than_min_assessed"
    assert mult[(0, 2)] < 1.0 < mult[(1, 2)]


def test_round_plan_missing_evidence_falls_back_to_flame_weight():
    rows = [row(0, 1, 0.9, 0.9, 0.9), row(1, 1, 0.3, 0.3, 0.3),
            row(2, 1, None, None, 0.9)]  # client 2: no stability / validation -> unassessed
    plan = round_aggregation_plan({0: 100, 1: 100, 2: 100}, rows, PRAAConfig())
    assert plan["reliability"][2]["status"] == "unassessed"
    assert plan["multipliers"][2] == 1.0
    assert plan["praa_weights"][1] < plan["flame_weights"][1]
    assert sum(plan["praa_weights"].values()) == pytest.approx(1.0)


def test_round_plan_with_no_evidence_is_exactly_flame():
    rows = [row(c, 1, None, None, 0.9) for c in range(3)]
    plan = round_aggregation_plan({0: 5, 1: 15, 2: 30}, rows, PRAAConfig())
    assert plan["praa_weights"] == pytest.approx(plan["flame_weights"])
    assert plan["max_weight_shift"] == pytest.approx(0.0)


def test_noisy_client_influence_is_reduced_end_to_end():
    """Synthetic noisy-prototype scenario: PRAA's global prototype is closer to the truth."""
    truth = torch.tensor([1.0, 0.0, 0.0, 0.0])
    protos = {c: {0: truth.clone() + 0.01 * c} for c in range(4)}
    protos[4] = {0: torch.tensor([-1.0, 5.0, 5.0, 5.0])}  # corrupted client
    counts = {c: {0: 10} for c in range(5)}
    rows = [row(c, 0, 0.95, 0.95, 0.95) for c in range(4)] + [row(4, 0, 0.1, 0.2, 0.3)]
    prs = compute_record_prs(rows, PRAAConfig(tau=2.0))
    mult, _ = class_multipliers(prs, PRAAConfig(tau=2.0))
    flame, _ = aggregate_prototypes(protos, counts, None)
    praa, w = aggregate_prototypes(protos, counts, mult)
    assert (praa[0] - truth).norm() < (flame[0] - truth).norm()
    assert w[0][4] < 0.2  # below the FLAME share of 1/5 = 0.2


def test_config_roundtrip():
    cfg = PRAAConfig(tau=2.0, client_reduce="mean", stability_window=None)
    assert PRAAConfig.from_dict(cfg.to_dict()) == cfg
