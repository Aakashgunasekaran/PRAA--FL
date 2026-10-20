"""Phase 6: PRS formula, missing-evidence handling, client reliability."""

import math

import pytest

from federated.praa import (PRAAConfig, client_reliability, compute_record_prs, record_prs)
from federated.prs import PRSConfig
from tests.helpers import row


def test_prs_equal_weights_known_value():
    r = record_prs(row(0, 1, v=0.9, s=0.8, c=0.7), PRAAConfig())
    assert r["prs_status"] == "complete"
    assert r["prs"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert r["used_components"] == ["validation", "consistency", "stability"]


def test_prs_custom_weights_known_value():
    cfg = PRAAConfig(prs=PRSConfig(validation_weight=2.0, stability_weight=1.0, consistency_weight=1.0))
    r = record_prs(row(0, 1, v=1.0, s=0.0, c=0.5), cfg)
    assert r["prs"] == pytest.approx((2 * 1.0 + 1 * 0.0 + 1 * 0.5) / 4)


@pytest.mark.parametrize("missing", ["v", "s", "c"])
def test_strict_policy_never_imputes(missing):
    values = {"v": 0.9, "s": 0.8, "c": 0.7}
    values[missing] = None
    r = record_prs(row(0, 1, **values), PRAAConfig())
    assert r["prs"] is None and r["prs_status"] == "unavailable"
    assert "components_not_valid" in r["reason"]


def test_degenerate_validation_blocks_prs_and_is_not_zero_or_one():
    r = record_prs(row(0, 1, v=None, v_status="degenerate", s=0.8, c=0.7), PRAAConfig())
    assert r["prs"] is None


def test_renormalize_extension_averages_only_valid_components():
    cfg = PRAAConfig(missing_component_policy="renormalize")
    r = record_prs(row(0, 1, v=0.6, s=None, c=0.8), cfg)
    assert r["prs_status"] == "partial"
    assert r["prs"] == pytest.approx(0.7)
    assert r["used_components"] == ["validation", "consistency"]


def test_renormalize_needs_two_valid_components():
    cfg = PRAAConfig(missing_component_policy="renormalize")
    assert record_prs(row(0, 1, v=None, s=None, c=0.9), cfg)["prs"] is None


def test_prs_is_within_unit_interval_for_extreme_components():
    lo = record_prs(row(0, 1, v=0.0, s=0.0, c=0.0), PRAAConfig())["prs"]
    hi = record_prs(row(0, 1, v=1.0, s=1.0, c=1.0), PRAAConfig())["prs"]
    assert lo == 0.0 and hi == 1.0


def test_prs_weight_validation():
    with pytest.raises(ValueError):
        PRSConfig(validation_weight=-1)
    with pytest.raises(ValueError):
        PRSConfig(0, 0, 0)


def test_client_reliability_sample_weighted_vs_mean():
    rows = [row(0, 1, 1.0, 1.0, 1.0, n=30), row(0, 2, 0.0, 0.0, 0.0, n=10),
            row(1, 1, None, 0.5, 0.5, n=5)]
    prs = compute_record_prs(rows, PRAAConfig())
    weighted = client_reliability(prs, [0, 1, 2], PRAAConfig())
    mean = client_reliability(prs, [0, 1, 2], PRAAConfig(client_reduce="mean"))
    assert weighted[0]["reliability"] == pytest.approx((30 * 1.0 + 10 * 0.0) / 40)
    assert mean[0]["reliability"] == pytest.approx(0.5)
    assert weighted[1]["status"] == "unassessed" and weighted[1]["reliability"] is None
    assert weighted[2]["status"] == "unassessed"  # client with no prototype records at all
    assert weighted[2]["reason"] == "client_has_no_prototype_records"


def test_reliability_ignores_unscored_records_instead_of_counting_them_as_zero():
    rows = [row(0, 1, 0.8, 0.8, 0.8, n=10), row(0, 2, None, 0.1, 0.1, n=1000)]
    rel = client_reliability(compute_record_prs(rows, PRAAConfig()), [0], PRAAConfig())
    assert rel[0]["reliability"] == pytest.approx(0.8)
    assert rel[0]["num_scored"] == 1 and rel[0]["num_records"] == 2


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PRAAConfig(tau=-1)
    with pytest.raises(ValueError):
        PRAAConfig(missing_component_policy="impute_zero")
    with pytest.raises(ValueError):
        PRAAConfig(weight_floor=2.0)
    assert math.isfinite(PRAAConfig().tau)
