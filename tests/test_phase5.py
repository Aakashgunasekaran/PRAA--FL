import pytest
import torch
import json
from pathlib import Path

from federated.prs import (
    PRSConfig,
    compute_prs,
    consistency_scores,
    stability_scores,
    validation_embeddings,
    validation_scores,
    validation_scores_from_embeddings,
)


def test_finite_prs_and_component_separation():
    result = compute_prs(
        {0: 0.8}, {0: 0.9}, {0: 1.0}, PRSConfig()
    )
    assert 0.0 <= result[0] <= 1.0


def test_identical_and_different_consistency():
    identical = consistency_scores({0: torch.tensor([1.0, 0.0])}, {0: torch.tensor([1.0, 0.0])})
    different = consistency_scores({0: torch.tensor([1.0, 0.0])}, {0: torch.tensor([0.0, 1.0])})
    assert identical[0] > different[0]


def test_stable_snapshots_are_high():
    scores = stability_scores(
        [{0: torch.tensor([1.0, 0.0])}, {0: torch.tensor([1.0, 0.0])}],
        [0],
    )
    assert scores[0] == pytest.approx(1.0)


def test_missing_stability_does_not_create_score():
    assert stability_scores(None, [0]) is None
    assert compute_prs({0: 1.0}, None, {0: 1.0})[0] is None


def test_invalid_and_mismatched_prototypes_rejected():
    with pytest.raises(FloatingPointError):
        consistency_scores({0: torch.tensor([float("nan"), 1.0])}, {0: torch.tensor([1.0, 0.0])})
    with pytest.raises(ValueError):
        consistency_scores({0: torch.tensor([1.0])}, {0: torch.tensor([1.0, 0.0])})
    with pytest.raises(ValueError):
        consistency_scores({0: torch.tensor([0.0, 0.0])}, {0: torch.tensor([1.0, 0.0])})


def test_validation_component_changes_prs_direction():
    low = compute_prs({0: 0.2}, {0: 0.8}, {0: 0.8})[0]
    high = compute_prs({0: 0.9}, {0: 0.8}, {0: 0.8})[0]
    assert high > low


def test_missing_reference_class_is_rejected():
    with pytest.raises(KeyError):
        consistency_scores({1: torch.tensor([1.0, 0.0])}, {0: torch.tensor([1.0, 0.0])})


def test_phase4_artifact_pipeline_is_consumable():
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "results" / "phase4" / "summary.json"
    artifact_path = root / "results" / "phase4" / "prototypes" / "client_0.pt"
    if not summary_path.exists() or not artifact_path.exists():
        pytest.skip("Phase 4 artifacts are not present")
    summary = json.loads(summary_path.read_text())
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert summary["phase"] == 4
    assert artifact["client_id"] == 0
    assert artifact["prototypes"]
    assert all(torch.isfinite(value).all() for value in artifact["prototypes"].values())


def test_repeated_calculation_is_reproducible():
    inputs = ({0: 0.7}, {0: 0.9}, {0: 0.8})
    assert compute_prs(*inputs) == compute_prs(*inputs)


def test_validation_score_uses_prototype_representation():
    class Representation(torch.nn.Module):
        def forward(self, images):
            return images, images

    loader = [(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))]
    scores = validation_scores(
        Representation(),
        loader,
        [0, 1],
        {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])},
    )
    assert scores == {0: 1.0, 1: 1.0}


def test_cached_validation_embeddings_preserve_scores():
    class Representation(torch.nn.Module):
        def forward(self, images):
            return images, images

    loader = [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[0.9, 0.1], [0.1, 0.9]]), torch.tensor([0, 1])),
    ]
    prototypes = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
    }
    model = Representation()
    direct = validation_scores(model, loader, [0, 1], prototypes)
    embeddings, labels = validation_embeddings(model, loader)
    cached = validation_scores_from_embeddings(
        embeddings, labels, [0, 1], prototypes
    )
    assert cached == direct
