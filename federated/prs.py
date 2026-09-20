"""Prototype Reliability Score (PRS) calculations for Phase 5.

PRS is a PRAA-FL research component, not part of the FLAME baseline.
Temporal stability is intentionally optional because Phase 4 currently
persists one prototype snapshot per client.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class PRSConfig:
    validation_weight: float = 1.0 / 3.0
    stability_weight: float = 1.0 / 3.0
    consistency_weight: float = 1.0 / 3.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        weights = (
            self.validation_weight,
            self.stability_weight,
            self.consistency_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("PRS weights must be non-negative and not all zero")
        if self.epsilon <= 0:
            raise ValueError("PRS epsilon must be positive")


def _finite_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return value.detach().to(dtype=torch.float32, device="cpu")


def cosine_score(left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Return cosine similarity mapped from [-1, 1] to the [0, 1] range."""
    left = _finite_vector(left, "left prototype")
    right = _finite_vector(right, "right prototype")
    if left.shape != right.shape:
        raise ValueError("Prototype dimensions do not match")
    left_norm = left.norm()
    right_norm = right.norm()
    if left_norm <= epsilon or right_norm <= epsilon:
        raise ValueError("Cosine similarity is undefined for a zero-norm prototype")
    cosine = torch.dot(left, right) / (left_norm * right_norm)
    return float(((cosine.clamp(-1.0, 1.0) + 1.0) / 2.0).item())


def validation_scores(
    model: torch.nn.Module,
    validation_loader: Iterable,
    classes: Sequence[int],
    prototypes: Mapping[int, torch.Tensor],
    device: str = "cpu",
) -> Dict[int, Optional[float]]:
    """Compute class-wise accuracy using the client's learned prototypes."""
    expected = {int(class_id) for class_id in classes}
    if not expected:
        return {}
    if set(prototypes) != expected:
        raise ValueError("Validation classes and prototype classes must match")
    prototype_ids = sorted(expected)
    prototype_matrix = torch.stack(
        [_finite_vector(prototypes[class_id], f"class {class_id} prototype") for class_id in prototype_ids]
    ).to(device)
    correct = {class_id: 0 for class_id in expected}
    counts = {class_id: 0 for class_id in expected}
    model.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.view(-1).to(device)
            _, embeddings = model(images)
            if embeddings.ndim != 2 or embeddings.size(1) != prototype_matrix.size(1):
                raise ValueError("Validation embedding dimension does not match prototypes")
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError("Validation embeddings contain NaN or Inf")
            distances = ((embeddings[:, None, :] - prototype_matrix[None, :, :]) ** 2).sum(dim=2)
            predictions = torch.tensor(
                [prototype_ids[index] for index in distances.argmin(dim=1).tolist()],
                device=device,
            )
            for class_id in expected:
                selected = labels == class_id
                counts[class_id] += int(selected.sum().item())
                correct[class_id] += int((predictions[selected] == labels[selected]).sum().item())
    return {
        class_id: (correct[class_id] / counts[class_id] if counts[class_id] else None)
        for class_id in expected
    }


def consistency_scores(
    prototypes: Mapping[int, torch.Tensor],
    reference_prototypes: Mapping[int, torch.Tensor],
    epsilon: float = 1e-8,
) -> Dict[int, float]:
    """Compare each client class prototype only with the same reference class."""
    scores = {}
    for class_id, prototype in prototypes.items():
        if class_id not in reference_prototypes:
            raise KeyError(f"Missing reference prototype for class {class_id}")
        scores[int(class_id)] = cosine_score(
            prototype, reference_prototypes[class_id], epsilon=epsilon
        )
    return scores


def stability_scores(
    snapshots: Optional[Sequence[Mapping[int, torch.Tensor]]],
    classes: Sequence[int],
    epsilon: float = 1e-8,
) -> Optional[Dict[int, float]]:
    """Measure temporal stability across at least two prototype snapshots.

    ``None`` means temporal stability is not measurable; it is not a score and
    must not be silently converted into a high-reliability value.
    """
    if snapshots is None or len(snapshots) < 2:
        return None
    scores = {}
    for class_id in classes:
        values = []
        for previous, current in zip(snapshots, snapshots[1:]):
            if class_id not in previous or class_id not in current:
                raise KeyError(f"Missing class {class_id} in stability snapshots")
            values.append(cosine_score(previous[class_id], current[class_id], epsilon))
        scores[int(class_id)] = sum(values) / len(values)
    return scores


def compute_prs(
    validation: Mapping[int, float],
    stability: Optional[Mapping[int, float]],
    consistency: Mapping[int, float],
    config: PRSConfig = PRSConfig(),
) -> Dict[int, Optional[float]]:
    """Combine component scores when all required components are available."""
    classes = set(validation) | set(consistency)
    if stability is not None:
        classes |= set(stability)
    result: Dict[int, Optional[float]] = {}
    weight_sum = (
        config.validation_weight
        + config.stability_weight
        + config.consistency_weight
    )
    for class_id in sorted(classes):
        stability_value = None if stability is None else stability.get(class_id)
        values = (validation.get(class_id), stability_value, consistency.get(class_id))
        if any(value is None for value in values):
            result[int(class_id)] = None
            continue
        score = (
            config.validation_weight * float(values[0])
            + config.stability_weight * float(values[1])
            + config.consistency_weight * float(values[2])
        ) / weight_sum
        if not 0.0 <= score <= 1.0 or not torch.isfinite(torch.tensor(score)):
            raise ValueError(f"Invalid PRS value for class {class_id}: {score}")
        result[int(class_id)] = score
    return result
