"""FLAME equations (4), (7), and (8) with explicit reference scaling."""

from typing import Tuple
import torch
import torch.nn.functional as F

from models.classifier import compute_prototypes, squared_euclidean_logits


def compute_mae_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstruction, target, reduction="mean")


def compute_proto_loss(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    prototypes = compute_prototypes(support_embeddings, support_labels, num_classes)
    logits = squared_euclidean_logits(query_embeddings, prototypes)
    present = torch.zeros(num_classes, dtype=torch.bool, device=query_labels.device)
    present[torch.unique(support_labels)] = True
    if not present[query_labels].all():
        raise ValueError("Every query class must have a support prototype")
    return F.cross_entropy(logits, query_labels)


def prescale_losses(
    mae_loss: torch.Tensor,
    proto_loss: torch.Tensor,
    mae_reference: float = 1.0,
    proto_reference: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mae_reference <= 0 or proto_reference <= 0:
        raise ValueError("loss references must be positive")
    return mae_loss / mae_reference, proto_loss / proto_reference


def combine_losses(
    mae_loss: torch.Tensor,
    proto_loss: torch.Tensor,
    alpha: float = 0.7,
    mae_reference: float = 1.0,
    proto_reference: float = 1.0,
) -> torch.Tensor:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    scaled_mae, scaled_proto = prescale_losses(
        mae_loss, proto_loss, mae_reference, proto_reference
    )
    return alpha * scaled_mae + (1 - alpha) * scaled_proto
