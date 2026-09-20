"""Deterministic support/query episode preparation."""

from typing import Tuple
import torch


def make_support_query(
    images: torch.Tensor,
    labels: torch.Tensor,
    support_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 < support_fraction < 1:
        raise ValueError("support_fraction must be between 0 and 1")
    generator = torch.Generator(device=images.device)
    generator.manual_seed(seed)
    support, query = [], []
    for class_id in torch.unique(labels).tolist():
        indices = torch.where(labels == class_id)[0]
        if len(indices) < 2:
            continue
        indices = indices[torch.randperm(len(indices), generator=generator, device=images.device)]
        split = min(max(1, int(len(indices) * support_fraction)), len(indices) - 1)
        support.append(indices[:split])
        query.append(indices[split:])
    if not support or not query:
        raise ValueError("Episode requires at least two samples for at least one class")
    support_idx = torch.cat(support)
    query_idx = torch.cat(query)
    return images[support_idx], labels[support_idx], images[query_idx], labels[query_idx]
