"""Feature extraction and local class-wise prototype generation."""

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch


def extract_features(model, dataloader, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract projected FLAME embeddings and labels without retaining graphs."""
    was_training = model.training
    model.eval()
    embeddings, labels = [], []
    try:
        with torch.no_grad():
            for images, batch_labels in dataloader:
                _, batch_embeddings = model(images.to(device))
                if not torch.isfinite(batch_embeddings).all():
                    raise FloatingPointError("Extracted embeddings contain NaN or Inf")
                embeddings.append(batch_embeddings.detach().cpu())
                labels.append(batch_labels.view(-1).detach().cpu())
    finally:
        model.train(was_training)
    if not embeddings:
        raise ValueError("Cannot extract features from an empty dataloader")
    return torch.cat(embeddings), torch.cat(labels)


def generate_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Return prototypes and sample counts only for classes present in labels."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, embedding_dim]")
    labels = labels.view(-1)
    if embeddings.size(0) != labels.size(0):
        raise ValueError("embeddings and labels must contain the same number of samples")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("Embeddings contain NaN or Inf")
    grouped = defaultdict(list)
    for embedding, label in zip(embeddings, labels.tolist()):
        grouped[int(label)].append(embedding)
    prototypes = {
        class_id: torch.stack(class_embeddings).mean(dim=0).detach().cpu()
        for class_id, class_embeddings in grouped.items()
    }
    counts = {class_id: len(class_embeddings) for class_id, class_embeddings in grouped.items()}
    return prototypes, counts


def generate_client_prototypes(
    model,
    client_datasets: Mapping[int, torch.utils.data.Dataset],
    batch_size: int = 64,
    device: str = "cpu",
    max_samples_per_client: Optional[int] = None,
) -> Dict[int, Dict[str, object]]:
    """Generate client-separated prototypes from a learned FLAME model.

    Each client is processed independently and only classes present in that
    client's data receive prototypes. The returned metadata is derived from
    the extracted embeddings and is suitable for serialization.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_samples_per_client is not None and max_samples_per_client <= 0:
        raise ValueError("max_samples_per_client must be positive when provided")

    results = {}
    for client_id, dataset in client_datasets.items():
        active_dataset = dataset
        if max_samples_per_client is not None:
            active_dataset = torch.utils.data.Subset(
                dataset, range(min(len(dataset), max_samples_per_client))
            )
        if len(active_dataset) == 0:
            raise ValueError(f"Client {client_id} has an empty dataset")
        loader = torch.utils.data.DataLoader(
            active_dataset, batch_size=batch_size, shuffle=False
        )
        embeddings, labels = extract_features(model, loader, device=device)
        prototypes, counts = generate_prototypes(embeddings, labels)
        embedding_dimension = embeddings.size(1)
        metadata = {
            class_id: {
                "client_id": int(client_id),
                "class_id": int(class_id),
                "sample_count": int(counts[class_id]),
                "embedding_dimension": int(embedding_dimension),
                "prototype_norm": float(prototype.norm().item()),
            }
            for class_id, prototype in prototypes.items()
        }
        results[int(client_id)] = {
            "prototypes": prototypes,
            "metadata": metadata,
            "samples_processed": int(len(active_dataset)),
            "embedding_dimension": int(embedding_dimension),
        }
    return results
