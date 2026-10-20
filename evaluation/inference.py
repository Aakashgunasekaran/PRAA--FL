"""Inference rules used for evaluation.

* FedAvg baseline: the supervised classifier head (``model.logits``).
* FLAME / PRAA: nearest-prototype classification in the projected embedding space,
  exactly the paper's Eq. (6)/(10): P(y=c|x) = softmax_c(-||g(f(x)) - p_c||^2) with
  GLOBAL prototypes p_c aggregated from the clients' prototypes
  (``federated.praa.aggregate_prototypes``).  Classes with no prototype get logit -inf
  (probability 0); this is reported, never silently filled.
"""

from typing import Mapping, Tuple

import torch


def embed_images(model, images: torch.Tensor, device="cpu", batch_size: int = 256) -> torch.Tensor:
    """Projected FLAME embeddings of an image tensor [N,3,H,W]; eval mode, no grad."""
    was_training = model.training
    model.eval()
    out = []
    try:
        with torch.no_grad():
            for start in range(0, images.size(0), batch_size):
                _, emb = model(images[start:start + batch_size].to(device))
                if not torch.isfinite(emb).all():
                    raise FloatingPointError("embeddings contain NaN or Inf")
                out.append(emb.detach().float().cpu())
    finally:
        model.train(was_training)
    if not out:
        raise ValueError("no images to embed")
    return torch.cat(out)


def classifier_logits_images(model, images: torch.Tensor, device="cpu", batch_size: int = 256) -> torch.Tensor:
    was_training = model.training
    model.eval()
    out = []
    try:
        with torch.no_grad():
            for start in range(0, images.size(0), batch_size):
                out.append(model.logits(images[start:start + batch_size].to(device)).detach().float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(out)


def prototype_logits(embeddings: torch.Tensor, prototypes: Mapping[int, torch.Tensor],
                     num_classes: int) -> Tuple[torch.Tensor, list]:
    """-squared Euclidean distance logits [N, num_classes]; missing classes -> -inf.

    Returns (logits, missing_class_ids).
    """
    if not prototypes:
        raise ValueError("no global prototypes available")
    dim = embeddings.size(1)
    logits = torch.full((embeddings.size(0), num_classes), float("-inf"), dtype=torch.float32)
    missing = []
    for class_id in range(num_classes):
        if class_id not in prototypes:
            missing.append(class_id)
            continue
        proto = prototypes[class_id].float()
        if proto.numel() != dim:
            raise ValueError("prototype dimension does not match embeddings")
        logits[:, class_id] = -((embeddings.float() - proto[None, :]) ** 2).sum(dim=1)
    return logits, missing


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)
