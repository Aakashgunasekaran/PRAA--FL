"""Validation metrics without updating model parameters."""

from typing import Dict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def evaluate_classifier(model, loader, device="cpu") -> Dict[str, float]:
    model.eval()
    predictions, targets, losses = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).to(device)
            logits = model.logits(images)
            losses.append(float(criterion(logits, labels).item()) * len(labels))
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    if not targets:
        raise ValueError("Validation loader is empty")
    return {
        "validation_loss": sum(losses) / len(targets),
        "validation_accuracy": float(accuracy_score(targets, predictions)),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "validation_macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    }
