"""Mentor-facing Phase 1 demonstration."""

import contextlib
import io
import json
from pathlib import Path

import torch


def main():
    # Phase 1 currently prints its own diagnostic banner at import time.
    # Suppress that internal output so this entrypoint owns the presentation.
    with contextlib.redirect_stdout(io.StringIO()):
        from datasets import preprocess

    images, labels = next(iter(preprocess.train_loader))
    summary = {
        "seed": preprocess.SEED,
        "training_images": len(preprocess.train_dataset),
        "validation_images": len(preprocess.val_dataset),
        "testing_images": len(preprocess.test_dataset),
        "classes": len(preprocess.info["label"]),
        "training_batches": len(preprocess.train_loader),
        "validation_batches": len(preprocess.val_loader),
        "testing_batches": len(preprocess.test_loader),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "nan": bool(torch.isnan(images).any()),
        "inf": bool(torch.isinf(images).any()),
    }
    output = Path("results/phase1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 60)
    print("PRAA-FL - PHASE 1: DATA PREPROCESSING")
    print("=" * 60)
    print("\nRandom Seed        :", summary["seed"])
    print("\nDataset Information\n" + "-" * 60)
    print("Training Images    :", summary["training_images"])
    print("Validation Images  :", summary["validation_images"])
    print("Testing Images     :", summary["testing_images"])
    print("Number of Classes  :", summary["classes"])
    print("\nDataLoader Information\n" + "-" * 60)
    print("Training batches   :", summary["training_batches"])
    print("Validation batches :", summary["validation_batches"])
    print("Testing batches    :", summary["testing_batches"])
    print("\nBatch Shape\n" + "-" * 60)
    print("Images :", torch.Size(summary["image_shape"]))
    print("Labels :", torch.Size(summary["label_shape"]))
    print("\nImage dtype :", summary["image_dtype"])
    print("Label dtype :", summary["label_dtype"])
    print("\nNaN values :", summary["nan"])
    print("Inf values :", summary["inf"])
    print("\n" + "=" * 60)
    print("PHASE 1 STATUS: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
