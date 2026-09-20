
import random

import numpy as np
import torch
from medmnist import INFO, PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset Information
# ============================================================

info = INFO["pathmnist"]
DataClass = PathMNIST


# ============================================================
# Image Preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# Load Datasets
# ============================================================

train_dataset = DataClass(
    split="train",
    transform=transform,
    download=True
)

val_dataset = DataClass(
    split="val",
    transform=transform,
    download=True
)

test_dataset = DataClass(
    split="test",
    transform=transform,
    download=True
)


# ============================================================
# Reproducible DataLoader Generator
# ============================================================

generator = torch.Generator()
generator.manual_seed(SEED)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# Verification Output
# ============================================================

print("=" * 60)
print("PRAA-FL - Phase 1 Data Preprocessing")
print("=" * 60)

print("Random Seed        :", SEED)

print()
print("Dataset Information")
print("-" * 60)

print("Training Images    :", len(train_dataset))
print("Validation Images  :", len(val_dataset))
print("Testing Images     :", len(test_dataset))
print("Number of Classes  :", len(info["label"]))

print()
print("DataLoader Information")
print("-" * 60)

print("Training batches   :", len(train_loader))
print("Validation batches :", len(val_loader))
print("Testing batches    :", len(test_loader))


# ============================================================
# Verify One Training Batch
# ============================================================

images, labels = next(iter(train_loader))

print()
print("Batch Shape")
print("-" * 60)

print("Images :", images.shape)
print("Labels :", labels.shape)

print()
print("Data Types")
print("-" * 60)

print("Image dtype :", images.dtype)
print("Label dtype :", labels.dtype)

print()
print("Value Check")
print("-" * 60)

print("Images contain NaN :", torch.isnan(images).any().item())
print("Images contain Inf :", torch.isinf(images).any().item())

print()
print("=" * 60)
print("Phase 1 preprocessing completed successfully.")
print("=" * 60)

