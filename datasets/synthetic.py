"""Synthetic PathMNIST-shaped data for OFFLINE tests and smoke runs ONLY.

Enabled by the environment variable ``PRAA_SYNTHETIC_DATA=1``.  The images are
class-conditional colour/stripe patterns plus noise.  They are NOT PathMNIST, any
accuracy measured on them says nothing about the paper's or the project's real
results, and every artefact produced with them must be labelled synthetic.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 9
DEFAULT_SIZES = (3000, 500, 500)  # train, val, test
_SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}


def synthetic_sizes() -> Tuple[int, int, int]:
    raw = os.environ.get("PRAA_SYNTHETIC_SIZES")
    if not raw:
        return DEFAULT_SIZES
    parts = tuple(int(x) for x in raw.split(","))
    if len(parts) != 3 or min(parts) < NUM_CLASSES:
        raise ValueError("PRAA_SYNTHETIC_SIZES must be 'train,val,test' with each >= 9")
    return parts


class SyntheticPathMNIST(Dataset):
    """Drop-in stand-in for ``medmnist.PathMNIST`` (same attributes used by this repo)."""

    is_synthetic = True

    def __init__(self, split: str, transform=None, size: int = None, noise: float = 0.12):
        if split not in _SPLIT_SEEDS:
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        self.transform = transform
        sizes = dict(zip(("train", "val", "test"), synthetic_sizes()))
        n = int(size or sizes[split])
        rng = np.random.RandomState(_SPLIT_SEEDS[split])
        labels = np.arange(n) % NUM_CLASSES
        rng.shuffle(labels)
        self.labels = labels.reshape(-1, 1).astype(np.int64)
        yy, xx = np.mgrid[0:28, 0:28] / 27.0
        images = np.empty((n, 28, 28, 3), dtype=np.uint8)
        palette = np.random.RandomState(7).uniform(0.2, 0.8, size=(NUM_CLASSES, 3))
        for i, c in enumerate(labels):
            freq = 2 + c
            phase = rng.uniform(0, 2 * np.pi)
            angle = np.pi * c / NUM_CLASSES
            stripes = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (np.cos(angle) * xx + np.sin(angle) * yy) + phase)
            img = palette[c][None, None, :] * (0.6 + 0.4 * stripes[..., None])
            img = img + rng.normal(0, noise, size=img.shape)
            images[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
        self.imgs = images

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label
