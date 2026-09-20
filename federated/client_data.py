import torch
from torch.utils.data import DataLoader, Subset

from federated.config import NUM_CLIENTS, DIRICHLET_ALPHA, SEED, BATCH_SIZE
from datasets import preprocess
from datasets.partition import partition_dataset


# Build the partition once at import time (deterministic)
_PARTITION = partition_dataset(preprocess.train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, SEED)


def build_partition(num_clients=NUM_CLIENTS, alpha=DIRICHLET_ALPHA, seed=SEED, min_samples_per_client=0):
    """Create a Phase 2 partition using the canonical Phase 2 algorithm."""
    return partition_dataset(
        preprocess.train_dataset,
        num_clients,
        alpha,
        seed,
        min_samples_per_client=min_samples_per_client,
    )


def get_client_dataset(client_id: int, partition=None):
    """Return a Subset of the global training dataset for the given client_id.

    Raises ValueError for invalid client ids or empty partitions.
    """
    active_partition = _PARTITION if partition is None else partition
    if not (0 <= client_id < len(active_partition)):
        raise ValueError(f"client_id must be in [0, {len(active_partition)}), got {client_id}")

    indices = active_partition.get(client_id, [])
    if len(indices) == 0:
        raise ValueError(f"Client {client_id} has an empty partition")

    return Subset(preprocess.train_dataset, indices)


def get_client_dataloader(client_id: int, partition=None):
    dataset = get_client_dataset(client_id, partition=partition)

    # Use a seeded generator for reproducible shuffling per client
    generator = torch.Generator()
    try:
        generator.manual_seed(SEED + int(client_id))
    except Exception:
        generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    return loader


# Expose the partition and a verify helper
def get_partition():
    return _PARTITION


def verify_partition():
    from datasets.partition import verify_partition as _verify
    total = len(preprocess.train_dataset)
    return _verify(_PARTITION, total)
