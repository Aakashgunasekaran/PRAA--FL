import pytest
import numpy as np
import torch

from federated import client_data
from federated.config import NUM_CLIENTS, DIRICHLET_ALPHA, SEED, BATCH_SIZE
from datasets import preprocess


def test_partition_completeness():
    partition = client_data.get_partition()
    total = len(preprocess.train_dataset)
    counts = sum(len(v) for v in partition.values())
    assert counts == total, f"Sum of client samples {counts} != total {total}"


def test_no_duplicate_indices():
    partition = client_data.get_partition()
    all_indices = [i for v in partition.values() for i in v]
    assert len(all_indices) == len(set(all_indices)), "Duplicate indices across clients"


def test_no_missing_indices():
    partition = client_data.get_partition()
    total = len(preprocess.train_dataset)
    all_indices = sorted([i for v in partition.values() for i in v])
    assert all_indices == list(range(total)), "Missing or out-of-range indices"


def test_indices_valid_range():
    partition = client_data.get_partition()
    total = len(preprocess.train_dataset)
    for v in partition.values():
        for idx in v:
            assert 0 <= idx < total


def test_same_seed_identical_partition():
    # Recreate partition with same seed
    from datasets.partition import partition_dataset
    p1 = partition_dataset(preprocess.train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, SEED)
    p2 = partition_dataset(preprocess.train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, SEED)
    assert p1 == p2


def test_different_seed_different_partition():
    from datasets.partition import partition_dataset
    p1 = partition_dataset(preprocess.train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, SEED)
    p2 = partition_dataset(preprocess.train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, SEED + 1)
    assert p1 != p2


def test_client_dataset_and_dataloader_creation():
    for client_id in range(NUM_CLIENTS):
        ds = client_data.get_client_dataset(client_id)
        assert len(ds) > 0
        dl = client_data.get_client_dataloader(client_id)
        batch = next(iter(dl))
        images, labels = batch
        assert images.shape[0] <= BATCH_SIZE
        assert images.shape[1:] == torch.Size([3, 28, 28])
        # labels may be shape [batch] or [batch,1]
        assert labels.ndim in (1, 2)
        # label values in 0..8
        lbls = labels.flatten().numpy()
        assert np.all((lbls >= 0) & (lbls <= 8))


def test_client_invalid_id():
    with pytest.raises(ValueError):
        client_data.get_client_dataset(NUM_CLIENTS)  # out of range


def test_edge_cases():
    from datasets.partition import partition_dataset
    with pytest.raises(ValueError):
        partition_dataset(preprocess.train_dataset, 0, DIRICHLET_ALPHA, SEED)
    with pytest.raises(ValueError):
        partition_dataset(preprocess.train_dataset, NUM_CLIENTS, 0.0, SEED)
