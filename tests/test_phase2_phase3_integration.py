import torch

from datasets import preprocess
from federated import client_data
from federated.client import Client
from federated.flame_config import (
    DIRICHLET_BETA,
    MIN_SAMPLES_PER_CLIENT,
    NUM_CLIENTS,
    RANDOM_SEED,
)
from federated.flame_data import PARTITION, get_flame_client_dataset
from models.classifier import FLAMEModel


def test_phase3_consumes_phase2_partition():
    canonical = client_data.build_partition(
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_BETA,
        seed=RANDOM_SEED,
        min_samples_per_client=MIN_SAMPLES_PER_CLIENT,
    )
    assert PARTITION == canonical
    assert set(PARTITION) == set(range(NUM_CLIENTS))
    assigned = [index for indices in PARTITION.values() for index in indices]
    assert len(assigned) == len(preprocess.train_dataset)
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(range(len(preprocess.train_dataset)))


def test_phase3_client_uses_exact_phase2_dataset():
    config = {
        "label_fraction": 0.01,
        "batch_size": 2,
        "seed": RANDOM_SEED,
        "embedding_dim": 16,
        "num_classes": 9,
        "local_epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "patch_size": 4,
        "mask_ratio": 0.8,
        "mae_weight": 0.7,
        "mae_reference": 1.0,
        "proto_reference": 1.0,
    }
    phase2_dataset = get_flame_client_dataset(0)
    client = Client(0, config, base_dataset=phase2_dataset)
    assert client.labeled_dataset.dataset is phase2_dataset
    assert client.unlabeled_dataset.dataset is phase2_dataset
    assert len(client.labeled_dataset) + len(client.unlabeled_dataset) == len(phase2_dataset)
    assert client.labeled_dataset.dataset.indices == phase2_dataset.indices
    assert client.unlabeled_dataset.dataset.indices == phase2_dataset.indices
    assert phase2_dataset.dataset is preprocess.train_dataset
    assert phase2_dataset is not preprocess.test_dataset
