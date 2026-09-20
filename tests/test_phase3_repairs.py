import torch
from torch.utils.data import TensorDataset

from federated.client import Client
from federated.flame import average_state_dicts
from training.episodes import make_support_query


def _config():
    return {
        "label_fraction": 0.5,
        "batch_size": 4,
        "seed": 42,
        "embedding_dim": 16,
        "num_classes": 3,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patch_size": 4,
        "mask_ratio": 0.8,
        "mae_weight": 0.7,
        "mae_reference": 1.0,
        "proto_reference": 1.0,
        "scheduler_factor": 0.1,
        "scheduler_patience": 0,
    }


def test_scheduler_changes_client_training_optimizer_learning_rate():
    images = torch.randn(4, 3, 28, 28)
    labels = torch.tensor([[0], [0], [1], [1]])
    client = Client(0, _config(), base_dataset=TensorDataset(images, labels))
    client._create_training_components()
    initial = client.learning_rate
    client.step_scheduler(1.0)
    client.step_scheduler(1.0)
    assert initial == 0.01
    assert client.learning_rate == initial * 0.1
    assert client.scheduler.optimizer is client.optimizer
    del client.model
    del client.optimizer
    del client.scheduler


def test_aggregation_preserves_integer_and_bool_buffer_dtypes():
    states = [
        {"weight": torch.tensor([1.0]), "counter": torch.tensor(2, dtype=torch.long),
         "flag": torch.tensor(True)},
        {"weight": torch.tensor([3.0]), "counter": torch.tensor(4, dtype=torch.long),
         "flag": torch.tensor(False)},
    ]
    result = average_state_dicts(states, [0.25, 0.75])
    assert result["weight"].dtype == torch.float32
    assert result["counter"].dtype == torch.long
    assert result["counter"].item() == 4
    assert result["flag"].dtype == torch.bool


def test_sparse_episode_excludes_singleton_class_and_keeps_sets_disjoint():
    images = torch.arange(5 * 3 * 28 * 28, dtype=torch.float32).view(5, 3, 28, 28)
    labels = torch.tensor([0, 1, 1, 2, 2])
    support_x, support_y, query_x, query_y = make_support_query(images, labels, seed=42)
    assert 0 not in support_y.tolist() + query_y.tolist()
    assert set(map(tuple, support_x.flatten(1).tolist())).isdisjoint(
        set(map(tuple, query_x.flatten(1).tolist()))
    )
    assert set(support_y.tolist()) == {1, 2}
    assert set(query_y.tolist()) == {1, 2}


def test_sparse_episode_with_no_eligible_class_is_explicitly_rejected():
    images = torch.randn(2, 3, 28, 28)
    labels = torch.tensor([0, 1])
    try:
        make_support_query(images, labels)
    except ValueError as error:
        assert "at least two samples" in str(error)
    else:
        raise AssertionError("an episode without an eligible class must be rejected")
