import torch
from torch.utils.data import TensorDataset

from federated.flame_data import PARTITION, get_flame_client_dataset
from run_phase4 import load_client_state
from training.prototypes import generate_client_prototypes, generate_prototypes


def test_prototype_mean():
    prototypes, counts = generate_prototypes(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 0])
    )
    assert torch.equal(prototypes[0], torch.tensor([2.0, 3.0]))
    assert counts[0] == 2


def test_multiple_classes_are_separate():
    prototypes, _ = generate_prototypes(
        torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0]]),
        torch.tensor([0, 0, 1]),
    )
    assert torch.equal(prototypes[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(prototypes[1], torch.tensor([10.0, 10.0]))


def test_missing_class_has_no_prototype():
    prototypes, _ = generate_prototypes(
        torch.tensor([[1.0, 2.0]]), torch.tensor([2])
    )
    assert 0 not in prototypes
    assert 1 not in prototypes
    assert 2 in prototypes


def test_prototype_dimension_and_sample_count():
    prototypes, counts = generate_prototypes(
        torch.randn(5, 8), torch.tensor([1, 1, 1, 3, 3])
    )
    assert prototypes[1].shape == (8,)
    assert counts[1] == 3
    assert counts[3] == 2


def test_client_separation_and_metadata():
    class Representation(torch.nn.Module):
        def forward(self, images):
            return images[:, :2], images[:, :2]

    datasets = {
        0: TensorDataset(
            torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
            torch.tensor([0, 0]),
        ),
        1: TensorDataset(torch.tensor([[9.0, 8.0, 0.0]]), torch.tensor([2])),
    }
    result = generate_client_prototypes(Representation(), datasets)
    assert set(result) == {0, 1}
    assert set(result[0]["prototypes"]) == {0}
    assert set(result[1]["prototypes"]) == {2}
    assert result[0]["metadata"][0]["sample_count"] == 2
    assert result[0]["metadata"][0]["embedding_dimension"] == 2


def test_nan_and_inf_are_rejected():
    for invalid in (
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
    ):
        try:
            generate_prototypes(invalid, torch.tensor([0]))
        except FloatingPointError:
            pass
        else:
            raise AssertionError("invalid embeddings must be rejected")


def test_empty_client_is_rejected():
    class Identity(torch.nn.Module):
        def forward(self, images):
            return images, images

    try:
        generate_client_prototypes(
            Identity(),
            {0: TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))},
        )
    except ValueError as error:
        assert "empty dataset" in str(error)
    else:
        raise AssertionError("empty clients must be rejected")


def test_phase2_partition_provides_client_datasets():
    assert len(PARTITION) == 50
    client_dataset = get_flame_client_dataset(0)
    assert len(client_dataset) == len(PARTITION[0])
    assert client_dataset.dataset is not None


def test_feature_extraction_does_not_access_test_dataset():
    class Identity(torch.nn.Module):
        def forward(self, images):
            return images, images

    train_like = TensorDataset(torch.ones(2, 2), torch.tensor([0, 0]))
    result = generate_client_prototypes(Identity(), {0: train_like})
    assert result[0]["samples_processed"] == 2


def test_local_state_is_preferred_over_global_fallback(tmp_path, monkeypatch):
    import run_phase4

    monkeypatch.setattr(run_phase4, "CLIENT_STATE_PATH", tmp_path)
    local_state = {"weight": torch.tensor([2.0])}
    torch.save(
        {"client_id": 0, "round": 3, "model_state_dict": local_state},
        tmp_path / "client_0.pt",
    )
    state, used_fallback = load_client_state(
        0, {"model_state_dict": {"weight": torch.tensor([9.0])}}, True
    )
    assert not used_fallback
    assert torch.equal(state["weight"], local_state["weight"])


def test_final_mode_rejects_missing_local_state(tmp_path, monkeypatch):
    import run_phase4

    monkeypatch.setattr(run_phase4, "CLIENT_STATE_PATH", tmp_path)
    try:
        load_client_state(1, {"model_state_dict": {}}, False)
    except FileNotFoundError as error:
        assert "client 1" in str(error)
    else:
        raise AssertionError("final mode must reject missing local state")


def test_client_state_identity_is_validated(tmp_path, monkeypatch):
    import run_phase4

    monkeypatch.setattr(run_phase4, "CLIENT_STATE_PATH", tmp_path)
    torch.save(
        {"client_id": 1, "model_state_dict": {}},
        tmp_path / "client_0.pt",
    )
    try:
        load_client_state(0, {"model_state_dict": {}}, False)
    except ValueError as error:
        assert "mismatch" in str(error).lower()
    else:
        raise AssertionError("a client must not use another client's state")