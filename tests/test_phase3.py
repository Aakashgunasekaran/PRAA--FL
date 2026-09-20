import torch

from federated.client import Client
from federated.flame_config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    LABEL_FRACTION,
    LEARNING_RATE,
    MAE_WEIGHT,
    MASK_RATIO,
    PATCH_SIZE,
)
from models.classifier import FLAMEModel, compute_prototypes, squared_euclidean_logits
from models.masking import RandomPatchMasker
from training.episodes import make_support_query
from training.losses import combine_losses, compute_mae_loss, compute_proto_loss


def test_resnet50_forward_and_decoder():
    model = FLAMEModel(embedding_dim=EMBEDDING_DIM)
    images = torch.randn(2, 3, 28, 28)
    features, embeddings = model(images)
    assert features.shape == (2, 2048)
    assert embeddings.shape == (2, EMBEDDING_DIM)
    assert model.reconstruct(features).shape == (2, 3, 28, 28)


def test_patch_masking_ratio_and_shape():
    masked, mask = RandomPatchMasker(PATCH_SIZE, MASK_RATIO)(torch.ones(2, 3, 28, 28))
    assert masked.shape == (2, 3, 28, 28)
    assert mask.shape == (2, 28, 28)
    assert mask[0, ::PATCH_SIZE, ::PATCH_SIZE].sum().item() == 39


def test_support_query_prototypes_and_query_loss():
    images = torch.randn(12, 3, 28, 28)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    sx, sy, qx, qy = make_support_query(images, labels, seed=42)
    assert len(sx) + len(qx) == len(images)
    model = FLAMEModel(embedding_dim=16)
    _, sh = model(sx)
    _, qh = model(qx)
    prototypes = compute_prototypes(sh, sy, 9)
    assert torch.allclose(prototypes[0], sh[sy == 0].mean(0))
    assert compute_proto_loss(sh, sy, qh, qy, 9).ndim == 0
    assert squared_euclidean_logits(qh, prototypes).shape == (len(qx), 9)


def test_combined_loss_and_shared_encoder_gradient():
    model = FLAMEModel(embedding_dim=16)
    images = torch.randn(4, 3, 28, 28)
    labels = torch.tensor([0, 0, 1, 1])
    features, emb = model(images)
    mae = compute_mae_loss(model.reconstruct(features), images)
    support_x, support_y, query_x, query_y = make_support_query(images, labels, seed=42)
    _, support_h = model(support_x)
    _, query_h = model(query_x)
    proto = compute_proto_loss(support_h, support_y, query_h, query_y, 2)
    loss = combine_losses(mae, proto, MAE_WEIGHT)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_client_sparse_split():
    config = {
        "label_fraction": LABEL_FRACTION,
        "batch_size": BATCH_SIZE,
        "seed": 42,
    }
    client = Client(0, {**config, "embedding_dim": 16, "num_classes": 9,
                        "local_epochs": 1, "learning_rate": LEARNING_RATE,
                        "weight_decay": 0.0, "patch_size": 4, "mask_ratio": 0.8,
                        "mae_weight": 0.7, "mae_reference": 1.0,
                        "proto_reference": 1.0})
    assert not set(client.labeled_indices) & set(client.unlabeled_indices)
