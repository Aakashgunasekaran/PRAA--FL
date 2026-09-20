"""FLAME model components.

The paper defines one shared encoder f_theta, a transposed-convolution
decoder f_theta' and a two-layer projection g_phi.  Prototypes are computed
from projected support embeddings, not classifier logits.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50


class Encoder(nn.Module):
    """ResNet-50 encoder adapted for 3x28x28 PathMNIST images."""

    def __init__(self, feature_dim: int = 2048, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("Pretrained weights are not supported for CPU/offline FLAME runs")
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Identity() if feature_dim == in_features else nn.Linear(in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))


class Decoder(nn.Module):
    """Transposed-convolution decoder producing exactly [B, 3, 28, 28]."""

    def __init__(self, feature_dim: int = 2048) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), 256, 7, 7)
        return self.deconv(x)


class ProtoHead(nn.Module):
    """Two fully connected layers with ReLU, as specified for g_phi."""

    def __init__(self, feature_dim: int = 2048, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FLAMEModel(nn.Module):
    """Shared ResNet-50 encoder with MAE and prototypical branches."""

    def __init__(
        self,
        feature_dim: int = 2048,
        embedding_dim: int = 128,
        num_classes: int = 9,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(feature_dim=feature_dim)
        self.decoder = Decoder(feature_dim=feature_dim)
        self.proto = ProtoHead(feature_dim=feature_dim, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return features, self.proto(features)

    def reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        _, embedding = self.forward(x)
        return self.classifier(embedding)


def compute_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute p_c as the mean projected embedding of support examples."""
    prototypes = embeddings.new_zeros((num_classes, embeddings.size(1)))
    for class_id in range(num_classes):
        selected = embeddings[labels == class_id]
        if selected.numel():
            prototypes[class_id] = selected.mean(dim=0)
    return prototypes


def squared_euclidean_logits(
    embeddings: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    """Return -||embedding - prototype||_2^2 logits."""
    return -((embeddings[:, None, :] - prototypes[None, :, :]) ** 2).sum(dim=-1)


def proto_logits(embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return squared_euclidean_logits(embeddings, prototypes)
