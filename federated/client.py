from typing import Dict
import torch
from torch import nn, optim

from federated.client_data import get_client_dataset
from models.classifier import FLAMEModel
from models.masking import RandomPatchMasker
from training.episodes import make_support_query
from training.losses import compute_mae_loss, compute_proto_loss, combine_losses


class Client:
    """Local FLAME client; prototypes never leave this object."""

    def __init__(self, client_id: int, config: Dict, device: str = "cpu", base_dataset=None, partition=None) -> None:
        if not 0 < config["label_fraction"] <= 1:
            raise ValueError("label_fraction must be in (0, 1]")
        self.client_id = client_id
        self.config = config
        self.device = device
        self._learning_rate = config["learning_rate"]
        self._scheduler_state = None
        self._pending_validation_loss = None
        base = base_dataset if base_dataset is not None else get_client_dataset(client_id, partition=partition)
        generator = torch.Generator().manual_seed(config["seed"] + client_id)
        order = torch.randperm(len(base), generator=generator).tolist()
        labeled_count = max(2, int(len(order) * config["label_fraction"]))
        labeled_count = min(labeled_count, len(order))
        self.labeled_indices = order[:labeled_count]
        self.unlabeled_indices = order[labeled_count:]
        self.labeled_dataset = torch.utils.data.Subset(base, self.labeled_indices)
        self.unlabeled_dataset = torch.utils.data.Subset(base, self.unlabeled_indices)
        self.labeled_loader = torch.utils.data.DataLoader(
            self.labeled_dataset, batch_size=config["batch_size"], shuffle=True,
            generator=generator
        )
        self.unlabeled_loader = torch.utils.data.DataLoader(
            self.unlabeled_dataset, batch_size=config["batch_size"], shuffle=True,
            generator=generator
        )

    def _create_training_components(self):
        torch.manual_seed(self.config["seed"] + self.client_id)
        self.model = FLAMEModel(
            embedding_dim=self.config["embedding_dim"],
            num_classes=self.config["num_classes"],
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self._learning_rate,
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.config.get("scheduler_factor", 0.1),
            patience=self.config.get("scheduler_patience", 5),
        )
        if self._scheduler_state is not None:
            self.scheduler.load_state_dict(self._scheduler_state)
        if self._pending_validation_loss is not None:
            self.scheduler.step(self._pending_validation_loss)
            self._pending_validation_loss = None

    def client_update(self, global_state_dict):
        torch.manual_seed(self.config["seed"] + self.client_id)
        self._create_training_components()
        self.model.load_state_dict(global_state_dict)
        masker = RandomPatchMasker(
            self.config["patch_size"], self.config["mask_ratio"]
        )
        self.model.train()
        stats = {"mae_loss": 0.0, "proto_loss": 0.0, "combined_loss": 0.0, "steps": 0}
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        steps = max(len(self.labeled_loader), len(self.unlabeled_loader), 1)
        for _ in range(self.config["local_epochs"]):
            for step in range(steps):
                if self.config.get("max_batches") is not None and step >= self.config["max_batches"]:
                    break
                if len(self.unlabeled_loader):
                    try:
                        unlabeled_images, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(self.unlabeled_loader)
                        unlabeled_images, _ = next(unlabeled_iter)
                else:
                    unlabeled_images, _ = next(iter(self.labeled_loader))
                try:
                    labeled_images, labeled_labels = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.labeled_loader)
                    labeled_images, labeled_labels = next(labeled_iter)
                unlabeled_images = unlabeled_images.to(self.device)
                labeled_images = labeled_images.to(self.device)
                labeled_labels = labeled_labels.view(-1).to(self.device)
                masked, _ = masker(unlabeled_images)
                features = self.model.encoder(masked)
                reconstruction = self.model.reconstruct(features)
                mae_loss = compute_mae_loss(reconstruction, unlabeled_images)
                try:
                    support_x, support_y, query_x, query_y = make_support_query(
                        labeled_images, labeled_labels, seed=self.config["seed"] + stats["steps"]
                    )
                except ValueError:
                    # A sparse batch with no class having two examples contributes MAE only.
                    proto_loss = mae_loss.new_zeros(())
                else:
                    _, support_h = self.model(support_x)
                    _, query_h = self.model(query_x)
                    proto_loss = compute_proto_loss(
                        support_h, support_y, query_h, query_y, self.model.num_classes
                    )
                loss = combine_losses(
                    mae_loss, proto_loss, self.config["mae_weight"],
                    self.config["mae_reference"], self.config["proto_reference"]
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                stats["mae_loss"] += float(mae_loss.item())
                stats["proto_loss"] += float(proto_loss.item())
                stats["combined_loss"] += float(loss.item())
                stats["steps"] += 1
        if stats["steps"]:
            for key in ("mae_loss", "proto_loss", "combined_loss"):
                stats[key] /= stats["steps"]
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._learning_rate = self.optimizer.param_groups[0]["lr"]
        self._scheduler_state = self.scheduler.state_dict()
        del self.model
        del self.optimizer
        del self.scheduler
        return state, len(self.labeled_dataset) + len(self.unlabeled_dataset), stats

    def step_scheduler(self, validation_loss: float) -> None:
        """Advance this client's scheduler using the global validation loss."""
        if hasattr(self, "scheduler"):
            self.scheduler.step(validation_loss)
            self._learning_rate = self.optimizer.param_groups[0]["lr"]
            self._scheduler_state = self.scheduler.state_dict()
        else:
            self._pending_validation_loss = validation_loss

    @property
    def learning_rate(self) -> float:
        return self._learning_rate
