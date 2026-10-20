"""Supervised FedAvg baseline (project definition).

Standard FedAvg (McMahan et al.): each selected client minimises cross-entropy of the
classifier head on its LABELLED samples only (the same 1 % labelled subsets FLAME uses;
unlabelled data is not used), and the server takes the sample-weighted parameter average
(weights proportional to each client's labelled-sample count).  It shares the network,
optimiser, local epochs and client selection with FLAME/PRAA so differences come from the
learning objective and the aggregation.
"""

import torch
import torch.nn.functional as F

from federated.client import Client


class SupervisedClient(Client):
    def client_update(self, global_state_dict, after_update=None):
        torch.manual_seed(self.config["seed"] + self.client_id)
        self._create_training_components()
        self.model.load_state_dict(global_state_dict)
        self.model.train()
        stats = {"ce_loss": 0.0, "steps": 0}
        for _ in range(self.config["local_epochs"]):
            for step, (images, labels) in enumerate(self.labeled_loader):
                if self.config.get("max_batches") is not None and step >= self.config["max_batches"]:
                    break
                images = images.to(self.device)
                labels = labels.view(-1).to(self.device)
                loss = F.cross_entropy(self.model.logits(images), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                stats["ce_loss"] += float(loss.item())
                stats["steps"] += 1
        if stats["steps"]:
            stats["ce_loss"] /= stats["steps"]
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._learning_rate = self.optimizer.param_groups[0]["lr"]
        self._scheduler_state = self.scheduler.state_dict()
        del self.model, self.optimizer, self.scheduler
        return state, len(self.labeled_dataset), stats
