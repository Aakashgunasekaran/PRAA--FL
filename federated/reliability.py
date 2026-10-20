"""Live reliability evidence during federated training (Phases 4-5 inside the round loop).

For every client that trains in a round we record *client-local* evidence:

* prototypes: class-wise means of the projected embeddings that the freshly trained LOCAL
  model produces for the client's labelled samples (same label policy as Phase 4);
* validation embeddings of the LOCAL model on the shared validation subset.

This is the evidence the preserved Phase 3 run never stored (it only has the final global
model), which is why PRAA needs a new training run: stability needs attested client-local
snapshots over several participations (``provenance.source_kind == "client_local_model"``).

PRAM components are computed with ``federated.pram`` (unchanged, formula ``pram/1``):

* validation  - nearest-prototype recall of the client's prototypes on the validation
                subset, using the embeddings of the model that produced the prototypes;
* consistency - (cos + 1)/2 against the unweighted self-inclusive mean of the same class
                over the clients being compared (this round's participants during
                training; all clients for the final global-model assessment);
* stability   - mean (cos + 1)/2 over the most recent consecutive attested snapshots.
"""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from evaluation.inference import embed_images
from federated.pram import assess_client, build_reference_prototypes
from training.prototypes import extract_features, generate_prototypes

ATTESTED_KIND = "client_local_model"
SNAPSHOT_SCHEMA = "praa-fl.snapshot/1"


def local_evidence(model, labeled_dataset, validation_images: torch.Tensor, device="cpu",
                   batch_size: int = 64) -> Dict:
    """Prototypes + validation embeddings of ONE local model (no gradients, eval mode)."""
    loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)
    embeddings, labels = extract_features(model, loader, device=device)
    prototypes, counts = generate_prototypes(embeddings, labels)
    return {
        "prototypes": prototypes,
        "counts": counts,
        "val_embeddings": embed_images(model, validation_images, device=device),
    }


def _record(prototypes: Mapping[int, torch.Tensor], counts: Mapping[int, int]) -> Dict:
    dims = {int(v.numel()) for v in prototypes.values()}
    if len(dims) != 1:
        raise ValueError("prototypes of one client must share an embedding dimension")
    return {
        "prototypes": {int(c): v for c, v in prototypes.items()},
        "metadata": {int(c): {"sample_count": int(counts[c])} for c in prototypes},
        "embedding_dim": dims.pop(),
    }


class ReliabilityTracker:
    """Keeps attested client-local prototype snapshots and runs PRAM on them."""

    def __init__(self, snapshot_dir: Optional[Path] = None, stability_window: Optional[int] = 3):
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.stability_window = stability_window
        self.history: Dict[int, List[Tuple[int, Dict[int, torch.Tensor]]]] = {}

    # -- snapshots ---------------------------------------------------------
    def _windowed(self, client_id: int):
        snaps = self.history.get(client_id, [])
        return snaps if self.stability_window is None else snaps[-self.stability_window:]

    def add_snapshot(self, client_id: int, round_id: int, prototypes: Mapping[int, torch.Tensor],
                     counts: Mapping[int, int]) -> None:
        snaps = self.history.setdefault(int(client_id), [])
        if snaps and snaps[-1][0] >= round_id:
            raise ValueError(f"client {client_id}: snapshot rounds must increase (got {round_id})")
        snaps.append((int(round_id), {int(c): v.detach().float().cpu().clone() for c, v in prototypes.items()}))
        if self.snapshot_dir is not None:
            path = self.snapshot_dir / "snapshots" / f"round_{round_id}" / f"client_{client_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema_version": SNAPSHOT_SCHEMA,
                "client_id": int(client_id),
                "round": int(round_id),
                "prototypes": {int(c): v.detach().float().cpu().clone() for c, v in prototypes.items()},
                "sample_counts": {int(c): int(counts[c]) for c in prototypes},
                "provenance": {"source_kind": ATTESTED_KIND,
                               "origin": "prototypes of the local model right after local training"},
            }, path)

    def load_from_disk(self, max_round: Optional[int] = None, delete_newer: bool = False) -> int:
        """Rebuild history from ``snapshots/`` (resume).  Files newer than ``max_round`` are
        ignored (and removed when ``delete_newer``) so a crashed round is never double counted.
        Returns the number of snapshots loaded."""
        self.history = {}
        if self.snapshot_dir is None:
            return 0
        root = self.snapshot_dir / "snapshots"
        if not root.exists():
            return 0
        loaded = 0
        for round_dir in sorted(root.glob("round_*"), key=lambda p: int(p.name.split("_")[1])):
            round_id = int(round_dir.name.split("_")[1])
            if max_round is not None and round_id > max_round:
                if delete_newer:
                    for f in round_dir.glob("*.pt"):
                        f.unlink()
                    round_dir.rmdir()
                continue
            for path in sorted(round_dir.glob("client_*.pt")):
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if (payload.get("provenance") or {}).get("source_kind") != ATTESTED_KIND:
                    raise ValueError(f"{path}: snapshot is not an attested client-local snapshot")
                cid = int(payload["client_id"])
                self.history.setdefault(cid, []).append(
                    (int(payload["round"]), {int(c): v for c, v in payload["prototypes"].items()}))
                loaded += 1
        for snaps in self.history.values():
            snaps.sort(key=lambda item: item[0])
        return loaded

    # -- PRAM during training ----------------------------------------------
    def assess_round(self, round_id: int, evidence: Mapping[int, Mapping],
                     validation_labels: torch.Tensor) -> List[Dict]:
        """Record this round's local snapshots and return PRAM rows for the participants."""
        records = {cid: _record(ev["prototypes"], ev["counts"]) for cid, ev in evidence.items()}
        for cid, ev in sorted(evidence.items()):
            self.add_snapshot(cid, round_id, ev["prototypes"], ev["counts"])
        reference = build_reference_prototypes(records)
        rows: List[Dict] = []
        for cid in sorted(records):
            rows.extend(assess_client(
                cid, records[cid], reference, evidence[cid]["val_embeddings"], validation_labels,
                self._windowed(cid), round_id))
        return rows

    # -- final / per-round assessment of GLOBAL-model prototypes -------------
    def assess_global_prototypes(
        self, round_id: int,
        client_prototypes: Mapping[int, Mapping[int, torch.Tensor]],
        client_counts: Mapping[int, Mapping[int, int]],
        validation_embeddings: torch.Tensor,
        validation_labels: torch.Tensor,
    ) -> List[Dict]:
        """PRAM over ALL clients' prototypes computed with the same global model.

        validation: global-model embeddings (as in Phase 5); consistency: reference over all
        clients; stability: from the attested local snapshots recorded during training
        (unavailable for a client with fewer than two snapshots)."""
        records = {cid: _record(p, client_counts[cid]) for cid, p in client_prototypes.items() if p}
        reference = build_reference_prototypes(records)
        rows: List[Dict] = []
        for cid in sorted(records):
            rows.extend(assess_client(
                cid, records[cid], reference, validation_embeddings, validation_labels,
                self._windowed(cid), round_id))
        return rows
