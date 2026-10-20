"""READ-ONLY Phase 5 audit. Writes nothing, trains nothing, never reads test data.

    python audit_phase5.py --phase3-dir /content/drive/MyDrive/phase3 \
        --phase4-dir /content/drive/MyDrive/phase4 \
        --phase5-dir /content/drive/MyDrive/phase5 --expect-mode final \
        --recompute-validation-clients 0,7,31

Labels: PASS (verified), FAIL (defect), WARN (methodological limitation). Exit status is
non-zero on any FAIL. Consistency is recomputed independently (NumPy float64) for EVERY
record from the raw Phase 4 files; validation is recomputed for the listed clients only,
because it needs the validation split.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from federated import flame_config as cfg
from federated import pram
from training.prototype_store import load_client_prototype_file, sha256_file


class Auditor:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append(("PASS" if ok else "FAIL", name, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    def warn(self, name, detail):
        self.results.append(("WARN", name, detail))
        print(f"[WARN] {name} -- {detail}")

    @property
    def failed(self):
        return [r for r in self.results if r[0] == "FAIL"]


def cosine01(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) + 1.0) / 2.0


def recompute_validation(model, dataset, n_used, prototypes):
    """Independent NumPy nearest-prototype recall for one client (validation prefix of n_used)."""
    subset = torch.utils.data.Subset(dataset, range(n_used))
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False):
            embeddings.append(model(images)[1].double().numpy())
            labels.append(batch_labels.view(-1).numpy())
    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    ids = sorted(prototypes)
    matrix = np.stack([np.asarray(prototypes[c], dtype=np.float64) for c in ids])
    distances = ((embeddings[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
    predicted = np.asarray(ids)[distances.argmin(axis=1)]
    out = {}
    for class_id in ids:
        mask = labels == class_id
        n = int(mask.sum())
        out[class_id] = (n, float((predicted[mask] == class_id).sum()) / n if n else None)
    return out


def audit(phase3_dir, phase4_dir, phase5_dir, expect_mode="final", validation_clients=(),
          validation_dataset=None, auditor=None):
    a = auditor or Auditor()
    phase3_dir, phase4_dir, phase5_dir = Path(phase3_dir), Path(phase4_dir), Path(phase5_dir)
    path = phase5_dir / "pram.json"
    if not a.check("pram.json exists", path.exists(), str(path)):
        return a
    try:
        document = pram.load_pram_output(path)  # schema + summary/readiness agreement
        a.check("pram.json passes schema and derived-section validation", True)
    except Exception as error:  # noqa: BLE001
        a.check("pram.json passes schema and derived-section validation", False, str(error))
        return a
    rows = document["records"]
    a.check(f"mode is '{expect_mode}'", document["mode"] == expect_mode, document["mode"])
    a.check("no duplicate (client, class) records", len({(r["client_id"], r["class_id"]) for r in rows}) == len(rows))
    a.check("test set not accessed (recorded)", document["provenance"]["test_set_accessed"] is False)
    a.check("status is not COMPLETE unless every record is complete",
            (document["status"] == "COMPLETE") == all(r["complete"] for r in rows), document["status"])

    # --- Phase 4 inputs: files, provenance, coverage ------------------------------------
    files = sorted((phase4_dir / "prototypes").glob("client_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    records = {}
    for file in files:
        record = load_client_prototype_file(file)
        records[int(record["client_id"])] = record
    expected_clients = cfg.NUM_CLIENTS if expect_mode == "final" else len(records)
    a.check(f"Phase 4 covers {expected_clients} clients", sorted(records) == list(range(expected_clients)),
            f"{len(records)} files")
    a.check("Phase 4 mode matches expectation (final needs final)",
            expect_mode == "smoke" or {r["config"]["mode"] for r in records.values()} == {"final"})
    recorded_hashes = document["provenance"]["phase4_prototype_file_sha256"]
    a.check("recorded Phase 4 file hashes equal actual files",
            recorded_hashes == {f.name: sha256_file(f) for f in files})
    a.check("recorded Phase 3 model hash equals actual file",
            document["provenance"]["phase3_model_sha256"] == sha256_file(phase3_dir / "final_model.pt"))
    first = next(iter(records.values()))["provenance"]
    a.check("Phase 4 records were built from the same Phase 3 model",
            first["source_model_sha256"] == document["provenance"]["phase3_model_sha256"])
    a.check("round_id equals Phase 4 source round", {r["round_id"] for r in rows} == {first["source_round"]})
    a.check("partition hash carried through", document["provenance"]["phase4_partition_sha256"] == first["partition_sha256"])

    expected_keys = {(c, k) for c, r in records.items() for k in r["class_ids"].tolist()}
    a.check("records cover exactly the Phase 4 (client, class) pairs",
            expected_keys == {(r["client_id"], r["class_id"]) for r in rows}, f"{len(expected_keys)} pairs")
    counts_ok = dims_ok = True
    for row in rows:
        record = records[row["client_id"]]
        i = record["class_ids"].tolist().index(row["class_id"])
        counts_ok &= row["sample_count"] == int(record["sample_counts"][i])
        dims_ok &= row["prototype_dimension"] == int(record["embedding_dim"])
    a.check("sample counts and dimensions equal Phase 4", counts_ok and dims_ok)

    # --- consistency: independent recompute for EVERY record ----------------------------
    holders = {}
    for c, r in records.items():
        for k, v in r["prototypes"].items():
            holders.setdefault(int(k), {})[c] = np.asarray(v, dtype=np.float64)
    worst, wrong_status, loo_worst = 0.0, [], 0.0
    for row in rows:
        mine = holders[row["class_id"]]
        part = row["consistency"]
        degenerate = len(mine) < 2
        expected_status = "degenerate" if degenerate else "valid"
        if part["status"] != expected_status:
            wrong_status.append((row["client_id"], row["class_id"], part["status"]))
            continue
        reference = np.mean(np.stack(list(mine.values())), axis=0)
        expected = cosine01(mine[row["client_id"]], reference)
        got = part["raw"]
        worst = max(worst, abs(got - expected))
        if not degenerate:
            others = [v for c, v in mine.items() if c != row["client_id"]]
            loo_worst = max(loo_worst, abs(part["details"]["leave_one_out_diagnostic"]
                                           - cosine01(mine[row["client_id"]], np.mean(np.stack(others), axis=0))))
    a.check("consistency status is degenerate iff only one client holds the class", not wrong_status, str(wrong_status[:3]))
    a.check("consistency recomputed independently for all records", worst < 1e-5, f"max |diff| = {worst:.2e}")
    a.check("leave-one-out diagnostic recomputed", loo_worst < 1e-5, f"max |diff| = {loo_worst:.2e}")
    singles = sum(1 for v in holders.values() if len(v) < 2)
    if singles:
        a.warn("classes held by a single client", f"{singles} classes have degenerate consistency")

    # --- validation status logic + optional full recompute ------------------------------
    bad_logic = [(r["client_id"], r["class_id"]) for r in rows
                 if (len(records[r["client_id"]]["class_ids"]) == 1) != (r["validation"]["status"] == "degenerate")
                 and r["validation"]["status"] != "unavailable"]
    a.check("validation is degenerate iff the client holds one class", not bad_logic, str(bad_logic[:3]))
    one_class = sum(len(r["class_ids"]) == 1 for r in records.values())
    if one_class:
        a.warn("single-class clients", f"{one_class} clients have degenerate validation scores")
    candidates = sorted({len(r["class_ids"]) for r in records.values()})
    a.warn("validation recall depends on candidate-class count",
           f"clients hold {candidates} classes; cross-client comparison is not normalized for chance")
    if validation_clients and validation_dataset is not None:
        from models.classifier import FLAMEModel
        state = torch.load(phase3_dir / "final_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        model = FLAMEModel(embedding_dim=document["config"]["embedding_dim"],
                           num_classes=document["config"]["num_classes"])
        model.load_state_dict(state)
        n_used = document["config"]["validation_samples_used"]
        for client_id in validation_clients:
            expected = recompute_validation(model, validation_dataset, n_used, records[client_id]["prototypes"])
            for row in (r for r in rows if r["client_id"] == client_id):
                n, recall = expected[row["class_id"]]
                part = row["validation"]
                got = part["raw"]
                same_n = part["details"]["num_eval_samples"] == n
                close = (recall is None and got is None) or (recall is not None and got is not None
                                                             and abs(got - recall) <= 1.0 / max(n, 1))
                a.check(f"client {client_id} class {row['class_id']}: validation recomputed",
                        same_n and close, f"n={n} saved={got} recomputed={recall}")
    elif validation_clients:
        a.warn("validation recompute skipped", "no validation dataset supplied")

    # --- stability evidence --------------------------------------------------------------
    stab = [r["stability"]["status"] for r in rows]
    n_valid = stab.count("valid")
    a.check("stability: every unavailable record has null value/raw and a reason",
            all(r["stability"]["value"] is None and r["stability"]["raw"] is None and r["stability"]["reason"]
                for r in rows if r["stability"]["status"] == "unavailable"))
    snaps = list((phase4_dir / "snapshots").glob("round_*/client_*.pt")) if (phase4_dir / "snapshots").exists() else []
    a.check("valid stability only where attested snapshots exist",
            n_valid == 0 or document["evidence"]["client_snapshot_files_accepted"] > 0)
    if n_valid == 0:
        a.warn("stability unavailable for all records",
               f"{len(snaps)} snapshot files on disk; needs attested client-local prototype snapshots at >= 2 rounds")
    ready = document["phase6_readiness"]
    a.check("readiness agrees with records", ready["records_with_all_three_valid"] == sum(r["complete"] for r in rows))
    if not ready["prs_computable_for_all_records"]:
        a.warn("complete three-component PRS not computable", f"{ready['records_with_all_three_valid']} of {len(rows)} records complete")
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--expect-mode", choices=["final", "smoke"], default="final")
    parser.add_argument("--recompute-validation-clients", default="")
    options = parser.parse_args()
    clients = [int(c) for c in options.recompute_validation_clients.split(",") if c]
    dataset = None
    if clients:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            from datasets import preprocess
        dataset = preprocess.val_dataset  # validation split only
    a = audit(options.phase3_dir, options.phase4_dir, options.phase5_dir, options.expect_mode, clients, dataset)
    print(f"\nAUDIT: {sum(r[0] == 'PASS' for r in a.results)} passed, {len(a.failed)} failed, "
          f"{sum(r[0] == 'WARN' for r in a.results)} warnings")
    sys.exit(1 if a.failed else 0)


if __name__ == "__main__":
    main()
