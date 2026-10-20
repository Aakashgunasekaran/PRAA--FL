"""Phase 8: global model evaluation and comparison of FedAvg, FLAME and PRAA.

Sub-commands
------------
evaluate        evaluate a finished run directory (fedavg | flame | praa) on val or test
evaluate-original
                evaluate the PRESERVED Phase 3 FLAME model (no retraining): FLAME's own rule
                (sample-weighted global prototypes from the Phase 4 files) and, as a clearly
                labelled post-hoc variant, PRAA's prototype aggregation
tune-tau        development sweep of the PRAA exponent tau on VALIDATION data only
compare         build the comparison table / convergence summary / figure from eval files

The test split is only ever read with ``--split test --final``; every such evaluation records
``test_set_accessed: true``.  Development (tau, weights, rounds) must use ``--split val``.

Inference rules: FedAvg uses the classifier head; FLAME and PRAA use nearest global prototype
(softmax over -squared distance, paper Eq. 6/10).  Metrics: accuracy, balanced accuracy,
macro-F1, macro one-vs-rest AUROC, NLL.
"""

import argparse
import contextlib
import csv
import datetime
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from evaluation.inference import embed_images, prototype_logits, softmax_probabilities  # noqa: E402
from evaluation.metrics import compute_metrics, convergence_summary  # noqa: E402
from federated.experiment import (World, assert_split, evaluate_classifier_head,  # noqa: E402
                                  evaluate_prototype_family, load_tensors)
from federated.experiment_config import validate_config  # noqa: E402
from federated.pram import load_pram_output  # noqa: E402
from federated.praa import (PRAAConfig, aggregate_prototypes, class_multipliers,  # noqa: E402
                            compute_record_prs)
from federated.reliability import ReliabilityTracker  # noqa: E402
from models.classifier import FLAMEModel, model_kwargs_from_config  # noqa: E402

PAPER_REFERENCE = {
    "source": "FLAME paper, Table I, PathMNIST, 1% labels (paper's own 80/20 split, mean of 3 runs)",
    "accuracy": 0.818, "balanced_accuracy": 0.791, "macro_f1": 0.79, "auroc_macro_ovr": None,
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_eval_split(preprocess, split: str, final: bool):
    """Return (images, labels) of ``val`` or ``test``.  Test needs the explicit ``final`` flag."""
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if split == "test" and not final:
        raise PermissionError("the test split is reserved for the final evaluation: pass --final")
    dataset = preprocess.val_dataset if split == "val" else preprocess.test_dataset
    assert_split(dataset, split)
    return load_tensors(dataset, range(len(dataset)))


def _import_preprocess():
    with contextlib.redirect_stdout(io.StringIO()):
        from datasets import preprocess
    return preprocess


def _load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    validate_config(config)
    state = torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=False)
    model = FLAMEModel(**model_kwargs_from_config(config))
    model.load_state_dict(state["model_state_dict"], strict=True)
    return config, model.eval(), state


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def evaluate_run(run_dir: Path, split: str, final: bool, device: str, praa_override=None) -> dict:
    config, model, state = _load_run(run_dir)
    model.to(device)
    world = World(config, device)
    eval_images, eval_labels = load_eval_split(world.preprocess, split, final)
    method = config["method"]
    result = {"method": method, "split": split, "evaluated_at": _now(),
              "test_set_accessed": split == "test", "rounds_trained": int(config["num_rounds"]),
              "synthetic_data": os.environ.get("PRAA_SYNTHETIC_DATA") == "1",
              "model_sha_hint": run_dir.name}
    if method == "fedavg":
        result["rules"] = {"classifier_head": evaluate_classifier_head(
            model, eval_images, eval_labels, device, config["num_classes"], full=True)}
        result["primary_rule"] = "classifier_head"
    else:
        praa_cfg = PRAAConfig.from_dict(praa_override or config["praa"])
        tracker = ReliabilityTracker(run_dir, praa_cfg.stability_window)
        loaded = tracker.load_from_disk()
        # reliability evidence ALWAYS comes from validation data
        rel_images, rel_labels = (world.val_images, world.val_labels)
        fam = evaluate_prototype_family(
            model, world.label_cache, rel_images, rel_labels, eval_images, eval_labels, tracker,
            praa_cfg, int(config["num_rounds"]), device, config["num_classes"], full=True)
        result["rules"] = {"flame_rule": fam["flame_rule"], "praa_rule": fam["praa_rule"]}
        result["primary_rule"] = "praa_rule" if method == "praa" else "flame_rule"
        result["reliability"] = fam["reliability"]
        result["praa_config"] = praa_cfg.to_dict()
        result["snapshots_loaded"] = loaded
        result["pram_rows"], result["prs_records"] = fam["pram_rows"], fam["prs_records"]
        result["praa_prototype_weights"] = fam["praa_prototype_weights"]
    result["primary"] = {k: result["rules"][result["primary_rule"]][k]
                         for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc_macro_ovr", "nll")}
    return result


# ---------------------------------------------------------------------------
# evaluate-original (preserved Phase 3 FLAME model)
# ---------------------------------------------------------------------------

def evaluate_original(phase3_dir: Path, phase4_dir: Path, pram_path: Path, split: str, final: bool,
                      device: str, praa_cfg: PRAAConfig) -> dict:
    from run_phase4 import expected_phase3_config
    from run_phase5 import load_phase4_records
    from training.prototype_store import validate_phase3_artifacts

    source = validate_phase3_artifacts(phase3_dir, expected_phase3_config(), require_rounds=200)
    records, _, _, _ = load_phase4_records(phase4_dir)
    config = source["config"]
    model = FLAMEModel(embedding_dim=config["embedding_dim"], num_classes=config["num_classes"])
    model.load_state_dict(source["state_dict"], strict=True)
    model.to(device).eval()
    images, labels = load_eval_split(_import_preprocess(), split, final)
    emb = embed_images(model, images, device=device)

    protos = {cid: {int(c): v for c, v in r["prototypes"].items()} for cid, r in records.items()}
    counts = {cid: {int(c): int(r["metadata"][c]["sample_count"]) for c in r["prototypes"]}
              for cid, r in records.items()}
    num_classes = int(config["num_classes"])
    out = {}
    gp_flame, _ = aggregate_prototypes(protos, counts, None)
    pram = load_pram_output(pram_path)
    prs = compute_record_prs(pram["records"], praa_cfg)
    mult, info = class_multipliers(prs, praa_cfg)
    gp_praa, weights = aggregate_prototypes(protos, counts, mult)
    for name, gp in (("flame_rule", gp_flame), ("praa_posthoc_rule", gp_praa)):
        logits, missing = prototype_logits(emb, gp, num_classes)
        m = compute_metrics(labels.numpy(), softmax_probabilities(logits).numpy(), num_classes)
        m["missing_prototype_classes"] = missing
        out[name] = m
    identical = all(torch.equal(gp_flame[c], gp_praa[c]) for c in gp_flame)
    return {
        "method": "flame_original_preserved", "split": split, "evaluated_at": _now(),
        "test_set_accessed": split == "test",
        "phase3_model_sha256": source["provenance"]["source_model_sha256"],
        "phase3_round": source["provenance"]["source_round"],
        "praa_config": praa_cfg.to_dict(),
        "praa_posthoc_identical_to_flame": identical,
        "praa_posthoc_note": (
            "strict PRS needs stability, which the preserved run cannot provide, so post-hoc PRAA "
            "equals FLAME" if identical else
            "post-hoc PRAA used the renormalize EXTENSION (validation + consistency only); exploratory"),
        "reliability": {"records": len(prs), "records_with_prs": sum(r["prs"] is not None for r in prs),
                        "classes_reweighted": sum(1 for i in info.values() if i["applied"])},
        "rules": out,
        "primary_rule": "flame_rule",
        "primary": {k: out["flame_rule"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                                      "auroc_macro_ovr", "nll")},
    }


# ---------------------------------------------------------------------------
# tune-tau (validation only)
# ---------------------------------------------------------------------------

def tune_tau(run_dir: Path, taus, device: str, policy: str) -> dict:
    config, model, _ = _load_run(run_dir)
    model.to(device)
    world = World(config, device)
    rows = []
    for tau in taus:
        cfg = dict(config["praa"], tau=float(tau), missing_component_policy=policy)
        praa_cfg = PRAAConfig.from_dict(cfg)
        tracker = ReliabilityTracker(run_dir, praa_cfg.stability_window)
        tracker.load_from_disk()
        fam = evaluate_prototype_family(model, world.label_cache, world.val_images, world.val_labels,
                                        world.val_images, world.val_labels, tracker, praa_cfg,
                                        int(config["num_rounds"]), device, config["num_classes"])
        rows.append({"tau": float(tau), "praa_rule": fam["praa_rule"], "flame_rule": fam["flame_rule"],
                     "records_with_prs": fam["reliability"]["records_with_prs"]})
    return {"split": "val", "test_set_accessed": False, "policy": policy, "run": str(run_dir), "sweep": rows}


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "auroc_macro_ovr")


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def build_comparison(entries: dict, histories: dict, split: str, targets=(0.5, 0.6, 0.7, 0.8)) -> dict:
    rows = []
    for name, ev in entries.items():
        rows.append({"name": name, "split": ev["split"], "rule": ev["primary_rule"],
                     "test_set_accessed": ev["test_set_accessed"],
                     **{k: ev["primary"][k] for k in METRIC_KEYS}})
        for rule, m in ev["rules"].items():
            if rule != ev["primary_rule"]:
                rows.append({"name": f"{name} [{rule}]", "split": ev["split"], "rule": rule,
                             "test_set_accessed": ev["test_set_accessed"],
                             **{k: m[k] for k in METRIC_KEYS}, "ablation": True})
    convergence = {}
    for name, hist in histories.items():
        accs = [h.get("validation_accuracy") for h in hist]
        rounds = [h["round"] for h in hist]
        if any(a is not None for a in accs):
            convergence[name] = convergence_summary(accs, rounds, targets)
    synthetic = any(e.get("synthetic_data") for e in entries.values())
    return {"split": split, "synthetic": synthetic,
            "paper_reference": None if synthetic else PAPER_REFERENCE, "rows": rows,
            "convergence_validation_accuracy": convergence,
            "caveat": ("Paper numbers use the paper's own 80/20 split and 3 seeds; this project uses the "
                       "official MedMNIST split and one seed, so differences are indicative only.")}


def write_comparison(doc: dict, output_dir: Path, histories: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(doc, indent=2, allow_nan=False))
    with open(output_dir / "comparison.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "split", "rule", *METRIC_KEYS])
        for r in doc["rows"]:
            writer.writerow([r["name"], r["split"], r["rule"], *[_fmt(r[k]) for k in METRIC_KEYS]])
    banner = ["**SYNTHETIC DATA (PathMNIST-shaped) - smoke test only; not PathMNIST results.**", ""] \
        if doc.get("synthetic") else []
    lines = [f"# Phase 8 comparison ({doc['split']} split)", "", *banner,
             "| Method | Rule | Accuracy | Balanced acc. | Macro-F1 | AUROC (macro OvR) |",
             "|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        lines.append(f"| {r['name']} | {r['rule']} | " + " | ".join(_fmt(r[k]) for k in METRIC_KEYS) + " |")
    ref = doc["paper_reference"]
    if ref:
        lines += ["", f"Paper reference (FLAME, 1% labels): accuracy {ref['accuracy']}, balanced accuracy "
                  f"{ref['balanced_accuracy']}, F1 {ref['macro_f1']}. {doc['caveat']}"]
    lines += ["", "## Convergence (validation accuracy per round)", "",
              "| Method | final | mean last 10 | best (round) | mean over rounds | rounds to 0.5/0.6/0.7/0.8 |",
              "|---|---|---|---|---|---|"]
    for name, c in doc["convergence_validation_accuracy"].items():
        rt = c["rounds_to_target"]
        lines.append(f"| {name} | {c['final']:.4f} | {c['final_mean_last_k']:.4f} | "
                     f"{c['best']:.4f} ({c['best_round']}) | {c['mean_over_rounds']:.4f} | "
                     f"{' / '.join(str(rt[k]) for k in rt)} |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, hist in histories.items():
            pts = [(h["round"], h["validation_accuracy"]) for h in hist if h.get("validation_accuracy") is not None]
            if pts:
                ax.plot(*zip(*pts), label=name)
        ax.set_xlabel("communication round")
        ax.set_ylabel("validation accuracy")
        ax.set_title("Convergence on the validation subset")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "convergence.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # plotting must never block the numeric outputs
        print(f"[warn] convergence figure skipped: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write(path: Path, document: dict, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False))
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_split(p):
        p.add_argument("--split", choices=("val", "test"), default="val")
        p.add_argument("--final", action="store_true", help="required to read the TEST split")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--device", default=None)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    add_split(p)

    p = sub.add_parser("evaluate-original")
    p.add_argument("--phase3-dir", type=Path, default=ROOT / "results" / "phase3")
    p.add_argument("--phase4-dir", type=Path, default=ROOT / "results" / "phase4")
    p.add_argument("--pram", type=Path, default=ROOT / "results" / "phase5" / "pram.json")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "phase8" / "eval_original.json")
    p.add_argument("--missing-component-policy", choices=("strict", "renormalize"), default="strict")
    p.add_argument("--tau", type=float, default=1.0)
    add_split(p)

    p = sub.add_parser("tune-tau")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--taus", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8])
    p.add_argument("--missing-component-policy", choices=("strict", "renormalize"), default="strict")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)

    p = sub.add_parser("compare")
    p.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR",
                   help="run directory holding eval_<split>.json and metrics.json")
    p.add_argument("--original", type=Path, default=None, help="eval_original.json of the preserved model")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase8")

    args = parser.parse_args(argv)
    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "evaluate":
        result = evaluate_run(args.run_dir, args.split, args.final, device)
        _write(args.output or args.run_dir / f"eval_{args.split}.json", result, args.overwrite)
        print({k: round(v, 4) if v is not None else None for k, v in result["primary"].items()})
    elif args.command == "evaluate-original":
        cfg = PRAAConfig(tau=args.tau, missing_component_policy=args.missing_component_policy)
        result = evaluate_original(args.phase3_dir, args.phase4_dir, args.pram, args.split, args.final,
                                   device, cfg)
        _write(args.output, result, args.overwrite)
        print(result["praa_posthoc_note"])
        for rule, m in result["rules"].items():
            print(rule, {k: (round(m[k], 4) if m[k] is not None else None) for k in METRIC_KEYS})
    elif args.command == "tune-tau":
        result = tune_tau(args.run_dir, args.taus, device, args.missing_component_policy)
        _write(args.output or args.run_dir / "tune_tau_val.json", result, args.overwrite)
        for row in result["sweep"]:
            print(f"tau={row['tau']:<4} val_acc(praa)={row['praa_rule']['accuracy']:.4f} "
                  f"val_acc(flame)={row['flame_rule']['accuracy']:.4f} prs_records={row['records_with_prs']}")
    elif args.command == "compare":
        entries, histories = {}, {}
        for item in args.run:
            name, _, path = item.partition("=")
            run_dir = Path(path)
            entries[name] = json.loads((run_dir / f"eval_{args.split}.json").read_text())
            histories[name] = json.loads((run_dir / "metrics.json").read_text())
        if args.original:
            entries["FLAME (original, preserved)"] = json.loads(args.original.read_text())
        if not entries:
            parser.error("nothing to compare")
        doc = build_comparison(entries, histories, args.split)
        write_comparison(doc, args.output_dir, histories)
        print((args.output_dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
