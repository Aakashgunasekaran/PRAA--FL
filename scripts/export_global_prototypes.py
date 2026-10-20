"""Export the global prototypes of a finished run so the web app can serve it.

    python scripts/export_global_prototypes.py --run-dir results/runs/praa            # PRAA rule (method=praa)
    python scripts/export_global_prototypes.py --run-dir results/runs/flame           # FLAME rule

Writes ``<run-dir>/global_prototypes.pt`` = {"prototypes": {class_id: Tensor[D]}, "rule": ..., "num_clients_per_class": ...}.
Same procedure as ``federated.experiment.evaluate_prototype_family`` (client prototypes from the final model
and the clients' labelled subsets; PRAA reliability evidence from the validation data).  Needs the dataset
(PathMNIST download, or PRAA_SYNTHETIC_DATA=1 for a synthetic smoke run).
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from evaluation.inference import embed_images  # noqa: E402
from federated.experiment import World, client_prototypes_with_model  # noqa: E402
from federated.experiment_config import validate_config  # noqa: E402
from federated.praa import PRAAConfig, aggregate_prototypes, class_multipliers, compute_record_prs  # noqa: E402
from federated.reliability import ReliabilityTracker  # noqa: E402
from models.classifier import FLAMEModel, model_kwargs_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rule", choices=("auto", "flame", "praa"), default="auto",
                        help="auto = the rule of the run's method (praa -> praa, flame -> flame)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    validate_config(config)
    if config["method"] == "fedavg":
        sys.exit("FedAvg runs classify with the classifier head and need no prototypes.")
    rule = config["method"] if args.rule == "auto" else args.rule

    state = torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=False)
    model = FLAMEModel(**model_kwargs_from_config(config))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(args.device).eval()

    with contextlib.redirect_stdout(io.StringIO()):
        world = World(config, args.device)
    protos, counts = client_prototypes_with_model(model, world.label_cache, args.device)
    multipliers = None
    if rule == "praa":
        praa_cfg = PRAAConfig.from_dict(config["praa"])
        tracker = ReliabilityTracker(run_dir, praa_cfg.stability_window)
        tracker.load_from_disk()
        rel_emb = embed_images(model, world.val_images, device=args.device)
        rows = tracker.assess_global_prototypes(int(config["num_rounds"]), protos, counts, rel_emb, world.val_labels)
        multipliers, _ = class_multipliers(compute_record_prs(rows, praa_cfg), praa_cfg)
    prototypes, weights = aggregate_prototypes(protos, counts, multipliers)
    out = run_dir / "global_prototypes.pt"
    torch.save({"prototypes": prototypes, "rule": rule,
                "num_clients_per_class": {c: len(w) for c, w in weights.items()}}, out)
    print(f"wrote {out}  ({len(prototypes)} classes, rule={rule})")


if __name__ == "__main__":
    main()
