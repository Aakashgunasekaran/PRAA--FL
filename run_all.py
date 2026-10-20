"""Run the whole PRAA-FL pipeline (Phases 1-8) from one command.

    python run_all.py --profile smoke            # offline, synthetic data, tiny CNN, ~3 minutes on CPU
    python run_all.py --profile full --dry-run   # print the plan and the compute estimate only
    python run_all.py --profile full --final     # real PathMNIST on a GPU; evaluates TEST at the end

Profiles
--------
smoke  Sets PRAA_SYNTHETIC_DATA=1 (PathMNIST-SHAPED synthetic images, NOT PathMNIST), uses the
       tiny CNN and a few rounds.  It proves that every phase is wired together; its numbers say
       nothing about PathMNIST or the paper.
full   Real PathMNIST, ResNet-50, 50 clients, paper hyper-parameters.  Trains FedAvg (minutes),
       the MATCHED FLAME baseline and PRAA (hours each on a T4, resumable), evaluates the
       PRESERVED Phase 3 model if results/phase3/final_model.pt exists, then compares.
       Validation is evaluated first; the TEST split is read only with --final.

Phase map: 1 data (datasets/preprocess.py) | 2 partition (federated/client_data.py) |
3 FLAME training (federated/experiment.py, method flame; original run preserved) |
4 prototypes + 5 PRAM (federated/reliability.py, federated/pram.py, every round) |
6 PRS (federated/praa.py, run_phase6.py) | 7 PRAA (federated/praa.py, method praa) |
8 evaluation (run_phase8.py).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="override the number of rounds")
    parser.add_argument("--config", type=str, default=None, help="JSON config overriding the profile default")
    parser.add_argument("--final", action="store_true", help="also evaluate the TEST split (final evaluation)")
    parser.add_argument("--resume", action="store_true", help="resume interrupted runs in --output-dir")
    parser.add_argument("--skip-train", action="store_true", help="reuse finished runs in --output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase3-dir", type=Path, default=ROOT / "results" / "phase3")
    parser.add_argument("--phase4-dir", type=Path, default=ROOT / "results" / "phase4")
    parser.add_argument("--pram", type=Path, default=ROOT / "results" / "phase5" / "pram.json")
    args = parser.parse_args(argv)

    if args.profile == "smoke":
        os.environ["PRAA_SYNTHETIC_DATA"] = "1"
        config_path = args.config or str(ROOT / "configs" / "smoke_synthetic.json")
        out = args.output_dir or ROOT / "results" / "smoke_e2e"
    else:
        os.environ.setdefault("PRAA_SYNTHETIC_DATA", "0")
        config_path = args.config or str(ROOT / "configs" / "paper_full.json")
        out = args.output_dir or ROOT / "results" / "runs"
    synthetic = os.environ["PRAA_SYNTHETIC_DATA"] == "1"

    import torch
    from federated.experiment import run_experiment
    from federated.experiment_config import load_config
    from run_phase6 import build_prs_document
    from run_phase8 import (build_comparison, evaluate_original, evaluate_run, write_comparison)
    from federated.pram import load_pram_output
    from federated.praa import PRAAConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = {"num_rounds": args.rounds} if args.rounds else {}
    configs = {m: load_config(config_path, {**overrides, "method": m}) for m in ("fedavg", "flame", "praa")}
    banner(f"PRAA-FL pipeline | profile={args.profile} | device={device} | synthetic_data={synthetic}")
    if synthetic:
        print("NOTE: synthetic PathMNIST-shaped data. Results below are NOT PathMNIST results.")
    c = configs["praa"]
    print(f"clients={c['num_clients']} per_round={c['clients_per_round']} rounds={c['num_rounds']} "
          f"encoder={c['encoder']} local_epochs={c['local_epochs']} tau={c['praa']['tau']} "
          f"policy={c['praa']['missing_component_policy']}")
    if args.profile == "full" and device == "cpu":
        print("WARNING: no GPU found. A full run needs a GPU (see estimate_compute.py: ~258 h on a 2-core CPU).")
    if args.dry_run:
        from estimate_compute import main as estimate
        estimate(["--rounds", str(c["num_rounds"]), "--metrics", str(args.phase3_dir / "metrics.json")])
        print("\n--dry-run: nothing trained.")
        return 0

    banner("Phases 1-2: data and client partition")
    from datasets import preprocess  # noqa: F401  (prints the Phase 1 summary)
    from federated.client_data import build_partition
    from datasets.partition import verify_partition
    part = build_partition(c["num_clients"], c["dirichlet_beta"], c["seed"], c["min_samples_per_client"])
    checks = verify_partition(part, len(preprocess.train_dataset))
    print("partition checks:", checks)
    assert all(checks.values()), "partition verification failed"

    runs = {}
    for method in ("fedavg", "flame", "praa"):
        run_dir = out / method
        runs[method] = run_dir
        if args.skip_train and (run_dir / "final_model.pt").exists():
            print(f"[skip] {method}: finished run found")
            continue
        banner(f"Phases 3-7: training {method.upper()}")
        t0 = time.time()
        run_experiment(configs[method], run_dir, device=device, resume=args.resume)
        print(f"{method} finished in {(time.time() - t0) / 60:.1f} min")

    banner("Phase 6: PRS of the preserved Phase 3 evidence (read-only)")
    if args.pram.exists() and not synthetic:
        pram = load_pram_output(args.pram)
        for policy in ("strict", "renormalize"):
            doc = build_prs_document(pram, PRAAConfig(missing_component_policy=policy))
            print(f"policy={policy:11s} status={doc['status']} records_with_prs="
                  f"{doc['summary']['records_with_prs']}/{doc['summary']['records']} "
                  f"clients_assessed={doc['summary']['clients_assessed']}/{doc['summary']['clients']}")
    else:
        print("skipped (synthetic profile or no pram.json)")

    splits = ["val"] + (["test"] if args.final else [])
    for split in splits:
        banner(f"Phase 8: evaluation on the {split.upper()} split")
        evals, histories = {}, {}
        for method, run_dir in runs.items():
            ev = evaluate_run(run_dir, split, args.final, device)
            (run_dir / f"eval_{split}.json").write_text(json.dumps(ev, indent=2, allow_nan=False))
            evals[method.upper() if method != "praa" else "PRAA"] = ev
            histories[method.upper() if method != "praa" else "PRAA"] = json.loads(
                (run_dir / "metrics.json").read_text())
            print(f"{method:7s} {({k: (round(v, 4) if v is not None else None) for k, v in ev['primary'].items()})}")
        original = args.phase3_dir / "final_model.pt"
        if original.exists() and not synthetic:
            ev = evaluate_original(args.phase3_dir, args.phase4_dir, args.pram, split, args.final, device,
                                   PRAAConfig())
            (out / f"eval_original_{split}.json").write_text(json.dumps(ev, indent=2, allow_nan=False))
            evals["FLAME (original, preserved)"] = ev
            print("original", {k: round(v, 4) for k, v in ev["primary"].items() if v is not None})
        elif not synthetic:
            print(f"[info] {original} not found: the preserved FLAME model is evaluated when its "
                  f"final_model.pt is placed there (see docs/CHECKPOINTS.md).")
        doc = build_comparison(evals, histories, split)
        write_comparison(doc, out / f"phase8_{split}", histories)
        print((out / f"phase8_{split}" / "comparison.md").read_text())

    banner("PIPELINE COMPLETE" + (" (SYNTHETIC SMOKE - not PathMNIST)" if synthetic else ""))
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
