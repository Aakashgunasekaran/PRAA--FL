"""Train one method (fedavg | flame | praa) with the unified runner.

    python run_experiment.py --method praa  --output-dir results/runs/praa
    python run_experiment.py --method flame --output-dir results/runs/flame --resume
    python run_experiment.py --config configs/smoke_synthetic.json --method praa --output-dir /tmp/smoke

Phases 6-7 are exercised by ``--method praa``; ``--method flame`` is the MATCHED FLAME
baseline (same code and seeds, size-weighted aggregation).  The preserved original Phase 3
run is never re-executed by this script.  The test split is never read.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from federated.experiment import run_experiment  # noqa: E402
from federated.experiment_config import METHODS, load_config, parse_overrides  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default=None, help="JSON config (see configs/)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config key, e.g. --set praa.tau=2 --set num_clients=10")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.overrides)
    overrides["method"] = args.method
    if args.rounds is not None:
        overrides["num_rounds"] = args.rounds
    config = load_config(args.config, overrides)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    synthetic = os.environ.get("PRAA_SYNTHETIC_DATA") == "1"
    print(f"method={config['method']} rounds={config['num_rounds']} device={device} "
          f"encoder={config['encoder']} synthetic_data={synthetic}")
    if synthetic:
        print("WARNING: PRAA_SYNTHETIC_DATA=1 -> synthetic PathMNIST-shaped data; results are NOT PathMNIST.")
    run_experiment(config, args.output_dir, device=device, resume=args.resume, overwrite=args.overwrite)
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
