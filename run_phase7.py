"""Phase 7: PRAA training (Prototype Reliability-Aware Adaptive Aggregation).

Thin entry point over ``federated.experiment`` with ``method=praa``: every round the selected
clients train locally (Phase 3 FLAME client, unchanged), their local prototypes and validation
embeddings are assessed by PRAM (Phase 5), turned into PRS and client multipliers (Phase 6),
and the server aggregates with  w_r ~ nu_r * m_r  (Phase 7).  FLAME stays a separate baseline:
``python run_experiment.py --method flame ...`` runs the matched size-weighted baseline.

    python run_phase7.py --output-dir results/runs/praa                      # full 200 rounds
    python run_phase7.py --output-dir results/runs/praa --resume             # after an interruption
    python run_phase7.py --config configs/pilot_continue_from_phase3.json --output-dir results/runs/praa_pilot
    python run_phase7.py --smoke --output-dir /tmp/praa_smoke                # synthetic, offline
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--smoke" in argv:
        argv.remove("--smoke")
        os.environ["PRAA_SYNTHETIC_DATA"] = "1"
        if "--config" not in argv:
            argv += ["--config", str(ROOT / "configs" / "smoke_synthetic.json")]
    from run_experiment import main as run_main
    run_main(["--method", "praa", *argv])


if __name__ == "__main__":
    main()
