"""Estimate compute for FedAvg / FLAME / PRAA runs BEFORE launching them.

Measures, on the machine it runs on, the FLOPs and wall time of one real FLAME local step
(64 masked unlabelled images through encoder+decoder, plus a <=64-image labelled
support/query episode through encoder+projection, forward and backward) and of one
evaluation forward pass, then combines them with the number of local steps.

Local steps per run: the exact total of the preserved Phase 3 history
(results/phase3/metrics.json, ``steps`` of every client every round) when present - PRAA and
the matched FLAME run use the same partition and the same client-selection seeds, so the
count carries over for the same number of rounds - otherwise ``--steps-per-round``.

The T4 projection is an ESTIMATE from FLOPs, not a measurement: time = FLOPs / (peak * util)
with the T4 fp32 peak (8.1 TFLOPS) and a utilisation range (default 0.25-0.50).  On the real
GPU (Colab) read ``measured_on_this_device.flame_run_hours_if_this_device`` instead: it is
computed from the step time actually measured there.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from models.classifier import FLAMEModel  # noqa: E402
from models.masking import RandomPatchMasker  # noqa: E402
from training.losses import compute_mae_loss  # noqa: E402

T4_PEAK_FP32 = 8.1e12


def one_step(model, masker, optimizer, device, batch=64, labeled=32):
    x = torch.randn(batch, 3, 28, 28, device=device)
    masked, _ = masker(x)
    recon = model.reconstruct(model.encoder(masked))
    loss = compute_mae_loss(recon, x)
    xs = torch.randn(labeled, 3, 28, 28, device=device)
    _, hs = model(xs)
    _, hq = model(xs)
    loss = loss + 0.01 * (hs.mean() + hq.mean())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def measure(device: str, steps: int = 3):
    model = FLAMEModel().to(device).train()
    masker = RandomPatchMasker(4, 0.8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    one_step(model, masker, opt, device)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one_step(model, masker, opt, device)
    if device == "cuda":
        torch.cuda.synchronize()
    step_time = (time.time() - t0) / steps
    with FlopCounterMode(display=False) as counter:
        one_step(model, masker, opt, device)
    step_flops = counter.get_total_flops()
    model.eval()
    imgs = torch.randn(256, 3, 28, 28, device=device)
    with torch.no_grad():
        model(imgs[:8])
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        model(imgs)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd_time_per_img = (time.time() - t0) / 256
        with FlopCounterMode(display=False) as fc:
            model(imgs)
    return {"step_time_s": step_time, "step_flops": step_flops,
            "fwd_time_per_image_s": fwd_time_per_img, "fwd_flops_per_image": fc.get_total_flops() / 256}


def local_steps(metrics_path: Path, rounds: int, steps_per_round: float):
    if metrics_path.exists():
        history = json.loads(metrics_path.read_text())
        per_round = [sum(c["steps"] for c in r["client_metrics"]) for r in history]
        if rounds <= len(per_round):
            return sum(per_round[:rounds]), f"exact, from {metrics_path} (first {rounds} rounds)"
        mean = sum(per_round) / len(per_round)
        return sum(per_round) + mean * (rounds - len(per_round)), "history + mean extrapolation"
    return steps_per_round * rounds, "assumed --steps-per-round"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "phase3" / "metrics.json")
    parser.add_argument("--steps-per-round", type=float, default=580.0)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--pram-val-samples", type=int, default=1000)
    parser.add_argument("--labeled-total", type=int, default=900)
    parser.add_argument("--util", type=float, nargs=2, default=[0.25, 0.50])
    parser.add_argument("--benchmark", action="store_true",
                        help="(informational) the step time of THIS device is always measured and reported "
                             "under measured_on_this_device; this flag only prints a reminder")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = measure(device)
    steps, steps_note = local_steps(args.metrics, args.rounds, args.steps_per_round)
    eval_imgs_per_round = (args.val_samples + args.labeled_total)           # global model eval
    praa_imgs_per_round = args.clients_per_round * (args.pram_val_samples + 50)  # local PRAM evidence
    eval_flops = eval_imgs_per_round * m["fwd_flops_per_image"] * args.rounds
    pram_flops = praa_imgs_per_round * m["fwd_flops_per_image"] * args.rounds
    train_flops = steps * m["step_flops"]
    def hours(flops, util): return flops / (T4_PEAK_FP32 * util) / 3600

    report = {
        "device_measured": device,
        "rounds": args.rounds, "local_steps": steps, "local_steps_source": steps_note,
        "measured_on_this_device": {
            "step_time_s": m["step_time_s"], "step_gflops": m["step_flops"] / 1e9,
            "fwd_ms_per_image": m["fwd_time_per_image_s"] * 1e3,
            "flame_run_hours_if_this_device": (steps * m["step_time_s"] + args.rounds * (
                eval_imgs_per_round + praa_imgs_per_round) * m["fwd_time_per_image_s"]) / 3600},
        "t4_projection_hours_ESTIMATE": {
            f"util_{u}": {"training": hours(train_flops, u), "evaluation_overhead_flame_family":
                          hours(eval_flops, u), "praa_extra_local_pram": hours(pram_flops, u),
                          "total_flame_run": hours(train_flops + eval_flops, u),
                          "total_praa_run": hours(train_flops + eval_flops + pram_flops, u)}
            for u in args.util},
        "fedavg_note": "supervised on ~900 labelled images: negligible (minutes) compared with FLAME/PRAA",
        "caveat": "T4 numbers are FLOP-based projections, not measurements; Colab sessions may be "
                  "interrupted, use --resume (checkpoints every 5 rounds).",
    }
    if args.benchmark:
        print(f"[benchmark] measured on {device}: {m['step_time_s']:.3f} s/step -> "
              f"{report['measured_on_this_device']['flame_run_hours_if_this_device']:.1f} h for the run")
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
