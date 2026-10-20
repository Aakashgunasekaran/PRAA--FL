# Experiments: settings, commands, compute estimate and actual results

## 1. What was actually run (measured in the build sandbox)

Sandbox: 2 CPU cores, 7 GB RAM, no GPU, no PathMNIST download, no Phase 3 model.

| Run | Data | Model | Outcome |
|---|---|---|---|
| `pytest -q tests` | synthetic / real preserved files | tiny CNN, one ResNet-50 stand-in | all tests pass (see `README.md` for the count) |
| `run_all.py --profile smoke --final` | **synthetic** PathMNIST-shaped (3 000/500/500) | tiny CNN, 10 clients, 5/round, 6 rounds | pipeline completes in ≈ 70 s; outputs in `results/smoke_e2e/` |
| 2-round PRAA smoke with the real architecture | synthetic | ResNet-50 (52.09 M params), 2 clients/round, 1 step | ≈ 20 s/round on 2 CPU cores; model file 200 MB, same size as the preserved one |
| `run_phase6.py` on `results/phase5/pram.json` | **real preserved evidence** | — | see §2 |
| `estimate_compute.py` | real Phase 3 step counts | ResNet-50 | see §3 |

Synthetic smoke numbers (validation / test split of the *synthetic* data, 6 rounds) are shown only to
prove the plumbing works. They are **not PathMNIST results** and must not be compared with the paper:

| Method | Val acc | Test acc (synthetic) |
|---|---|---|
| FedAvg (supervised, 5 % labels, 6 rounds) | 0.110 | 0.110 |
| FLAME (matched) | 0.930 | 0.938 |
| PRAA | 0.932 | 0.940 |

(PRAA ≈ FLAME here because the synthetic evidence is nearly uniform; FedAvg barely moves in 6 tiny rounds.
A separate 40-round, 5-epoch check showed the FedAvg baseline does learn: validation accuracy 0.22 → 0.81.)

## 2. Phase 6 on the real preserved evidence (measured)

Input: `results/phase5/pram.json` (126 (client, class) records from the original 200-round model).

| Policy | PRS computed | Clients assessed | Outcome |
|---|---|---|---|
| strict (documented rule) | **0 / 126** | 0 / 50 | stability missing for every record → every client keeps its FLAME weight (`results/phase6/prs.json`, status `NO_PRS_COMPUTABLE`) |
| `renormalize` (extension, exploratory) | 115 / 126 | 39 / 50 | client reliability R_r 0.594-0.989 (mean 0.807); multipliers 0.736-1.225; largest client-weight change 0.0093 (`results/phase6_partial_exploratory/prs.json`) |

Meaning: PRAA on the preserved model cannot differ from FLAME under the documented rule; real PRS values
require the new run, which records client-local snapshots.

## 3. Compute estimate (before any full run)

Measured on this 2-core CPU with the real model (`results/compute_estimate.json`): one FLAME local step
(64 masked unlabelled + ≤ 64 labelled images, fwd+bwd) = **835 GFLOP, 7.6 s**; one forward pass 15.8 ms/image.
The preserved history has **116 064 local steps** over its 200 rounds (580/round); PRAA and the matched FLAME run
use the same partition and selection seeds, so they have the same count.

| Run | This CPU | T4 (ESTIMATE = FLOPs / (8.1 TFLOPS × 25-50 % utilisation)) |
|---|---|---|
| FLAME (matched), 200 rounds | ≈ 258 h (infeasible) | ≈ 6.7 – 13.5 h |
| PRAA, 200 rounds | — | ≈ 7.0 – 14.1 h (+0.3-0.6 h local PRAM) |
| Pilot: warm start from the preserved model, 30 rounds, per arm | — | ≈ 1 – 2 h |
| FedAvg (≈ 881 labelled images) | minutes | minutes |

The T4 figures are projections, not measurements. Replace them with `python estimate_compute.py --benchmark` on
the GPU. Colab disconnects are handled by `--resume` (checkpoint every 5 rounds, bit-exact resume).

## 4. Experiment plan (run on a GPU; development on validation only)

Settings (paper / Phase 3 unless noted): 50 clients, 10 per round, Dirichlet β = 0.1, min 10 samples, ResNet-50 encoder,
projection 128-d, MAE ratio 0.8, α = 0.7, Adam 1e-3, batch 64, 2 local epochs, ReduceLROnPlateau(0.1, 5), 1 % labels,
seed 42. Additions: per-round validation on a seeded 2 000-image subset (also drives the scheduler),
PRAA τ = 1, floor 0.05, cap 5, strict PRS, stability window 3 (`configs/paper_full.json`).

```bash
# 0. estimate on your GPU
python estimate_compute.py --benchmark
# 1. cheap controlled pilot from the PRESERVED model (30 rounds, selection seeds continue at 201)
python run_experiment.py --method flame --config configs/pilot_continue_from_phase3.json --output-dir results/runs/flame_pilot
python run_phase7.py                  --config configs/pilot_continue_from_phase3.json --output-dir results/runs/praa_pilot
python run_phase8.py evaluate --run-dir results/runs/flame_pilot --split val
python run_phase8.py evaluate --run-dir results/runs/praa_pilot  --split val
# 2. tune tau on VALIDATION only (frozen run; also try --missing-component-policy renormalize, labelled extension)
python run_phase8.py tune-tau --run-dir results/runs/praa_pilot --taus 0 0.5 1 2 4 8
# 3. full 200-round runs (resumable)
python run_experiment.py --method fedavg --output-dir results/runs/fedavg
python run_experiment.py --method flame  --output-dir results/runs/flame
python run_phase7.py --set praa.tau=<chosen on validation> --output-dir results/runs/praa
# 4. evaluation: validation first, test exactly once
for m in fedavg flame praa; do python run_phase8.py evaluate --run-dir results/runs/$m --split val; done
python run_phase8.py evaluate-original --split val --output results/phase8/eval_original_val.json
python run_phase8.py compare --run FedAvg=results/runs/fedavg --run FLAME=results/runs/flame --run PRAA=results/runs/praa \
       --original results/phase8/eval_original_val.json --split val
for m in fedavg flame praa; do python run_phase8.py evaluate --run-dir results/runs/$m --split test --final; done
python run_phase8.py evaluate-original --split test --final --output results/phase8/eval_original_test.json
python run_phase8.py compare --run FedAvg=results/runs/fedavg --run FLAME=results/runs/flame --run PRAA=results/runs/praa \
       --original results/phase8/eval_original_test.json --split test --output-dir results/phase8_test
```
`python run_all.py --profile full --final` chains steps 3-4 (without tuning).

Decision rules: choose τ, `client_reduce`, `apply_to_*` and any round budget on validation only; freeze them; then run
the test evaluation once per method and report it with the seed. Use ≥ 3 seeds (`--set seed=43` etc., new output dirs)
before claiming a difference between PRAA and FLAME.

## 5. Improvement levers investigated (none validated yet on real data)

| Lever | Status | Why |
|---|---|---|
| Valid validation signal for LR scheduling (`scheduler_signal=method_val_loss`) | implemented, default in new runs | preserved run's LR collapsed to ~1e-8 (see `LIMITATIONS.md` #2) |
| Paper's prototype inference rule for evaluating FLAME | implemented | classifier head is untrained in FLAME |
| Reliability-aware parameter aggregation (PRAA) | implemented | Phase 7 |
| Reliability-aware prototype aggregation | implemented | reduces noisy prototypes in the inference prototypes |
| τ / client-reduction / policy sweep on validation | tool provided (`tune-tau`) | only τ = 1 is the documented formula |
| Warm-start pilot instead of 200 more rounds | implemented | cuts the cost to ~1-2 h per arm |
| More rounds, other seeds | via `--rounds`, `--set seed=` | measure variance before concluding |
| Changing the FLAME objective (e.g. cross-client prototype alignment) | **not implemented** | outside the paper's method; would be an unrelated algorithm |

No improvement is guaranteed; any gain over the paper's 81.8 % must come out of the experiments above.
