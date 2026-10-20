# PRAA-FL — Prototype Reliability-Aware Adaptive Aggregation for Federated Medical Image Classification

Base paper: *FLAME: Federated Learning With Masked Autoencoders and Mean-Prototypes Embedding for
Sparsely Labeled Medical Images* (Kundalwal & Mishra, IEEE TETCI 2025). Dataset: PathMNIST
(50 non-IID clients, Dirichlet β = 0.1, 10 clients/round, 1 % labels).

PRAA-FL extends FLAME's server step. Clients train exactly like FLAME (MAE + prototypical loss);
the server assesses how reliable each client's prototypes are (PRAM → PRS) and aggregates with
reliability-aware weights (PRAA) instead of sample counts alone.

## Status at a glance (read this first)

| Phase | What | State |
|---|---|---|
| 1 Data preprocessing | `datasets/preprocess.py` | done (original), unchanged except an opt-in synthetic switch for offline tests |
| 2 Federated client training setup | `datasets/partition.py`, `federated/client*.py` | done (original) |
| 3 Masked autoencoder (FLAME) | `federated/client.py`, `models/`, `training/` | done (original 200 rounds **preserved, not re-run**) |
| 4 Prototype generation | `training/prototype*.py`, `run_phase4.py` | done (original, 50 clients preserved in `results/phase4`) |
| 5 PRAM | `federated/pram.py`, `federated/reliability.py`, `run_phase5.py` | done; live per-round PRAM added |
| 6 PRS | `federated/praa.py`, `federated/prs.py`, `run_phase6.py` | done |
| 7 PRAA | `federated/praa.py`, `federated/experiment.py`, `run_phase7.py` | done and tested |
| 8 Evaluation | `evaluation/`, `run_phase8.py` | done and tested |

**What has and has not been run.** The sandbox that produced this ZIP had no GPU, no PathMNIST
download access and no copy of the 199 MB Phase 3 model. Therefore:

* Run and verified here: the 171-test suite, a complete offline pipeline on synthetic
  PathMNIST-shaped data (`run_all.py --profile smoke`), a two-round smoke run with the real
  ResNet-50 architecture, Phase 6 on your real preserved PRAM evidence, a compute estimate.
* **Not run here (needs your GPU / data):** any real PathMNIST training or evaluation, hence **no
  accuracy / macro-F1 / AUROC / convergence numbers for FedAvg, FLAME or PRAA on PathMNIST exist yet,
  and none are claimed.** `docs/EXPERIMENTS.md` has the exact commands and a compute estimate
  (≈ 7–14 h per FLAME/PRAA run on a T4, estimated).
* No claim is made that PRAA beats the paper's 81.8 %. That is decided by the experiments only.

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                       # 171 tests, offline (synthetic data), ~3 min on CPU
python run_all.py --profile smoke --final       # whole pipeline offline in ~1-2 min (SYNTHETIC data)
python run_all.py --profile full --dry-run      # plan + compute estimate for the real experiment
```

## Web app: upload an image and classify it

A FastAPI backend (`app/server.py`) serves a single-page frontend (`app/static/`) from the same address,
so there is nothing to build and no CORS to configure.

```bash
pip install -r requirements.txt -r requirements-app.txt
python run_app.py                  # open http://127.0.0.1:8000   (API docs: /docs)
```

The app looks for a model in this order: `PRAA_CHECKPOINT` env var, `results/runs/{praa,flame,fedavg}/final_model.pt`,
then `results/phase3/final_model.pt` (your preserved 200-round FLAME model, see `docs/CHECKPOINTS.md`).
FLAME/PRAA models classify by the nearest global prototype (paper Eq. 6/10); a FedAvg model uses its classifier head.

| Checkpoint | What else it needs |
|---|---|
| `results/phase3/final_model.pt` | nothing: global prototypes are aggregated from `results/phase4/prototypes/` |
| a run from `run_experiment.py` / `run_phase7.py` | `python scripts/export_global_prototypes.py --run-dir results/runs/praa` once (needs the dataset) |
| a FedAvg run | nothing |

**No checkpoint present: demo mode.** The app still starts so you can try the upload flow, but it uses an untrained
random model, shows a banner, and returns `"demo": true`. Its predictions mean nothing.

API: `GET /api/health`, `GET /api/classes`, `POST /api/predict` (multipart field `file`; PNG/JPG/BMP/TIFF/WEBP, max 10 MB):

```bash
curl -F "file=@patch.png" http://127.0.0.1:8000/api/predict
```

Images are centre-cropped to a square and resized to 28 x 28, then normalised exactly like `datasets/preprocess.py`.
The model was trained on PathMNIST (H&E colorectal tissue patches), so other kinds of photos give meaningless output.
This is a research prototype, not a medical device. Tests: `pytest tests/test_app.py`.

Colab / GPU, real data (resumable, validation first, test only with `--final`):

```bash
python run_experiment.py --method fedavg --output-dir results/runs/fedavg
python run_experiment.py --method flame  --output-dir results/runs/flame   # matched FLAME baseline
python run_phase7.py                     --output-dir results/runs/praa    # PRAA   (add --resume after a disconnect)
python run_phase8.py evaluate --run-dir results/runs/praa --split val      # development numbers
python run_phase8.py evaluate-original --split val                         # preserved Phase 3 model (needs final_model.pt)
python run_phase8.py compare --run FedAvg=results/runs/fedavg --run FLAME=results/runs/flame --run PRAA=results/runs/praa
python run_phase8.py evaluate --run-dir results/runs/praa --split test --final   # ONCE, at the very end
```

Always run from the project root: the local `datasets/` package shadows HuggingFace `datasets`
(`conftest.py` and every `run_*.py` put the project root first on `sys.path`).

## Layout

```
datasets/      Phase 1-2 data + Dirichlet partition (+ synthetic.py for offline tests)
models/        FLAME model (ResNet-50 encoder, decoder, projection head, classifier), patch masker
training/      losses, episodes, prototype extraction/store, validation
federated/     client.py (FLAME client), praa.py (PRS weights + aggregation), pram.py, reliability.py,
               experiment.py (FedAvg/FLAME/PRAA loop), fedavg_baseline.py, flame.py (FLAME aggregation, kept)
evaluation/    inference rules (classifier head / nearest global prototype) and metrics
app/          FastAPI backend (server.py, inference_service.py) + static frontend; started with run_app.py
run_phase{1..8}.py, run_experiment.py, run_all.py, estimate_compute.py, audit_phase{4,5}.py
configs/       paper_full.json, pilot_continue_from_phase3.json, smoke_synthetic.json
results/       preserved Phase 3-5 evidence (+ manifest), Phase 6 outputs, compute estimate, smoke run
docs/          METHOD.md (formulas), EXPERIMENTS.md, CHECKPOINTS.md, LIMITATIONS.md, per-phase notes
tests/         unit, integration and end-to-end tests
```

## Documentation

* `docs/METHOD.md` — all eight phases and every formula
* `docs/EXPERIMENTS.md` — settings, commands, compute estimate, actual measured results, ablations
* `docs/CHECKPOINTS.md` — where the large original checkpoints are and how to verify them
* `docs/LIMITATIONS.md` — what is incomplete, assumptions, findings about the preserved run
