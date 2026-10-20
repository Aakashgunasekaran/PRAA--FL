# Limitations, assumptions and findings

## What is not done
* **No real PathMNIST training or evaluation was run for Phases 6-8.** The build sandbox had no GPU,
  could not download PathMNIST (dataset host blocked) and had no copy of the Phase 3 model. The code
  paths are tested on synthetic PathMNIST-shaped data, with a 2-round ResNet-50 smoke run, and on the
  real preserved Phase 4/5 files (see `tests/test_evaluate_original.py`, which uses a randomly
  initialised stand-in model, so its metric values are meaningless).
* Consequently there are **no measured accuracy / macro-F1 / AUROC / convergence results on PathMNIST**
  for FedAvg, FLAME (matched), the preserved FLAME model or PRAA, and no statement about beating the
  paper's 81.8 % can be made. `docs/EXPERIMENTS.md` lists what to run.
* Not tested on a GPU (no CUDA available); device handling follows the original code (`cuda` if available).

## Findings about the preserved Phase 3 run (from your files)
1. **Its validation signal is not informative.** `run_phase3.py` validated with the *classifier head*,
   which FLAME's loss never trains, on the first **64** validation images (all accuracies are multiples
   of 1/64). Reported validation accuracy stays 0.14-0.25 (chance 0.11) for all 200 rounds; it says
   nothing about FLAME's prototype-based quality. Phase 8 therefore evaluates FLAME with the paper's
   inference rule (nearest global prototype).
2. **The LR scheduler was driven by that signal.** ReduceLROnPlateau saw a flat loss (≈ 2.2) and cut the
   clients' learning rate from 1e-3 down to ≈ 1e-8 (reported rate in `metrics.json`: 1e-5 by round ~100,
   1e-7 to 1e-8 by round ~170-200). The preserved model was therefore probably under-trained, and any
   PRAA-vs-"original FLAME" gain would mix the scheduler repair with the aggregation change. New runs use
   `scheduler_signal = method_val_loss` (the method's own validation NLL on a 2 000-image subset) for FedAvg,
   FLAME and PRAA alike; `classifier_head` reproduces the old behaviour. **The matched FLAME run, not the
   original model, is the fair baseline for PRAA.**
3. **The prototypical loss is mostly trivial.** Median local proto loss is 0.0000 in rounds 11-200
   (mean 0.02-0.06, max 2.2): with 1 % labels (881 labelled images in total) and Dirichlet β = 0.1 each client
   holds 1-4 classes, so within-client episodes carry little cross-class signal. This is inherent to the
   specified setup, not changed here.
4. **Phase 5 evidence of the preserved prototypes.** Validation recall (115 valid records) averages 0.579
   vs 0.339 chance for the candidate sets (sample-weighted 0.637; 21 records below chance, 10 ≥ 0.9).
   Consistency saturates (mean 0.998, min 0.963) because class prototypes are nearly collinear
   (mean between-class cosine 0.974), so consistency discriminates little; validation recall carries the signal.
5. **Stability does not exist for the preserved run** (no client-local snapshots), so strict PRS is null for
   all 126 records and post-hoc PRAA equals FLAME. PRAA therefore needs a new training run that records snapshots.

## Assumptions / decisions (all configurable, none from the paper)
* Client reliability R_r = sample-weighted mean of PRS; relative multiplier m_r = R_r^τ / mean over assessed;
  floor 0.05, cap 5, τ = 1, `min_assessed` = 2; unassessed clients keep their FLAME weight.
* Stability window = last 3 snapshots; live consistency reference = this round's participants only.
* Live validation component uses the local model's embeddings of a 1 000-image validation prefix; the validation
  labels therefore influence aggregation weights (by design of PRAM). The test split never does.
* Per-round validation is a seeded 2 000-image subset; the final development evaluation uses all 10 004 images.
* FedAvg = supervised cross-entropy on the same labelled subsets (project definition; the paper has no FedAvg row).
* `renormalize` PRS is an extension of the documented formula, off by default, always labelled.
* The reliability multiplier is bounded and near 1 when PRS values are compressed (e.g. 0.74-1.22 on the preserved
  evidence with the extension on); with τ = 1 the effect on accuracy may be small. τ is tuned on validation only.

## Comparability with the paper
The paper reserves its own 20 % test split and averages 3 runs. This project uses the official MedMNIST
split (test = 7 180 images, a different centre than train/val) and one seed. Differences to 81.8 % / 79.1 % / 0.79
are indicative at best. Report results with the seed and rounds used; repeat with several seeds
(`--set seed=...`) before drawing conclusions.

## Other notes
* The local package `datasets/` shadows HuggingFace `datasets`; always run from the project root.
* Resume is bit-exact (loader RNG is checkpointed) for runs made with this code; the original Phase 3
  resume logic did not restore loader RNG and is untouched.
* `legacy/run_phase5.py.pre_pram_backup` is the superseded earlier Phase 5 runner, kept for reference only.
