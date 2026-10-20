# Large original checkpoints (not inside the ZIP)

The original Phase 3 artefacts are ~199 MB each and stay on the author's Google Drive. The ZIP
carries only their small evidence files (config, 200-round metrics history, Phase 4 prototypes,
Phase 5 `pram.json`) plus a SHA-256 manifest.

| Artefact | Location (as used in the Colab notebook) | Size | SHA-256 |
|---|---|---|---|
| `final_model.pt` (keys: `model_state_dict`, `history`) | `MyDrive/phase3/final_model.pt` | 199.2 MB | `cafc30c356db9f41fbf21674e040e526073d0a7645b5a5b9666fd4fe4869d2fd` |
| `checkpoint_round_200.pt` (weights identical to `final_model.pt`; also client LR/scheduler state) | `MyDrive/phase3/checkpoints/` | 199.18 MB | — |
| `checkpoint_round_199.pt` | `MyDrive/phase3/checkpoints/` | 199.18 MB | — |
| Phase 4 prototypes (also copied into `results/phase4/`) | `MyDrive/phase4/` | 1.2 MB | per-file hashes in `results/phase5/pram.json` |

The hash and the "round 200 / weights identical" cross-check come from the Phase 4/5 provenance
records that were produced from the real checkpoint.

## Using them

```bash
# 1. put the model next to the preserved evidence (copy, do not move)
cp /content/drive/MyDrive/phase3/final_model.pt results/phase3/final_model.pt
# 2. verify it is the original
python scripts/verify_preserved.py --phase3-model results/phase3/final_model.pt
# 3. evaluate the preserved FLAME model with the paper's prototype inference rule
python run_phase8.py evaluate-original --split val
# 4. warm-start a controlled FLAME-vs-PRAA continuation from it (optional)
python run_experiment.py --method flame --config configs/pilot_continue_from_phase3.json --output-dir results/runs/flame_pilot
python run_phase7.py                  --config configs/pilot_continue_from_phase3.json --output-dir results/runs/praa_pilot
```

`results/phase3/final_model.pt` is git-ignored. Nothing in this repository writes to the original
Phase 3 directory; the preserved small files are protected by `results/PRESERVED_MANIFEST.json`
(checked by `scripts/verify_preserved.py` and by `tests/test_preserved_artifacts.py`).
