# Phase 4: Prototype Generation (global-model prototypes)

## What Phase 4 produces

Phase 3 saved only the final global model (`final_model.pt`: `model_state_dict` +
`history`) and `config.json`. It did not save per-client local models or per-round
client snapshots. Phase 4 therefore applies the saved global model to each simulated
client's Phase 2 partition and stores class-wise mean embeddings.

These are **prototypes computed by the global model on each client's partition data**.
They are not representations from client-locally trained models. Local model states and
temporal stability are recorded as `unavailable`; nothing is synthesized.

The run is labelled `GLOBAL_MODEL_PROTOTYPES_NOT_FINAL` (mode `global_model_prototypes`)
and `final_experiment_ready` is `false`.

## Representation

```text
PathMNIST image -> shared ResNet-50 encoder -> FLAME projection head
                -> 128-dimensional projected embedding   (model.proto(model.encoder(x)))
```

No separate pretrained MAE encoder exists; the shared encoder is the MAE-trained one.
The model runs in `eval()` mode under `torch.no_grad()`.

## Paths (Colab defaults)

| item | path |
|---|---|
| Phase 3 input | `/content/drive/MyDrive/phase3/` (`final_model.pt`, `config.json`) |
| Phase 4 output | `/content/drive/MyDrive/phase4/` (`prototypes/client_<id>.pt`, `summary.json`) |
| Smoke output | `/content/phase4_smoke/` (separate, never mixed with full output) |

## Checkpoint validation (before anything is computed)

Config equals the fixed project config; history is 200 consecutive rounds; weights are
finite; the state dict is compared with a fresh `FLAMEModel` (missing / unexpected keys,
shape mismatches) and loaded with `strict=True`; the report is stored in
`summary.json["state_dict_check"]` together with the checkpoint's top-level keys.

## Which samples contribute (`--label-policy`)

* `labeled_subset` (default, the project's 1% policy): the client's labeled samples,
  reconstructed exactly as `federated.client.Client` selects them (`seed + client_id`,
  `max(2, int(n * 0.01))` samples).
* `all_labels`: every sample in the client's partition.

## Saved format

One file per client, loadable with `weights_only=True`: `schema_version`, `client_id`,
`class_ids`, `prototype_vectors [K, 128]`, `sample_counts`, `absent_class_ids`,
`label_policy`, `provenance`, `config`, plus the legacy `prototypes` / `metadata` view.
`provenance` includes the source model file, SHA-256, round, partition hash,
`prototype_origin = global_model_applied_to_client_partition`,
`client_local_model_states = unavailable`, `temporal_stability_history = unavailable`.
Absent classes get no prototype and are listed in `absent_class_ids`.

## Running

```text
python run_phase4.py --smoke      # 5 clients, <= 64 samples  -> /content/phase4_smoke
python run_phase4.py              # all 50 clients            -> /content/drive/MyDrive/phase4
```

All records are computed first and written afterwards; `summary.json` is written last,
so a directory without it is an incomplete run. If the destination already holds
results the run stops before loading data and writes nothing; `--overwrite` replaces
them (and still refuses to mix files from clients outside the run). `--temporal` is
rejected: no client history exists.

## Audit

`python audit_phase4.py` (read-only; Colab defaults) checks paths, hashes, strict state
dict compatibility, 9 classes / 128-d shapes, status labels, provenance wording, and that
the run is not reported as final or as client-local.

## Downstream note

Phase 5 final mode and the Phase 6 numeric-PRS gate currently require a Phase 4 run in
mode `final`. Global-model prototypes are intentionally not that, so Phase 5 `--smoke`
(512 validation samples) is the only runnable Phase 5 mode until that policy is revisited.
