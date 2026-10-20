# Phase 5: Prototype Reliability Assessment Module (PRAM)

PRAM measures three signals per (client, class) from the frozen Phase 3 model and
the Phase 4 prototype files, both read-only. It does not compute PRS, change
aggregation, or touch the test split. Implementation: `federated/pram.py`;
runner: `run_phase5.py`; output: `results/phase5/pram.json`.

## Evidence states

Every component is `{status, value, raw, reason, details}`.

| status | value | meaning |
|---|---|---|
| `valid` | normalized score in [0, 1] | computed from sufficient evidence |
| `degenerate` | null (`raw` holds the number) | computable but uninformative by construction; Phase 6 must not use it |
| `unavailable` | null, raw null | required evidence is missing; `reason` says why |

A low score is `valid` with a small value. A missing score is never 0, 1, or a
neutral constant.

## Components (formula version `pram/1`)

Definitions follow `docs/PHASE_5_PROTOTYPE_RELIABILITY_SCORE.md`. They are PRAA-FL
project formulations, not claims of the FLAME paper.

* **Validation.** Per-class recall of nearest-prototype classification
  (squared Euclidean, ties to the lowest class id) on the shared PathMNIST
  validation split, using the global model's embeddings and the client's own
  prototypes as candidates. No client-specific validation data exists, so this
  scores a client's *prototypes in the global embedding space*, not a
  client-trained model. `degenerate` when the client holds one class (recall is
  1.0 by construction); `unavailable` when the validation split has no sample of
  the class. Recorded: `num_eval_samples`, `num_candidate_classes`, `chance_level`.
* **Consistency.** `(cos(client prototype, reference) + 1) / 2`, where the
  reference is the unweighted mean of all clients' prototypes of the same class
  (self-inclusive, as documented). `degenerate` when only one client holds the
  class (the reference is then the client itself). A leave-one-out value is stored
  as a diagnostic only. Dimension mismatches and non-finite or malformed vectors
  are rejected; zero-norm vectors are `unavailable`.
* **Stability.** Mean over consecutive snapshots of `(cos + 1) / 2` for the same
  client and class. Requires at least two snapshots under
  `<phase4>/snapshots/round_<r>/client_<id>.pt` whose `client_id` and `round`
  match their location and whose `provenance.source_kind == "client_local_model"`.
  Anything else is rejected and counted in `evidence`. Non-participation is not
  instability: snapshots exist only for rounds the client trained, and
  `round_gaps` is recorded.

## Training-history availability

The Phase 3 history stores scalar per-round losses for participating clients only.
The saved checkpoints hold the global model; the repository's Phase 3 never wrote
per-client or per-round prototype snapshots. Copies of one global model are not
independent client observations. Unless attested client-local snapshots exist,
stability is `unavailable` for every record and `evidence.stability_conclusion`
says so. PRS cannot be computed for those records
(`phase6_readiness.prs_computable_for_all_records == false`).

## Output (`pram.json`, schema `praa-fl.phase5.pram/1`)

Top level: `schema_version`, `formula_version`, `mode`, `status`
(`COMPLETE` or `PARTIAL_EVIDENCE_MISSING`), `formulas`, `config`, `provenance`
(Phase 3 model hash and round, Phase 4 partition hash, per-file prototype hashes),
`evidence`, `reference_prototypes`, `summary`, `phase6_readiness`, `records`.
Each record: `client_id`, `class_id`, `round_id`, `sample_count`,
`prototype_dimension`, `validation`, `consistency`, `stability`, `complete`.
Phase 6 should call `federated.pram.load_pram_output` and combine components only
for records with `complete == true`. Writes are atomic and never overwrite without
`--overwrite`.

## Running

```text
python run_phase5.py --smoke --phase3-dir <phase3> --phase4-dir <phase4>
python run_phase5.py --phase3-dir <phase3> --phase4-dir <phase4> --output-dir <phase5>
```

Smoke uses 512 validation samples and writes `results/phase5_smoke`. Final mode
requires a final-mode Phase 4 run covering all 50 clients and the complete
validation split.

## Decisions awaiting approval

1. Which stability evidence to use, since none exists for the finished run.
2. Whether validation recall should be corrected for the number of candidate
   classes (it favors clients holding few classes).
3. Whether the consistency reference should be leave-one-out and/or sample-count
   weighted.
4. How Phase 6 treats `degenerate` records.
