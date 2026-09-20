# Phase 4: Prototype Generation

## Objective

Phase 4 generates client-specific, class-wise prototypes from the learned
FLAME representation. It does not calculate reliability or change federated
aggregation.

## Input and representation

The runner loads the learned Phase 3 representation. Phase 3 now persists the
latest trained state for each selected client after every round, including the
client ID and round number. For each Phase 2 client training subset, it uses:

```text
PathMNIST image -> shared ResNet-50 encoder -> FLAME projection head
             -> 128-dimensional projected embedding
```

No new encoder or neural network is created.

## Prototype generation

For client `k` and class `c`:

```text
P(k,c) = mean(f(x_i)) for all client samples with y_i = c
```

Only classes present in a client dataset receive prototypes. Absent classes
are omitted; no zero or synthetic prototypes are created.

## Extraction and storage

Feature extraction uses evaluation mode and `torch.no_grad()`. Images are
processed sequentially in batches, and non-finite embeddings raise an
explicit `FloatingPointError`. Each client is saved independently under:

```text
results/phase4/prototypes/client_<id>.pt
```

Each file contains the client ID, class-to-prototype tensors, sample counts,
embedding dimensions, prototype norms, and representation description.
`results/phase4/summary.json` contains measured run totals.

Two explicit execution modes are supported:

* Smoke mode (`python run_phase4.py`) processes available local states and
  explicitly reports global-checkpoint fallback for missing clients. It is
  CPU-friendly and does not run the 200-round experiment.
* Final mode (`python run_phase4.py --final`) rejects missing or mismatched
  local states and never falls back to the global checkpoint. It is ready only
  when all 50 clients have actual local states.

The current smoke artifacts are therefore not final research results:
10 local states and 40 global fallbacks were available from the one-round
Phase 3 smoke run. A full Phase 3 execution must be run separately to produce
the required 50 local states.

## Memory strategy and isolation

Clients are processed sequentially. No optimizer is created, embeddings are
detached to CPU, and no client model collection is retained. The Phase 4
runner uses only Phase 2 training subsets and does not access the PathMNIST
test dataset.

## Scope boundary

```text
Phase 4 = Prototype Generation
Phase 5 = Prototype Reliability Score (PRS) — implemented separately
Phase 6 = PRAM/PRAA adaptive aggregation — not implemented
Phase 7 = PRAA — not implemented
Phase 8 = Final Evaluation — not implemented
```
