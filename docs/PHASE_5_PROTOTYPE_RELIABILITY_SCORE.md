# Phase 5: Prototype Reliability Score

Phase 5 adds the PRAA-FL Prototype Reliability Assessment Module (PRAM). It
does not modify FLAME training or perform adaptive aggregation.

## Inputs

Phase 5 consumes the Phase 4 files under
`results/phase4/prototypes/client_<id>.pt`. Each class is compared only with
the same class in a reference prototype. The reference is the mean of the
available Phase 4 client prototypes for that class.

## Reliability components

* **Validation score:** class-wise accuracy of the client's learned FLAME
  representation on the PathMNIST validation split. Validation embeddings are
  classified by nearest squared-Euclidean distance to that client's Phase 4
  prototypes. This avoids using the separate `model.classifier` head, which is
  not optimized by the Phase 3 FLAME objective. Test data is never loaded.
* **Prototype consistency:** cosine similarity between a client prototype and
  the same-class reference prototype, mapped from `[-1, 1]` to `[0, 1]`.
* **Prototype stability:** temporal cosine similarity between consecutive
  prototype snapshots, also mapped to `[0, 1]`.

The original Phase 4 artifacts contained one snapshot per client. Consequently,
temporal stability was explicitly unavailable and was not replaced by a
fabricated value. With the new round-scoped compatibility path, stability is
calculated only for the same client and class when at least two snapshots are
available. Other records retain `prs: null`.

## PRS formulation

When all three components exist:

```text
PRS = (w_v * validation + w_s * stability + w_c * consistency)
      / (w_v + w_s + w_c)
```

The default weights are one-third each and are configurable through
`federated.prs.PRSConfig`. This is a PRAA-FL experimental formulation, not a
claim about the FLAME paper.

## Outputs

`results/phase5/prs.json` stores client ID, class ID, all component values,
PRS, prototype dimension, sample count, state-source counts, and data-isolation
metadata. The artifact is structured for later Phase 6 consumption but Phase 6
is not implemented.

## Sanity result

The deterministic smoke execution used the existing Phase 4 artifacts and a
64-sample prefix of the validation split:

```text
Clients evaluated: 50
Class records: 81
Temporal snapshot files: 20
Temporal rounds: 2
Local states: 10
Global fallbacks: 40
Prototype consistency: finite
Prototype stability: finite for clients with round 1 and round 2 snapshots
PRS: finite for records with all three components
Test set accessed: False
Status: INCOMPLETE_CLIENT_COVERAGE for the smoke run because 40 clients lack local temporal states
```

## Limitations

The current smoke artifacts contain ten local states and forty global
fallbacks. A final experiment requires a full Phase 3 run and at least two
prototype snapshots per client to measure temporal stability and produce
finite PRS values.
