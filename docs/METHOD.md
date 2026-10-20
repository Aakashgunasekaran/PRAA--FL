# Method: the eight phases and their formulas

Notation: clients r = 1..R (R = 50), classes c = 1..9, n_r = size of client r's partition,
n_{r,c} = number of labelled support samples of class c on client r (1 % labelled subset),
f_θ = shared ResNet-50 encoder, g_φ = projection head (128-d), p_{r,c} = client prototype.
Items marked **[paper]** follow the FLAME paper, **[repo]** follow the project's earlier
documents/code (`docs/PHASE_5_*.md`), **[decision]** are choices made while completing Phases 6-8
(each is configurable and listed in `docs/LIMITATIONS.md`).

## Phase 1 — Data preprocessing
PathMNIST (MedMNIST v2, 28×28×3, 9 classes): 89 996 train / 10 004 validation / 7 180 test;
normalisation (x−0.5)/0.5. `datasets/preprocess.py`. Validation is used for all development
decisions; test only for the final evaluation (`--split test --final`).

## Phase 2 — Federated client training setup
Dirichlet (β = 0.1) non-IID partition over 50 clients, minimum 10 samples per client, seed 42;
10 clients sampled per round with `random.Random(seed + round)`; 1 % labelled subset per client,
`max(2, int(0.01 n_r))` labelled samples chosen with `seed + client_id`; the rest is unlabelled.

## Phase 3 — Masked autoencoder + prototypical network (FLAME) [paper]
Local loss per step: L = α·L_MAE + (1−α)·L_Proto with α = 0.7, masking ratio 0.8 (patch 4),

    L_MAE   = mean || x − f_θ'(f_θ(M(x))) ||²                       (paper Eq. 4)
    p_c     = mean_{i in support, y_i=c} g_φ(f_θ(x_i))               (Eq. 5)
    P(y=c|x)= softmax_c( − || g_φ(f_θ(x)) − p_c ||² )                (Eq. 6/10)
    L_Proto = − mean_{(x,y) in query} log P(y|x)                     (Eq. 7)

Adam lr 1e-3, batch 64, 2 local epochs, ReduceLROnPlateau (×0.1, patience 5), 200 rounds.
Server (FLAME): θ ← Σ_r ν_r θ_r, ν_r = n_r / Σ n  (`federated/flame.py`, unchanged).
The **original 200-round run is preserved and not re-run** (`results/phase3/`, model on Drive).

## Phase 4 — Prototype generation
Class-wise mean of projected embeddings of the labelled samples a client holds
(`training/prototypes.py`, `training/prototype_store.py`). Preserved: prototypes of the final global
model on every client's labelled subset (`results/phase4/`, 126 (client, class) prototypes, 881
labelled samples). In the new PRAA/FLAME runs the same extraction is applied to each **local**
model right after local training (`federated/reliability.py::local_evidence`), which produces the
attested client-local snapshots the preserved run never stored.

## Phase 5 — Prototype Reliability Assessment Module (PRAM) [repo]
Three components per (client r, class c), each `valid` (value ∈ [0,1]), `degenerate` (computable
but uninformative) or `unavailable` (evidence missing) — never replaced by 0, 1 or a constant
(`federated/pram.py`, formula version `pram/1`):

    V_{r,c} validation  = recall of class c under nearest-prototype classification (squared Euclidean,
                          candidates = classes client r holds) on the shared validation set.
                          degenerate if r holds one class (recall = 1 by construction).
    C_{r,c} consistency = ( cos(p_{r,c}, p̄_c) + 1 ) / 2,  p̄_c = unweighted self-inclusive mean of the
                          clients' class-c prototypes. degenerate if only r holds c.
    S_{r,c} stability   = mean over consecutive attested client-local snapshots of ( cos + 1 ) / 2.
                          unavailable with < 2 attested snapshots containing c.

Where the evidence comes from:

* *Post-hoc, preserved run* (`run_phase5.py`): validation uses the global model's embeddings; no
  snapshots exist → stability `unavailable` for all 126 records.
* *Live, PRAA/FLAME runs*: after local training, client r's prototypes and the validation embeddings
  **of its local model** are assessed against the round's participants (reference = participants
  holding c). Stability uses the last `stability_window` = 3 snapshots **[decision]**; participation
  gaps are recorded, non-participation is not instability. Snapshots are written to
  `snapshots/round_<t>/client_<r>.pt` with `provenance.source_kind = "client_local_model"`.
* *Global assessment* (final evaluation, and every round for logging): PRAM over all 50 clients'
  prototypes computed with the same global model; stability from the stored local snapshots.

## Phase 6 — Prototype Reliability Score (PRS) and client contribution
    PRS_{r,c} = ( w_v V + w_s S + w_c C ) / ( w_v + w_s + w_c ),  w = 1/3 each        [repo]

**Strict policy (default, the documented rule):** PRS exists only when V, S and C are all `valid`.
Otherwise it is `null` with a reason — nothing is imputed. `--missing-component-policy renormalize`
averages the valid components only (needs ≥ 2); this is an **extension, not in the documents**, off
by default and labelled `extension_notice` wherever used.

    R_r = Σ_c n_{r,c} PRS_{r,c} / Σ_c n_{r,c}   over classes with a PRS          [decision]
          (`client_reduce="mean"` = plain mean). A client with no PRS is "unassessed".

## Phase 7 — Prototype Reliability-Aware Adaptive Aggregation (PRAA)
Relative reliability multiplier (mean 1 over assessed clients) **[decision]**:

    m_r = clip( R_r^τ / mean_{r' assessed} R_{r'}^τ , floor, cap ),   τ = 1, floor = 0.05, cap = 5
    m_r = 1 for unassessed clients, and for everyone if fewer than 2 clients are assessed or τ = 0

Server step (the FLAME baseline stays a separate method):

    FLAME:  w_r = ν_r                       = n_r / Σ n
    PRAA :  w_r = ν_r m_r / Σ_j ν_j m_j,    θ ← Σ_r w_r θ_r

"No evidence ⇒ no re-weighting": a client without evidence keeps its FLAME weight; PRS itself is
never imputed. With τ = 0, or when no evidence exists (round 1, the preserved run), PRAA ≡ FLAME
exactly (tested). Global prototypes used at inference:

    FLAME:  P_c = Σ_r n_{r,c} p_{r,c} / Σ_r n_{r,c}                  (pooled class mean)
    PRAA :  P_c = Σ_r a_{r,c} p_{r,c},  a_{r,c} ∝ n_{r,c} m_{r,c},  m_{r,c} = PRS_{r,c}^τ / mean over assessed holders of c

Two switches (`praa.apply_to_model`, `praa.apply_to_prototypes`) allow model-only / prototype-only
ablations. FedAvg baseline (project definition): supervised cross-entropy of the classifier head on
the labelled subsets only, size-weighted by labelled count; same network, optimiser, epochs and
client sampling.

## Phase 8 — Global model evaluation
Inference: FedAvg → classifier head; FLAME/PRAA → nearest global prototype, probabilities
softmax(−‖z − P_c‖²) (paper Eq. 6/10). Metrics (`evaluation/metrics.py`): accuracy, balanced
accuracy, macro-F1, macro one-vs-rest AUROC (classes lacking positives or negatives are reported,
not zero-filled), NLL. Convergence from the per-round validation accuracy curve: final, mean of the
last 10 rounds, best and its round, mean over rounds (normalised AUC), rounds to reach 0.5/0.6/0.7/0.8,
rounds to 95 % of best. Per-round validation uses a seeded 2 000-image subset of the validation set;
the final development evaluation uses the whole validation split.
