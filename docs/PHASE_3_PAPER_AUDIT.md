# Phase 3 FLAME Paper Audit

This audit distinguishes paper-specified behavior from implementation choices.

| Requirement | Paper/specification | Implementation | Status | Notes |
|---|---|---|---|---|
| FL objective | Eq. (1), Section III | `Server.aggregate` | IMPLEMENTATION DETAIL | Sample-count weights are used because the paper leaves `nu_r` abstract. |
| Shared encoder | Section III-B/C/E | `FLAMEModel.encoder` | PAPER-SPECIFIED | One ResNet-50 instance feeds both branches. |
| Encoder | Section III-E | `models/classifier.py:Encoder` | PAPER-SPECIFIED | ResNet-50 adapted for 28x28 inputs. |
| Patch masking | Eq. (3), Section III-A | `models/masking.py:RandomPatchMasker` | PAPER-SPECIFIED | 4x4 patches and configurable ratio. |
| MAE reconstruction | Eq. (4) | `training/losses.py:compute_mae_loss` | PAPER-SPECIFIED | MSE against original images. |
| Decoder | Section III-E | `models/classifier.py:Decoder` | EXPERIMENTAL ADAPTATION | Transposed-convolution decoder sized for PathMNIST. |
| Projection head | Section III-B/E | `models/classifier.py:ProtoHead` | PAPER-SPECIFIED | Two fully connected layers with ReLU. |
| Prototypes | Eq. (5) | `compute_prototypes` | PAPER-SPECIFIED | Support embeddings only. |
| Probability | Eq. (6), Eq. (10) | `squared_euclidean_logits` | PAPER-SPECIFIED | Negative squared Euclidean logits. |
| Proto loss | Eq. (7) | `compute_proto_loss` | PAPER-SPECIFIED | Cross entropy on query examples only. |
| Combined loss | Eq. (8) | `combine_losses` | PAPER-SPECIFIED | Configurable 0.7/0.3 defaults. |
| Loss pre-scaling | Section III-C | `prescale_losses` | IMPLEMENTATION DETAIL | Explicit reference scales; paper does not specify numerical scale factors. |
| Client update | Algorithm 2 | `Client.client_update` | EXPERIMENTAL ADAPTATION | Balanced paired labeled/unlabeled iterators. |
| Client sampling | Algorithm 1 | `select_clients` | EXPERIMENTAL ADAPTATION | Deterministic seeded sample; exact stratification is unspecified. |
| Aggregation | Algorithm 1 | `Server.aggregate` | IMPLEMENTATION DETAIL | Normalized sample-count weighting. |
| Local prototypes | Section III-B | `Client.client_update` | PAPER-SPECIFIED | Prototypes are not returned to the server. |
| Sparse labels | Specification | `Client` split | EXPERIMENTAL ADAPTATION | Deterministic configurable label fraction. |
| Scheduler | Section IV-A | `run_federated_training` | PAPER-SPECIFIED | ReduceLROnPlateau factor 0.1, patience 5. |
| Metrics | Specification | `training.validation` | PAPER-SPECIFIED | Loss, accuracy, balanced accuracy, macro F1. |
| Persistence | Specification | `run_federated_training` | IMPLEMENTATION DETAIL | JSON config/metrics and final model checkpoint. |
| PRS/PRAM/PRAA | Phase boundary | No implementation | NOT IMPLEMENTED | Intentionally deferred to later phases. |
