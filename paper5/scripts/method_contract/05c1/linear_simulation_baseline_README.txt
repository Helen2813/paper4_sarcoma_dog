Paper 5 linear simulation baseline benchmark
========================================================================

Script version: 05c1-fit-linear-simulation-baselines-v1-no-cli

Scope
-----
Confirmatory architecture-screening datasets: 720
Scenarios: 36
Latent rank sensitivities: [5, 10, 15]
Primary linear rank: 10

Models
------
PCA_RNA
PCA_METHYLATION
PCA_CNV
BLOCK_PCA with equal total training variance per modality

Evaluation
----------
Five-fold deterministic held-out reconstruction.
Full-data simulation-truth recovery is reported separately.
Cross-view retrieval is scored against true row_source pairing.
Complete simulation truth is used only for evaluation, never preprocessing.

Scientific state
----------------
Outcome access: false
TARGET model fitting: false
MOFA fitting: false
Neural fitting: false
Architecture selection: false
Final gate-calibration simulations: not used

Interpretation
--------------
Linear baselines are reported whether they perform well or poorly. No prespecified model family is silently omitted.
The false-shared-discovery gate is diagnostic for ungated PCA; it is not used to hide or remove the linear baseline.
