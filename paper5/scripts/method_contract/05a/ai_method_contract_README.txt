Paper 5 AI method contract
========================================================================

Script version: 05a-freeze-ai-method-contract-v1
Created UTC: 2026-08-11T18:43:01.039109+00:00
Provisional method name: SABLE-MO

Scientific state
----------------
outcome_loaded: false
clinical_labels_loaded: false
model_fitting_performed: false
biological labels used for selection: false
post-standardization variance used for selection: false

Frozen target evaluation
------------------------
5 repeats of 5-fold outer CV; 4-fold inner CV.
Splits are balanced and use no outcomes or clinical labels.
Neural fits use a separate deterministic early-stopping subset.
The metric-validation subset is never reused for early stopping.
All data-dependent preprocessing is fitted within training folds.

Model families
--------------
- modality-specific PCA
- block-scaled PCA
- MOFA+ rank sweep
- ridge generalized CCA (secondary)
- V0-V4 SABLE-MO ablation family
- SABLE-VAE (secondary exploratory)

Method-development rule
-----------------------
Neural architecture and regularization profiles are selected on ground-truth simulations before TARGET benchmarking.
TARGET inner CV uses an outcome-blind composite of held-out reconstruction, paired retrieval, missing-view recovery, and modality balance.
The one-standard-error rule selects the simplest eligible model.

Transparency rule
-----------------
All prespecified model families and failed variants remain in the benchmark audit. Poor results may be summarized in the supplement but may not be silently omitted.

Next scripts
------------
05b_generate_small_n_multiomics_simulations.py
05c_fit_linear_mofa_baselines.py
05d_fit_sable_mo_variants.py
05e_compare_ablate_and_freeze_method.py
