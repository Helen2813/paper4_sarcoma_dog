Paper 5 MOFA out-of-sample projection protocol
========================================================================

Script version: 05c2a-freeze-mofa-out-of-sample-projection-protocol-v1-no-cli
Created UTC: 2026-08-11T23:43:18.048430+00:00

Status
------
Frozen before the first MOFA simulation fit.
No architecture was selected.
No outcome, clinical label, or biological program label was accessed.

Why this stage exists
---------------------
The Paper 5 benchmark requires genuinely held-out reconstruction, cross-view retrieval, and missing-view reconstruction.
Held-out MOFA factor scores are therefore inferred with a fixed-parameter Gaussian posterior mean under the training-fitted MOFA model.
Raw multiplication of held-out data by fitted MOFA weights is explicitly forbidden as the benchmark projection rule.

Frozen MOFA models
------------------
MOFA_K5
MOFA_K10
MOFA_K15
MOFA_K20

Preprocessing
-------------
Training-fold eligibility and pre-z-score variance ranking only.
Primary budgets: RNA 5000; methylation 10000; CNV 5000.
No imputation for MOFA; missing observations remain NaN.
Training-fold finite-value centering and scaling are applied externally.
mofapy2 internal centering/scaling is disabled.

Held-out projection
-------------------
A = I + sum W_obs^T diag(tau_obs) W_obs
b = sum W_obs^T diag(tau_obs) x_obs
z_hat = solve(A, b)

Paired benchmark rule
---------------------
05c2b must reuse the exact 05c1 five-fold split registry so that linear-versus-MOFA comparisons use identical held-out samples.

Next stage
----------
05c2b_fit_mofa_simulation_baselines.py
