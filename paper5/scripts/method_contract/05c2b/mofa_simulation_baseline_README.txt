Paper 5 MOFA simulation baseline benchmark
========================================================================

Script version: 05c2b-fit-mofa-simulation-baselines-v1-no-cli

Scope
-----
Confirmatory sable_mo_v2 architecture-screening bank only.
720 datasets; MOFA K=5,10,15,20.
Five held-out fits plus one full-data truth fit per dataset and rank.
Total frozen MOFA fits: 17280.

Held-out preprocessing
----------------------
Feature eligibility and pre-z-score variance ranking use training samples only.
Primary feature budgets: RNA 5000; methylation 10000; CNV 5000.
No median or KNN imputation is used for MOFA.
Missing observations remain NaN.
Centering and scaling parameters come from the training fold only.

Out-of-sample inference
-----------------------
Raw X @ W is forbidden.
Test scores use the 05c2a fixed-parameter Gaussian posterior mean.
Held-out samples never update W, Tau, sparsity, ARD, or preprocessing.

Truth evaluation
----------------
Shared factor/subspace recovery uses the same definitions as 05c1.
Private recovery means view-specific private-signal capture, not an explicit MOFA private block.
Technical-factor contamination is evaluated from known simulation truth.

Transparency
------------
Failed fits remain in the audit.
G09 is diagnostic for the MOFA baseline.
No architecture is selected in this stage.
Final gate-calibration simulations are not accessed.
