Paper 5 selected neural profile freeze
========================================================================

Script version: 05d3-freeze-selected-neural-profile-v1-no-cli
Freeze status: FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION

Purpose
-------
Freeze the outcome-blind simulation-selected neural architecture and
regularization profile before any confirmatory neural ablation is run.
No model is trained in this stage.

Selected profile
----------------
Profile ID: ARCH_COMPACT__REG_WEAK
Architecture: compact
Regularization: weak
Shared latent dimension: 5
Private latent dimension per view: 2
Hidden dimension: 32
PCA components: RNA=10, METHYLATION=10, CNV=10

Selection evidence
------------------
Status: PROFILE_SELECTED_BY_FROZEN_ONE_SE_RULE
Independent one-SE replay: PASS
Best hard-gate-eligible profile: ARCH_COMPACT__REG_WEAK
Best mean utility: 0.674453590656
Best-profile SE: 0.0271911923123
One-SE threshold: 0.647262398344
Selected mean utility: 0.674453590656
Selected mean stability-gated false biological discovery fraction: 0.0723039215686

Guardrails
----------
TARGET accessed: NO
Outcomes/clinical labels accessed: NO
Confirmatory replicates 11-20 accessed: NO
Final gate-calibration data accessed: NO
Representation method selected: NO
Neural fitting performed by 05d3: NO

Confirmatory use
---------------
Use this exact frozen profile for V4 SABLE-MO and reuse it without
profile retuning for V0, V1, V2, V3, and V5.

Next stage
----------
05d4_fit_neural_ablation_benchmark_gpu.py
