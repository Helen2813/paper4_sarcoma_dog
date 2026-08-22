Paper 5 prospective simulation and technical-confound addendum
============================================================================

Script version: 05a1-freeze-simulation-technical-confound-addendum-v3-no-cli
Created UTC: 2026-08-11T20:35:13.885598+00:00

Status
------
Freeze status: FROZEN_BEFORE_MODEL_FITTING_AFTER_DISCLOSED_PILOT_MATERIALIZATION
Frozen before simulation model fitting: yes
Frozen before TARGET representation fitting: yes
Simulation data materialized before addendum: yes; disclosed 05b v1 pilot excluded from confirmatory analyses
Outcome loaded: no
Biological annotation used for selection: no

Why this addendum exists
------------------------
The base 05a contract is retained unchanged. This prospective addendum adds the realistic n=62 operating point, rotationally ambiguous factors, partial sharing, correlated-private nulls, structured missingness, sparse/dense loadings, and technical batch-confound scenarios.

Two-stage simulation policy
---------------------------
Architecture screening: 36 combined scenarios and 720 datasets.
Final gate calibration: 24 scenarios and 3100 planned datasets, executed only after the architecture is frozen.

Stability interpretation
------------------------
Factor-level recovery uses Hungarian matching and sign alignment.
Subspace-level recovery uses principal angles and projection distance.
Rotationally stable subspaces may be retained without assigning stable identities to individual factors.

Technical-confound rule
-----------------------
Reproducibility does not establish biological meaning. Stable factors are audited against usable sample-level technical metadata after representation freeze and before primary biological claims.

Next required scripts
---------------------
04d_target_technical_metadata_preflight.py
05b_generate_small_n_multiomics_simulations.py v2

Do not use or rerun 05b v1 for confirmatory analyses after creating this addendum.
The required v2 generator must write to paper5/simulations/sable_mo_v2.
