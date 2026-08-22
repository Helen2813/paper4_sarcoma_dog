Paper 5 neural simulation protocol
========================================================================

Script version: 05d1-freeze-neural-simulation-protocol-v1-no-cli

Scientific state
----------------
Neural model fitting performed: NO
Neural profile selected: NO
TARGET model fitting performed: NO
Final gate-calibration simulations accessed: NO
Outcome access: NO

GPU rule
--------
All neural forward/backward/optimizer work must run on frozen cuda:0.
Quiet CPU fallback is forbidden.
CPU is allowed for deterministic preprocessing, I/O, tables, and hashing.

Simulation split
----------------
Replicates 1-10 of every scenario: neural profile development only.
Replicates 11-20 of every scenario: confirmatory V0-V5 neural ablation only.
This split is frozen before any neural result exists.

Development partition
---------------------
Reuse exact 05c1 folds.
Folds 3-5: fit.
Fold 2: early stopping.
Fold 1: untouched profile-validation metrics.

Profile grid
------------
3 frozen 05a architecture profiles x 3 frozen 05a regularization profiles = 9.
Profile is selected for V4 SABLE-MO and reused by V0-V3 and V5.

Stability
---------
All 9 candidate profiles receive the frozen 20-resample development stability audit.
Confirmatory V4 receives an independent 20-resample stability audit on replicates 11-20.

Next stage
----------
05d2_screen_neural_profiles_gpu.py
