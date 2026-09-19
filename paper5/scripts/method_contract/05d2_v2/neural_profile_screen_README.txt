Paper 5 neural profile simulation screen
========================================================================

Script version: 05d2v2-screen-neural-profiles-gpu-syncfree-v2-no-cli

Scope
-----
V4 SABLE-MO profile selection only.
Development simulation replicates 1-10 only.
Confirmatory replicates 11-20 were not accessed.
TARGET-OS data were not accessed.
Outcomes and biological annotation were not used for selection.

GPU execution
-------------
All neural forward passes, backward passes, AdamW updates, contrastive
similarity matrices, and neural reconstruction operations ran on cuda:0.
Quiet CPU fallback was forbidden.
CPU was used only for feature eligibility/ranking, median imputation, audit I/O and hashing; dual PCA remained on CUDA.

Runtime optimization v2
-----------------------
The interrupted v1 execution was excluded from profile selection.
v2 removes per-epoch CUDA-to-host synchronization while preserving the frozen scientific protocol.

Profile screen
--------------
Candidate profiles: 9
Development datasets: 360
Dataset x profile shards: 3240
Passed shards: 3240
Failed shards: 0
Stability resamples per shard: 20

Selection
---------
Status: PROFILE_SELECTED_BY_FROZEN_ONE_SE_RULE
Selected profile: ARCH_COMPACT__REG_WEAK
Selection used the frozen hard gates and one-standard-error rule.
The selected profile is not yet the final Paper 5 representation method.

Next stage
----------
05d3_freeze_selected_neural_profile.py
