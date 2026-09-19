Paper 5 neural confirmatory ablation benchmark
========================================================================

Script version: 05d4-fit-neural-ablation-benchmark-gpu-v1.2-verified-fixture-hotfix-no-cli

Scope
-----
Confirmatory simulation replicate indices 11-20 only.
Confirmatory datasets: 360
Variants: V0-V5.
Five frozen folds x two frozen seeds = 21,600 held-out fold-seed evaluations.
Early-stop fit + selected-epoch refit = 43,200 planned neural optimization runs.

Frozen profile
--------------
Profile: ARCH_COMPACT__REG_WEAK
Architecture: compact
Regularization: weak
The same 05d3-frozen profile is reused by all V0-V5 variants without retuning.

Confirmatory separation
-----------------------
Development simulation replicates 1-10 are not used as confirmatory evidence.
TARGET-OS matrices, outcomes, clinical labels, and biological program annotations are not accessed.
Final gate-calibration simulations are not accessed.

GPU execution
-------------
All neural forward/backward/AdamW work runs on frozen cuda:0 in float32.
Dual-PCA Gram/eigendecomposition/projection runs in float64 on cuda:0.
AMP: OFF; TF32: OFF; torch.compile: OFF; quiet CPU fallback: FORBIDDEN.

Important V4 rule
-----------------
The V4 stability gate is not a training component and is not executed in 05d4.
V4-versus-V3 is therefore not interpreted here as the effect of the stability gate.
Independent confirmatory V4 stability resampling is reserved for 05d5.

Results
-------
Passed fold-seed shards: 21600
Failed fold-seed shards: 0
No final representation method is selected in this stage.

Next stage
----------
05d5_evaluate_sable_mo_stability_gpu.py
