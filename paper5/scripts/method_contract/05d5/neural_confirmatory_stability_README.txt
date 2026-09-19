Paper 5 V4 confirmatory stability benchmark
========================================================================

Script version: 05d5-evaluate-sable-mo-confirmatory-stability-gpu-v1-no-cli

Scope
-----
Confirmatory simulation replicate indices 11-20 only.
Confirmatory datasets: 360
Variant: V4_SABLE_MO
Frozen profile: ARCH_COMPACT__REG_WEAK
Reference fits: 360
Resample fits: 7200
Total neural optimization runs if all datasets pass: 7560

Epoch bridge
------------
05d1 did not explicitly specify how the ten 05d4 V4 fold/seed-selected epoch counts
should transfer to one full-data 05d5 stability epoch count. Before any 05d5 stability
result is generated, this script freezes a deterministic dataset-specific registry:
mean of the ten 05d4 selected_epoch diagnostics, rounded nearest integer half-up.
No 05d4 utility/truth metric, 05d5 stability result, TARGET value, or outcome is used by this rule.

Stability
---------
Each dataset receives one frozen-seed full-data V4 reference fit and 20 frozen-seed 80% subsample refits.
Each resample refits preprocessing and the neural model.
Shared scores are aligned by orthogonal Procrustes and Hungarian factor matching.
The already-frozen 05a stability thresholds are applied without retuning.

Interpretation
--------------
Factor-level primary summaries are restricted to factor-level-identifiable datasets and true biological nulls.
Rotational non-null factor-level results remain descriptive; subspace stability remains interpretable.
This stage does not select the final representation method.

GPU execution
-------------
All neural forward/backward/AdamW work runs on frozen cuda:0 in float32.
Dual-PCA Gram/eigendecomposition/projection runs in float64 on cuda:0.
AMP: OFF; TF32: OFF; torch.compile: OFF; quiet CPU fallback: FORBIDDEN.

Results
-------
Passed dataset shards: 360
Failed dataset shards: 0

Next stage
----------
05e_compare_ablate_and_freeze_method.py
