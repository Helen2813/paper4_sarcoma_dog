Paper 5 05x2b — post-unblinding exploratory retention ablation
========================================================================

Script version: 05x2b-run-stability-only-retention-ablation-fast-cpu-v2-no-cli
JSON backend: stdlib-json
No model refit; no GPU computation; 400 frozen shards only.

Results
-------
CORRELATED_PRIVATE_N62 | FULL_FROZEN_RETENTION: 70/200 = 35.0% [95% CI 28.7%, 41.8%]
CORRELATED_PRIVATE_N62 | NO_MODALITY_SUPPORT: 70/200 = 35.0% [95% CI 28.7%, 41.8%]
CORRELATED_PRIVATE_N62 | PURE_FACTOR_REPRODUCIBILITY: 70/200 = 35.0% [95% CI 28.7%, 41.8%]
INDEPENDENT_NULL_N62 | FULL_FROZEN_RETENTION: 0/200 = 0.0% [95% CI 0.0%, 1.9%]
INDEPENDENT_NULL_N62 | NO_MODALITY_SUPPORT: 0/200 = 0.0% [95% CI 0.0%, 1.9%]
INDEPENDENT_NULL_N62 | PURE_FACTOR_REPRODUCIBILITY: 0/200 = 0.0% [95% CI 0.0%, 1.9%]

Exploratory context only; frozen 05f3e remains authoritative.
No D01 reweighting, threshold retuning, TARGET access, or outcome access.
