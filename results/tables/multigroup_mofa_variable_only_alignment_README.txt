Variable-only frozen-module alignment to multi-group MOFA
Script version: 39-align-variable-only-mofa-v1

Purpose
-------
Test whether frozen canine programs recur in an outcome-blind factor space in
which frozen-module membership was not used for feature selection.

The factor models were frozen by script 38 before this alignment was performed.

Primary tests
-------------
- Maximum absolute cosine with any retained factor, assessed against the
  maximum-over-factors null from variability-matched random gene sets.
- Rotation-invariant capture by shared, ubiquitous, non-FFPE-shared, and
  GSE39055-associated factor subspaces.
- Stability across all prespecified initial factor ranks.
- Sensitivity to GSE39055 exclusion and Detection-P-aware feature filtering.

Random controls
---------------
Random panels are matched approximately on cross-study variability. All genes
belonging to any primary frozen module are excluded from the random candidate
pool. Frozen loading magnitudes and signs are permuted across matched genes.

Interpretability rule
---------------------
A module is interpreted only when at least 5 naturally
selected genes and at least 20% of the frozen module
are present. This intentionally excludes low-coverage M24 analyses.

Important distinction
---------------------
This is stronger than the targeted MOFA audit because no frozen genes were
forced into the factor-analysis feature space. Alignment is nevertheless a
post-fit test, not a new independent patient cohort or outcome validation.

No clinical endpoint or outcome label is loaded.
