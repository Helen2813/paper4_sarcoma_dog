Paper 5 06b0 - TARGET pre-fit input/policy lock
================================================

This stage performs NO TARGET model fitting and reads NO clinical outcomes or biological annotations.
It freezes the exact 78-case molecular-input identities, upstream 04b/04c preprocessing contracts,
05a nested-CV assignments/model-selection policy, 05e proposed SABLE-MO implementation, 06a q10
retention policy, and the frozen 04d technical-metadata boundary.

Critical leakage guard:
Global harmonized z-score matrices are integrity/provenance anchors only for held-out CV. Nested CV must
re-estimate ranking/selection, missing-value handling, scaling, PCA/learned preprocessing only from each
training partition, then apply those fitted transformations to validation/test.

Critical scope guard:
The 06a q10=0.60 primary and q10=0.40 historical sensitivity retention rules are SABLE-MO-specific and
must not be transferred to other representation engines.

06b0 does NOT authorize a real TARGET fit. A separate source-hash/implementation lock is required next.
