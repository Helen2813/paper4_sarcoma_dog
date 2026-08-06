Variable-only outcome-blind MOFA input preparation
Script version: 37-prepare-variable-only-mofa-inputs-v1

Purpose
-------
Create a stricter baseline for independent latent-factor recurrence.

Unlike script 34, this script never forces frozen M34, M11, M24, or M40 genes
into the factor-analysis feature space. Genes are selected only by:
- strict one-to-one ortholog status,
- availability in the requested cohorts,
- outcome-blind cross-study variability,
- and, for the detection-aware set, GSE39055 Detection P-value coverage.

Analysis sets
-------------
- four_cohort_variable_only_1500
- four_cohort_detection_aware_variable_only_700
- three_cohort_no_ffpe_variable_only_1500

Interpretation
--------------
Any frozen-module genes present in these sets were selected naturally by the
same outcome-blind variability rule as all other genes. Subsequent module-to-
factor alignment is therefore a stronger test of latent recurrence than the
frozen-program-enriched analysis.

A module with very low natural gene coverage is not interpretable in this
baseline. No outcome is loaded or used.
