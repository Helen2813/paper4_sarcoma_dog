Outcome-blind multi-study factor input preparation
Script version: 34-prepare-multistudy-factor-inputs-v1

Purpose
-------
Prepare harmonized canine and human osteosarcoma matrices for multi-study
factor analysis that separates shared and study-specific latent transcriptional
variation.

No outcomes are loaded or used by this script.

Feature universe
----------------
Strict symbol-concordant one-to-one canine-human orthologs from the locked
ortholog-QC table.

Analysis sets
-------------
1. four_cohort_core_plus_frozen
   Top 500 cross-study variable background genes plus all
   available frozen primary-module genes across DOG2, TARGET-OS, GSE21257,
   and GSE39055.

2. four_cohort_detection_aware
   Top 350 cross-study variable genes among probes detected
   at P < 0.01 in at least 50% of GSE39055
   samples, plus available frozen primary-module genes that meet the same rule.

3. three_cohort_no_ffpe
   DOG2, TARGET-OS, and GSE21257 sensitivity analysis excluding the FFPE DASL
   cohort.

Transformation
--------------
Within each cohort, each gene is transformed by a rank-based inverse-normal
transformation and standardized. Cohort mean differences are therefore removed;
the analysis targets shared versus cohort-specific covariance structure.

Interpretation
--------------
The prepared matrices support unsupervised statistical machine learning.
Subsequent latent factors must be related to outcomes only after model fitting.
Factor-space alignment with frozen modules should use rotation-invariant
subspace projection in addition to individual-factor correlations.
