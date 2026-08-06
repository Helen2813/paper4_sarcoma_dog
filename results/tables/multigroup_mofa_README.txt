Python-native multi-group MOFA analysis
Script version: 35-mofapy2-multigroup-factor-v1

Method
------
This script uses the official Python packages mofapy2 and mofax.

It is not an exact implementation of the De Vito multi-study factor analysis
or the R-only VIMSFA package. Instead, it uses MOFA2 multi-group inference with
one RNA view and one group per cohort.

In multi-group MOFA, feature weights are common, while group-wise factor
activity is learned through factor scores and group-specific ARD. Factors can
therefore be active in all groups, in a subset of groups, or in one group only.

Input
-----
Outcome-blind rank-Gaussian matrices created by script 34.

Analysis sets
-------------
- four_cohort_core_plus_frozen
- four_cohort_detection_aware
- three_cohort_no_ffpe

Initial factor grids
--------------------
[8, 12, 16]

Factor activity
---------------
The primary descriptive threshold is at least
1.0% variance explained within a group.
Sensitivity thresholds are [0.5, 1.0, 2.0]%.

Model-selection guardrail
-------------------------
No rank is selected using an outcome. All ranks and analysis sets are retained.
Standard variational MOFA uses PCA-based initialization; one fixed seed is
recorded for each rank. Rank sensitivity is evaluated through rotation-invariant
weight-subspace comparisons.

Interpretation
--------------
Call this analysis "multi-group MOFA2" or "unsupervised multi-group factor
analysis" in the manuscript. Do not call it VIMSFA or exact MSFA.
