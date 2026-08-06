Cross-cohort frozen-module representation preservation audit
Script version: 27-cross-cohort-module-preservation-v1

Purpose
-------
This analysis distinguishes structural transport of a frozen module from
transport of its outcome association. It evaluates canine DOG2, TARGET-OS,
GSE21257, and GSE39055 without changing frozen genes, weights, directions,
or validation tiers.

Structural metrics
------------------
- PC1 variance explained
- concordance of cohort PC1 loadings with frozen canine risk loadings
- mean signed pairwise gene correlation
- correlation among signed-mean, canine-weighted, and human-PC1 scores
- leave-one-gene-out or repeated gene-subset score reliability
- expression-matched random-module controls

Interpretation
--------------
A module can preserve its co-expression representation while its prognostic
association changes across endpoints or cohorts. Such a result indicates
outcome heterogeneity rather than proof that the module definition failed.
Conversely, weak structural preservation can implicate platform or
representation instability.

Random controls
---------------
500 expression-matched random gene sets are used per
cohort-module pair. Their p-values are descriptive specificity controls.
