Conservative module-preservation audit
Script version: 28-conservative-module-preservation-audit-v1

Why this audit was added
------------------------
The script 27 signed-mean/weighted-score correlation and overlapping-subset
reliability metrics can be high partly because the compared scores share most
or all genes. They are useful diagnostics but should not, by themselves,
define cross-cohort structural preservation.

Primary preservation evidence in this audit
-------------------------------------------
1. Spearman preservation of the full within-module gene-correlation matrix
   between canine DOG2 and each human cohort.
2. Concordance of human PC1 loadings with frozen canine risk-oriented loadings.
3. Correlation between scores formed from non-overlapping gene halves.

Permutation tests
-----------------
Gene-label permutations are used for correlation-matrix and loading
concordance. BH correction is applied across the 12 human cohort-module
comparisons for each test family.

Interpretation
--------------
Outcome association and representation preservation remain separate.
Frozen score direction is never reversed after viewing a human outcome.
