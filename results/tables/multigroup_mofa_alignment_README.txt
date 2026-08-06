Frozen-module alignment to outcome-blind multi-group MOFA
Script version: 36-align-frozen-modules-to-mofa-v2

Purpose
-------
Project the four frozen primary canine program-loading vectors onto the
outcome-blind MOFA weight subspaces frozen by script 35.

Primary metrics
---------------
1. Signed and absolute cosine similarity with individual factors.
2. Rotation-invariant squared projection of each frozen module vector onto:
   - all retained factors
   - factors active in at least two cohorts
   - factors active in every cohort
   - partially shared factors
   - shared factors excluding GSE39055
   - shared factors including GSE39055
   - GSE39055-specific factors
3. Rank sensitivity across 8, 12, and 16 initial factors.
4. Sensitivity to Detection-P-aware feature filtering and exclusion of GSE39055.
5. Comparison with the corresponding stacked-PCA subspace.

Matched random controls
-----------------------
For each module, random gene sets are matched approximately on the outcome-blind
cross-study variability percentile. The frozen loading values are permuted across
the matched genes. These controls assess representation specificity within the
fitted feature space.

Important limitation
--------------------
The core feature sets were enriched with available frozen primary-module genes.
Therefore, this analysis is a targeted latent-representation audit, not an
independent rediscovery of the modules. Matched random controls reduce but do
not eliminate this design dependence. A later variable-only factor sensitivity
can test independent recurrence without forced module inclusion.

Guardrails
----------
No clinical endpoint or outcome label is loaded.
The MOFA models and frozen module weights are read after their hashes were fixed.
No factor rank is selected using an outcome.
