Paper 4 project-wide multiplicity and assay-aware evidence lock
Script version: 33-projectwide-multiplicity-assay-aware-lock-v1

Purpose
-------
1. Apply one Benjamini-Hochberg correction across all 12 frozen primary human
   outcome tests: four modules in TARGET-OS, GSE21257, and GSE39055.
2. Integrate the outcome-blind GSE39055 Detection P-value diagnostic.
3. Add same-cohort Patkar TME-subtype biological convergence annotations.
4. Update the representation-outcome decoupling typology without changing any
   frozen program, score orientation, primary outcome model, or raw result.

Interpretation hierarchy
------------------------
- Script 23 remains the prespecified TARGET-OS and GSE21257 primary analysis.
- Script 26 remains the locked GSE39055 primary RFS analysis.
- Script 31 determines whether GSE39055 direction is robust to assay rules.
- Script 28 remains the conservative structure-preservation analysis.
- Script 32 is biological convergence within overlapping DOG2 samples, not
  independent validation.
- Script 33 updates manuscript wording and multiplicity accounting only.

Important restriction
---------------------
A diagnostic detection-aware sensitivity cannot replace a locked primary
analysis. However, assay-rule-sensitive or non-estimable direction should not
be presented as definitive biological heterogeneity.
