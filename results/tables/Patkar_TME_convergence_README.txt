Patkar TME-subtype convergence analysis
Script version: 32-patkar-tme-convergence-v1

Purpose
-------
This analysis tests whether frozen canine risk-oriented transcriptional
programs align with the independently developed deconvolution-based TME
subtypes reported by Patkar et al.

Important limitation
--------------------
The frozen programs and Patkar subtypes are evaluated in overlapping DOG2
samples and both use the same bulk transcriptomic data. Therefore, this is
not an independent external validation. It is an orthogonal-method biological
convergence/annotation analysis.

Primary targeted convergence questions
--------------------------------------
1. Does M34 differ between immune-enriched tumors (IE or IE-ECM) and
   immune-desert tumors (ID)?
2. Does M11 differ between IE-ECM and IE tumors?

Additional analyses
-------------------
- Kruskal-Wallis tests across ID, IE, and IE-ECM for M34, M11, M24, and M40.
- Pairwise Mann-Whitney tests with multiplicity correction.
- Strict signed-mean score is primary.
- Frozen canine PCA-weighted score is a sensitivity variant.

Guardrails
----------
No subtype label is used to select genes, alter weights, orient scores, revise
validation tiers, or change the locked human outcome evidence hierarchy.
