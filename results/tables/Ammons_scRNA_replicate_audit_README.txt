Ammons single-cell replicate-unit audit
Script version: 43-audit-lock-ammons-single-cell-v1

Purpose
-------
1. Audit whether orig.ident can safely be treated as an independent dog.
2. Identify any metadata column that collapses the 8 orig.ident
   levels to six biological dogs.
3. Verify the sign consistency underlying the exact paired tests.
4. Remove non-estimable M11/M24 negative-component rankings.
5. Lock manuscript-ready M34 and M40 biological localization statements.

Current replicate status
------------------------
candidate_six_dog_mapping_found

Candidate six-dog mapping column
--------------------------------
name

Important statistical point
---------------------------
For eight paired units, the minimum attainable two-sided exact sign-flip
P value is 2 / 2^8 = 0.0078125. All four targeted contrasts reached this
floor because every paired difference had the same sign. This is strong
directional consistency, but the four tests are biologically related and
should not be presented as four independent replications.

Biological interpretation
-------------------------
- M34: negative-loading genes localize to IFN-TAM/TAM/DC/TIM populations,
  while the signed risk score is highest in osteoblast-lineage populations.
  Because 153 of 155 detected genes have negative loadings, the score is
  principally an inverse immune-lineage axis.
- M40: highest in cycling osteoclast, cycling T-cell, and cycling osteoblast
  populations. The program is a broad cycling/proliferation axis with clear
  cycling-osteoblast enrichment, not an osteoblast-specific program.
- M11/M24 negative-component rankings from script 42 were non-estimable
  artifacts of all-NaN components and are excluded from cleaned outputs.

No clinical outcome is loaded or reanalyzed.
