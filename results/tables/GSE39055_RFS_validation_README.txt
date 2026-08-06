GSE39055 frozen-program recurrence-free-survival validation
Script version: 26-gse39055-rfs-validation-v1

Primary analysis
----------------
- Four frozen primary canine programs: M34, M11, M24, M40
- Strict one-to-one signed-mean human score
- Recurrence-free survival using the GEO recurrence/follow-up time and recurrence event
- Cox HR per SD, fixed-direction C-index, BH FDR across four modules

Zero-time handling
------------------
The primary Cox analysis excludes nonpositive recorded times.
A prespecified sensitivity analysis replaces zero time with one day
(0.033333 months).

Adjustment hierarchy
--------------------
Age and sex are baseline sensitivity covariates.
Human proliferation PC1 is a mechanistic sensitivity adjustment.
Percent necrosis is post-treatment and is not treated as a primary baseline confounder.

Interpretation
--------------
GSE39055 is a small third human cohort. Bootstrap, leave-one-out,
score-variant, and expression-matched random-panel analyses are robustness diagnostics.
No result may be used to change frozen module membership, weights, direction, or tier.
