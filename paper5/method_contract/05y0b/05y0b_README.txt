Paper 5 05y0b — supplemental blind diagnostic-schema preflight
==================================================================

No failure labels were inspected and no diagnostic comparison was computed.

Realized private-correlation candidate
--------------------------------------
Exact path: scientific_payload.factor_rows[0].best_private_absolute_correlation
Leaf: best_private_absolute_correlation
Scope: ONE_NON_NULL_DIRECT_VALUE_PER_G19_DATASET
Seed reconstruction required: NO

G19 factor-row availability
---------------------------
Datasets: 200
Factor rows: 1000
Retained rows: 77
Non-retained rows: 923

All factor rows have median/q10/inclusion:
  True

All non-retained rows have median/q10/inclusion:
  True

Bidirectional threshold frontier available from sealed payload:
  True

Frozen STANDARD (G07_IND_N62_S1P00)
-------------------------------
Datasets: 100
Factor rows: 500
Retained rows: 451
Non-retained rows: 49

Matched non-retained scenario-signature control schema available:
  True

Next step
---------
Freeze a new 05y1 diagnostic contract before computing any 70-vs-130,
factor-level classifier, anchor calibration, or threshold-frontier result.
