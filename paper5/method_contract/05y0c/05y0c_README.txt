Paper 5 05y0c — semantic source preflight
==============================================

No failure labels, 70-vs-130 association, threshold frontier, or model fit is computed.

05y0 candidate
--------------
Path: scientific_payload.factor_rows[0].best_private_absolute_correlation
Structural classification:
  FIRST_FACTOR_PROJECTION_OF_A_FACTOR_LEVEL_FIELD__NOT_A_UNIQUE_DATASET_SCALAR

Dataset-row field present:
  False

Same leaf present on factor rows in first G19 shard:
  5/5

Permitted as the primary realized generator rho:
  False

Why this matters
----------------
A factor_rows[0] value must not be promoted to a dataset-level realized generator
correlation merely because exactly one value is encountered per shard. If the field
exists on every factor row, factor 0 is only an arbitrary projection.

Next step
---------
Use the source-context export to identify the actual generator-level realized
private cross-view correlation or the exact deterministic reconstruction rule.
Do not freeze 05y1 primary realized-rho analysis until that source is unambiguous.
