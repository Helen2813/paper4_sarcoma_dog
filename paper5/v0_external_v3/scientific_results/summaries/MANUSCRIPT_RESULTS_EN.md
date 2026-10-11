# V0 external addendum: fixed-procedure Results

This addendum was specified after the original external V1/V4 results, before the new external V0 fits.
Only V0 was trained. V1 results were reused without modification. This compares the entire fixed architecture/objective transition, not an equal-capacity causal effect.
Folds and seeds are not independent external replications. Differences and fold SDs are descriptive; no inference or new selection is performed.

| Cohort | Metric | V0 | V1 (historical) | V1 minus V0 | Folds favor V1 |
|---|---|---:|---:|---:|---:|
| SARC | reconstruction_nmse | 0.747577 | 0.721596 | -0.025982 | 5/5 |
| SARC | omitted_view_nmse | 0.844502 | 0.864307 | +0.019805 | 0/5 |
| SARC | adjusted_cross_view_mrr | 0.091288 | 0.151202 | +0.059914 | 5/5 |
| KIRC | reconstruction_nmse | 0.744681 | 0.712948 | -0.031733 | 5/5 |
| KIRC | omitted_view_nmse | 0.922994 | 0.983446 | +0.060452 | 0/5 |
| KIRC | adjusted_cross_view_mrr | 0.033626 | 0.050143 | +0.016517 | 5/5 |

NMSE: lower is better; adjusted MRR: higher is better. All endpoints/folds are retained, irrespective of direction or collapse flags.
Two-seed subspace agreement is not resampling stability or retention-gate validation. The 400-epoch cap does not establish convergence. No historic claim or manuscript file was automatically changed.
