Paper 5 - 05a2 inference and final-gate implementation clarification
==============================================================================

Script version: 05a2-freeze-inference-final-gate-clarification-v1-no-cli
Amendment version: sable-mo-inference-final-gate-clarification-v1
Created UTC: 2026-08-21T21:38:31+00:00

Status
------
FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION
Final-gate datasets materialized at freeze: NO
TARGET representation results available at freeze: NO
Outcomes available at freeze: NO
05d5 state at freeze: IN_PROGRESS
05d5 results used to define this amendment: NO

Honest timing disclosure
------------------------
05a2 is frozen after 05d4 confirmatory ablation and the 05d4a read-only audit were available.
It is prospective with respect to the unmaterialized final-gate bank, future TARGET representation
benchmarking, pairing-permutation results, biology, and outcomes.

Effective clarifications
------------------------
1. RC01 pairing permutations: 100 -> 1,000 for Monte Carlo resolution only.
2. D01: all frozen-selector null datasets are pooled for the primary decision (N=1,000).
3. D01: upper endpoint of a two-sided 95% Wilson score interval, z=1.959963984540054,
   no continuity correction; PASS iff upper endpoint <= 0.10.
4. D01 reporting: point estimate and numerator/denominator first, CI second, PASS/FAIL third;
   independent-null, correlated-private, and per-n breakdowns are mandatory descriptive results.
5. D02/D03: exact STANDARD scenario is G07_IND_N62_S1P00 only.
6. D02: pooled factor-level denominator is 400 true-factor opportunities across 100 datasets.
7. D03: exactly 1-D02 on the same denominator; it is mirrored, not independent evidence.
8. Factor-level 0.70/0.40/0.80 stability thresholds remain point-estimate rules.
9. PASS/FAIL consequences are frozen before results; failures change the conclusion, not thresholds.
10. Runtime pilot shards are reusable only under an unchanged numerical implementation hash.

D02/D03 favorable-direction disclosure
---------------------------------------
The selector clarification excludes mixed-support and sparse/dense loading stress scenarios and therefore
probably makes the STANDARD TPR easier to pass. It also reduces selected datasets from 400 to 100.
This is accepted before final-gate data exist because the frozen decision/target names explicitly said
STANDARD and standard_n62_signal1; stress scenarios remain mandatory descriptive analyses.

Unchanged
---------
Final-gate datasets: 3,100
Final-gate stability resamples per dataset: 100
D01 limit: 0.10
D02 TPR minimum: 0.80
D03 FNR maximum: 0.20
Neural profile/architecture: unchanged
Outcome blinding: unchanged

Downstream rule
---------------
Every final-gate script must read 05a + 05a1 + 05a2 and treat the 05a2 effective registries as
the implementation clarification layer. 05a and 05a1 remain immutable provenance artifacts.
