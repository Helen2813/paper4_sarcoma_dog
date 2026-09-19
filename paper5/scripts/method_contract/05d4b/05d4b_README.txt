Paper 5 - 05d4b read-only final-gate decision-design audit
==============================================================================

Script version: 05d4b-read-only-final-gate-design-audit-v1-no-cli
Audit version: sable-mo-final-gate-design-audit-v1
Created UTC: 2026-08-21T21:07:38+00:00

Scientific state
----------------
GPU computation: NO
Model fitting: NO
Final-gate simulation data accessed: NO
TARGET accessed: NO
Outcomes accessed: NO
Threshold retuning: NO

Purpose
-------
This audit reads only frozen 05a/05a1/05b-v2 registries and contracts.
It evaluates whether the already-frozen final-gate decision rules are operationally unambiguous and how much resolution they have before any final-gate simulation dataset is materialized.

D01
----
Current frozen selector resolves to 5 scenarios / 1000 datasets.
Independent null component: 800 datasets.
Correlated-private stress component: 200 datasets.
Frozen event formula: probability of retaining at least one false biological shared factor in a null dataset
Frozen CI text: Wilson 95% binomial interval
The CI text does not specify sidedness, z, or continuity correction; this is an implementation ambiguity that must be frozen before final-gate materialization.
The design audit therefore reports both common interpretations and uses two-sided 95% Wilson only for prospective power/reference calculations, not as a rewritten frozen rule.

D02 / D03
-----------
Current selector matches: G07_IND_N62_S1P00, G18_MIXED_SUPPORT_N62_S1P00, G23_SPARSE_LOADING_N62_S1P00, G24_DENSE_LOADING_N62_S1P00
Exact STANDARD independent candidate matches: G07_IND_N62_S1P00
Current matching datasets: 400
Exact STANDARD matching datasets: 100
Current matching true-factor opportunities: 1600
Exact STANDARD true-factor opportunities: 400
D03 is an algebraic mirror of D02 on the same selector: True
TPR formula is factor-based: True
Cross-dataset aggregation unit explicit in frozen formula: False
The exact D02/D03 aggregation denominator must therefore be frozen in 05a2 before results.

Pairing permutation resolution
------------------------------
Frozen RC01 pairing permutations: 100
Frozen multiplicity adjustment: Benjamini-Hochberg
A prospective 100 -> 1,000 resolution amendment is recommended without changing the null construction or multiplicity family.

Reference design calculation
----------------------------
Under a two-sided 95% Wilson score interval with no continuity correction:
N=200 permits at most 11 false-acceptance datasets while keeping the upper CI <= 0.10.
N=1000 permits at most 81 false-acceptance datasets while keeping the upper CI <= 0.10.
Minimum N for at least 80% probability of passing D01 when the true rate is exactly 8% is 1617 under this reference implementation.
This is a design-resolution diagnostic, not a recommendation to alter the frozen 3,100 datasets.

Next step
---------
Use this audit plus 05d4a to write a narrow 05a2 inference/final-gate implementation clarification amendment before any final-gate data are materialized.
