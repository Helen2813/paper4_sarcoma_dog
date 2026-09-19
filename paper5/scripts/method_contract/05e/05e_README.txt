Paper 5 - 05e primary proposed representation-method freeze
==============================================================================

Script version: 05e-compare-ablate-and-freeze-method-v1-no-cli
Status: FROZEN_PRIMARY_PROPOSED_REPRESENTATION_METHOD_BEFORE_FINAL_GATE_AND_TARGET
Created UTC: 2026-08-22T17:42:07+00:00

What is frozen
--------------
Primary proposed method: V4_SABLE_MO
Frozen neural profile: ARCH_COMPACT__REG_WEAK
Role: fixed SABLE-MO implementation entering the preregistered final gate and later outcome-blind TARGET benchmark.

What 05e does NOT claim
-----------------------
- no cross-family winner is declared;
- no new cross-family score or threshold is introduced;
- V5 remains secondary exploratory/ineligible;
- 05d4 V4-minus-V3 is not a stability-gate ablation;
- the final gate has not yet passed;
- the final TARGET-selected representation has not yet been chosen.

Confirmatory V4 stability snapshot
---------------------------------
Datasets: 360
Passed: 360
Failed: 0
Latent collapse: 0
Factor-level primary-eligible datasets: 340
Aggregate stability metrics are descriptive; they are not reinterpreted as factor-level hard thresholds.

TARGET policy preserved
-----------------------
The frozen 05a outcome-blind TARGET nested-CV and final-full-data method-selection rules remain unchanged.
05e does not pre-select a TARGET winner and does not use TARGET data.

Secondary comparators
---------------------
05cmp1 froze iCluster2b and MOVE before 05e. No scientific secondary-comparator result was used here.
After 05e they may be run only as secondary post-freeze benchmarking and cannot reopen 05e/final-gate rules.

Next primary scientific stage
-----------------------------
Materialize/evaluate the frozen 3,100-dataset final-gate calibration bank under 05a + 05a1 + 05a2.
If a preregistered gate fails, change the scientific conclusion rather than the threshold.
