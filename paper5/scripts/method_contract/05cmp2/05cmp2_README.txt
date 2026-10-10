Paper 5 - 05cmp2 secondary comparator confirmatory benchmark contract
================================================================================

Status: FROZEN_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_BEFORE_SCIENTIFIC_FITS

Scope
-----
Methods: iCluster2b, MOVE
Dataset bank: exact 360 confirmatory simulations used by frozen 05e context
Scenarios: 36
Replicates: 11-20

Common cross-method context
---------------------------
shared_subspace_recovery
factor_alignment_correlation
scenario_robustness_summary
missing_view_robustness

Method-specific
---------------
MOVE latent_subspace_stability: allowed under frozen 3-seed architecture/stability policy
MOVE heldout_reconstruction: descriptive only
iCluster2b latent_subspace_stability: NOT ESTIMABLE in 05cmp2 because 05cmp1 froze only
one deterministic scientific seed and no iCluster2b resampling schedule.

Prohibited
----------
No TARGET comparator run in this contract.
No outcome/clinical/biology access.
No 05e/final-gate/TARGET selection consequence.
No cross-family winner.
No new composite score.
No fallback activation after dataset-specific performance.

Next
----
Build a separately source-authorized scientific comparator runner that consumes this exact contract.
