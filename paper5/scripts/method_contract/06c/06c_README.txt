Paper 5 06c - frozen TARGET V1 full-data refit contract
================================================================
Script version: 06c-freeze-target-v1-full-data-refit-contract-v1-no-cli

06b selected V1_SHARED_PRIVATE_AE in 17/25 outer splits (68%), satisfying
the frozen >=40% unique-primary rule. All 700 outer benchmark rows passed.

This stage freezes the first all-78 refit protocol BEFORE that refit occurs:
- model: V1_SHARED_PRIVATE_AE
- profile: ARCH_COMPACT__REG_WEAK
- primary feature budgets: RNA 5000 / METHYLATION 10000 / CNV 5000
- all 78 cases, outcome-blind
- primary seed: 1103 (first already-frozen 05a outer seed)
- reproducibility seeds: 2207 and 3301; never allowed to replace primary
- fixed epochs: 400, inherited from the frozen profile maximum_epochs
- no early stopping or epoch selection on the full 78 cases
- no post-fit seed selection
- no SABLE-MO q10=.60/.40 or stability gate applied to V1

The 06b V1 outer selected-epoch distribution is recorded as outcome-blind
consistency evidence only. It does not tune the final epoch count.

06c performs no full-data model fit and reads no clinical outcomes or biological annotations.
