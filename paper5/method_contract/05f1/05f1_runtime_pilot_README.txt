Paper 5 - 05f1 final-gate GPU runtime pilot
============================================================================

Script version: 05f1-run-final-gate-runtime-pilot-gpu-v2-seed-collision-aware-no-cli
Numerical implementation hash: e16deabe0cb733bf4d6a2dbf8b0a977eb7d5a959d7a6331a69d59dda16144b1a

Scope
-----
The pilot executes the first three dataset IDs frozen by 05f0.
Each dataset traverses the complete final-gate path: one full-data V4 reference fit
plus all 100 frozen 80% stability resample fits.
Final-gate simulation inputs are generated deterministically in memory and not persisted as an input bank.

Scientific visibility
---------------------
Per-dataset scientific payloads are sealed in reusable result shards but are not printed or aggregated by 05f1.
No D01/D02/D03, TPR, FNR, FDP, familywise false acceptance, subgroup result, or scientific PASS/FAIL is evaluated here.

Numerical freeze
----------------
The exact generator, neural helper, linear/PCA helper, fixed-epoch bridge, full 310,000-row
stability seed registry, profile, thresholds, dtype/runtime settings, and local final-gate numerical wrappers
are hash-frozen before the first pilot fit.
The frozen 32-bit stable_seed derivation is preserved. Seed integers must be unique within each dataset's
100 resamples; equal seed integers across different datasets are allowed and audit-recorded as expected
birthday collisions in a 2**32-1 seed space.
Pilot shards are reusable only under the unchanged numerical implementation hash.

Runtime-only summary
--------------------
Pilot datasets: 3
Pilot optimization runs: 303
Mean wall seconds/dataset: 588.416
Peak CUDA allocation: 68.0 MiB
Rough full-run extrapolation: 21.11 days

Next
----
If runtime is acceptable, run the full final-gate streaming evaluation using this exact numerical implementation hash.
If runtime-only optimization is needed, create a new versioned numerical path and recompute all pilot shards.
