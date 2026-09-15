Paper 5 - 05f0 final-gate streaming execution plan
==============================================================================

Script version: 05f0-freeze-final-gate-streaming-execution-plan-v1-no-cli
Status: FROZEN_FINAL_GATE_STREAMING_EXECUTION_PLAN_BEFORE_RUNTIME_PILOT

Correction captured by this stage
---------------------------------
The frozen 05a1 final-gate materialization policy is:
  stream_or_temporary_after_architecture_freeze
Therefore Paper 5 will NOT create a permanent 3,100-NPZ final-gate input bank.
The earlier 05f0 persistent-materialization implementation is withdrawn before any final-gate dataset was created.

Execution model
---------------
Each final-gate dataset is generated deterministically from its frozen registry seed only when needed.
The preferred path is in-memory generation. A temporary NPZ is allowed only as an execution cache,
must not become a frozen scientific input bank, and must be discarded after the validated result shard exists.
Persistent outputs are per-dataset result/provenance shards, not regenerated simulation inputs.

Frozen scope
------------
Final-gate scenarios: 24
Final-gate datasets: 3100
Stability resamples per dataset: 100
Full planned reference+resample fits: 313,100
Scientific thresholds/decisions: unchanged from 05a1 + 05a2
TARGET/outcomes/biology: forbidden

Runtime pilot
-------------
Pilot datasets: first 3 rows in the frozen replicate registry
Pilot full-path fits: 303
Pilot IDs: G01_IND_N50_S0P00|R001|S0269184890, G01_IND_N50_S0P00|R002|S0643022940, G01_IND_N50_S0P00|R003|S2850811377
Pilot console output may contain timing, VRAM and progress only.
No D01/D02/D03, TPR, FNR, FDP, false-acceptance or PASS/FAIL may be printed/aggregated during pilot execution.

Numerical versioning
--------------------
05f1 must freeze its numerical implementation hash immediately before the first pilot fit.
Pilot shards may be reused in the full run only if that hash remains unchanged.
Any numerical-path change invalidates and recomputes pilot shards, per frozen 05a2.
