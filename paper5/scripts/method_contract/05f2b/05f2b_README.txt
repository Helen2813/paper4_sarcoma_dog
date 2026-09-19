Paper 5 05f2b — post-start/pre-unblinding final-gate aggregation freeze
=========================================================================

Script version: 05f2b-freeze-final-gate-aggregation-poststart-preunblinding-v2-runtime-portable-ast-equivalence-no-cli
Status: FROZEN_FINAL_GATE_AGGREGATION_IMPLEMENTATION_POSTSTART_PREUNBLINDING

Purpose
-------
Freeze the exact final-gate aggregation implementation after 05f2 has started
but before any scientific shard has been opened or aggregated. The exact
scientific aggregation core already existed in the pre-existing 05f3 source;
05f2b transparently records this timing deviation rather than calling it a
before-data-generation freeze.

Scientific visibility
---------------------
05f2b reads frozen contracts/registries, 05f2 progress metadata, and 05f3/05f3b source only.
It does not read any final-gate scientific result shard, compute any gate
metric, or evaluate PASS/FAIL.

Execution order
---------------
1. Freeze 05f2b (this post-start/pre-unblinding stage).
2. Run/resume 05f2 until 3100/3100 sealed shards are complete.
3. Run the unchanged SHA256-locked 05f3b aggregation script exactly once.
4. Route downstream work through the already-frozen 05a2 failure policy.

Frozen 05f3b SHA256: fcf1a2e1fec76b0ea15ef93be813851153d763a437c95d46ce1c41d148b9e435
Frozen 05f3b version: 05f3b-aggregate-final-gate-results-cpu-v1-poststart-preunblinding-locked-no-cli
Bootstrap replicates: 10000
Bootstrap master seed: 20260822
