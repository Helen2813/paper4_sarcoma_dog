Paper 5 05x0 — read-only retention-ablation preflight
=========================================================

Script version
--------------
05x0-retention-ablation-readonly-preflight-v1-no-cli

Purpose
-------
Check whether the already-sealed 05f2 factor payloads contain enough separately
stored stability and modality-support information to define a no-refit,
post-unblinding exploratory ablation of the SABLE-MO retention rule.

This stage computes NO ablation result and fits NO model.

Verified upstream artifacts
---------------------------
05f3e lock: paper5/method_contract/05f3e/final_gate_results_lock.json
05f4b lock present: True
05f2 progress manifest: paper5/method_contract/05f2/full_run_progress_manifest.csv

Scan
----
Datasets: 3100
Factor rows: 15500
Distinct factor fields: 66

Interpretation rule
-------------------
05x0 does not infer or invent an ablation formula from field names. The exported
source context must establish the exact frozen retention logic. Only if that logic
can be reconstructed unambiguously from fields already stored in the sealed
payload should a separately versioned 05x1 exploratory contract be written.

If refitting is required, do not use 05x0 to justify it; proceed to the frozen
TARGET branch and treat a refit-based ablation as later exploratory work.
