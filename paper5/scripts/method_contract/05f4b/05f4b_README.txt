Paper 5 05f4 — manuscript asset export from locked 05f3e
================================================================

Script version
--------------
05f4b-export-final-gate-manuscript-assets-v2-read-only-plot-tolerance-no-cli

Scope
-----
05f4b is a read-only packaging stage with a plotting-only floating-point tolerance recovery. It verifies the frozen 05f3e results lock
and every artifact covered by that lock, checks JSON/CSV consistency, and exports
manuscript-facing tables, figures, and LaTeX macros.

05f4b performs:
- no GPU computation;
- no model fitting;
- no threshold or selector changes;
- no denominator changes;
- no reweighting of the frozen D01 null bank;
- no post-hoc replacement of the non-estimable D01 logistic trend;
- no new D05 chance model.

Key locked observations copied for manuscript use
-------------------------------------------------
D01 pooled false acceptance: 7.00%
D01 independent-null false acceptance: 0.00%
D01 correlated-private false acceptance: 35.00%
D02 standard sensitivity: 92.75%
D03 standard FNR: 7.25%
D04 rotational-subspace recovery: 0.828986
D05 view-support accuracy: 67.50% (frozen threshold 80%; FAIL)
D06 batch false biological acceptance: 0.00%

D01 sample-size trend
---------------------
Manuscript status: NON_ESTIMABLE_ZERO_EVENTS
Events: 0/800
The raw prespecified model record remains preserved in 05f3e for audit, but 05f4
does not export its OR≈1 or huge odds-scale Wald interval as a manuscript result
when the model is non-estimable due to zero events.

D05 chance baseline
-------------------
05f4b intentionally does not invent a post-unblinding chance model. If a D05 chance
baseline is desired, define its estimand first and compute it in a separately
versioned exploratory script from frozen mixed-support truth labels.

Recovery audit
--------------
postunblinding_recovery_audit.csv records the observed 05f3b→05f3e recovery chain.
Execution-note text is explicitly labeled as a post-hoc audit note from the
observed terminal logs; script hashes are computed directly from the current files.

Authoritative source
--------------------
paper5/method_contract/05f3e/final_gate_results_lock.json
SHA256: 6d691237662d633a4f8d8294126a33b32c5694372a6e58de71bd45731d67c397
