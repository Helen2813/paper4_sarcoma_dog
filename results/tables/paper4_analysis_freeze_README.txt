Paper 4 locked cross-species evidence synthesis
Script version: 29-lock-cross-species-evidence-v1

Purpose
-------
This script performs no new feature selection, score orientation, model fitting,
or hypothesis testing. It freezes the interpretation of the human validation
and conservative structure-preservation analyses.

Evidence layers
---------------
1. TARGET-OS overall survival.
2. GSE21257 metastasis within five years.
3. GSE39055 recurrence-free survival.
4. Conservative canine-human representation preservation from script 28.

Important restriction
---------------------
The three human endpoints are scientifically related but not identical.
They are summarized by triangulation, not pooled into a formal meta-analysis.

Manuscript hierarchy
--------------------
- Script 23 provides the prespecified primary TARGET-OS and GSE21257 analyses.
- Script 24 provides robustness diagnostics.
- Script 26 provides the third-cohort GSE39055 RFS analysis.
- Script 28 provides manuscript-ready conservative representation classes.
- Script 29 locks the final descriptive evidence hierarchy.
