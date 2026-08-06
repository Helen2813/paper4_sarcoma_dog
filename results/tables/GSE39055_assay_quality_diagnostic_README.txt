GSE39055 assay-quality diagnostic
Script version: 31-gse39055-assay-quality-diagnostic-v2

Purpose
-------
This script audits the FFPE WG-DASL assay layer without changing the frozen
program definitions or the locked primary analyses.

Data used
---------
- GEO sample-level normalized VALUE
- GEO sample-level Detection PVal
- GPL14951 probe-to-gene annotation
- Frozen strict canine-to-human genes and risk-oriented signs

Outcome-blind probe rules
-------------------------
1. Highest-variance probe per unambiguous gene.
2. Best-detected probe per unambiguous gene.
3. Highest-variance probe filtered to Detection PVal < 0.01 in at least 50% of samples.
4. Best-detected probe filtered to Detection PVal < 0.01 in at least 50% of samples.
5. Best-detected probe filtered to Detection PVal < 0.01 in at least 80% of samples.

Interpretation restriction
--------------------------
RFS associations under alternative assay rules are diagnostic sensitivities.
They do not replace script 26, change frozen weights, reverse score direction,
or reopen the locked evidence hierarchy from script 29.

Canonical-gene restriction
--------------------------
MKI67, TOP2A, BIRC5, UBE2C, and EZR are descriptive assay-direction checks.
They are not treated as gold-standard positive controls, and no result is used
to select or orient a module.
