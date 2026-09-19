Paper 5 confirmatory architecture-screening simulations v2
============================================================================

Script version: 05b-generate-small-n-multiomics-simulations-v2-no-cli
Generator version: sable-mo-simulation-generator-v2

Scientific state
----------------
outcomes generated: false
clinical labels generated: false
representation models fitted: false
stability thresholds retuned: false

Materialized scope
------------------
Architecture-screening scenarios: 36
Architecture-screening datasets: 720
Total bytes: 2291729180
Final gate-calibration datasets: not materialized

Truth retained in every dataset
-------------------------------
- biological shared scores, loadings and view support
- modality-private scores and loadings
- technical batch scores, labels, loadings and view support
- complete matrices before missing-value masking
- intentional patient-pairing row maps
- structured missingness reason codes
- factor-level and subspace-level identifiability flags

Technical metadata note
-----------------------
Frozen parsed source-inventory rows: 16
04d manifest discovery count: 17
Cosmetic source-discovery accounting discrepancy; the frozen parsed source inventory is authoritative for downstream audit provenance.

Important
---------
The earlier sable_mo_v1 datasets are a disclosed developmental pilot and are excluded from all confirmatory comparisons.
The 3,100 final stability-gate calibration datasets remain frozen as a registry only and are generated after architecture freeze.
