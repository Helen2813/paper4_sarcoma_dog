Paper 4 GSE239948 independent canine evidence lock
Script version: 47-lock-gse239948-independent-canine-evidence-v2

Purpose
-------
Lock the outcome-blind GSE239948 representation results from script 46 v2,
audit possible sample reuse between GSE238110 DOG2 and GSE239948, and append
the external canine evidence to the multidimensional Paper 4 master table.

Inputs
------
External representation manifest: C:\Users\olegk\Desktop\paper4_sarcoma_dog\results\tables\GSE239948_external_representation_manifest.json
Locked multidimensional master: C:\Users\olegk\Desktop\paper4_sarcoma_dog\results\tables\paper4_locked_multidimensional_transport_evidence_with_single_cell_six_dogs.csv
Reference expression: C:\Users\olegk\Desktop\paper4_sarcoma_dog\data\processed\GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv
External expression: C:\Users\olegk\Desktop\paper4_sarcoma_dog\data\processed\canine_validation_GSE239948_expression_log2_symbol.csv

Independence audit
------------------
Exact normalized identifier overlaps: 0
Expression-fingerprint overlap flags: 0
Independence evidence class: no_identifier_or_near_duplicate_expression_overlap_detected

The expression-fingerprint audit uses jointly variable genes outside the four
primary frozen modules. It compares sample-wise rank profiles and cohort-centered
gene-z profiles. Conservative thresholds are intended to flag possible exact or
near-duplicate expression profiles for manual review. The audit cannot prove donor
independence when the two studies use unrelated identifier coding systems.

Locked interpretation
---------------------
- M34: strong external canine representation preservation.
- M40: exceptionally strong external canine representation preservation.
- M11: reliable external score without clear cross-cohort structural preservation.
- M24: no clear preservation evidence; small-module inference remains limited.

Guardrails
----------
- No outcome or treatment-response endpoint is loaded.
- Frozen genes, loadings, signs, and validation tiers are unchanged.
- GSE239948 is a representation-validation cohort, not a prognostic-validation cohort.
- Leave-one-gene-out correlations for large modules are not independent evidence.
- Existing human multiplicity and outcome conclusions are unchanged.
