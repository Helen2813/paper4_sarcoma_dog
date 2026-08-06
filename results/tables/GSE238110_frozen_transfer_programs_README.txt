Frozen canine-to-human osteosarcoma transfer programs

Canine primary endpoint: DFI.
Canine OS is a concordance/sensitivity endpoint.
Program definitions, gene membership, score orientation, and validation tiers are frozen after script 20.
No human outcome may be used to change module membership, gene weights, score direction, or tier assignment.

Primary confirmatory programs:
  M11: angiogenesis_ecm_remodeling_like | primary_clean_non_proliferation
  M24: developmental_neural_signaling_like | primary_clean_non_proliferation
  M34: immune_myeloid_inflammatory_like | primary_clean_non_proliferation
  M40: proliferation_cell_cycle_deviation | primary_proliferation_deviation_axis

Secondary prespecified programs:
  M17: stress_hypoxia_like | secondary_sensitivity
  M25: secondary_program_M25 | secondary_sensitivity
  M28: secondary_program_M28 | secondary_sensitivity
  M38: secondary_program_M38 | secondary_sensitivity

Primary human score: strict one-to-one signed mean z-score using frozen canine loading signs.
Secondary zero-shot score: frozen canine PCA-weighted score.
M40 residualized score is a mechanistic sensitivity analysis, not a replacement for raw M40 or proliferation scores.
TARGET-OS and GSE21257 must be treated as external datasets; cohort-specific preprocessing may not use outcomes.
External validation, not canine repeated-CV p-values, determines translational support.