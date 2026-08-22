Paper 5 - 05dev0 synthetic TARGET input fixture freeze
==============================================================================
Script version: 05dev0-freeze-target-input-synthetic-fixtures-v1-no-cli
Fixture version: paper5-target-input-synthetic-fixture-v1
Status: FROZEN_SYNTHETIC_TARGET_INPUT_FIXTURES_BEFORE_TARGET_CODE

WHY THIS EXISTS
---------------
This package is created BEFORE downstream TARGET code so that future loaders,
validators, pairing checks, missing-view logic, and split/seed plumbing can be
written and tested without reading any real TARGET value.

WHY THIS IS NOT TOO EARLY
-------------------------
05dev0 freezes only the INPUT/DEVELOPMENT interface. It does NOT freeze the
final representation method or latent-output schema. 05e has not yet selected
the primary representation method, so any selected-method output fixture must
be created later as a new versioned artifact.

REAL DATA ACCESS
----------------
Real TARGET values accessed: NO
TARGET outcomes accessed: NO
Biological annotations accessed: NO

SCIENTIFIC CHOICES NOT MADE HERE
--------------------------------
- final representation method
- latent dimension
- selected-method output schema
- scientific TARGET nested-CV splits
- scientific TARGET fit seeds
- feature budgets
- outcomes
- biology/program scores

VALID FIXTURES
--------------
paper5\development_fixtures\target_input_synthetic_v1\sample_manifest.csv
paper5\development_fixtures\target_input_synthetic_v1\view_availability.csv
paper5\development_fixtures\target_input_synthetic_v1\rna.npy
paper5\development_fixtures\target_input_synthetic_v1\methylation.npy
paper5\development_fixtures\target_input_synthetic_v1\cnv.npy
paper5\development_fixtures\target_input_synthetic_v1\pairing_map_correct.csv
paper5\development_fixtures\target_input_synthetic_v1\pairing_map_corrupted.csv
paper5\development_fixtures\target_input_synthetic_v1\development_folds.csv
paper5\development_fixtures\target_input_synthetic_v1\development_seed_registry.csv
paper5\development_fixtures\target_input_synthetic_v1\expected_input_schema.json

MALFORMED NEGATIVE-TEST FIXTURES
--------------------------------
paper5\development_fixtures\target_input_synthetic_v1\malformed\duplicate_sample_ids.csv
paper5\development_fixtures\target_input_synthetic_v1\malformed\all_views_missing_sample.csv
paper5\development_fixtures\target_input_synthetic_v1\malformed\mask_matrix_inconsistency.json
paper5\development_fixtures\target_input_synthetic_v1\malformed\pairing_duplicate_target.csv

RULE AFTER 05e
--------------
If the selected representation method needs method-specific expected outputs,
create a NEW versioned fixture artifact after 05e. Never rewrite 05dev0.
