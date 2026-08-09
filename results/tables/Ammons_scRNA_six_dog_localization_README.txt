Ammons six-dog single-cell localization
Script version: 44-recompute-ammons-six-dog-localization-v1

Why this analysis is required
-----------------------------
The official study generated eight raw scRNA-seq samples from six dogs.
Dogs 1 and 2 each had two technical replicate libraries. The Cell Browser
metadata reflects this structure:
- 8 orig.ident library identifiers
- 6 name values

This script combines technical replicate libraries by name before any
biological inference. The earlier eight-orig.ident targeted tests are retained
only for audit comparison and are superseded by this six-dog analysis.

Statistical implication
-----------------------
For six paired biological dogs, the minimum attainable two-sided exact sign-flip
P value is 2 / 2^6 = 0.03125.

Mapping
-------
               orig.ident dog_id
run_count_tumor_no_tx_1_1     N1
run_count_tumor_no_tx_1_2     N1
run_count_tumor_no_tx_2_1     N2
run_count_tumor_no_tx_2_2     N2
  run_count_tumor_no_tx_4     N3
  run_count_tumor_no_tx_5     N4
  run_count_no_tx_tumor_6     N5
  run_count_tumor_no_tx_7     N6

Corrected targeted results
--------------------------
                                                         contrast_name                           score_column         group_column            group_a            group_b                    difference_definition         hypothesis_direction  n_paired_dogs  n_positive_differences  n_negative_differences  n_zero_differences  all_same_nonzero_sign  mean_paired_difference  median_paired_difference  bootstrap_ci_low  bootstrap_ci_high  exact_sign_flip_p  minimum_attainable_two_sided_p                                                               dog_level_differences  estimable  primary_bh_q
                     M34 signed risk: osteoblast lineage versus immune             M34_signed_risk_score_mean analysis_compartment osteoblast_lineage    immune_combined osteoblast_lineage minus immune_combined Higher in osteoblast lineage              6                       6                       0                   0                   True                0.243371                  0.249147          0.216846           0.269451            0.03125                         0.03125 N1:0.26619881;N2:0.23308344;N3:0.26521032;N4:0.20007476;N5:0.29300701;N6:0.20265199       True       0.03125
     M34 negative-loading expression: immune versus osteoblast lineage M34_negative_component_expression_mean analysis_compartment    immune_combined osteoblast_lineage immune_combined minus osteoblast_lineage Higher in immune compartment              6                       6                       0                   0                   True                0.242186                  0.248135          0.214274           0.269155            0.03125                         0.03125 N1:0.26615773;N2:0.23203694;N3:0.26423226;N4:0.19595016;N5:0.29317374;N6:0.20156465       True       0.03125
                M40 signed risk: cycling versus non-cycling osteoblast             M40_signed_risk_score_mean          celltype.l1 Osteoblast_cycling         Osteoblast      Osteoblast_cycling minus Osteoblast Higher in cycling osteoblast              6                       6                       0                   0                   True                0.385975                  0.362595          0.316778           0.463719            0.03125                         0.03125 N1:0.45753183;N2:0.31881679;N3:0.54539045;N4:0.40637395;N5:0.28223609;N6:0.30550122       True       0.03125
M40 positive-loading expression: cycling versus non-cycling osteoblast M40_positive_component_expression_mean          celltype.l1 Osteoblast_cycling         Osteoblast      Osteoblast_cycling minus Osteoblast Higher in cycling osteoblast              6                       6                       0                   0                   True                0.435544                  0.407016          0.354484           0.527101            0.03125                         0.03125 N1:0.51679161;N2:0.35806646;N3:0.62600264;N4:0.45596612;N5:0.31183886;N6:0.34459667       True       0.03125

Guardrails
----------
- Dog is the biological replicate.
- Individual cells and technical libraries are not independent replicates.
- The four targeted contrasts are biologically related and are not four
  independent replications.
- Single-cell localization supports biological interpretation, not causal or
  external prognostic validation.
