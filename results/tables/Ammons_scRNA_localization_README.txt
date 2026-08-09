Ammons canine osteosarcoma single-cell localization
Script version: 42-score-ammons-single-cell-localization-v1

Data
----
- 32,613 cells from the UCSC Cell Browser "All Cells" leaf.
- Six dog/sample identifiers are expected in orig.ident.
- Three annotation levels: celltype.l1, celltype.l2, celltype.l3.

Scoring
-------
Only frozen primary-program genes are extracted from the compressed matrix.
Each gene is rank-Gaussian transformed across all cells. Frozen risk-oriented
loadings are then applied without outcome information.

For each module:
- signed_risk_score uses signed frozen loadings normalized by their absolute sum;
- positive_component_expression uses positive loadings only;
- negative_component_expression uses the absolute magnitude of negative
  loadings and therefore represents expression of the protective/opposite-sign
  component.

Biological replication
----------------------
Primary summaries use dog x cell type or dog x broad compartment pseudobulks.
Individual cells are never treated as independent biological replicates.

Primary targeted contrasts
--------------------------
1. M34 signed risk: osteoblast lineage versus immune compartment.
2. M34 negative-loading expression: immune versus osteoblast lineage.
3. M40 signed risk: cycling versus non-cycling osteoblast.
4. M40 positive-loading expression: cycling versus non-cycling osteoblast.

The exact sign-flip test uses paired dog-level differences, and BH correction
is applied across the four targeted contrasts.

Important guardrails
--------------------
- M34 has only two detected positive-loading genes; that positive component is
  descriptive and is not a primary test.
- M11 and M24 contain no detected negative-loading genes and remain secondary.
- Single-cell localization is same-species biological annotation, not external
  outcome validation.
- A high M34 score in osteoblast-lineage cells and high negative-component
  expression in immune cells would support an immune-depletion/exclusion
  interpretation, not prove causal immune exclusion.

Score coverage:
module_label  n_frozen_genes  n_detected_genes  coverage_fraction  n_positive_detected  n_negative_detected  signed_score_estimable  positive_component_estimable  negative_component_estimable                                                                          component_guardrail
         M34             162               155           0.956790                    2                  153                    True                          True                          True M34 positive component is descriptive because only two positive-loading genes are available.
         M40             111               107           0.963964                   89                   18                    True                          True                          True                                                                                             
         M11               7                 6           0.857143                    6                    0                    True                          True                         False                                                                                             
         M24               7                 6           0.857143                    6                    0                    True                          True                         False                                                                                             

Targeted tests:
                                                         contrast_name                           score_column      group_column            group_a            group_b                    difference_definition         hypothesis_direction  n_paired_dogs  mean_paired_difference  median_paired_difference  bootstrap_ci_low  bootstrap_ci_high  exact_sign_flip_p                                                   dog_level_differences  estimable  primary_bh_q
                     M34 signed risk: osteoblast lineage versus immune             M34_signed_risk_score_mean broad_compartment osteoblast_lineage    immune_combined osteoblast_lineage minus immune_combined Higher in osteoblast lineage              8                0.244923                  0.250143          0.224688           0.265360           0.007812  0.293007;0.265314;0.266972;0.231081;0.235076;0.26521;0.200075;0.202652       True      0.007812
     M34 negative-loading expression: immune versus osteoblast lineage M34_negative_component_expression_mean broad_compartment    immune_combined osteoblast_lineage immune_combined minus osteoblast_lineage Higher in immune compartment              8                0.243898                  0.249125          0.222732           0.264980           0.007812  0.293174;0.265267;0.266934;0.230047;0.234018;0.264232;0.19595;0.201565       True      0.007812
                M40 signed risk: cycling versus non-cycling osteoblast             M40_signed_risk_score_mean       celltype.l1 Osteoblast_cycling         Osteoblast      Osteoblast_cycling minus Osteoblast Higher in cycling osteoblast              8                0.386538                  0.365832          0.329152           0.450217           0.007812   0.282236;0.464207;0.45069;0.325291;0.312611;0.54539;0.406374;0.305501       True      0.007812
M40 positive-loading expression: cycling versus non-cycling osteoblast M40_positive_component_expression_mean       celltype.l1 Osteoblast_cycling         Osteoblast      Osteoblast_cycling minus Osteoblast Higher in cycling osteoblast              8                0.436017                  0.411561          0.369096           0.510782           0.007812 0.311839;0.522758;0.510466;0.367155;0.349355;0.626003;0.455966;0.344597       True      0.007812
