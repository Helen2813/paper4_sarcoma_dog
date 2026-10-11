# Paper 5 follow-up: scope and use

This is post-result descriptive sensitivity and replay, not a new confirmatory
experiment. Historical TARGET primary selection remains unchanged. No models
were fitted and no old source, result, threshold or authorization was rewritten.

## Weight sensitivity
The global grid uses step 0.05 on the complete nonnegative four-weight simplex.
The local grid uses step 0.01 within +/-0.05 of each historical weight, with sum
one. The local box is an explicitly chosen arithmetic neighborhood, not a claim
about plausible scientific preferences. All ten frozen candidates are retained.
At each weight, two seeds are averaged within each inner fold, then four inner
folds supply the original mean/SE heuristic and complexity-resolved one-SE choice.
Outer values never select a weight or model. Grid counts are not probabilities.
The continuous L1 margins concern V4/V1 mean equality only, not one-SE selection.
Integer labels in selected/best tables use model_codebook.csv. Co-max tables use
bitmasks (bit j identifies model j), so V3/V4 ties remain explicit.
A hypothetical choice is not a new authorized primary representation. No outer
performance of a post-hoc selected weight is presented as unbiased validation.

## Completed 06e4 replay
The 78-case full-data V1 representations and existing four-program score table
are used to reproduce only the completed 06e4 statistics. The complete 5-value
canonical spectra and all four programs/all three seeds are shown. Association
maxima can be reached on different axes: repeated max-|rho| magnitudes do not
establish identity of a factor. A narrow range can also arise around zero.
Primary matched-null support stays at seed 1103; it is not copied to other seeds.
No new seed-specific p/q, program selection or program-to-subspace biology
statistic is computed. Procrustes residual is unscaled, with the original
reference-norm denominator; it is not a pure rotation-invariant angle measure.
Full-data TARGET diagnostics are not merged with held-out SARC/KIRC diagnostics.
Three seed pairs from the same cases are not independent replications. No
association of subspace agreement with biological reproducibility is tested.

## Interpretation
TARGET reconstruction scores are clipped PCA-space scores, unlike SARC/KIRC
molecular-target NMSE. Do not put them on a single error scale or infer an
isolated sample-size effect. Manuscript and README edits require author review.
