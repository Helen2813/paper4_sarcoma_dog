# Paper 5: proposed interpretation edits (not applied to the manuscript)

## Status and provenance
This is a post-result descriptive audit of existing outputs, not a new prespecified
experiment. The historical selection, utility weights, models, seeds, thresholds
and frozen result files have not been changed. No new model fitting, p-values,
inferential confidence intervals or cohort pooling were performed.

## Suggested short-paper title (author decision still required)
Latent-Factor Stability Is Not Sufficient: A Simulation Study and Multi-Omics
Applications of SABLE-MO

## TARGET component decomposition
The frozen TARGET utility is U = 0.30 R + 0.30 Q + 0.25 M + 0.15 B.
Reported V4-minus-V1 differences were paired by repeat, outer fold and seed.
For the primary outer description the three original seeds were averaged within
each split; the 25 split means were summarized with equal weights. The separate
1103/2207 table is a post-result common-seed sensitivity, not a replacement.

| TARGET component | Mean V4 minus V1 | Contribution to delta U |
|---|---:|---:|
| `heldout_reconstruction_score` | -0.109115 | -0.032734 |
| `random_adjusted_cross_view_mrr` | +0.022468 | +0.006740 |
| `missing_view_reconstruction_score` | +0.007545 | +0.001886 |
| `modality_balance_score` | +0.024147 | +0.003622 |

The recorded mean outer utility difference is -0.020486.
These contributions explain the arithmetic of the fixed utility; they do not
estimate a causal contribution of any training component. Inner and outer
analyses remain separate. Inner component/replay status: `ALL_1900_CURRENT_SHARDS_SELECTION_REPLAY_PASS`.
The historical one-SE rule uses inner-fold means and its standard-error rule;
that rule is replayed for provenance, not offered as a new inferential test.

## V3 and V4
V3 and V4 share the specified representation-training components. The V4 post-fit
retention layer was not applied during TARGET cross-family selection or during
the limited SARC/KIRC experiments. Numerical row equality is reported separately
from checkpoint/latent identity. If numerical maxima are tied, a label resolved
by complexity is not evidence that the tied method never reached the maximum.
Use `target_v3_v4_equality.csv` and, when available, `target_inner_selection_replay.csv`.

## Cross-cohort interpretation
Describe task- and cohort-context-dependent behavior, not a threshold sample size
at which V4 becomes effective. TARGET reconstruction components are clipped PCA-
space scores, whereas SARC/KIRC use observed molecular-target NMSE. Do not pool
these numbers into a common reconstruction ranking or convert 1-score to NMSE.
SARC/KIRC use cohort-specific refitting, not zero-shot application of TARGET
weights. The TARGET V1 selection remains unchanged.

## Representation versus retention
Retention operating properties were evaluated in simulations. The real-cohort
applications evaluate representations before the retention layer. Agreement of
two seeds is initialization reproducibility in its specified fit/assessment
setting, not subsampling stability, a q10 test or biological factor validation.
A Boolean variant specification alone does not demonstrate execution of a gate.

## Study-method coverage
TARGET: historical block PCA, MOFA K5/K10/K15/K20, and the specified neural models.
SARC and KIRC: only budgeted block PCA K10, V1, and pre-retention V4.
MOVE and iCluster2b: a separate confirmatory simulation branch. Do not say that
all these comparators were evaluated in all three real cohorts.

## Seed-dependent biological alignment
Keep the previously fixed primary seed. Report any existing seed-specific M34/M40
rank changes descriptively without selecting a more favorable seed. A larger
absolute correlation is not automatically stronger matched-random specificity.
This script does not reread biological scores, recompute matched-random tests,
or verify the quoted 06e4 M40/M34 values; insert those from their audited source.
Preselecting a seed prevents favorable-run selection but does not establish stability.

## KIRC chronology
The KIRC protocol was frozen after inspection of SARC results and technical KIRC
QC, but before KIRC model-performance evaluation. It is a fixed-procedure
replication, not an untouched preregistration predating both external studies.

## Training budget and dimension
Report the 400-epoch cap and actual selected epochs. Convergence is not established.
Best and final objectives cannot substitute for a full learning curve or a terminal
slope. Do not extend training solely for this descriptive audit. Fixing the PCA
rank in development documents provenance, not a sensitivity analysis of that rank.

## Code/Data Availability: replacement scope for editing
Paper 5 uses matched RNA expression, methylation and copy-number data from
TARGET-OS, TCGA-SARC and TCGA-KIRC with cohort-specific manifests and preprocessing.
Describe exact code, environment, input manifests and access procedures actually
released. Add only a verified repository URL, release identifier and license.
Do not claim that large matrices or data are in Git unless they are actually
included. Remove any copied Availability statement that identifies this study as
Molecular Transport Audit or identifies canine accessions as its primary multi-omics
cohorts. Canine program references elsewhere may be legitimate and must not be
blindly deleted. Repository/deposition details remain an author-supplied item.

## Submission checklist (manual manuscript edits; no source file was changed)
Check title/abstract scope, unfilled numeric placeholders, figure cross-references,
metric directions and denominators; preserve D05 and correlated-private limitations.
Neither completion of these edits nor a checksum audit guarantees publication.
