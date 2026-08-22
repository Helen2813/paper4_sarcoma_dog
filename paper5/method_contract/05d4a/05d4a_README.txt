Paper 5 - 05d4a read-only paired confirmatory ablation audit
==============================================================================

Script version: 05d4a-read-only-paired-ablation-audit-v1-no-cli
Audit version: sable-mo-05d4a-paired-confirmatory-ablation-audit-v1
Created UTC: 2026-08-21T19:51:56+00:00

Purpose
-------
05d4a is a derived READ-ONLY audit of the already frozen 05d4 confirmatory ablation results.
It performs no model fitting, no GPU computation, no TARGET access, no outcome access, and no method selection.

Pairing
-------
All contrasts are computed within the same confirmatory simulation dataset, paired by scenario_id, replicate_index, and simulation_seed.
The primary C1-C3 contrasts therefore use 360 paired dataset-level differences rather than differences between unpaired grand means.

Uncertainty
-----------
Two-sided 95% percentile bootstrap intervals use 5,000 deterministic paired bootstrap resamples.
Bootstrap resampling is performed over confirmatory datasets within the relevant reporting group.
These intervals are descriptive confirmatory uncertainty summaries; 05d4a introduces no new hypothesis-testing gate.

Interpretation guardrails
------------------------
1. C1 tests the descriptive contribution of shared/private decomposition.
2. C2 is interpreted primarily using cross-view retrieval/alignment metrics.
3. C3 is interpreted primarily using missing-view reconstruction/robustness metrics.
4. Monotone improvement in one composite utility is NOT required for every component.
5. V4 training metrics in 05d4 are PRE-GATE. The V4 resampling-based stability component is evaluated in 05d5.
6. V4-minus-V3 must NOT be presented as a stability-gate ablation from 05d4.
7. V5 remains secondary exploratory and cannot be promoted to the final primary method by 05d4a.
8. Frozen scenario_family labels are reported as descriptive strata; no subgroup result changes the frozen method.

Frozen source provenance
------------------------
05d4 lock SHA-256: fdeeaf06846a3f45b451766056f355897c4f2cc375d6702bc4a99fa854aa26df
05d4 result manifest SHA-256: a5faaf5ce6111acd51db9b575d6943d2ee11fdddd281772c86bf5a237effb11a
05d4 dataset-variant table SHA-256: 9249e41112eb4401e5e0e34ab8c7b3ede2440b5b018cba8e14ab93ed10c43a3f

Headline overall paired contrasts
---------------------------------
C1_SHARED_PRIVATE | private_subspace_recovery | n=0 | mean delta=nan | 95% bootstrap CI [nan, nan] | fraction positive=nan
C1_SHARED_PRIVATE | shared_subspace_recovery | n=310 | mean delta=0.0460837 | 95% bootstrap CI [0.0415094, 0.0506285] | fraction positive=0.845
C1_SHARED_PRIVATE | simulation_utility | n=360 | mean delta=0.159467 | 95% bootstrap CI [0.15509, 0.163909] | fraction positive=1.000
C2_CONTRASTIVE | random_adjusted_cross_view_mrr | n=360 | mean delta=0.105387 | 95% bootstrap CI [0.0965813, 0.114349] | fraction positive=0.892
C2_CONTRASTIVE | same_patient_similarity_margin | n=360 | mean delta=0.0834774 | 95% bootstrap CI [0.0768599, 0.0900826] | fraction positive=0.878
C2_CONTRASTIVE | simulation_utility | n=360 | mean delta=0.0445681 | 95% bootstrap CI [0.0411329, 0.0482075] | fraction positive=0.897
C3_DROPOUT_MISSING_VIEW | heldout_reconstruction_score | n=360 | mean delta=0.00385052 | 95% bootstrap CI [0.00253465, 0.00505352] | fraction positive=0.686
C3_DROPOUT_MISSING_VIEW | missing_view_reconstruction_score | n=360 | mean delta=0.0216007 | 95% bootstrap CI [0.0200685, 0.0231708] | fraction positive=0.936
C3_DROPOUT_MISSING_VIEW | simulation_utility | n=360 | mean delta=0.00650137 | 95% bootstrap CI [0.00503727, 0.00802593] | fraction positive=0.667
S1_VAE_VS_V3 | heldout_reconstruction_score | n=360 | mean delta=-0.0154237 | 95% bootstrap CI [-0.0171738, -0.0137278] | fraction positive=0.164
S1_VAE_VS_V3 | missing_view_reconstruction_score | n=360 | mean delta=0.0124237 | 95% bootstrap CI [0.0111262, 0.0137032] | fraction positive=0.825
S1_VAE_VS_V3 | random_adjusted_cross_view_mrr | n=360 | mean delta=0.0385996 | 95% bootstrap CI [0.0338597, 0.04337] | fraction positive=0.806
S1_VAE_VS_V3 | simulation_utility | n=360 | mean delta=0.0025666 | 95% bootstrap CI [0.000884948, 0.00434266] | fraction positive=0.578
S2_VAE_VS_V4_PRE_GATE | heldout_reconstruction_score | n=360 | mean delta=-0.0149984 | 95% bootstrap CI [-0.0167201, -0.0133574] | fraction positive=0.178
S2_VAE_VS_V4_PRE_GATE | missing_view_reconstruction_score | n=360 | mean delta=0.0130405 | 95% bootstrap CI [0.011743, 0.01432] | fraction positive=0.828
S2_VAE_VS_V4_PRE_GATE | random_adjusted_cross_view_mrr | n=360 | mean delta=0.0398199 | 95% bootstrap CI [0.0349598, 0.0447406] | fraction positive=0.806
S2_VAE_VS_V4_PRE_GATE | simulation_utility | n=360 | mean delta=0.00323025 | 95% bootstrap CI [0.00138471, 0.00512577] | fraction positive=0.583

Next methodological use
-----------------------
Use these derived paired summaries to present C1-C3 by hypothesis-matched endpoints and to describe V5 without changing its frozen exploratory status.
05d4a itself does not select or freeze the representation method.
