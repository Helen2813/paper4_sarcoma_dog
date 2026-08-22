Paper 5 - 05cmp1 secondary comparator implementation freeze
================================================================================
Script version: 05cmp1-freeze-secondary-comparator-implementations-v1-no-cli
Status: FROZEN_SECONDARY_COMPARATOR_IMPLEMENTATIONS_BEFORE_05E_AND_SCIENTIFIC_RUNS

Frozen required external comparators
------------------------------------
Classical: iCluster2b (iClusterPlus 1.47.3 official Windows binary; R 4.6.1; Bioconductor 3.23)
Deep: MOVE (move-dl 1.5.0; PyTorch 2.12.1+cu130; CUDA 13.0)
Fallbacks: iClusterBayes2 and AUTOENCODIX varix remain locked inactive because both required methods passed synthetic smoke tests.

Common scientific representation dimension
-------------------------------------------
Integrated/shared latent dimension: 5

Scientific boundaries
---------------------
- No scientific comparator results were read or generated here.
- No TARGET values, outcomes, biology, or final-gate data were accessed.
- Secondary comparators cannot influence 05e or the final gate.
- Scientific comparator execution remains forbidden until 05e is frozen.

Exact iCluster2b policy
-----------------------
K=5; lasso; tune.iCluster2b; min.lambda=10; max.lambda=500; lambda.iter=25;
EM.iter=25; min.shrinkage.rate=0.05 per view; eps=1e-4; eps2=1e-6.
No performance-based fallback or post-hoc retuning.

Exact MOVE policy
-----------------
Latent=5; beta=0.01; dropout=0.10; lr=1e-3; epochs=300; equal continuous weights.
Candidate hidden architectures: [64,32], [128,64], [256,128].
Architecture selection is outcome-blind: held-out reconstruction first, then latent-subspace stability,
with deterministic simplicity tie-breaking. Canonical scientific seed is fixed prospectively.

Missingness / preprocessing
---------------------------
Use the same frozen outcome-blind Paper 5 inputs. Standardization is fit-split only using observed entries.
After standardization, feature-level missing values and whole missing views are zero-coded. No extra missingness
indicator features are added by either comparator.

Next
----
05e_compare_ablate_and_freeze_method.py
