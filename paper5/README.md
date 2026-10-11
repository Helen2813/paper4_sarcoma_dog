# SABLE-MO — Paper 5

**Latent-Factor Stability Is Not Sufficient: SABLE-MO for Prespecified Factor Retention in Small-Sample Multi-Omics**  
*Manuscript authors: Elena Krikun and Abedalrhman Alkhateeb.*

Research workflow for learning and evaluating shared and modality-private representations of matched **RNA expression, DNA methylation, and gene-level copy number**. The study separates three questions: whether a representation is useful, whether a factor is reproducible, and whether the available evidence supports its biological interpretation.

SABLE-MO combines a shared/private representation engine with a **separate post-fit factor-retention procedure**. Its evaluation includes controlled simulations, an outcome-blind TARGET-OS application, and limited fixed-procedure applications in TCGA-SARC and TCGA-KIRC.

> **Interpretation boundary:** a successful run, strong reconstruction, or agreement between random seeds is not a certificate of biological validity. The external SARC/KIRC experiments evaluate the **V4 representation before retention**, not the complete SABLE-MO retention gate. This is research software, not a clinical decision tool.

[Study design](#study-design) · [Data](#data-and-cohorts) · [Results](#results) · [Reproduction](#reproducing-and-inspecting-the-work) · [Limitations](#limitations) · [Citation](#citation-and-licensing)

## Study design

The project keeps the following stages distinct:

```text
Development simulations → neural-profile selection
                         ↓
Held-back component ablation + separate resampling assessment
                         ↓
Frozen final-gate evaluation → documented targeted recalibration
                         ↓
Outcome-blind TARGET-OS representation selection and application
                         ↓
Limited fixed-procedure applications: TCGA-SARC, then TCGA-KIRC
```

Secondary simulation comparators are a separate analysis branch. They do not reopen the historical method freeze, final gate, or TARGET selection.

The confirmatory simulation comparison uses **360 datasets: 36 scenarios × 10 held-back replicates**. Where linear/MOFA tables contain both development and confirmatory records, replicates **11–20** identify this confirmatory subset; replicates 1–10 must not be silently mixed into it. The historical final gate and the subsequent 400-dataset recalibration bank are separate collections with their own denominators and decisions.

### Representation and retention

The representation engine uses modality-specific encoders, shared and private latent components, block-balanced reconstruction, shared alignment, and—where enabled—paired contrastive alignment, modality dropout, and missing-view reconstruction. The patient-level shared embedding is the mean of the available per-view shared embeddings. When a modality is omitted, it is excluded from that mean and its private decoder input is zeroed.

Post-fit retention is a different operation: refit the preprocessing and model under resampling, align latent spaces, match factors, and apply the applicable frozen rules for aligned correlations, inclusion frequency, factor variance, and modality support. Rotational-subspace recovery and exact modality-support attribution have separate interpretations; neither establishes the biological identity of a named factor by itself.

| Identifier | Role in the study |
|---|---|
| `V0_BALANCED_AE` | Block-balanced autoencoder component baseline |
| `V1_SHARED_PRIVATE_AE` | Shared/private decomposition; historically selected TARGET representation |
| `V2_CONTRASTIVE_SPAE` | Adds paired contrastive alignment |
| `V3_DROPOUT_SPAE` | Adds modality dropout and missing-view reconstruction |
| `V4_SABLE_MO` | Same representation engine/objective as V3, with a separate post-fit retention layer where that layer is explicitly executed |
| `V5_SABLE_VAE` | Exploratory variational extension; not eligible for retrospective primary promotion |

**V4 versus V3 before retention is not an ablation of the retention gate.** The external experiments use only V1, pre-retention V4, and `BLOCK_PCA_BUDGETED_K10`. MOFA, MOVE, and iCluster2b belong to other specified comparison stages; they were not added to these external three-model experiments.

## Data and cohorts

| Cohort | Matched cases in the documented analysis | Role |
|---|---:|---|
| TARGET-OS | 78 | Primary outcome-blind representation-selection/application cohort |
| TCGA-SARC | 250 | Limited external fixed-procedure application |
| TCGA-KIRC | 304 | Limited external fixed-procedure application outside sarcomas |

RNA, methylation, and CNV are **three modalities within each cohort**, not three independent patient cohorts. The cohorts are not pooled into one training dataset or one cross-cohort performance average. SARC/KIRC models are fitted within their respective cohorts; these are not zero-shot transfers of TARGET weights.

### External input construction

The SARC/KIRC acquisition manifests identify exact GDC file UUIDs, checksums, workflows, and patient/sample/portion relationships. The selected workflows are **STAR – Counts**, **SeSAMe HM450 methylation**, and **ASCAT3 gene-level CNV**. HM27 is not substituted for HM450, and other CNV workflows are not substituted case by case.

The documented inputs use:

| Modality | Selected channel | Fixed transformation | Features before fold-local filtering |
|---|---|---|---:|
| RNA | `tpm_unstranded` | `log2(TPM + 1)` | 60,660 |
| Methylation | `beta_value`, `cg` probes only | `log2(beta / (1 - beta))` after the frozen clipping operation | 482,421 |
| CNV | `copy_number` | `log2(max(copy_number, 10^-6) / 2)` | 60,623 |

The exact clipping constants, dtypes, and arithmetic order are defined in the input-preparation source. Missing entries remain missing until the applicable training-only imputation step. Gene identifiers retain their versions; repeated gene symbols are not used to merge rows. `min_copy_number` and `max_copy_number` do not replace `copy_number`.

Data-dependent feature filtering and preprocessing are learned **inside each training split**. The external feature budgets are 5,000 RNA, 10,000 methylation, and 5,000 CNV features. Training missingness limits are 0 for RNA/CNV and 0.20 for methylation; variance ranking precedes z-standardization. Imputation uses training medians, and scaling uses training means and sample standard deviations. Neural PCA is fitted within the relevant training subset, with 10 components per modality in the compact profile.

Exact duplicate CNV fingerprints and constant profiles are recorded as QC findings, not proof of duplicated patients. No additional patient exclusions were made during the limited model experiments.

The workflow acquires data through GDC manifests. Large molecular files, generated arrays, checkpoints, and local environments are **not assumed to be included in a Git checkout**. Inspect the release contents and its data manifests rather than assuming that every generated directory below is version-controlled. A new unrestricted API search is not an exact replacement for a frozen acquisition manifest.

## Results

### Completed external fixed-procedure applications

Each cohort uses five fixed outer folds, two neural seeds (`1103`, `2207`), and one PCA fit per fold: **25 model evaluations per cohort**. Each neural evaluation has an epoch-selection fit followed by same-seed reinitialization and refitting, giving 40 neural optimization stages per cohort.

| Cohort / folds | Preliminary fit | Epoch-selection holdout | Final outer-training refit | Test |
|---|---:|---:|---:|---:|
| SARC, folds 1–5 | 160 | 40 | 200 | 50 |
| KIRC, folds 1–4 | 194 | 49 | 243 | 61 |
| KIRC, fold 5 | 195 | 49 | 244 | 60 |

Preprocessing is fitted first on the preliminary-fit cases for epoch selection, then independently refitted on the complete outer-training set for final training. Test cases are not used to select features, preprocessing parameters, or epochs.

The neural profile is `ARCH_COMPACT__REG_WEAK`: shared dimension 5, private dimension 2 per view, hidden dimension 32, and maximum 400 epochs with patience 30 and `min_delta = 0.0001`. The PCA baseline uses K=10 and the same feature budgets, but is **not an equal-capacity model**. It also differs from the historical all-eligible-feature TARGET block PCA.

Means below first average neural seeds within each fold and then average the five folds equally. Smaller NMSE is better; larger adjusted MRR is better.

| Cohort | Model | Reconstruction NMSE ↓ | Omitted-view NMSE ↓ | Adjusted cross-view MRR ↑ |
|---|---|---:|---:|---:|
| TCGA-SARC | Budgeted block PCA, K=10 | 0.5845 | 0.7867 | 0.1820 |
| TCGA-SARC | V1: shared/private AE | 0.7216 | 0.8643 | 0.1512 |
| TCGA-SARC | V4: pre-retention representation | 0.7322 | 0.8192 | 0.2113 |
| TCGA-KIRC | Budgeted block PCA, K=10 | 0.4391 | 0.8941 | 0.0706 |
| TCGA-KIRC | V1: shared/private AE | 0.7129 | 0.9834 | 0.0501 |
| TCGA-KIRC | V4: pre-retention representation | 0.7270 | 0.8888 | 0.1198 |

The table is drawn from the completed cohort-specific `scientific_results/summaries/model_summary.csv` records and their result-audit replays; the paths and audit entry points are listed below.

**What these results support.** In both external applications, V4 has higher adjusted retrieval and lower omitted-view NMSE than V1 in every fold, but higher ordinary reconstruction NMSE in every fold. The additional V4 components therefore exhibit a consistent task-dependent trade-off relative to V1 in these experiments. The V1–V4 comparison changes several components together and does not isolate one component's causal contribution.

**What they do not support.** PCA has lower ordinary reconstruction error than both neural procedures in both cohorts. For omitted-view reconstruction, PCA is better than V4 on SARC, whereas V4 has only a small average advantage on KIRC (V4 minus PCA: −0.005305; four of five folds favor V4). Retrieval favors V4 over PCA in four of five SARC folds and all five KIRC folds. These are descriptive comparisons, not a universal ranking or a claim of statistical significance.

Same-fold two-seed subspace agreement is 0.6920 for V1 and 0.7523 for V4 on SARC; on KIRC it is 0.5653 and 0.7909, respectively. Neural held-out collapse flags are 0/10 for each model in each external cohort. These findings do not replace resampling stability, biological-truth evaluation, or retention validation.

All 20 epoch-selection runs in each cohort reached the 400-epoch cap. SARC selected epoch 400 in 19 runs and 399 in one; KIRC selected 400 in 16 runs, 399 in two, 397 in one, and 396 in one. **The experiments are training-budget-limited; numerical convergence is not established.**

### Metric definitions for the external experiments

**Reconstruction NMSE** is computed on the same training-selected, standardized molecular-feature targets for all three models within each fold. Neural predictions are inverse-projected from per-view PCA coordinates. Squared errors are summed only over observed test entries, divided by their number, and normalized by the centered observed-entry training variance with a floor of `1e-12`. The three modality-specific NMSEs receive equal weight.

**Omitted-view NMSE** uses the same target space, observation rules, and training normalizer while excluding the target modality from the prediction inputs. Missing target values are not replaced with imputed values for scoring.

**Cross-view retrieval** searches for the corresponding patient across six ordered modality pairs. For `n` test candidates, the recorded chance-adjusted score is `clip((MRR - H_n/n) / (1 - H_n/n), 0, 1)`, where `H_n` is the nth harmonic number. It uses 50 candidates on SARC and the actual 60/61 on KIRC. Adjusted MRR is not diagnostic accuracy, and the adjustment does not make differently sized retrieval tasks identical.

Training-reference NMSE is **not R²** and is not a directly evaluated test-set mean-predictor comparison. Fold standard deviations are descriptive dispersion, not confidence intervals. Folds share training observations; folds and seeds are not independent cohorts.

### Historical core results remain separate

The documented TARGET-OS outcome-blind selection selected **`V1_SHARED_PRIVATE_AE`**, not V4. External results do not revise that decision or transfer the SABLE-MO retention thresholds to V1. Downstream TARGET outputs must be interpreted through their own contracts; the limited SARC/KIRC reports do not certify completion of every later TARGET analysis.

The secondary MOVE/iCluster2b confirmatory run completed **720 records: 360 datasets × two methods**, with zero recorded dataset execution failures. Its results are context for the specified endpoints, not a declaration of an overall winning method.

<details>
<summary><strong>Historical final gate and targeted recalibration</strong></summary>

The historical final-gate route was `CORE_PASS`, with narrower claims limited by their own tests:

| Endpoint | Recorded result | Interpretation |
|---|---|---|
| D01: pooled dataset-level false biological acceptance | 70/1000 = 7.00%; 95% Wilson upper endpoint 8.75% | Passed the frozen pooled criterion; does not establish uniform control over every null mechanism |
| D02/D03: matched-and-retained true factors / complementary false-negative rate | 371/400 = 92.75%; 29/400 = 7.25% | One recovery family with a shared denominator, not two independent confirmations |
| D04: rotational-subspace recovery | 0.829 | Passed its own ≥0.80 criterion; not validation of individual axis identities |
| D05: exact modality support conditional on truth matching and retention | 81/120 = 67.5% | **Failed** the ≥0.80 criterion; support labels remain exploratory |
| D06: false biological acceptance per technical-factor opportunity | 0/400 | Result in the specified simulated technical-confound conditions, not a general real-data guarantee |

The pooled D01 result contains a critical subgroup difference: **0/800 independent-null datasets versus 70/200 correlated-private datasets (35%)**. The pooled pass must not conceal the correlated-private failure mode.

A subsequent single-threshold recalibration changed the lower-decile aligned-correlation requirement from `q10 = 0.40` to `0.60` and was fixed before evaluation on a newly seeded validation bank. Within that fresh bank, correlated-private false acceptance changed from 66/200 (33%) to 38/200 (19%); standard true-factor recovery changed from 755/800 (94.375%) to 752/800 (94.0%). The remaining 19% is a limitation, not a solved failure mode. These are independently seeded simulations under the existing generative/numerical design, not an independent laboratory replication.

This targeted validation does not replace the historical final gate or repair D05. The `q10` rule is engine- and stage-specific; it was not applied in the limited SARC/KIRC experiments and is not automatically transferred to TARGET-selected V1.

Source records: `paper5/method_contract/05f3e/final_gate_operating_metric_summary.csv`, `final_gate_mandatory_reporting.csv`, and `final_gate_results_lock.json`; recalibration: `paper5/method_contract/05y6/prospective_q10_validation_result.json` and `05y6_results_lock.json`.

</details>

## Repository and workspace layout

Paths below follow the recorded working-tree organization. Stage outputs are generated directories; this map is not a claim that all data are shipped in the repository.

```text
workspace-root/
├── README.md
├── paper5/
│   ├── scripts/                  # Core methods, simulations, TARGET, comparators
│   ├── method_contract/          # Protocols, source authorizations, result locks
│   ├── data_manifest/            # Input inventories and data provenance
│   ├── model_inputs/             # Prepared core inputs; large local artifacts
│   ├── simulations/              # Generated simulation banks
│   ├── results/                  # Core scientific outputs
│   └── comparator_envs/          # Local environments; not source-code payloads
├── paper5_parallel_sarc/
│   ├── sarc5.py                  # Exact numerical helper reused by KIRC
│   ├── acquisition_v1/
│   ├── pointwise_inputs_v1/
│   ├── limited_model_v1/         # Protocol, splits, authorization, scientific_results
│   └── manuscript_assets_v1/
└── paper5_parallel_kirc/
    ├── acquisition_v1/
    ├── schema_qc_v1/
    ├── pointwise_inputs_v1/
    ├── limited_model_v1/         # Protocol, splits, authorization, scientific_results
    └── manuscript_assets_v1/
```

Some original standalone launchers were run from Windows Downloads and archived as source copies by their stage. Their location is not interchangeable with the location of imported numerical helpers. In particular, the KIRC workflow binds `paper5_parallel_sarc/sarc5.py` and specific `paper5/scripts/` sources. A standalone Paper 5 release must preserve these dependencies or document an explicit relocation; simply renaming or moving pinned helpers breaks the recorded source identity.

## Reproducing and inspecting the work

### Start with the completed evidence

For each external cohort, inspect:

```text
limited_model_v1/fixed_protocol.json
limited_model_v1/split_registry.json
limited_model_v1/synthetic_preflight_lock.json
limited_model_v1/authorization_v1/
limited_model_v1/scientific_results/completion_lock.json
limited_model_v1/scientific_results/summaries/model_summary.csv
limited_model_v1/scientific_results/summaries/summary.json
```

These paths are relative to the appropriate `paper5_parallel_sarc/` or `paper5_parallel_kirc/` directory. The immutable per-evaluation results and receipts are under `scientific_results/shards/`. Post-result audits and manuscript tables are separate outputs under `manuscript_assets_v1/`.

For example, from the workspace root, this command reads the existing KIRC summary without fitting a model:

```powershell
Import-Csv -LiteralPath ".\paper5_parallel_kirc\limited_model_v1\scientific_results\summaries\model_summary.csv" | Format-Table model_id,reconstruction_nmse_mean,omitted_view_nmse_mean,adjusted_cross_view_mrr_mean -AutoSize
```

The core anchors include `paper5/method_contract/05e/`, the historical gate in `05f3e/`, targeted recalibration in `05y6/`, and the TARGET result lock in `06b/`. The completed secondary-comparator lock is under `paper5/method_contract/05cmp4_v7_batch_rebalance/`.

### Stage order for the external applications

| Purpose | SARC entry points | KIRC entry points |
|---|---|---|
| Metadata inventory and acquisition | `sarc0.py`, `sarc1.py` | `kirc0.py`, `kirc1.py` |
| Technical QC and fixed inputs | `sarc2.py`, `sarc3.py`, `sarc4.py` | `kirc2.py`, `kirc3.py` |
| Fixed protocol and artificial model preflight | `sarc5.py` | `kirc4.py` |
| Separate exact-source authorization | `sarc6a.py` | `kirc5a.py` |
| Real fixed-procedure experiment | `sarc6.py` | `kirc5.py` |
| Result audit and manuscript assets | `sarc7.py` | `kirc6.py` |

This is a dependency map, **not a “run every script” command sequence**. The supplied runners are snapshot-bound research scripts rather than a general-purpose command-line package. Check the exact entry point, working-directory requirement, input manifest, source hashes, and authorization state before execution. Several stages have no CLI arguments and deliberately refuse replacement of existing artifacts. A complete clean-checkout installer is not established by the supplied evidence.

Use the existing successful preflight and runtime records for the intended stage. An authorization is a local reproducibility control over an exact source/configuration, not an ethical approval, data-access permission, or biological validation. Do not edit expected hashes, delete failed records, or change seeds to force a pass. Resume only where the unchanged executor explicitly supports it; an unresolved technical-stop marker requires review.

### Recorded execution environment

The successful external runs recorded **Windows, Python 3.10.0, NumPy 1.26.4, pandas 2.3.3, PyTorch 2.12.1+cu130, CUDA build 13.0, and an NVIDIA GeForce RTX 4080 Laptop GPU**. AMP, TF32, and `torch.compile` were disabled; the recorded CPU thread limit was two.

These are **observed run properties**, not a complete installation specification or a guarantee of support on other systems. Numerical helpers also use SciPy, whose version is not supplied in the external execution record. MOFA and the R/iClusterPlus comparator require their own recorded dependencies. The neural executors do not silently fall back to CPU. Do not update a locked environment in place to run an old snapshot.

### Provenance and audit limits

Data UUIDs, file sizes, checksums, ordered patient IDs, splits, source identities, preprocessing digests, evaluation receipts, and completion locks are retained separately from manuscript summaries. Exact-byte source hashes include formatting and line endings; Git or editor normalization can therefore invalidate a historical source pin. Preserve the recorded source copies and verify them after checkout rather than rewriting the pins.

The external result audits reproduce saved error arithmetic, chance adjustments, aggregation, split alignment, and diagnostics from small stored test-latent matrices. They **do not re-run the molecular pipeline or trained models**. Raw MRR is not reconstructed from per-view embeddings because those embeddings are absent from the compact evidence. Checksums show agreement with a recorded snapshot, not proof that the experiment is scientifically correct.

## Limitations

**Different experiments answer different questions.** Historical neural recovery uses held-out fold/seed evaluations, while several linear/factor/comparator recovery endpoints use full-data fits. Some historical reconstruction scores also differ in prediction space and normalization. The common external molecular-target metrics do not retroactively harmonize those older numbers. No pooled cross-family leaderboard is implied.

**Representation is not retention.** No external biological-truth endpoint, full resampling retention test, survival analysis, or `q10` validation was performed in the limited SARC/KIRC applications. Their positive findings cannot repair the historical D05 or correlated-private limitations. Two seeds do not establish robustness to patient resampling.

**Execution and diagnostics are distinct.** Neural collapse flags can coexist with successful execution. Historical `05d4` V4 evaluations contained 135 flagged fold–seed evaluations out of 3,600; these were not removed from the frozen summaries. The external zero-flag results must not be generalized to all project analyses.

**Chronology matters.** KIRC was a limited replication of the SARC procedure specified after core/SARC results were available and before KIRC model fitting. It is not an untouched preregistration predating both external applications. Source locks and later audit exports also have distinct timestamps; a post-result audit is not a pre-result protocol.

**Known comparator amendments are preserved.** The completed comparator run uses the separately authorized `v7_batch_rebalance` results rather than replacing failed v6 records. A documented v7 export issue affected the signed numeric `simulation_seed` field in 352 method–dataset records (176 datasets). Join by the frozen dataset identifier/registry and retain the separate seed-correction annotation; do not silently rewrite the frozen outputs. This is a provenance caveat, not permission to alter model fits.

## Citation and licensing

The accompanying working manuscript is:

> Elena Krikun and Abedalrhman Alkhateeb. **Latent-Factor Stability Is Not Sufficient: SABLE-MO for Prespecified Factor Retention in Small-Sample Multi-Omics.**

Cite the manuscript version and exact software commit or archived release used. No publication DOI, accepted venue, or final bibliographic year is asserted by this README. Dataset and comparator acknowledgments should follow the manuscript and the actual source manifests.

A software license is not established in the supplied project records and is not assigned by this README. Include the repository's approved licensing terms with the released code; do not infer a license from this documentation.
