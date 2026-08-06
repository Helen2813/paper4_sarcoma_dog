Human external validation of frozen canine osteosarcoma programs

Script version: 23-human-external-validation-v1

Primary external score:
- strict one-to-one ortholog signed-mean z-score
- fixed canine risk direction
- no human outcome used for gene selection, weighting, score orientation, or validation-tier revision

Primary external settings:
1. TARGET-OS overall survival: continuous fixed score, Cox HR per SD, and fixed-score Harrell C-index
2. GSE21257 metastasis within five years: continuous fixed score, logistic OR per SD, ROC-AUC, and PR-AUC

Multiplicity:
- BH correction within each primary setting across M34, M11, M24, and M40
- additional global BH correction across all eight primary tests

Sensitivity analyses:
- strict canine-PCA weighted score
- broad mapped score
- human-cohort PC1
- available clinical adjustment
- proliferation adjustment
- M40 residual to disjoint proliferation
- GSE21257 overall-survival association
- expression-matched random gene-set controls

Interpretation:
- External association is not proof of causality or clinical utility.
- GSE21257 is small; metastasis and OS results require replication.
- TARGET-OS has limited sample size and public clinical covariates.
- Random-gene-set controls are descriptive specificity diagnostics.
