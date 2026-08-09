GSE239948 blind de novo rediscovery audit
=========================================

Script version
--------------
49-gse239948-blind-de-novo-rediscovery-v2

Purpose
-------
Test whether de novo co-expression modules formed in GSE239948 recover frozen canine programs without using frozen membership during module discovery.

Discovery procedure
-------------------
- Only GSE239948 expression is used during feature selection and clustering.
- The top 3000 genes by GSE239948 variance define the discovery universe.
- Gene-gene Spearman correlation is clustered by average linkage.
- Genes with zero or near-zero variance inside a sample subsample are excluded from that subsample clustering and retained as unassigned genes.
- The full-cohort correlation cut corresponds to a one-sided nominal positive-correlation P value of 0.01; the observed full-cohort r cut was 0.3536.
- Clusters smaller than 5 genes are not treated as discovered modules.
- Stability is evaluated across 30 outcome-blind 80% sample subsamples.

Frozen-program matching
-----------------------
Frozen M34, M11, M24, and M40 gene sets are loaded only after de novo modules have been constructed.
Each frozen program is compared with every discovered module, and the best F1 overlap is retained.
Expression-variance-matched random panels also take their maximum match across all discovered modules, so the empirical null includes the best-of-many-module search.
BH correction is applied across the four frozen primary programs.

Interpretation guardrails
-------------------------
- This is a representation rediscovery analysis, not an outcome validation analysis.
- Variable-gene filtering is outcome-blind but may reduce coverage of frozen programs.
- A significant best-match result supports independent rediscovery of a related co-expression structure; it does not imply identical module boundaries.
- The cohort has 43 samples, so module boundaries and small-module results require caution.
- No human or canine outcome is loaded or used.
