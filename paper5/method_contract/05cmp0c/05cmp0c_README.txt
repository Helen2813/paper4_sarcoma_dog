Paper 5 - iCluster2b CPU synthetic implementation smoke test
==============================================================================
Script version: 05cmp0c-smoke-test-icluster2b-cpu-v4-cache-safe-windows-binary-no-cli
Status: PASS_SYNTHETIC_ICLUSTER2B_CPU_SMOKE_BEFORE_05CMP1

PURPOSE
-------
Verify that the frozen required external classical comparator, iCluster2b, can
be loaded from the isolated R/Bioconductor environment and fit to ONLY the
frozen 05dev0 synthetic three-view fixture.

EXPECTED IMPLEMENTATION
-----------------------
R: R version 4.6.1*
Bioconductor: 3.23
iClusterPlus Windows binary: 1.47.3
Bioconductor source release listed upstream: 1.48.0
Package policy: official platform-native Windows binary; no source compilation for this smoke stage.
method: iCluster2b
same-package fallback: iClusterBayes2 (NOT activated by a PASS smoke test)

WHAT THIS PROVES
----------------
- the isolated Paper 5 R library is used;
- iCluster2b and tune.iCluster2b expose the expected Bioconductor 3.23 API;
- the 24-row RNA/methylation/CNV synthetic fixture is accepted after a
  development-only NaN-to-zero complete-matrix adapter;
- iCluster2b returns finite 24 x 3 meanZ and finite coefficient matrices
  with the expected feature-by-factor dimensions;
- a second independent R process with the same development seed provides a
  repeatability diagnostic for 05cmp1 design.

WHAT THIS DOES NOT DO
---------------------
- no real TARGET value is read;
- no outcome, biology or clinical label is read;
- no scientific comparator performance estimate is generated;
- tune.iCluster2b is NOT used to select a scientific lambda here;
- no scientific K, lambda, tuning, missing-view, preprocessing or stability
  policy is frozen here;
- no meanZ or beta numeric array is persisted;
- iCluster2b cannot influence 05e or the final-gate decision;
- iClusterBayes2 fallback is not activated automatically by this script.

DEVELOPMENT-ONLY SMOKE CONFIGURATION
------------------------------------
seed: 1103
K: 3
lambda per view: [10.0, 10.0, 10.0]
method per view: ['lasso', 'lasso', 'lasso']
EM.iter: 10
eps: 0.0001
eps2: 1e-06

NEXT
----
Create 05cmp1 and prospectively freeze exact comparator implementations and
scientific adapter/tuning rules before any scientific secondary-comparator run.
