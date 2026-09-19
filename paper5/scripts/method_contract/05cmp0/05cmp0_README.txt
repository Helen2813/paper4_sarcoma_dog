Paper 5 - 05cmp0 secondary comparator selection contract
==============================================================================
Script version: 05cmp0-freeze-secondary-comparator-selection-contract-v1-no-cli
Contract version: sable-mo-secondary-comparator-selection-contract-v1
Status: FROZEN_SECONDARY_COMPARATOR_SELECTION_BEFORE_05E_AND_TARGET_RESULTS
Literature search date: 2026-08-21

WHY THIS CONTRACT EXISTS
------------------------
The primary Paper 5 benchmark was frozen before these external comparators and
already contains PCA-by-view, BLOCK_PCA, multiple MOFA configurations, ridge
GCCA, and V0-V5. 05cmp0 adds a small secondary post-freeze benchmark to address
the methods-paper question: does the frozen SABLE-MO result remain informative
next to established published external multi-omics methods?

SELECTED METHODS
----------------
Required classical external comparator: iClusterBayes2
Required deep external comparator: MOVE
Predeclared deep fallback: AUTOENCODIX varix

WHY THESE TWO
-------------
iClusterBayes2:
- established Bayesian latent-variable multi-omics method;
- current official Bioconductor implementation;
- 2026 official tutorial demonstrates an 84-sample GBM multi-omics example,
  close to the TARGET sample-size regime.

MOVE:
- peer-reviewed Multi-Omics Variational autoEncoder;
- Nature Biotechnology publication;
- public PyTorch/PyPI implementation;
- supports continuous multi-modal inputs and latent-space analysis;
- published workflow explicitly uses reconstruction and stability for
  unsupervised model development.

FALLBACK
--------
AUTOENCODIX varix is frozen only as the MOVE fallback. It cannot replace MOVE
because of a favorable result; it activates only if MOVE fails the predefined
implementation smoke-test policy.

IMPORTANT TIMING
----------------
05d5 state when 05cmp0 was frozen: IN_PROGRESS
05d5 result used to select comparators: NO
05e result available: NO
TARGET representation result available: NO
TARGET outcome accessed: NO

ALLOWED BEFORE 05E
------------------
Installation work and smoke tests on 05dev0 synthetic fixtures only.

FORBIDDEN BEFORE 05E / 05CMP1
-----------------------------
- scientific comparator simulation results;
- scientific comparator TARGET results;
- comparator-driven change to SABLE-MO;
- outcome/biology-guided comparator tuning.

NEXT COMPARATOR STAGE
---------------------
After installation/smoke tests, create 05cmp1 to freeze exact versions,
environments, adapters, hyperparameter-selection rules, seeds, and comparable
metrics BEFORE the first scientific comparator run.
