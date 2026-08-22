Paper 5 - 05cmp0a classical comparator continuous-data amendment
==============================================================================
Script version: 05cmp0a-amend-classical-comparator-continuous-data-path-v1-no-cli
Status: FROZEN_COMPARATOR_METHOD_CORRECTION_BEFORE_SCIENTIFIC_RUNS
Literature check date: 2026-08-21

CORRECTION
----------
05cmp0 selected iClusterBayes2 as the required classical external comparator.

Current official iClusterPlus documentation recommends:
- iCluster2b / tune.iCluster2b for continuous multi-omics;
- iClusterPlus2 / iClusterBayes2 for mixed data types.

Paper 5 RNA, methylation and CNV analysis matrices are continuous-valued.
Therefore the effective required classical comparator is now iCluster2b.

FALLBACK
--------
iClusterBayes2 is retained only as a same-package fallback if iCluster2b fails
the frozen synthetic installation/smoke-test policy.

UNCHANGED
---------
MOVE remains the required deep external comparator.
AUTOENCODIX varix remains the MOVE-only deep fallback.
Secondary comparators remain ineligible to influence 05e or final-gate rules.

PROSPECTIVE STATUS
------------------
Scientific comparator results viewed: NO
TARGET comparator results viewed: NO
TARGET outcomes accessed: NO
05d5 results used to make this correction: NO

NEXT
----
Install/smoke-test iCluster2b and MOVE on 05dev0 synthetic fixtures only.
Then freeze exact versions and adapters in 05cmp1 before scientific runs.
