Paper 5 05f3b — final-gate scientific aggregation
================================================

05f3b is the first stage that opens and aggregates scientific values from all
3,100 sealed final-gate shards. It performs no model fitting and accesses no
TARGET data, clinical outcomes, or biological annotations.

Core gates
----------
D01: familywise false biological acceptance over the frozen N=1,000 null pool;
PASS iff upper endpoint of the two-sided 95% Wilson interval <= 0.10.
D02: G07 STANDARD pooled matched-and-retained true factors / 400 opportunities;
PASS iff point estimate >= 0.80.
D03: 1-D02 on the same denominator; PASS iff point estimate <= 0.20.

Narrow gates
------------
D04 rotational subspace point estimate >= 0.80.
D05 mixed-support classification accuracy >= 0.80.
D06 batch false biological acceptance <= 0.10.
D07 confirms no post-hoc retuning.

The frozen 05a2 failure-consequence policy determines downstream interpretation.
A failed criterion changes the conclusion, not the criterion.
