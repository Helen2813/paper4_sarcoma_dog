External human validation robustness audit
Script version: 24-external-validation-robustness-audit-v1

Purpose
-------
1. Recover finite descriptive logistic odds ratios without requiring statsmodels.
2. Add stratified-bootstrap coefficient intervals and label-permutation AUC tests.
3. Evaluate age and proliferation adjustment in TARGET-OS.
4. Evaluate proliferation adjustment in GSE21257.
5. Quantify leave-one-out stability in both human cohorts.
6. Convert expression-matched random-panel controls to empirical p-values and BH q-values.
7. Produce a frozen evidence summary without changing modules, weights, directions, or tiers.

Interpretation
--------------
The primary prespecified inference remains the strict signed-mean analysis from script 23.
This audit supplies robustness and effect-size diagnostics. It must not be used to revise the
frozen canine programs after seeing human outcomes.
