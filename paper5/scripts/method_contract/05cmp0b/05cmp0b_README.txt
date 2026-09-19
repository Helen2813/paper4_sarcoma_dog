Paper 5 - MOVE GPU synthetic implementation smoke test
==============================================================================
Script version: 05cmp0b-smoke-test-move-gpu-v1-no-cli
Status: PASS_SYNTHETIC_MOVE_GPU_SMOKE_BEFORE_05CMP1

PURPOSE
-------
Verify that the frozen required external deep comparator, MOVE, can be imported,
trained and projected on CUDA using ONLY the frozen 05dev0 synthetic fixture.

WHAT THIS PROVES
----------------
- move-dl 1.5.0 imports under the active isolated environment;
- CUDA is visible and MOVE model parameters/train tensors execute on cuda:0;
- the 24-row synthetic three-view input is accepted after the development-only
  NaN-to-zero MOVE missing-value adapter;
- whole-view missingness and feature-level methylation missingness are exercised;
- training completes with finite diagnostics;
- the latent projection has one finite row per synthetic sample;
- the decoder produces finite continuous output;
- same-seed repeatability is recorded for 05cmp1 design, but is NOT a 05cmp0b
  scientific/fallback decision criterion.

WHAT THIS DOES NOT DO
---------------------
- no real TARGET value is read;
- no outcome, biology or clinical label is read;
- no scientific comparator performance estimate is generated;
- no MOVE hyperparameter, feature-weight, preprocessing or stability policy is
  frozen for scientific analysis;
- no latent array is persisted;
- MOVE cannot influence 05e or the final-gate decision.

DEVELOPMENT-ONLY SMOKE CONFIGURATION
------------------------------------
seed: 1103
batch size: 8
epochs: 12
hidden: [32, 16]
latent dim: 5
beta: 0.01
dropout: 0.1
learning rate: 0.001
continuous weights: [1, 1, 1]

NEXT
----
Complete the iCluster2b synthetic smoke test, then create 05cmp1 to freeze exact
comparator implementations and scientific adapters BEFORE scientific comparator
runs.
