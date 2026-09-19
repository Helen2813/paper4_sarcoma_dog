Paper 5 05f2 — resumable full final-gate shard generation
==========================================================

Execution path
--------------
05f2 uses the original frozen 05f1 sequential numerical implementation.
The exact-equivalent two-stream alternative was not adopted because the
completed three-dataset full-path runtime pilot was slower. This choice
used runtime information only; no final-gate scientific metric was viewed.

Resumption
----------
Each completed dataset is stored as an atomic sealed shard. The progress
manifest is updated atomically after each completed dataset. If execution is
interrupted, rerun the same 05f2 command. Completed contiguous rows are
hash-verified and skipped. A shard written immediately before interruption
but not yet entered into the progress manifest is recovered automatically.

Scientific blinding
-------------------
05f2 does not aggregate, print or evaluate D01, D02, D03, FDP, TPR, FNR or
final PASS/FAIL. Those values remain sealed in per-dataset shards until 05f3.

Datasets complete: 3100
Optimization runs represented: 313100
Stored shard volume: 2.650 GiB
Scientific contract hash: bd5830f967d58c9e5afa70d183df1d22b581877b27688d52c50de60edc165fc2
Numerical implementation hash: e16deabe0cb733bf4d6a2dbf8b0a977eb7d5a959d7a6331a69d59dda16144b1a
