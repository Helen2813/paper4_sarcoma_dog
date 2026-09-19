Paper 5 05f1d — two-stream full-path validation pilot
======================================================

Purpose
-------
Validate the single-process two-CUDA-stream resample-training execution path
on the same three frozen 05f1 pilot datasets before any 3,100-dataset full run.

Scientific design
-----------------
No scientific design element changes. Seeds, selected epoch, preprocessing,
architecture, optimizer, alignment, stability thresholds and scientific
aggregation remain frozen. Only independent resample neural fits are scheduled
two-at-a-time in one CUDA context. The canonical stability_audit is replayed
using exact cached model states.

Validation criterion
--------------------
For each of the three pilot datasets, the complete scientific payload must hash
exactly to the already-frozen 05f1 payload after removing only wall_seconds and
created_at_utc. Scientific values are not printed or aggregated.

Two-stream numerical implementation hash: 3d7a27fcaf8ff33abb50e74277c3c9553c0d341d15f82d972cb611616a77df21
Mean full-path runtime per dataset: 627.375 s
Rough 3,100-dataset extrapolation: 22.510 days
All-three exact equivalence: True
