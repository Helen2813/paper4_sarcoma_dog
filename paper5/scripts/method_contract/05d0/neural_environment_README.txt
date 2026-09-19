Paper 5 neural runtime environment lock
========================================================================

Script version: 05d0-freeze-pytorch-neural-environment-v1-no-cli

Purpose
-------
Freeze the PyTorch/CUDA execution environment before any confirmatory neural fit.

Runtime policy
--------------
CUDA device 0 only.
float32 only.
Automatic mixed precision disabled.
torch.compile disabled.
TF32 disabled.
torch.use_deterministic_algorithms(True, warn_only=False).
cuDNN benchmark disabled.
cuDNN deterministic enabled.
CUBLAS_WORKSPACE_CONFIG=:4096:8
Primary DataLoader workers = 0.

Scientific state
----------------
Neural simulation model fitting performed: NO
Architecture selected: NO
Final gate-calibration data used: NO
Outcomes accessed: NO

Next stage
----------
05d1_freeze_neural_simulation_protocol.py
