#!/usr/bin/env python3
"""
Paper 5 - 05cmp3 run post-05e scientific secondary-comparator confirmatory benchmark.

TWO-STAGE / SOURCE-AUTHORIZED / RESUMABLE

Before a 05cmp3c source-authorization lock exists:
    - ONLY a synthetic preauthorization self-test may run.
    - No frozen 360-dataset simulation NPZ is opened.
    - No scientific comparator result is produced.

After a separate 05cmp3c lock authorizes this exact runner SHA:
    - run iCluster2b and MOVE on the exact 360 confirmatory simulation datasets
      frozen by 05cmp2;
    - write sealed method-by-dataset shards;
    - retry a caught per-dataset method failure once with the identical frozen
      numerical configuration and identical deterministic seed(s);
    - never activate iClusterBayes2/AUTOENCODIX based on scientific performance;
    - truth is used only after a fit is complete for eligible simulation metrics;
    - no TARGET/outcome/clinical/biology access;
    - no influence on 05e, final gate, or TARGET model selection.

The runner is intentionally no-CLI so the authorized SHA has exactly one
scientific execution path.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import random
import subprocess
import tempfile
import time
import traceback
import stat
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


SCRIPT_VERSION = (
    '05cmp3-run-secondary-comparator-confirmatory-benchmark-v7-singleton-tail-rebalance-preserved-v6-history-isolated-results-no-cli'
)

RUNNER_FILENAME = "05cmp3_run_secondary_comparator_confirmatory_benchmark_v7.py"
AMENDMENT_REVIEWED_INPUTS = {'paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v5.py': '802c131dfda75effb4e60dd38ebf05512fa5da6c1b3eec97c0c499b3c4e869f5', 'paper5/method_contract/05cmp3a/secondary_comparator_runner_source_authorization_lock.json': '5dbb075b3467fa865cb2b0570a79c2419e952a2c8c001eb04287d8ade5fba319', 'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v5.json': '7ec4e687151a4af67b718a1ee13e521109a0071be91de2e751e9d0bb02b96b94', 'paper5/scripts/05b_generate_small_n_multiomics_simulations_v2.py': 'b4adb1fcb32b65a998ee70e1b72d52e13490194c7ad8e5ddd802b95bf05a28db', 'paper5/scripts/05c1_fit_linear_simulation_baselines.py': 'c2c91b4ad274f78ccc369e6fca765102285cf3d9c8191f3c33f4a450b7426367', 'paper5/scripts/05cmp3a_freeze_secondary_comparator_runner_v5_source_authorization_v1.py': '526bf30c101dea38cb26d07c9acd625892febebcd697fb466be0c4c1fe653bfa'}


# Post-failure history is retained, not erased or relabelled as a pre-fit freeze.
BN_EVIDENCE_SHA256 = 'b782b74dd0bfd1a58a2a6b999b303e75a42fb5c99bbdb50c510658e1d7536819'
BN_AMENDMENT = {'historical_access_disclosure': {'R_called': False,
                                  'authorization_created': False,
                                  'candidate_loader_substituted_in_memory_for_synthetic_probe_only': True,
                                  'existing_sources_results_locks_modified': False,
                                  'frozen_npz_or_truth_opened': False,
                                  'installed_MOVE_source_bytes_copied_for_review': True,
                                  'old_numeric_metrics_compared_or_exported': False,
                                  'old_shard_JSON_bytes_read_and_parsed': True,
                                  'old_shard_hash_status_and_errors_inspected': True,
                                  'runner_imported_under_non_main_name': True,
                                  'scientific_fit_performed': False},
 'historical_shards': {'other_files': [],
                       'records': [{'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R11__S1956851480',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R11__S1956851480.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '9711723037ff3c88a5e76a05442c704a782eed20c7499029e8d097afc312d25d',
                                    'size_bytes': 8025,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R12__S2151775064',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R12__S2151775064.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '3051f6a1910454527f12c2baad7712c61dd7c86885c95daea8c483bd0fc1d7a4',
                                    'size_bytes': 8027,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R13__S2503617495',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R13__S2503617495.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'a5fba24fb9553687eec7e2172c0b25240147246c90adc1f8a4502b5a923e9620',
                                    'size_bytes': 8027,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R14__S248501019',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R14__S248501019.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'dbeceb9c3f00621857bb9cc5a73035b2b7959481a67d2c14234279a9c026389e',
                                    'size_bytes': 8023,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R15__S2956704822',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R15__S2956704822.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '9e14fdaed2ebb1a4737f69894d8425d6e417a6a123f18b97b90955d72fe25005',
                                    'size_bytes': 8027,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R16__S4220044469',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R16__S4220044469.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '1637bf5e57be83ba35aa71225ceceda8a579aeb771fd2b5942f2294e37721e11',
                                    'size_bytes': 8025,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R17__S3787896917',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R17__S3787896917.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '75d933ca479f69e806d8c22906614800fab8f2deef769469c99e356263c29425',
                                    'size_bytes': 8026,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R18__S1834030575',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R18__S1834030575.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '4322e8c88dc5d44febeac770424ba59e759fe3f1fd9abb124a5e624eaa072e45',
                                    'size_bytes': 8026,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R19__S676714901',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R19__S676714901.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '637ba7a6d611cc2c5c7b673f81e97c5d764f09563734aaaf2fe802a7e2f6c632',
                                    'size_bytes': 8023,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A01_N62_STANDARD__R20__S3775451817',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A01_N62_STANDARD__R20__S3775451817.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'd102f954d091b167603880f199fdc10357b58e63f2b932042d9fc807a7f9685b',
                                    'size_bytes': 8026,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R11__S736203244',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R11__S736203244.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '94c6b085dcd091b2f8ec1fc0770c832f8289ae3a278c2b06ece353ba536c60ba',
                                    'size_bytes': 8032,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R12__S825591582',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R12__S825591582.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'fd8718d5259b33f48beff7ade222e4d8ed1640a9a5537a48016f6e5577c449cd',
                                    'size_bytes': 8033,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R13__S2035530975',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R13__S2035530975.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'c37dc9ceb314d7eb6738fb24eaf1e24f6a5673e0d5a41b54aeccfd1f4c73e179',
                                    'size_bytes': 8035,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R14__S2194705193',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R14__S2194705193.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'f44c5377bd0e9e46a31bf20563a270cf695aa0ffa4d81bd22d333e60441d9287',
                                    'size_bytes': 8037,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R15__S2774419216',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R15__S2774419216.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'eae6203d000f57c0d9ea763ba14dd94b020efdeeef3db2b3b78ca9bca22bc3e2',
                                    'size_bytes': 8037,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R16__S3430676483',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R16__S3430676483.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '53affe1700ba6ddf70124c8b165a1df1ce08575e85cf6e26a006e0dc599fba5f',
                                    'size_bytes': 8036,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R17__S3398246703',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R17__S3398246703.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '71b6ce3871333936c0da06fa3e40064b36e6eadb1a5d0d3f5115e30ab31d1713',
                                    'size_bytes': 8036,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R18__S3480902876',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R18__S3480902876.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '79fcdfb050bfbd6f27bc53a79d988a4e0c4d9c6c95d5d91f113ad416a5b879ff',
                                    'size_bytes': 8036,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R19__S3600454974',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R19__S3600454974.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '90720ccf45f57311832717dc63f6db61648a1d9db35ececf9ee30a614cc7c394',
                                    'size_bytes': 8036,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A02_SHARED_RHO05__R20__S1881944730',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A02_SHARED_RHO05__R20__S1881944730.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'bc6b4eaf770ad8263c85fc33920643b9c030d5f628ccccb48d06f22edbef64e3',
                                    'size_bytes': 8036,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R11__S2231306999',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R11__S2231306999.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'd73261b5bfee7ff764e6c275f705f0a9856f848c24e3913e51920610f9ed6f01',
                                    'size_bytes': 8030,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R12__S1500210888',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R12__S1500210888.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'f2e2cebf2ebaac95f703d0aa31e9d278e87755ab50184bf753fbd9a02dbe374b',
                                    'size_bytes': 8028,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R13__S596820742',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R13__S596820742.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '949e5ebb3baf46088dfc348ead45a56e2ab9d4e4275391e2cbf9d3bb13f9bbd5',
                                    'size_bytes': 8027,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R14__S1692752220',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R14__S1692752220.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '18d07b97d038b284d877790e02d86f29f6f64f4c2d5f44e104d2979e26f03889',
                                    'size_bytes': 8028,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R15__S3780506108',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R15__S3780506108.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '4a46b4bb1c97f78555250d3b3f880d48f010a7b089b61b960f7019289ba34022',
                                    'size_bytes': 8028,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R16__S2233887688',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R16__S2233887688.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'db5455ce0add7214db6cb785d3634e208215cfc55dbeba4587b1375dbd94f7ae',
                                    'size_bytes': 8030,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R17__S2046008660',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R17__S2046008660.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '3edaeb6a5dd9cd57b2885c820d4ef156d8b113369685b0e063a4e18917cfaa0f',
                                    'size_bytes': 8029,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R18__S1306969488',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R18__S1306969488.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '69a98cdcaca836a8dde0d39741f8d957b2aa68b3d9a041aec8119c24137788e0',
                                    'size_bytes': 8027,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R19__S4025994634',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R19__S4025994634.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '6fcd43f6b12d6471862b44bc5ae089c986793d5e8ca6504a3d2af2fec3971a75',
                                    'size_bytes': 8029,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R20__S4284512387',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A03_ROTATIONAL_RHO08__R20__S4284512387.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'e63c764abafa021c83d643afffc96354a4781bc168506270e011b4e4c4036b0b',
                                    'size_bytes': 8028,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R11__S3837701889',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R11__S3837701889.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'e016443a84c4ad2eec483e98743b5a368175f5da93cfa23753c20a613df160a8',
                                    'size_bytes': 8039,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R12__S3798729190',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R12__S3798729190.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'd74df4e74694e88a126152d50c5752db6f0ea2b5643146a0ae9e6405107d0b28',
                                    'size_bytes': 8039,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R13__S1328920472',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R13__S1328920472.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '5eeb37e4964681057c0f6398b784e411281d1c10c7b720e721e951f61c36faf5',
                                    'size_bytes': 8039,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R14__S2229041992',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R14__S2229041992.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '61e5db523d889b2740f259670bb26384f4a36aafd9cbf3025462f0ea35adbb22',
                                    'size_bytes': 8040,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R15__S633780796',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R15__S633780796.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'fc9acfe6e3bc4c5cb6d9f4b43e30adb6f1b6c3777b315fc73a6e45c476709e0e',
                                    'size_bytes': 8037,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R16__S414669577',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R16__S414669577.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'e05c42a88b680576b9023bfdc9bd86d754f20933b70ec8de77ced374371e2acf',
                                    'size_bytes': 8037,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R17__S3123056078',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R17__S3123056078.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '97f3b0f542649fb68b111cb5dc96a23b6e00fda52a9cd0f1613372ffa4a60654',
                                    'size_bytes': 8040,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R18__S2625537233',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R18__S2625537233.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'a15a3cb98073a9b7f777c2972b1f624e0cec9d95f68f915545d48306da0f5267',
                                    'size_bytes': 8040,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R19__S1486566318',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R19__S1486566318.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': 'db2c887a32b9f9a9dfd3c64da8ea357eca4c5c47c58a65cb55e38387fecc073e',
                                    'size_bytes': 8038,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R20__S1003318659',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A04_PARTIAL_RNA_METH__R20__S1003318659.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '8a799d50c47bcfb12166c81bc312750727992bf62507fa5a99dd569d87d0a29e',
                                    'size_bytes': 8039,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': None,
                                    'dataset_id': 'A05_PARTIAL_RNA_CNV__R11__S1811950922',
                                    'first_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'method': 'MOVE',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/MOVE/A05_PARTIAL_RNA_CNV__R11__S1811950922.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'second_error_class': 'SINGLETON_BATCHNORM_DURING_MOVE_SELECTION',
                                    'sha256': '16bcde95b636254eca17dc3862ada4e4fc290de7961bfb6606a0e76c133b239e',
                                    'size_bytes': 8036,
                                    'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R11__S1956851480',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R11__S1956851480.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '63136999957249f8d4a41a5e4b694f777710bee6afa1877ca28be98895a91efc',
                                    'size_bytes': 3622,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R12__S2151775064',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R12__S2151775064.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '608adf6d335ee0418438383f6bd06a021093965a7797ac0e51bedffd570dc357',
                                    'size_bytes': 3623,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R13__S2503617495',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R13__S2503617495.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '5f18229253e78870c161249f6cd84db3bda60285d76fc86d912dd16540ee3ee0',
                                    'size_bytes': 3621,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R14__S248501019',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R14__S248501019.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '619f22e7f38895b880b7687d361ed62d2d7f961840198d546d77886fdd278e8b',
                                    'size_bytes': 3620,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R15__S2956704822',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R15__S2956704822.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '3b2e4f2d122678f2e81f4dccb1a6c3fcf21c39f69f0356158500da1ea0f8a888',
                                    'size_bytes': 3622,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R16__S4220044469',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R16__S4220044469.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '1e91bd6612500308d8a114eabaea382a712e5834d58c0b93f694f7d038cebb9d',
                                    'size_bytes': 3617,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R17__S3787896917',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R17__S3787896917.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'a2c6e10ae42eaa360753eea6f03afc3877599212e25a9b6717800fc4531db185',
                                    'size_bytes': 3622,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R18__S1834030575',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R18__S1834030575.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '5534e1f53a89225f09f512db2f0bfbec51973cb8b7c1b0f5a5e9a925e8fa93f1',
                                    'size_bytes': 3622,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R19__S676714901',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R19__S676714901.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '3b5c71cb73c605ecccf783a888a583462720a4e14bc22a4020040955c512381b',
                                    'size_bytes': 3618,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A01_N62_STANDARD__R20__S3775451817',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A01_N62_STANDARD__R20__S3775451817.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '6ea298ba63496f5ad80a11ace3e96811293a0dd9787929bcf25ed5b3931cf615',
                                    'size_bytes': 3623,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R11__S736203244',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R11__S736203244.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'b78d0924cb210d92d51d16c357311667f658d088dde9e33203c748065c1480da',
                                    'size_bytes': 3625,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R12__S825591582',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R12__S825591582.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '79d5fd7d3bfed17a6361d638d992f3553fe913e698815628648b1e00ca3df11c',
                                    'size_bytes': 3627,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R13__S2035530975',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R13__S2035530975.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'd017dfcb7e89230722d5792744fd46ba8c2e9c91db1f09181e27cd645cf4e644',
                                    'size_bytes': 3630,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R14__S2194705193',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R14__S2194705193.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '0abbe38d2d65800473e7e9627061aab55d4be8963ec88601625526588c5d8d08',
                                    'size_bytes': 3630,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R15__S2774419216',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R15__S2774419216.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'b3e97a831d6bd9d24ee79f3d563695a012802f8ddd0c3091312733effeced08b',
                                    'size_bytes': 3633,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R16__S3430676483',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R16__S3430676483.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'c7c1bfe355d2607838424fef183b1a3f101b232b41912a9180f71a79aa82f270',
                                    'size_bytes': 3634,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R17__S3398246703',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R17__S3398246703.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '1217a0b98c6a22d96441ea22e5d6857ea35e94b6a0f2ba49d2903c74934943c8',
                                    'size_bytes': 3632,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R18__S3480902876',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R18__S3480902876.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '0bbe00e4bdbbfb17d86fb06d033f7af8b3af98cc27df8f71e22509700e245989',
                                    'size_bytes': 3633,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R19__S3600454974',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R19__S3600454974.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '3a1d53c6559d5f1d47f30353c8176f098266330f04afe4f67720501eb2fadbe6',
                                    'size_bytes': 3632,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A02_SHARED_RHO05__R20__S1881944730',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A02_SHARED_RHO05__R20__S1881944730.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '17d76c6ab4650493be79eedf21634ba0151dc3c87186f306e40ec2bc6005942b',
                                    'size_bytes': 3631,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R11__S2231306999',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R11__S2231306999.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '84ecfc6f5b0a8f8c17787cb24bb4422826b6555b76ef087632afadafe452cfe3',
                                    'size_bytes': 2885,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R12__S1500210888',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R12__S1500210888.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'd89c7af39206b09361f2f6a7d55b1620c4f63a712b4f2f4638ac7c5b49e53e2c',
                                    'size_bytes': 2884,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R13__S596820742',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R13__S596820742.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '16603a38b4664af727d99e4a67e56470bd8deffb6f0e9ca5e8f3a4a741c5c0a8',
                                    'size_bytes': 2883,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R14__S1692752220',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R14__S1692752220.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'e2b495bbf753bd001202bf0bfbceb64641a6557058a4ea08fabd9621ffd0f8bd',
                                    'size_bytes': 2884,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R15__S3780506108',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R15__S3780506108.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'b20f78e57816379b7c8bd8bc2ac13ce6e83b86c4d2f4b74a43908881fb85ebc8',
                                    'size_bytes': 2884,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R16__S2233887688',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R16__S2233887688.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'bbef8e6189dea037c68d2d357ddf3c6a2bf5211338e03bc851c12e48ff5b5359',
                                    'size_bytes': 2884,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R17__S2046008660',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R17__S2046008660.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'ee517b1e017a952ce31150100d1861c7a1b11dfd8886357a2c8842b61a760741',
                                    'size_bytes': 2884,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R18__S1306969488',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R18__S1306969488.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'b6aceb7740f6d142c75b76c08d0b970a618989090b3a1a23d516830bbb76b0a6',
                                    'size_bytes': 2883,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R19__S4025994634',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R19__S4025994634.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '64b113ffc36d489a25dea2597b8d39846849885421c9b9d8496e153e0b82df50',
                                    'size_bytes': 2883,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A03_ROTATIONAL_RHO08__R20__S4284512387',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A03_ROTATIONAL_RHO08__R20__S4284512387.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '743c5862ccfad3ca55aa0a9ac4763fca7223ed4e35d8116a02a00cd468b83249',
                                    'size_bytes': 2882,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R11__S3837701889',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R11__S3837701889.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '53ba8628a914c64ef14c48af4f9c5ebbd0cc005fa4579a311b29bb4ce8d0632b',
                                    'size_bytes': 3255,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R12__S3798729190',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R12__S3798729190.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '3d567794d3408ccf9228235e1c20d8ce247ecac622fcdb713f37ec6588dc2729',
                                    'size_bytes': 3257,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R13__S1328920472',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R13__S1328920472.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '07348619b946a0c4c453be0ae2b564ae5d872508e37a89c1162277116200f8dc',
                                    'size_bytes': 3260,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R14__S2229041992',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R14__S2229041992.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '54e98d1a2edb4f72a4ff33dcb1932d2ae4364a5823328b7793ae5be8cea3c456',
                                    'size_bytes': 3260,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R15__S633780796',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R15__S633780796.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'ce0fc89b6e944425c5392f1227281eb95accec8bf0b76752e18115011cc34ad6',
                                    'size_bytes': 3256,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R16__S414669577',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R16__S414669577.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': '7b8f919ff085bd12a22b528df68ad2dcf1c94f5b746a169fc70a567aaa6107f7',
                                    'size_bytes': 3256,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R17__S3123056078',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R17__S3123056078.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'cc06e06c398474b1cbdd545d9b95f708291b80278fdb53a0c91e6bcf71e08d58',
                                    'size_bytes': 3261,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R18__S2625537233',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R18__S2625537233.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'e94d4f482d8eba83e91dbc9acdd2dbdbca647860b224b8e5a8f4c3e84c3384bc',
                                    'size_bytes': 3262,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R19__S1486566318',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R19__S1486566318.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'ed251f6bf6ac577809e0eb2bb15f97c4fe0417ff3f7fdff061e8ac6cdf42a604',
                                    'size_bytes': 3259,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A04_PARTIAL_RNA_METH__R20__S1003318659',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A04_PARTIAL_RNA_METH__R20__S1003318659.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'b15562e5331e04ac28c45e9dd19a67f8b933a4d3d5912ad113072cf5d1ebc4f3',
                                    'size_bytes': 3261,
                                    'status': 'PASS'},
                                   {'attempt_count': 1,
                                    'dataset_id': 'A05_PARTIAL_RNA_CNV__R11__S1811950922',
                                    'method': 'iCluster2b',
                                    'path': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1/shards/iCluster2b/A05_PARTIAL_RNA_CNV__R11__S1811950922.json',
                                    'runner_sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                    'sha256': 'e41525b9c8260629bfc34143686bbcb364d29a298726e647a6c5c5aef0109eec',
                                    'size_bytes': 3258,
                                    'status': 'PASS'}],
                       'status_counts': [{'count': 41,
                                          'method': 'MOVE',
                                          'status': 'FAILED_AFTER_IDENTICAL_RETRY'},
                                         {'count': 41, 'method': 'iCluster2b', 'status': 'PASS'}]},
 'policy': {'action': 'move the final index of the penultimate batch to the front of the final singleton '
                      'batch',
            'all_samples_exactly_once_per_epoch': True,
            'base_batch_size_rule': 'min(16, n_fit_samples)',
            'batch_count_preserved': True,
            'batchnorm_training_retained': True,
            'drop_last_enabled': False,
            'example_n49_after': [16, 16, 15, 2],
            'example_n49_before': [16, 16, 16, 1],
            'flattened_random_order_preserved': True,
            'historical_final_lock_scope': 'paper5/method_contract/05cmp4',
            'new_execution': 'recompute BOTH methods for all 360 datasets into a fresh namespace; never '
                             'transplant old PASS records',
            'new_result_lock_dir': 'paper5/method_contract/05cmp4_v7_batch_rebalance',
            'new_result_root': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v7_batch_rebalance',
            'numerical_training_equivalence_claimed': False,
            'old_result_root': 'paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1',
            'outcome_based_policy_search': False,
            'policy_id': 'MOVE_TRAIN_SINGLETON_TAIL_REBALANCE_V1',
            'prior_failure_policy': 'v6 records remain immutable and disclosed; v7 is a distinct amended '
                                    'execution',
            'projection_loader_changed': False,
            'sampler_generator_stream_preserved': True,
            'scientific_epochs': 300,
            'scope': 'all 360 datasets under the same rule; no per-dataset rescue',
            'training_batch_boundaries_changed': True,
            'trigger': 'training only, final batch contains exactly one sample'},
 'probe_status': 'SYNTHETIC_CANDIDATE_PASS_NOT_AUTHORIZED',
 'qualification': 'Actual v6 scientific iCluster2b outputs already exist. Amendment was selected from '
                  'runtime failure mechanics and artificial probes, not comparative numeric results. This is '
                  'not a pre-first-fit freeze.',
 'reviewed_report_sha256': '6141365b39ea92b6003ca359c94b8129fff11b0269df8a33a522a0d59e6f3d28',
 'reviewed_zip_sha256': '1bcdd158a6255ddc85857b2b3a6694fb51b912c2557ba268bd7e633eb8766a3f',
 'source_snapshots': {'paper5/comparator_envs/move_1_5_0/Lib/site-packages/move/data/dataloaders.py': {'path': 'paper5/comparator_envs/move_1_5_0/Lib/site-packages/move/data/dataloaders.py',
                                                                                                       'sha256': 'c33791e4a4b468d8081a7aea7975e8f93b33251a64bd3844bb77bd54bd75c982',
                                                                                                       'size_bytes': 6780},
                      'paper5/comparator_envs/move_1_5_0/Lib/site-packages/move/models/vae.py': {'path': 'paper5/comparator_envs/move_1_5_0/Lib/site-packages/move/models/vae.py',
                                                                                                 'sha256': 'a73248339aef802a80e484849f0ccf72fc87b147c883c35bdb41561ed9b071a9',
                                                                                                 'size_bytes': 26970},
                      'paper5/comparator_envs/move_1_5_0/Lib/site-packages/move/training/training_loop.py': {'path': 'paper5/comparator_envs/move_1_5_0/Lib/site-packages/move/training/training_loop.py',
                                                                                                             'sha256': '6dbd05437950e6a33f612f5bf969bb42fee30302d0a7395c56f558437b24caf6',
                                                                                                             'size_bytes': 3724},
                      'paper5/method_contract/05b_v2/architecture_screening_data_manifest.csv': {'path': 'paper5/method_contract/05b_v2/architecture_screening_data_manifest.csv',
                                                                                                 'sha256': 'f3980e88aff408b65cbfce96aca50bad4fc08ecece2bc42bbc04f365c35d9c87',
                                                                                                 'size_bytes': 507713},
                      'paper5/method_contract/05cmp1/secondary_comparator_implementation_contract.json': {'path': 'paper5/method_contract/05cmp1/secondary_comparator_implementation_contract.json',
                                                                                                          'sha256': '418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95',
                                                                                                          'size_bytes': 12328},
                      'paper5/method_contract/05cmp1/secondary_comparator_implementation_contract_lock.json': {'path': 'paper5/method_contract/05cmp1/secondary_comparator_implementation_contract_lock.json',
                                                                                                               'sha256': '37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648',
                                                                                                               'size_bytes': 4937},
                      'paper5/method_contract/05cmp1/secondary_comparator_implementation_registry.csv': {'path': 'paper5/method_contract/05cmp1/secondary_comparator_implementation_registry.csv',
                                                                                                         'sha256': '85dbc68289bae71b9cf4f8a01af34b09d50750fde97f0b57ef3b6c57ebf299b0',
                                                                                                         'size_bytes': 955},
                      'paper5/method_contract/05cmp1/secondary_comparator_metric_eligibility_registry.csv': {'path': 'paper5/method_contract/05cmp1/secondary_comparator_metric_eligibility_registry.csv',
                                                                                                             'sha256': '8802d8d4bba20cfd4acc3263b12f0a119bcef4e4107164e6f944d0a01d9bef46',
                                                                                                             'size_bytes': 1474},
                      'paper5/method_contract/05cmp1/secondary_comparator_scientific_adapter_registry.csv': {'path': 'paper5/method_contract/05cmp1/secondary_comparator_scientific_adapter_registry.csv',
                                                                                                             'sha256': 'f30c73d19627d357073786b75f7b8141a757becf9d2413491518948cc099ea48',
                                                                                                             'size_bytes': 714},
                      'paper5/method_contract/05cmp1/secondary_comparator_seed_registry.csv': {'path': 'paper5/method_contract/05cmp1/secondary_comparator_seed_registry.csv',
                                                                                               'sha256': '2c70c516ad07496a71fe513139b357b92f46fd746bea5258f1fb237028c4e87c',
                                                                                               'size_bytes': 978},
                      'paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract.json': {'path': 'paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract.json',
                                                                                                                  'sha256': '8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2',
                                                                                                                  'size_bytes': 9406},
                      'paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract_lock.json': {'path': 'paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract_lock.json',
                                                                                                                       'sha256': 'a55d3e3498d9cc6a4acaaab64e5098007b093071b60999c82cf8e124a0301b5b',
                                                                                                                       'size_bytes': 1329},
                      'paper5/method_contract/05cmp2/secondary_comparator_confirmatory_dataset_registry.csv': {'path': 'paper5/method_contract/05cmp2/secondary_comparator_confirmatory_dataset_registry.csv',
                                                                                                               'sha256': '4b72b94c4df49dbbbd1a90d035855e628a4fa50eede0451dfe796227bda53314',
                                                                                                               'size_bytes': 269402},
                      'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v5.json': {'path': 'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v5.json',
                                                                                                                          'sha256': '7ec4e687151a4af67b718a1ee13e521109a0071be91de2e751e9d0bb02b96b94',
                                                                                                                          'size_bytes': 3062},
                      'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v6.json': {'path': 'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v6.json',
                                                                                                                          'sha256': 'e213d756bfcbef5ee01e46e72cec361ca39b3656674b8cba32afd5b821e49179',
                                                                                                                          'size_bytes': 5876},
                      'paper5/method_contract/05cmp3a/secondary_comparator_runner_source_authorization_lock.json': {'path': 'paper5/method_contract/05cmp3a/secondary_comparator_runner_source_authorization_lock.json',
                                                                                                                    'sha256': '5dbb075b3467fa865cb2b0570a79c2419e952a2c8c001eb04287d8ade5fba319',
                                                                                                                    'size_bytes': 6616},
                      'paper5/method_contract/05cmp3b/secondary_comparator_runner_v6_source_authorization_lock.json': {'path': 'paper5/method_contract/05cmp3b/secondary_comparator_runner_v6_source_authorization_lock.json',
                                                                                                                       'sha256': 'cca95c069ae27b1c8adf7ad17d577ba38e80f3f5df778fe60068c67e3062675e',
                                                                                                                       'size_bytes': 8458},
                      'paper5/scripts/05b_generate_small_n_multiomics_simulations_v2.py': {'path': 'paper5/scripts/05b_generate_small_n_multiomics_simulations_v2.py',
                                                                                           'sha256': 'b4adb1fcb32b65a998ee70e1b72d52e13490194c7ad8e5ddd802b95bf05a28db',
                                                                                           'size_bytes': 109309},
                      'paper5/scripts/05c1_fit_linear_simulation_baselines.py': {'path': 'paper5/scripts/05c1_fit_linear_simulation_baselines.py',
                                                                                 'sha256': 'c2c91b4ad274f78ccc369e6fca765102285cf3d9c8191f3c33f4a450b7426367',
                                                                                 'size_bytes': 126641},
                      'paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v5.py': {'path': 'paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v5.py',
                                                                                                      'sha256': '802c131dfda75effb4e60dd38ebf05512fa5da6c1b3eec97c0c499b3c4e869f5',
                                                                                                      'size_bytes': 113760},
                      'paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v6.py': {'path': 'paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v6.py',
                                                                                                      'sha256': '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a',
                                                                                                      'size_bytes': 128331},
                      'paper5/scripts/05cmp3a_freeze_secondary_comparator_runner_v5_source_authorization_v1.py': {'path': 'paper5/scripts/05cmp3a_freeze_secondary_comparator_runner_v5_source_authorization_v1.py',
                                                                                                                  'sha256': '526bf30c101dea38cb26d07c9acd625892febebcd697fb466be0c4c1fe653bfa',
                                                                                                                  'size_bytes': 23750},
                      'paper5/scripts/cmp3bncheck.py': {'path': 'paper5/scripts/cmp3bncheck.py',
                                                        'sha256': 'de6255d94b7b200f8dfcdfa6eb15a1c88b887e011572be45ab73bcf52dad3c79',
                                                        'size_bytes': 25768}},
 'status': 'REVIEWED_POST_V6_BATCH_IMPLEMENTATION_AMENDMENT_EVIDENCE',
 'synthetic_probe_summary': {'canonical_latent_shape': [62, 5],
                             'completed_actual_synthetic_fit_count': 24,
                             'fixture': {'features': [500, 4000, 500],
                                         'fold_test_sizes': [13, 13, 12, 12, 12],
                                         'fold_train_sizes': [49, 49, 50, 50, 50],
                                         'fresh_seed': 740091,
                                         'n': 62},
                             'frozen_scientific_epochs_unchanged': 300,
                             'scientific_metric_values_evaluated': False,
                             'status': 'PASS_CANDIDATE_ACTUAL_MOVE_SYNTHETIC_SELECTION',
                             'synthetic_epochs': 2}}

PREAUTH_STATUS = "05CMP3_SECONDARY_COMPARATOR_RUNNER_SYNTHETIC_PREAUTH_PASS"
AUTH_STATUS = "05CMP3C_SECONDARY_COMPARATOR_RUNNER_V7_BATCH_AMENDMENT_AUTHORIZED"
RESULT_LOCK_STATUS = "05CMP4_V7_SECONDARY_COMPARATOR_AMENDED_RESULTS_FROZEN"

EXPECTED = {
    "cmp2_contract_sha256":
        "8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2",
    "cmp2_lock_sha256":
        "a55d3e3498d9cc6a4acaaab64e5098007b093071b60999c82cf8e124a0301b5b",
    "cmp2_registry_sha256":
        "4b72b94c4df49dbbbd1a90d035855e628a4fa50eede0451dfe796227bda53314",
    "cmp2_contract_status":
        "FROZEN_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_BEFORE_SCIENTIFIC_FITS",
    "cmp2_lock_status":
        "05CMP2_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_CONTRACT_FROZEN",
    "cmp1_contract_sha256":
        "418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95",
    "cmp1_lock_sha256":
        "37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648",
    "cmp1_impl_sha256":
        "85dbc68289bae71b9cf4f8a01af34b09d50750fde97f0b57ef3b6c57ebf299b0",
    "cmp1_adapter_sha256":
        "f30c73d19627d357073786b75f7b8141a757becf9d2413491518948cc099ea48",
    "cmp1_seed_sha256":
        "2c70c516ad07496a71fe513139b357b92f46fd746bea5258f1fb237028c4e87c",
    "cmp1_metric_sha256":
        "8802d8d4bba20cfd4acc3263b12f0a119bcef4e4107164e6f944d0a01d9bef46",
    "manifest05b_sha256":
        "f3980e88aff408b65cbfce96aca50bad4fc08ecece2bc42bbc04f365c35d9c87",
    "n_datasets": 360,
    "n_scenarios": 36,
    "replicates": list(range(11, 21)),
    "methods": ["iCluster2b", "MOVE"],
    "latent_dim": 5,
    "move_version": "1.5.0",
    "torch_version": "2.12.1+cu130",
    "cuda_build": "13.0",
    "move_source_hashes": {
        "dataloaders":
            "c33791e4a4b468d8081a7aea7975e8f93b33251a64bd3844bb77bd54bd75c982",
        "training_loop":
            "6dbd05437950e6a33f612f5bf969bb42fee30302d0a7395c56f558437b24caf6",
        "vae":
            "a73248339aef802a80e484849f0ccf72fc87b147c883c35bdb41561ed9b071a9",
    },
    "rscript_sha256":
        "d829bcf7e9fa1d7e3e828c565c3cdbb1ed416f551f4fa6fd4dfcdf231e33e5e8",
    "icluster_tree_sha256":
        "cad0c995150e778026fb227ac533a974d77106e431c65a3802aeda4254f6fd70",
    "icluster_version": "1.47.3",
    "bioc_version": "3.23",
    "r_version_prefix": "R version 4.6.1",
}

MOVE_CANDIDATES = [
    ("MOVE_H64_32", [64, 32]),
    ("MOVE_H128_64", [128, 64]),
    ("MOVE_H256_128", [256, 128]),
]
MOVE_EPOCHS = 300
MOVE_LR = 0.001
MOVE_BETA = 0.01
MOVE_DROPOUT = 0.1
MOVE_KLD_WARMUP_STEPS = [1, 75, 150, 300]
MOVE_STABILITY_BASE_SEEDS = [1103, 2207, 3301]

ICLUSTER_K = 5
ICLUSTER_MIN_LAMBDA = 10.0
ICLUSTER_MAX_LAMBDA = 500.0
ICLUSTER_LAMBDA_ITER = 25
ICLUSTER_EM_ITER = 25
ICLUSTER_MIN_SHRINKAGE = [0.05, 0.05, 0.05]
ICLUSTER_EPS = 1e-4
ICLUSTER_EPS2 = 1e-6

VIEW_KEYS = ["rna", "methylation", "cnv"]
NPZ_VIEW_KEYS = {
    "rna": "X_rna",
    "methylation": "X_methylation",
    "cnv": "X_cnv",
}


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("utf-8"))
    h.update(str(tuple(x.shape)).encode("utf-8"))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(project_root.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def tree_sha256(root: Path) -> tuple[str, int, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory missing: {root}")
    files = sorted(
        [p for p in root.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(root)).replace("\\", "/"),
    )
    h = hashlib.sha256()
    total = 0
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        data = path.read_bytes()
        total += len(data)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
        h.update(b"\0")
    return h.hexdigest(), len(files), total


def derive_seed(parts: list[str], base_seed: int) -> int:
    payload = "|".join([*parts, str(int(base_seed))])
    value = int(
        hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8],
        16,
    )
    return 1 + (value % 2147483646)


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


# =============================================================================
# Exact frozen upstream verification
# =============================================================================

def build_paths(script_path: Path) -> dict[str, Path]:
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"
    return {
        "project_root": project_root,
        "paper5": paper5,
        "cmp2_contract":
            paper5 / "method_contract/05cmp2/"
            "secondary_comparator_confirmatory_benchmark_contract.json",
        "cmp2_lock":
            paper5 / "method_contract/05cmp2/"
            "secondary_comparator_confirmatory_benchmark_contract_lock.json",
        "cmp2_registry":
            paper5 / "method_contract/05cmp2/"
            "secondary_comparator_confirmatory_dataset_registry.csv",
        "cmp1_contract":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_implementation_contract.json",
        "cmp1_lock":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_implementation_contract_lock.json",
        "cmp1_impl":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_implementation_registry.csv",
        "cmp1_adapter":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_scientific_adapter_registry.csv",
        "cmp1_seed":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_seed_registry.csv",
        "cmp1_metric":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_metric_eligibility_registry.csv",
        "manifest05b":
            paper5 / "method_contract/05b_v2/"
            "architecture_screening_data_manifest.csv",
        "linear_helper":
            paper5 / "scripts/05c1_fit_linear_simulation_baselines.py",
        "preflight":
            paper5 / "method_contract/05cmp3_preflight/"
            "secondary_comparator_runner_synthetic_preflight_v7.json",
        "auth":
            paper5 / "method_contract/05cmp3c/"
            "secondary_comparator_runner_v7_batch_amendment_authorization_lock.json",
        "result_root":
            paper5 / "results/simulation_benchmark/"
            "05cmp_secondary_comparators/v7_batch_rebalance",
        "result_lock_dir":
            paper5 / "method_contract/05cmp4_v7_batch_rebalance",
    }


def verify_schema_amendment_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    """Retain exact v5 history and the reviewed generator/helper; no execution."""
    records: dict[str, Any] = {}
    for relative, expected in AMENDMENT_REVIEWED_INPUTS.items():
        path = require_file(paths["project_root"] / relative)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Schema-amendment reviewed input changed: {relative}: {actual}")
        records[relative] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return records



def bn_require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def bn_project_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    bn_require(not rel.is_absolute() and ".." not in rel.parts, "Invalid evidence path")
    cur = root
    for part in rel.parts:
        cur = cur / part
        try:
            info = cur.lstat()
        except FileNotFoundError:
            continue
        bn_require(not stat.S_ISLNK(info.st_mode) and not (
            getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ), f"Evidence symlink/reparse path rejected: {cur}")
    bn_require(cur.resolve().is_relative_to(root.resolve()), "Evidence path escapes project")
    return cur


def bn_file_inventory(root: Path, relative: str) -> set[str]:
    path = bn_project_path(root, relative)
    if not path.exists():
        return set()
    bn_require(path.is_dir(), f"Not a directory: {path}")
    found: set[str] = set()
    def failed(error: OSError) -> None:
        raise RuntimeError(f"Cannot inventory history: {error}")
    for directory, dirs, files in os.walk(path, followlinks=False, onerror=failed):
        for name in dirs + files:
            p = Path(directory) / name
            rel = p.relative_to(root).as_posix()
            bn_project_path(root, rel)
            if name in files:
                bn_require(p.is_file(), f"Not a regular history file: {rel}")
                found.add(rel)
    return found


def verify_bn_history(paths: dict[str, Path]) -> dict[str, Any]:
    """Hash old bytes only: never decode prior numeric metric payloads or open NPZ."""
    root = paths["project_root"]
    encoded = json.dumps(BN_AMENDMENT, sort_keys=True, separators=(",", ":")).encode("utf-8")
    bn_require(hashlib.sha256(encoded).hexdigest() == BN_EVIDENCE_SHA256,
               "BN amendment evidence constant changed")
    records = BN_AMENDMENT["historical_shards"]["records"]
    expected_files = {r["path"] for r in records}
    bn_require(len(records) == 82 and len(expected_files) == 82, "Historical inventory count mismatch")
    actual_files = bn_file_inventory(root, BN_AMENDMENT["policy"]["old_result_root"])
    bn_require(actual_files == expected_files,
               "Historical v6 file set changed. Do not delete files or resume v6; report the discrepancy.")
    bn_require(not bn_file_inventory(root, BN_AMENDMENT["policy"]["historical_final_lock_scope"]),
               "Unexpected historical v6 final lock/files; review required")
    all_records = list(BN_AMENDMENT["source_snapshots"].values()) + records
    for record in all_records:
        p = bn_project_path(root, record["path"])
        bn_require(p.is_file() and p.stat().st_size == record["size_bytes"],
                   f"Historical/source file absent or size changed: {record['path']}")
        bn_require(sha256_file(p) == record["sha256"],
                   f"Historical/source SHA changed: {record['path']}")
    return {"status": "PASS_EXACT_V6_HISTORY_PRESERVED", "evidence_sha256": BN_EVIDENCE_SHA256,
            "old_shard_count": 82, "old_iCluster2b_PASS": 41, "old_MOVE_singleton_failures": 41,
            "old_shard_bytes_hashed": True, "old_metric_values_decoded": False,
            "old_records_transplanted_into_new_results": False}


def verify_upstream(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], pd.DataFrame]:
    exact = {
        "cmp2_contract": EXPECTED["cmp2_contract_sha256"],
        "cmp2_lock": EXPECTED["cmp2_lock_sha256"],
        "cmp2_registry": EXPECTED["cmp2_registry_sha256"],
        "cmp1_contract": EXPECTED["cmp1_contract_sha256"],
        "cmp1_lock": EXPECTED["cmp1_lock_sha256"],
        "cmp1_impl": EXPECTED["cmp1_impl_sha256"],
        "cmp1_adapter": EXPECTED["cmp1_adapter_sha256"],
        "cmp1_seed": EXPECTED["cmp1_seed_sha256"],
        "cmp1_metric": EXPECTED["cmp1_metric_sha256"],
        "manifest05b": EXPECTED["manifest05b_sha256"],
    }
    for key, expected_sha in exact.items():
        path = require_file(paths[key])
        observed = sha256_file(path)
        if observed != expected_sha:
            raise RuntimeError(
                f"Frozen upstream SHA mismatch for {key}: "
                f"{observed} != {expected_sha}"
            )

    contract = load_json(paths["cmp2_contract"])
    lock = load_json(paths["cmp2_lock"])
    if contract.get("status") != EXPECTED["cmp2_contract_status"]:
        raise RuntimeError("05cmp2 contract status mismatch.")
    if lock.get("status") != EXPECTED["cmp2_lock_status"]:
        raise RuntimeError("05cmp2 lock status mismatch.")

    registry = pd.read_csv(
        paths["cmp2_registry"],
        dtype=str,
    ).fillna("")
    if registry.shape[0] != EXPECTED["n_datasets"]:
        raise RuntimeError(
            f"05cmp2 registry rows changed: {registry.shape[0]}"
        )
    if registry["scenario_id"].nunique() != EXPECTED["n_scenarios"]:
        raise RuntimeError("05cmp2 registry scenario count changed.")
    reps = sorted(
        pd.to_numeric(
            registry["replicate_index"],
            errors="raise",
        ).astype(int).unique().tolist()
    )
    if reps != EXPECTED["replicates"]:
        raise RuntimeError(
            f"05cmp2 confirmatory replicates changed: {reps}"
        )
    if registry["comparator_dataset_id"].duplicated().any():
        raise RuntimeError(
            "05cmp2 comparator_dataset_id is not unique."
        )

    return contract, registry


# =============================================================================
# Environment / source bindings
# =============================================================================

def import_linear_helper(path: Path) -> Any:
    require_file(path)
    spec = importlib.util.spec_from_file_location(
        "paper5_cmp_linear_helper",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not build import spec for frozen linear helper."
        )
    module = importlib.util.module_from_spec(spec)
    import sys as _paper5_loader_sys

    _paper5_loader_missing = object()
    _paper5_loader_previous = _paper5_loader_sys.modules.get(
        spec.name, _paper5_loader_missing
    )
    _paper5_loader_sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if _paper5_loader_previous is _paper5_loader_missing:
            _paper5_loader_sys.modules.pop(spec.name, None)
        else:
            _paper5_loader_sys.modules[spec.name] = _paper5_loader_previous
        raise
    for name in [
        "subspace_metrics",
        "standardized_columns",
        "integrated_anchor_truth",
    ]:
        if not hasattr(module, name):
            raise RuntimeError(
                f"Frozen linear helper no longer exposes {name}."
            )
    return module


def import_move_api() -> dict[str, Any]:
    try:
        move_version = importlib.metadata.version("move-dl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "move-dl is not installed in the active environment. "
            "Activate paper5/comparator_envs/move_1_5_0."
        ) from exc

    if move_version != EXPECTED["move_version"]:
        raise RuntimeError(
            f"MOVE version mismatch: {move_version}"
        )
    if torch.__version__ != EXPECTED["torch_version"]:
        raise RuntimeError(
            f"PyTorch version mismatch: {torch.__version__}"
        )
    if str(torch.version.cuda) != EXPECTED["cuda_build"]:
        raise RuntimeError(
            f"CUDA build mismatch: {torch.version.cuda}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "cuda:0 is required; CPU fallback is forbidden."
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from move.data.dataloaders import make_dataloader
    from move.models.vae import VAE
    from move.training.training_loop import training_loop

    source_paths = {
        "dataloaders":
            Path(inspect.getfile(make_dataloader)).resolve(),
        "vae":
            Path(inspect.getfile(VAE)).resolve(),
        "training_loop":
            Path(inspect.getfile(training_loop)).resolve(),
    }
    for key, path in source_paths.items():
        observed = sha256_file(require_file(path))
        expected = EXPECTED["move_source_hashes"][key]
        if observed != expected:
            raise RuntimeError(
                f"MOVE source SHA mismatch for {key}: "
                f"{observed} != {expected}"
            )

    return {
        "make_dataloader": make_dataloader,
        "VAE": VAE,
        "training_loop": training_loop,
        "source_paths": source_paths,
        "move_version": move_version,
    }


def find_exact_rscript() -> Path:
    candidates = [
        Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"),
        Path(r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"),
    ]
    explicit = os.environ.get("PAPER5_RSCRIPT", "").strip()
    if explicit:
        candidates.insert(0, Path(explicit))

    for path in candidates:
        if (
            path.is_file()
            and sha256_file(path) == EXPECTED["rscript_sha256"]
        ):
            return path.resolve()

    raise RuntimeError(
        "Exact frozen Rscript 4.6.1 executable was not "
        "found/hash-matched."
    )


def icluster_tree_sha256_frozen_05cmp0c(root: Path) -> tuple[str, int, int]:
    # Reproduce the exact 05cmp0c v4 format, including native Windows Path order.
    # Do not substitute sorted POSIX strings, omit size, or use raw digest bytes.
    # This helper is used ONLY for the frozen iClusterPlus package verification.
    if not root.is_dir():
        raise FileNotFoundError(f"Directory missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = int(path.stat().st_size)
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes


def verify_r_library(paper5: Path) -> dict[str, Any]:
    r_library = (
        paper5 / "comparator_envs/r_4_6_bioc_3_23"
    ).resolve()
    package_root = r_library / "iClusterPlus"
    tree_hash, file_count, total_bytes = icluster_tree_sha256_frozen_05cmp0c(
        package_root
    )
    if tree_hash != EXPECTED["icluster_tree_sha256"]:
        raise RuntimeError(
            "iClusterPlus package-tree SHA mismatch: "
            f"{tree_hash}"
        )

    rscript = find_exact_rscript()

    expression = r"""
options(warn=1, digits=17)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BiocManager))
suppressPackageStartupMessages(library(iClusterPlus))
cat("R_VERSION=", R.version.string, "\n", sep="")
cat("BIOC_VERSION=", as.character(BiocManager::version()), "\n", sep="")
cat("ICLUSTERPLUS_VERSION=", as.character(packageVersion("iClusterPlus")), "\n", sep="")
cat("ICLUSTER2B_EXISTS=", exists("iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE), "\n", sep="")
cat("TUNE_ICLUSTER2B_EXISTS=", exists("tune.iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE), "\n", sep="")
"""

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(r_library)
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[key] = "1"

    with tempfile.TemporaryDirectory(
        prefix="paper5_05cmp3_rprobe_"
    ) as td:
        rp = Path(td) / "probe.R"
        rp.write_text(
            expression,
            encoding="utf-8",
            newline="\n",
        )
        completed = subprocess.run(
            [str(rscript), "--vanilla", str(rp)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            "Frozen R/iClusterPlus environment probe failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()

    if not parsed.get("R_VERSION", "").startswith(
        EXPECTED["r_version_prefix"]
    ):
        raise RuntimeError(
            f"R version mismatch: {parsed.get('R_VERSION')}"
        )
    if parsed.get("BIOC_VERSION") != EXPECTED["bioc_version"]:
        raise RuntimeError(
            "Bioconductor version mismatch."
        )
    if (
        parsed.get("ICLUSTERPLUS_VERSION")
        != EXPECTED["icluster_version"]
    ):
        raise RuntimeError(
            "iClusterPlus version mismatch."
        )
    if parsed.get("ICLUSTER2B_EXISTS", "").lower() != "true":
        raise RuntimeError(
            "iCluster2b unavailable."
        )
    if (
        parsed.get("TUNE_ICLUSTER2B_EXISTS", "").lower()
        != "true"
    ):
        raise RuntimeError(
            "tune.iCluster2b unavailable."
        )

    return {
        "rscript": rscript,
        "r_library": r_library,
        "package_tree_sha256": tree_hash,
        "package_tree_file_count": file_count,
        "package_tree_total_bytes": total_bytes,
        "probe": parsed,
        "subprocess_env": env,
    }


# =============================================================================
# Adapter / preprocessing
# =============================================================================

def fit_observed_standardizer(
    x: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    fit = x[np.asarray(fit_indices, dtype=int)]

    observed = np.isfinite(fit)
    counts = observed.sum(axis=0)

    sums = np.where(
        observed,
        fit,
        0.0,
    ).sum(axis=0)

    means = np.divide(
        sums,
        counts,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=counts > 0,
    )

    centered_sq = np.where(
        observed,
        (fit - means) ** 2,
        0.0,
    ).sum(axis=0)

    variance = np.divide(
        centered_sq,
        counts,
        out=np.ones(x.shape[1], dtype=np.float64),
        where=counts > 0,
    )

    scale = np.sqrt(variance)

    bad = (~np.isfinite(scale)) | (scale <= 0.0)
    scale[bad] = 1.0
    means[~np.isfinite(means)] = 0.0

    return {
        "mean": means,
        "scale": scale,
        "fit_observed_counts": counts.astype(np.int64),
        "n_constant_or_absent_fit_features":
            np.asarray(bad, dtype=np.uint8),
    }


def transform_with_standardizer(
    x: np.ndarray,
    prep: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    observed = np.isfinite(x)

    z = (
        x - prep["mean"]
    ) / prep["scale"]

    z[~np.isfinite(z)] = 0.0
    return z, observed


def adapt_views(
    raw_views: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    output_indices: np.ndarray,
    dtype: np.dtype,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[dict[str, Any]],
]:
    outputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []

    for view in VIEW_KEYS:
        raw = np.asarray(
            raw_views[view],
            dtype=np.float64,
        )
        prep = fit_observed_standardizer(
            raw,
            fit_indices,
        )
        transformed, observed = transform_with_standardizer(
            raw[np.asarray(output_indices, dtype=int)],
            prep,
        )

        transformed = transformed.astype(
            dtype,
            copy=False,
        )
        outputs.append(transformed)
        masks.append(observed)

        audits.append(
            {
                "view": view,
                "n_fit_rows": int(len(fit_indices)),
                "n_output_rows": int(len(output_indices)),
                "n_features": int(raw.shape[1]),
                "n_fit_features_no_observed":
                    int(
                        np.sum(
                            prep["fit_observed_counts"] == 0
                        )
                    ),
                "n_fit_features_constant_or_absent":
                    int(
                        np.sum(
                            prep[
                                "n_constant_or_absent_fit_features"
                            ]
                        )
                    ),
                "output_missing_entries_zero_filled":
                    int(np.sum(~observed)),
            }
        )

    return outputs, masks, audits


# =============================================================================
# Frozen simulation truth metrics
# =============================================================================

def own_mean_canonical_correlation(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> float:
    estimated = np.asarray(
        estimated,
        dtype=np.float64,
    )
    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    if (
        estimated.ndim != 2
        or truth.ndim != 2
        or estimated.shape[0] != truth.shape[0]
    ):
        raise RuntimeError(
            "Subspace metric shape mismatch."
        )
    if (
        estimated.shape[1] == 0
        or truth.shape[1] == 0
    ):
        return float("nan")
    if (
        not np.isfinite(estimated).all()
        or not np.isfinite(truth).all()
    ):
        raise RuntimeError(
            "Non-finite latent/truth matrix in subspace metric."
        )

    def basis(x: np.ndarray) -> np.ndarray:
        xc = x - x.mean(
            axis=0,
            keepdims=True,
        )
        u, s, _ = np.linalg.svd(
            xc,
            full_matrices=False,
        )
        if s.size == 0:
            return np.zeros(
                (x.shape[0], 0),
                dtype=np.float64,
            )
        tol = (
            float(s[0])
            * max(xc.shape)
            * np.finfo(np.float64).eps
        )
        rank = int(np.sum(s > tol))
        return u[:, :rank]

    qa = basis(estimated)
    qb = basis(truth)

    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return 0.0

    values = np.linalg.svd(
        qa.T @ qb,
        compute_uv=False,
    )
    values = np.clip(
        values,
        0.0,
        1.0,
    )

    # A rank-deficient estimated space receives zero recovery for
    # missing truth dimensions so collapse cannot be rewarded.
    denom = int(truth.shape[1])
    total = float(np.sum(values))
    return total / float(denom)


def flatten_numeric_scalars(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []

    if isinstance(value, dict):
        for k, v in value.items():
            p = (
                f"{prefix}.{k}"
                if prefix
                else str(k)
            )
            out.extend(
                flatten_numeric_scalars(
                    v,
                    p,
                )
            )

    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            p = f"{prefix}[{i}]"
            out.extend(
                flatten_numeric_scalars(
                    v,
                    p,
                )
            )

    elif (
        isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
        and not isinstance(value, bool)
    ):
        x = float(value)
        if math.isfinite(x):
            out.append((prefix, x))

    return out


def paper5_shared_subspace_recovery(
    linear: Any,
    estimated: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    if truth.shape[1] == 0:
        return (
            float("nan"),
            {
                "status":
                    "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS"
            },
        )

    authoritative = linear.subspace_metrics(
        np.asarray(
            estimated,
            dtype=np.float64,
        ),
        np.asarray(
            truth,
            dtype=np.float64,
        ),
    )

    numeric = flatten_numeric_scalars(
        authoritative
    )

    preferred = [
        (
            name,
            value,
        )
        for name, value in numeric
        if (
            "mean" in name.lower()
            and "canonical" in name.lower()
        )
        or "shared_subspace_recovery" in name.lower()
    ]

    if len(preferred) == 1:
        return (
            float(preferred[0][1]),
            {
                "status":
                    "PASS_FROZEN_HELPER_NAMED_SCALAR",
                "helper_scalar_path":
                    preferred[0][0],
            },
        )

    own = own_mean_canonical_correlation(
        estimated,
        truth,
    )

    matches = [
        (
            name,
            value,
        )
        for name, value in numeric
        if np.isclose(
            value,
            own,
            rtol=0.0,
            atol=1e-10,
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Frozen Paper 5 subspace_metrics did not yield "
            "one unambiguous scalar matching the "
            "mean-canonical-correlation definition. "
            f"Own={own}; authoritative={authoritative!r}; "
            f"matches={matches!r}"
        )

    return (
        float(matches[0][1]),
        {
            "status":
                "PASS_FROZEN_HELPER_NUMERIC_REPLAY",
            "helper_scalar_path":
                matches[0][0],
        },
    )


def factor_alignment_correlation(
    linear: Any,
    estimated: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    if truth.shape[1] == 0:
        return float("nan"), []

    e = np.asarray(
        linear.standardized_columns(
            np.asarray(
                estimated,
                dtype=np.float64,
            )
        )
    )
    t = np.asarray(
        linear.standardized_columns(
            np.asarray(
                truth,
                dtype=np.float64,
            )
        )
    )

    if (
        e.ndim != 2
        or t.ndim != 2
        or e.shape[0] != t.shape[0]
    ):
        raise RuntimeError(
            "standardized_columns returned incompatible matrices."
        )
    if (
        not np.isfinite(e).all()
        or not np.isfinite(t).all()
    ):
        raise RuntimeError(
            "Non-finite standardized columns in factor alignment."
        )

    corr = np.corrcoef(
        e.T,
        t.T,
    )[
        : e.shape[1],
        e.shape[1] :
    ]

    corr = np.asarray(
        corr,
        dtype=np.float64,
    )
    if not np.isfinite(corr).all():
        raise RuntimeError(
            "Factor-alignment correlation matrix is non-finite."
        )

    row_ind, col_ind = linear_sum_assignment(
        -np.abs(corr)
    )
    values = np.abs(
        corr[
            row_ind,
            col_ind,
        ]
    )

    details = [
        {
            "estimated_factor_index": int(i),
            "truth_factor_index": int(j),
            "signed_pearson":
                float(corr[i, j]),
            "absolute_pearson":
                float(abs(corr[i, j])),
        }
        for i, j in zip(
            row_ind.tolist(),
            col_ind.tolist(),
            strict=True,
        )
    ]

    return float(np.mean(values)), details


# =============================================================================
# MOVE low-level execution
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_loader_before_bn_amendment(
    make_dataloader: Any,
    views: list[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        int(seed)
    )

    return make_dataloader(
        con_list=[
            np.asarray(
                x,
                dtype=np.float32,
            )
            for x in views
        ],
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


class RebalancedTailBatchSampler:
    """Keep the existing random index sequence; fix only a terminal singleton.

    Production rule requires the separate 05cmp3c batch-amendment authorization.
    The source sampler and its generator are reused, with no additional draw.
    """
    def __init__(self, original: Any, n: int, batch_size: int):
        self.original = original
        self.n = int(n)
        self.batch_size = int(batch_size)
        bn_require(self.n >= 2, "BatchNorm training requires at least two fit rows")
        bn_require(self.batch_size == min(16, self.n), "Unexpected base batch size")

    def __len__(self) -> int:
        return len(self.original)

    def __iter__(self):
        batches = [list(batch) for batch in self.original]
        bn_require(batches, "Empty sampler")
        before = [i for batch in batches for i in batch]
        bn_require(sorted(before) == list(range(self.n)), "Sampler must visit each row once")
        bn_require(len(batches) == len(self), "Sampler batch count mismatch")
        if len(batches[-1]) == 1:
            bn_require(len(batches) >= 2 and len(batches[-2]) >= 3,
                    "Cannot rebalance tail under the fixed batch cap")
            moved = batches[-2].pop()
            batches[-1].insert(0, moved)
        after = [i for batch in batches for i in batch]
        bn_require(after == before, "Rebalancing changed flattened sample order")
        bn_require(all(2 <= len(batch) <= self.batch_size for batch in batches),
                "Candidate produced an invalid training batch")
        yield from batches


def move_loader(
    make_dataloader: Any, views: list[np.ndarray], batch_size: int, shuffle: bool, seed: int,
) -> Any:
    """Use the validated singleton-tail rule only for affected training loaders."""
    loader = _move_loader_before_bn_amendment(make_dataloader, views, batch_size, shuffle, seed)
    if not shuffle:
        return loader
    n = len(loader.dataset)
    bn_require(n >= 2, "Cannot train BatchNorm on fewer than two fit rows")
    bn_require(batch_size == min(16, n), "Base batch size differs from frozen rule")
    if n % batch_size != 1:
        return loader
    bn_require(loader.num_workers == 0 and not loader.drop_last, "Unexpected loader settings")
    return torch.utils.data.DataLoader(
        dataset=loader.dataset,
        batch_sampler=RebalancedTailBatchSampler(loader.batch_sampler, n, batch_size),
        num_workers=0, collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory, generator=loader.generator,
    )


def fit_move(
    api: dict[str, Any],
    views: list[np.ndarray],
    hidden: list[int],
    seed: int,
    epochs: int,
    kld_warmup_steps: list[int],
) -> dict[str, Any]:
    VAE = api["VAE"]
    training_loop = api["training_loop"]
    make_dataloader = api["make_dataloader"]

    # Initialize CUDA before seeding, cache cleanup, and allocator statistics.
    torch.cuda.init()
    seed_everything(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    n = int(
        views[0].shape[0]
    )
    if any(
        int(v.shape[0]) != n
        for v in views
    ):
        raise RuntimeError(
            "MOVE view row counts differ."
        )

    batch_size = min(
        16,
        n,
    )

    train_loader = move_loader(
        make_dataloader,
        views,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    project_loader = move_loader(
        make_dataloader,
        views,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    model = VAE(
        categorical_shapes=None,
        continuous_shapes=[
            int(v.shape[1])
            for v in views
        ],
        categorical_weights=None,
        continuous_weights=[
            1.0,
            1.0,
            1.0,
        ],
        num_hidden=list(hidden),
        num_latent=EXPECTED["latent_dim"],
        beta=MOVE_BETA,
        dropout=MOVE_DROPOUT,
        cuda=True,
    )

    model = model.to(
        model.device
    )

    if sorted(
        {
            p.device.type
            for p in model.parameters()
        }
    ) != ["cuda"]:
        raise RuntimeError(
            "MOVE model parameters are not exclusively CUDA."
        )

    started = time.perf_counter()

    (
        losses,
        bce_losses,
        sse_losses,
        kld_losses,
        final_kld_weight,
    ) = training_loop(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=None,
        lr=MOVE_LR,
        num_epochs=int(epochs),
        batch_dilation_steps=[],
        kld_warmup_steps=list(
            kld_warmup_steps
        ),
        early_stopping=False,
        patience=0,
    )

    torch.cuda.synchronize(0)
    elapsed = (
        time.perf_counter()
        - started
    )

    for label, values in {
        "loss": losses,
        "bce": bce_losses,
        "sse": sse_losses,
        "kld": kld_losses,
    }.items():
        x = np.asarray(
            values,
            dtype=np.float64,
        )
        if (
            x.shape
            != (int(epochs),)
            or not np.isfinite(x).all()
        ):
            raise RuntimeError(
                f"MOVE {label} history invalid: "
                f"shape={x.shape}"
            )

    latent = np.asarray(
        model.project(
            project_loader
        ),
        dtype=np.float32,
    )

    if (
        latent.shape
        != (
            n,
            EXPECTED["latent_dim"],
        )
        or not np.isfinite(latent).all()
    ):
        raise RuntimeError(
            "MOVE latent output invalid: "
            f"{latent.shape}"
        )

    parameter_count = int(
        sum(
            int(p.numel())
            for p in model.parameters()
        )
    )
    peak_bytes = int(
        torch.cuda.max_memory_allocated(0)
    )

    return {
        "model": model,
        "latent": latent,
        "parameter_count":
            parameter_count,
        "elapsed_seconds":
            float(elapsed),
        "peak_cuda_bytes":
            peak_bytes,
        "final_kld_weight":
            float(final_kld_weight),
        "final_loss":
            float(
                np.asarray(
                    losses,
                    dtype=float,
                )[-1]
            ),
    }


def move_reconstruction_sse(
    model: Any,
    adapted_views: list[np.ndarray],
    observed_masks: list[np.ndarray],
) -> float:
    concat = np.concatenate(
        [
            np.asarray(
                v,
                dtype=np.float32,
            )
            for v in adapted_views
        ],
        axis=1,
    )

    mask = np.concatenate(
        observed_masks,
        axis=1,
    )
    if not mask.any():
        raise RuntimeError(
            "Held-out fold contains zero observed entries."
        )

    tensor = torch.from_numpy(
        concat
    ).to(
        device=model.device,
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(
            tensor
        )
        _, recon = model.decode(
            mu
        )

    if recon is None:
        raise RuntimeError(
            "MOVE decoder returned no continuous reconstruction."
        )

    rec = (
        recon.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if (
        rec.shape != concat.shape
        or not np.isfinite(rec).all()
    ):
        raise RuntimeError(
            "MOVE held-out reconstruction invalid."
        )

    diff = (
        rec
        - concat.astype(
            np.float64,
            copy=False,
        )
    )

    return float(
        np.sum(
            np.square(
                diff[mask]
            )
        )
    )


def deterministic_sample_folds(
    sample_ids: np.ndarray,
    n_folds: int = 5,
) -> list[np.ndarray]:
    ids = [
        str(x)
        for x in np.asarray(
            sample_ids
        ).tolist()
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Sample IDs are not unique."
        )

    order = sorted(
        range(len(ids)),
        key=lambda i: ids[i],
    )

    fold_by_row = np.empty(
        len(ids),
        dtype=int,
    )

    for pos, row_index in enumerate(
        order
    ):
        fold_by_row[row_index] = (
            pos % n_folds
        )

    return [
        np.flatnonzero(
            fold_by_row == fold
        )
        for fold in range(
            n_folds
        )
    ]


def pairwise_mean_canonical(
    latents: list[np.ndarray],
) -> tuple[list[float], float]:
    if len(latents) != 3:
        raise RuntimeError(
            "MOVE stability requires exactly three full-data seeds."
        )

    pair_values: list[float] = []

    for i in range(3):
        for j in range(
            i + 1,
            3,
        ):
            pair_values.append(
                own_mean_canonical_correlation(
                    latents[i],
                    latents[j],
                )
            )

    return (
        pair_values,
        float(
            np.median(
                np.asarray(
                    pair_values,
                    dtype=float,
                )
            )
        ),
    )


def select_move_architecture(
    api: dict[str, Any],
    raw_views: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    dataset_id: str,
) -> dict[str, Any]:
    n = len(sample_ids)
    all_idx = np.arange(
        n,
        dtype=int,
    )
    folds = deterministic_sample_folds(
        sample_ids,
        n_folds=5,
    )

    stage1_rows: list[
        dict[str, Any]
    ] = []

    for candidate_id, hidden in MOVE_CANDIDATES:
        fold_sses: list[float] = []

        for fold_index, test_idx in enumerate(
            folds
        ):
            train_idx = np.setdiff1d(
                all_idx,
                test_idx,
                assume_unique=True,
            )

            train_views, _, _ = adapt_views(
                raw_views,
                fit_indices=train_idx,
                output_indices=train_idx,
                dtype=np.float32,
            )

            test_views, test_masks, _ = adapt_views(
                raw_views,
                fit_indices=train_idx,
                output_indices=test_idx,
                dtype=np.float32,
            )

            seed = derive_seed(
                [
                    "MOVE",
                    dataset_id,
                    str(fold_index),
                    candidate_id,
                    "architecture_selection_cv",
                ],
                1103,
            )

            fit = fit_move(
                api,
                train_views,
                hidden=hidden,
                seed=seed,
                epochs=MOVE_EPOCHS,
                kld_warmup_steps=
                    MOVE_KLD_WARMUP_STEPS,
            )

            sse = move_reconstruction_sse(
                fit["model"],
                test_views,
                test_masks,
            )

            fold_sses.append(
                sse
            )

            del fit["model"]
            del fit
            gc.collect()
            torch.cuda.empty_cache()

        values = np.asarray(
            fold_sses,
            dtype=np.float64,
        )

        stage1_rows.append(
            {
                "candidate_id":
                    candidate_id,
                "hidden":
                    list(hidden),
                "fold_sse":
                    values.tolist(),
                "mean_heldout_reconstruction_sse":
                    float(np.mean(values)),
                "se_heldout_reconstruction_sse":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                        / math.sqrt(
                            len(values)
                        )
                    ),
            }
        )

    means = np.asarray(
        [
            r[
                "mean_heldout_reconstruction_sse"
            ]
            for r in stage1_rows
        ],
        dtype=float,
    )

    best_idx = int(
        np.argmin(
            means
        )
    )

    cutoff = (
        float(
            stage1_rows[
                best_idx
            ][
                "mean_heldout_reconstruction_sse"
            ]
        )
        + float(
            stage1_rows[
                best_idx
            ][
                "se_heldout_reconstruction_sse"
            ]
        )
    )

    retained = [
        r
        for r in stage1_rows
        if float(
            r[
                "mean_heldout_reconstruction_sse"
            ]
        )
        <= cutoff
    ]

    if not retained:
        raise RuntimeError(
            "MOVE one-SE stage retained zero candidates."
        )

    full_views, _, full_audit = adapt_views(
        raw_views,
        fit_indices=all_idx,
        output_indices=all_idx,
        dtype=np.float32,
    )

    stage2_rows: list[
        dict[str, Any]
    ] = []

    canonical_latents: dict[
        str,
        np.ndarray,
    ] = {}

    for candidate in retained:
        candidate_id = str(
            candidate[
                "candidate_id"
            ]
        )
        hidden = list(
            candidate[
                "hidden"
            ]
        )

        latents: list[
            np.ndarray
        ] = []

        parameter_count: int | None = None
        seed_rows: list[
            dict[str, Any]
        ] = []

        for base_seed in (
            MOVE_STABILITY_BASE_SEEDS
        ):
            seed = derive_seed(
                [
                    "MOVE",
                    dataset_id,
                    candidate_id,
                    "full_data_stability",
                ],
                base_seed,
            )

            fit = fit_move(
                api,
                full_views,
                hidden=hidden,
                seed=seed,
                epochs=MOVE_EPOCHS,
                kld_warmup_steps=
                    MOVE_KLD_WARMUP_STEPS,
            )

            latent = np.asarray(
                fit["latent"],
                dtype=np.float64,
            )
            latents.append(
                latent
            )

            if parameter_count is None:
                parameter_count = int(
                    fit[
                        "parameter_count"
                    ]
                )
            elif (
                parameter_count
                != int(
                    fit[
                        "parameter_count"
                    ]
                )
            ):
                raise RuntimeError(
                    "MOVE parameter count changed across seeds."
                )

            if base_seed == 1103:
                canonical_latents[
                    candidate_id
                ] = latent.copy()

            seed_rows.append(
                {
                    "base_seed":
                        int(base_seed),
                    "derived_seed":
                        int(seed),
                    "elapsed_seconds":
                        float(
                            fit[
                                "elapsed_seconds"
                            ]
                        ),
                    "peak_cuda_bytes":
                        int(
                            fit[
                                "peak_cuda_bytes"
                            ]
                        ),
                    "latent_sha256":
                        sha256_array(
                            latent
                        ),
                }
            )

            del fit["model"]
            del fit
            gc.collect()
            torch.cuda.empty_cache()

        (
            pair_values,
            stability,
        ) = pairwise_mean_canonical(
            latents
        )

        stage2_rows.append(
            {
                "candidate_id":
                    candidate_id,
                "hidden":
                    hidden,
                "parameter_count":
                    int(
                        parameter_count
                    ),
                "pairwise_mean_canonical_correlations":
                    pair_values,
                "median_pairwise_mean_canonical_correlation":
                    stability,
                "seed_runs":
                    seed_rows,
            }
        )

    stage2_sorted = sorted(
        stage2_rows,
        key=lambda r: (
            -float(
                r[
                    "median_pairwise_mean_canonical_correlation"
                ]
            ),
            int(
                r[
                    "parameter_count"
                ]
            ),
            str(
                r[
                    "candidate_id"
                ]
            ),
        ),
    )

    selected = stage2_sorted[0]
    selected_id = str(
        selected[
            "candidate_id"
        ]
    )

    selected_stage1 = next(
        r
        for r in stage1_rows
        if r["candidate_id"]
        == selected_id
    )

    return {
        "selected_candidate_id":
            selected_id,
        "canonical_latent":
            canonical_latents[
                selected_id
            ],
        "stage1":
            stage1_rows,
        "stage1_best_candidate_id":
            str(
                stage1_rows[
                    best_idx
                ][
                    "candidate_id"
                ]
            ),
        "stage1_one_se_cutoff":
            float(cutoff),
        "stage1_retained_candidate_ids":
            [
                str(
                    r[
                        "candidate_id"
                    ]
                )
                for r in retained
            ],
        "stage2":
            stage2_rows,
        "selected_latent_subspace_stability":
            float(
                selected[
                    "median_pairwise_mean_canonical_correlation"
                ]
            ),
        "selected_cv_mean_heldout_reconstruction_sse":
            float(
                selected_stage1[
                    "mean_heldout_reconstruction_sse"
                ]
            ),
        "full_data_preprocessing_audit":
            full_audit,
    }


# =============================================================================
# iCluster2b exact scientific execution
# =============================================================================

ICLUSTER_R_SCRIPT = r"""
options(warn=1, digits=17)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BiocManager))
suppressPackageStartupMessages(library(iClusterPlus))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4) stop("Expected input_dir output_dir seed expected_rows")
input_dir <- args[[1]]
output_dir <- args[[2]]
seed <- as.integer(args[[3]])
expected_rows <- as.integer(args[[4]])

if (as.character(packageVersion("iClusterPlus")) != "1.47.3") {
  stop("iClusterPlus version mismatch")
}
if (as.character(BiocManager::version()) != "3.23") {
  stop("Bioconductor version mismatch")
}

read_matrix <- function(name) {
  path <- file.path(input_dir, paste0(name, ".csv"))
  x <- as.matrix(read.csv(path, header=FALSE, check.names=FALSE))
  storage.mode(x) <- "double"
  if (nrow(x) != expected_rows) stop("Row mismatch for ", name)
  if (!all(is.finite(x))) stop("Non-finite adapted input: ", name)
  x
}

xList <- list(
  read_matrix("rna"),
  read_matrix("methylation"),
  read_matrix("cnv")
)

set.seed(seed)
started <- proc.time()[["elapsed"]]

fit <- iClusterPlus::tune.iCluster2b(
  xList=xList,
  K=5,
  method=rep("lasso", 3),
  min.lambda=10,
  max.lambda=500,
  lambda.iter=25,
  EM.iter=25,
  min.shrinkage.rate=rep(0.05, 3),
  eps=1e-4,
  eps2=1e-6
)

elapsed <- proc.time()[["elapsed"]] - started

if (is.null(fit$meanZ)) stop("tune.iCluster2b returned no meanZ")
meanZ <- as.matrix(fit$meanZ)
storage.mode(meanZ) <- "double"

if (nrow(meanZ) != expected_rows) stop("meanZ row mismatch")
if (ncol(meanZ) != 5) stop("meanZ latent dimension mismatch")
if (!all(is.finite(meanZ))) stop("meanZ non-finite")

lambda <- unlist(fit$lambda, use.names=FALSE)
storage.mode(lambda) <- "double"

if (length(lambda) != 3) stop("Expected three final lambda values")
if (!all(is.finite(lambda))) stop("Non-finite lambda")

dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

write.table(
  meanZ,
  file=file.path(output_dir, "meanZ.csv"),
  sep=",",
  row.names=FALSE,
  col.names=FALSE,
  quote=FALSE
)

write.table(
  matrix(lambda, nrow=1),
  file=file.path(output_dir, "lambda.csv"),
  sep=",",
  row.names=FALSE,
  col.names=FALSE,
  quote=FALSE
)

metadata <- data.frame(
  key=c(
    "r_version",
    "bioconductor_version",
    "iclusterplus_version",
    "seed",
    "elapsed_seconds"
  ),
  value=c(
    R.version.string,
    as.character(BiocManager::version()),
    as.character(packageVersion("iClusterPlus")),
    as.character(seed),
    as.character(elapsed)
  ),
  stringsAsFactors=FALSE
)

write.table(
  metadata,
  file=file.path(output_dir, "metadata.tsv"),
  sep="\t",
  row.names=FALSE,
  col.names=TRUE,
  quote=FALSE
)

cat("ICLUSTER2B_SCIENTIFIC_PASS\n")
"""


def run_icluster_fit(
    r_env: dict[str, Any],
    views: list[np.ndarray],
    seed: int,
) -> dict[str, Any]:
    n = int(
        views[0].shape[0]
    )

    if any(
        v.shape[0] != n
        for v in views
    ):
        raise RuntimeError(
            "iCluster2b view row counts differ."
        )

    with tempfile.TemporaryDirectory(
        prefix="paper5_05cmp3_icluster_"
    ) as td:
        root = Path(td)
        input_dir = root / "input"
        output_dir = root / "output"

        input_dir.mkdir(
            parents=True
        )
        output_dir.mkdir(
            parents=True
        )

        runner = root / "run_icluster2b.R"

        for name, matrix in zip(
            VIEW_KEYS,
            views,
            strict=True,
        ):
            np.savetxt(
                input_dir / f"{name}.csv",
                np.asarray(
                    matrix,
                    dtype=np.float64,
                ),
                delimiter=",",
                fmt="%.17g",
            )

        runner.write_text(
            ICLUSTER_R_SCRIPT,
            encoding="utf-8",
            newline="\n",
        )

        command = [
            str(
                r_env[
                    "rscript"
                ]
            ),
            "--vanilla",
            str(runner),
            str(input_dir),
            str(output_dir),
            str(int(seed)),
            str(n),
        ]

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=r_env[
                "subprocess_env"
            ],
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "iCluster2b scientific subprocess failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        if (
            "ICLUSTER2B_SCIENTIFIC_PASS"
            not in completed.stdout
        ):
            raise RuntimeError(
                "iCluster2b scientific PASS sentinel missing."
            )

        mean_z = np.loadtxt(
            output_dir / "meanZ.csv",
            delimiter=",",
            ndmin=2,
        )

        lambdas = np.loadtxt(
            output_dir / "lambda.csv",
            delimiter=",",
            ndmin=1,
        )

        if (
            mean_z.shape
            != (
                n,
                ICLUSTER_K,
            )
            or not np.isfinite(
                mean_z
            ).all()
        ):
            raise RuntimeError(
                "Invalid iCluster2b meanZ: "
                f"{mean_z.shape}"
            )

        lambdas = np.asarray(
            lambdas,
            dtype=np.float64,
        ).reshape(-1)

        if (
            lambdas.shape != (3,)
            or not np.isfinite(
                lambdas
            ).all()
        ):
            raise RuntimeError(
                "Invalid iCluster2b lambda vector: "
                f"{lambdas}"
            )

        metadata_df = pd.read_csv(
            output_dir / "metadata.tsv",
            sep="\t",
            dtype=str,
        )

        metadata = {
            str(r.key):
                str(r.value)
            for r in metadata_df.itertuples(
                index=False
            )
        }

    return {
        "latent":
            np.asarray(
                mean_z,
                dtype=np.float64,
            ),
        "lambda":
            lambdas.tolist(),
        "elapsed_seconds":
            float(
                metadata[
                    "elapsed_seconds"
                ]
            ),
        "derived_seed":
            int(seed),
    }


# =============================================================================
# Simulation dataset loading and post-fit metrics
# =============================================================================

def resolve_registry_dataset_path(
    project_root: Path,
    raw: str,
) -> Path:
    normalized = str(
        raw
    ).replace(
        "\\",
        "/",
    )

    path = Path(
        normalized
    )

    if path.is_absolute():
        return path

    return (
        project_root / path
    ).resolve()


def checked_simulation_path(project_root: Path, row: pd.Series) -> Path:
    """Integrity check for either observed-input or post-fit truth loading."""
    path = resolve_registry_dataset_path(project_root, row["path"])
    require_file(path)
    if path.stat().st_size != int(row["size_bytes"]):
        raise RuntimeError(f"Simulation dataset size mismatch: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != str(row["sha256"]):
        raise RuntimeError(
            f"Simulation dataset SHA mismatch: {observed_sha} != {row['sha256']}"
        )
    return path


def check_v2_npz_keys(npz: Any) -> None:
    """Check names only; do not decode truth or other unused array payloads."""
    keys = list(npz.files)
    if len(keys) != len(set(keys)):
        raise RuntimeError("Simulation NPZ contains duplicate array names.")
    required = {
        *NPZ_VIEW_KEYS.values(), "sample_index", "Z_biological_shared",
        "biological_factor_ids", "row_source_rna",
    }
    missing = required - set(keys)
    if missing:
        raise RuntimeError(f"Simulation NPZ missing required v2 arrays: {sorted(missing)}")
    if "Z_shared" in keys:
        raise RuntimeError("Unexpected legacy Z_shared in frozen v2 NPZ; no alias fallback allowed.")


def load_scientific_npz(
    project_root: Path,
    row: pd.Series,
) -> dict[str, np.ndarray]:
    """Load fit inputs only. Biological truth is decoded separately AFTER fit."""
    path = checked_simulation_path(project_root, row)
    n = int(row["n_samples"])
    with np.load(path, allow_pickle=False) as npz:
        check_v2_npz_keys(npz)
        # Deliberately do not index any truth/row_source array in this stage.
        views = {
            view: np.asarray(npz[key], dtype=np.float64)
            for view, key in NPZ_VIEW_KEYS.items()
        }
        sample_index = np.asarray(npz["sample_index"])
    for view, x in views.items():
        expected = (n, int(row[f"n_features_{view}"]))
        if x.shape != expected:
            raise RuntimeError(f"Simulation {view} shape {x.shape} != {expected}.")
        if np.isinf(x).any():
            raise RuntimeError(f"Simulation {view} contains infinite input values.")
    if (sample_index.dtype.kind not in "iu" or sample_index.shape != (n,)
            or not np.array_equal(sample_index, np.arange(n))):
        raise RuntimeError("Frozen v2 sample_index is not the canonical integer 0..n-1 sequence.")
    # NaNs, missing rows and intentionally corrupted view ordering remain untouched.
    return {**views, "sample_index": sample_index}


def load_scientific_truth(
    project_root: Path,
    row: pd.Series,
    linear: Any,
) -> np.ndarray:
    """Post-fit evaluation target: exact 05c1 integrated biological anchor truth.

    All biological columns (ALL3 and pair-view factors) are retained. No technical
    or private factors are substituted/concatenated. Observed rows are not repaired.
    """
    path = checked_simulation_path(project_root, row)
    n = int(row["n_samples"])
    k = int(row["n_biological_shared_factors"])
    with np.load(path, allow_pickle=False) as npz:
        check_v2_npz_keys(npz)
        arrays = {key: np.asarray(npz[key]) for key in (
            "Z_biological_shared", "biological_factor_ids", "row_source_rna",
        )}
    z = arrays["Z_biological_shared"]
    ids = arrays["biological_factor_ids"]
    anchor = arrays["row_source_rna"]
    if z.shape != (n, k) or z.dtype.kind != "f" or not np.isfinite(z).all():
        raise RuntimeError(f"Frozen biological truth geometry/values invalid: {z.shape} != {(n, k)}.")
    if ids.shape != (k,) or ids.dtype.kind not in "US":
        raise RuntimeError("Frozen biological_factor_ids geometry/dtype invalid.")
    if len(set(ids.tolist())) != k:
        raise RuntimeError("Duplicate biological_factor_ids.")
    # 05b v2 explicitly preserves RNA in canonical order, including corruption cases.
    if (anchor.dtype.kind not in "iu" or anchor.shape != (n,)
            or not np.array_equal(anchor, np.arange(n))):
        raise RuntimeError("Frozen v2 RNA anchor is not canonical; no pairing repair allowed.")
    truth, truth_ids = linear.integrated_anchor_truth(arrays, "biological")
    if (truth.shape != (n, k) or not np.array_equal(truth, z)
            or truth_ids != [str(value) for value in ids.tolist()]):
        raise RuntimeError("Frozen integrated biological anchor helper replay mismatch.")
    return np.asarray(truth, dtype=np.float64)


def evaluate_truth_metrics(
    linear: Any,
    latent: np.ndarray,
    z_shared: np.ndarray,
    factor_level_identifiable: bool,
    subspace_level_identifiable: bool,
) -> dict[str, Any]:
    latent = np.asarray(
        latent,
        dtype=np.float64,
    )

    z_shared = np.asarray(
        z_shared,
        dtype=np.float64,
    )

    out: dict[str, Any] = {
        "n_true_shared_factors":
            int(
                z_shared.shape[1]
            ),
        "shared_subspace_recovery":
            None,
        "shared_subspace_recovery_status":
            "NOT_APPLICABLE",
        "factor_alignment_correlation":
            None,
        "factor_alignment_status":
            "NOT_APPLICABLE",
        "factor_alignment_matches":
            [],
    }

    if z_shared.shape[1] == 0:
        out[
            "shared_subspace_recovery_status"
        ] = (
            "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS"
        )

        out[
            "factor_alignment_status"
        ] = (
            "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS"
        )

        return out

    if subspace_level_identifiable:
        (
            subspace_value,
            helper,
        ) = paper5_shared_subspace_recovery(
            linear,
            latent,
            z_shared,
        )

        out[
            "shared_subspace_recovery"
        ] = float(
            subspace_value
        )

        out[
            "shared_subspace_recovery_status"
        ] = helper[
            "status"
        ]

        out[
            "shared_subspace_helper_replay"
        ] = helper

    else:
        out[
            "shared_subspace_recovery_status"
        ] = (
            "NOT_PRIMARY_INTERPRETABLE_"
            "SUBSPACE_LEVEL_IDENTIFIABLE_FALSE"
        )

    if factor_level_identifiable:
        (
            factor_value,
            matches,
        ) = factor_alignment_correlation(
            linear,
            latent,
            z_shared,
        )

        out[
            "factor_alignment_correlation"
        ] = float(
            factor_value
        )

        out[
            "factor_alignment_status"
        ] = (
            "PASS_SIGN_PERMUTATION_INVARIANT"
        )

        out[
            "factor_alignment_matches"
        ] = matches

    else:
        out[
            "factor_alignment_status"
        ] = (
            "NOT_PRIMARY_INTERPRETABLE_"
            "FACTOR_LEVEL_IDENTIFIABLE_FALSE"
        )

    return out


def registry_metadata(
    row: pd.Series,
) -> dict[str, Any]:
    def maybe_float(
        name: str,
    ) -> float | None:
        text = str(
            row.get(
                name,
                "",
            )
        ).strip()

        if text == "":
            return None

        return float(
            text
        )

    partial_view_count = 0.0

    for col in [
        "rna_missing_partial_view",
        "methylation_missing_partial_view",
        "cnv_missing_partial_view",
    ]:
        value = maybe_float(
            col
        )

        if value is not None:
            partial_view_count += value

    return {
        "comparator_dataset_id":
            str(
                row[
                    "comparator_dataset_id"
                ]
            ),
        "scenario_id":
            str(
                row[
                    "scenario_id"
                ]
            ),
        "replicate_index":
            int(
                row[
                    "replicate_index"
                ]
            ),
        "simulation_seed":
            int(
                row[
                    "simulation_seed"
                ]
            ),
        "scenario_family":
            str(
                row.get(
                    "scenario_family",
                    "",
                )
            ),
        "n_samples":
            int(
                row[
                    "n_samples"
                ]
            ),
        "missingness_mode":
            str(
                row.get(
                    "missingness_mode",
                    "",
                )
            ),
        "batch_mode":
            str(
                row.get(
                    "batch_mode",
                    "",
                )
            ),
        "factor_level_identifiable":
            parse_bool(
                row[
                    "factor_level_identifiable"
                ]
            ),
        "subspace_level_identifiable":
            parse_bool(
                row[
                    "subspace_level_identifiable"
                ]
            ),
        "primary_truth_target":
            str(
                row.get(
                    "primary_truth_target",
                    "",
                )
            ),
        "is_missing_view_scenario":
            bool(
                partial_view_count
                > 0.0
            ),
        "source_npz_sha256":
            str(
                row[
                    "sha256"
                ]
            ),
    }


# =============================================================================
# Per-method scientific shard functions
# =============================================================================

def run_icluster_dataset(
    row: pd.Series,
    arrays: dict[str, np.ndarray],
    linear: Any,
    r_env: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    metadata = registry_metadata(
        row
    )

    dataset_id = metadata[
        "comparator_dataset_id"
    ]

    n = metadata[
        "n_samples"
    ]

    all_idx = np.arange(
        n,
        dtype=int,
    )

    raw_views = {
        view:
            arrays[
                view
            ]
        for view in VIEW_KEYS
    }

    (
        views,
        _,
        prep_audit,
    ) = adapt_views(
        raw_views,
        fit_indices=all_idx,
        output_indices=all_idx,
        dtype=np.float64,
    )

    derived_seed = derive_seed(
        [
            "iCluster2b",
            dataset_id,
            "dataset_scientific_tuning_and_fit",
        ],
        1103,
    )

    fit = run_icluster_fit(
        r_env,
        views,
        derived_seed,
    )

    # Truth is first touched here, after the scientific fit returns.
    truth_metrics = evaluate_truth_metrics(
        linear,
        latent=fit[
            "latent"
        ],
        z_shared=load_scientific_truth(project_root, row, linear),
        factor_level_identifiable=
            metadata[
                "factor_level_identifiable"
            ],
        subspace_level_identifiable=
            metadata[
                "subspace_level_identifiable"
            ],
    )

    return {
        "status":
            "PASS",
        "method":
            "iCluster2b",
        **metadata,
        "fit": {
            "derived_seed":
                int(
                    derived_seed
                ),
            "final_lambda":
                fit[
                    "lambda"
                ],
            "elapsed_seconds":
                fit[
                    "elapsed_seconds"
                ],
            "latent_shape":
                list(
                    fit[
                        "latent"
                    ].shape
                ),
            "latent_sha256":
                sha256_array(
                    fit[
                        "latent"
                    ]
                ),
            "preprocessing_audit":
                prep_audit,
            "latent_subspace_stability":
                (
                    "NOT_ESTIMABLE_WITH_"
                    "FROZEN_SINGLE_SEED_"
                    "POLICY_IN_05CMP2"
                ),
        },
        "metrics":
            truth_metrics,
        "scientific_firewalls": {
            "truth_used_during_fit_or_tuning":
                False,
            "outcome_clinical_biology_used":
                False,
            "TARGET_used":
                False,
            "05e_or_final_gate_changed":
                False,
        },
    }


def run_move_dataset(
    row: pd.Series,
    arrays: dict[str, np.ndarray],
    linear: Any,
    move_api: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    metadata = registry_metadata(
        row
    )

    raw_views = {
        view:
            arrays[
                view
            ]
        for view in VIEW_KEYS
    }

    selection = select_move_architecture(
        move_api,
        raw_views=raw_views,
        sample_ids=arrays[
            "sample_index"
        ],
        dataset_id=metadata[
            "comparator_dataset_id"
        ],
    )

    # Truth is first used here, after architecture selection and all fits finish.
    truth_metrics = evaluate_truth_metrics(
        linear,
        latent=selection[
            "canonical_latent"
        ],
        z_shared=load_scientific_truth(project_root, row, linear),
        factor_level_identifiable=
            metadata[
                "factor_level_identifiable"
            ],
        subspace_level_identifiable=
            metadata[
                "subspace_level_identifiable"
            ],
    )

    return {
        "status":
            "PASS",
        "method":
            "MOVE",
        **metadata,
        "architecture_selection": {
            k: v
            for k, v in selection.items()
            if k != "canonical_latent"
        },
        "fit": {
            "canonical_latent_shape":
                list(
                    selection[
                        "canonical_latent"
                    ].shape
                ),
            "canonical_latent_sha256":
                sha256_array(
                    selection[
                        "canonical_latent"
                    ]
                ),
            "selected_candidate_id":
                selection[
                    "selected_candidate_id"
                ],
            "latent_subspace_stability":
                selection[
                    "selected_latent_subspace_stability"
                ],
            "heldout_reconstruction":
                selection[
                    "selected_cv_mean_heldout_reconstruction_sse"
                ],
        },
        "metrics":
            truth_metrics,
        "scientific_firewalls": {
            "truth_used_during_architecture_selection_or_fit":
                False,
            "outcome_clinical_biology_used":
                False,
            "TARGET_used":
                False,
            "05e_or_final_gate_changed":
                False,
        },
    }


def failure_payload(
    method: str,
    row: pd.Series,
    first_error: str,
    second_error: str,
) -> dict[str, Any]:
    return {
        "status":
            "FAILED_AFTER_IDENTICAL_RETRY",
        "method":
            method,
        **registry_metadata(
            row
        ),
        "first_attempt_error":
            first_error,
        "second_attempt_error":
            second_error,
        "fallback_activated":
            False,
        "retuning_or_rescue":
            False,
        "scientific_firewalls": {
            "outcome_clinical_biology_used":
                False,
            "TARGET_used":
                False,
            "05e_or_final_gate_changed":
                False,
        },
    }


def execute_with_one_identical_retry(
    method: str,
    fn: Any,
    row: pd.Series,
) -> dict[str, Any]:
    errors: list[str] = []

    for attempt in [
        1,
        2,
    ]:
        try:
            result = fn()
            result[
                "attempt_count"
            ] = attempt
            result[
                "identical_retry_used"
            ] = bool(
                attempt == 2
            )
            return result

        except Exception:
            errors.append(
                traceback.format_exc()
            )
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return failure_payload(
        method,
        row,
        errors[0],
        errors[1],
    )


# =============================================================================
# Synthetic preauthorization tests
# =============================================================================

def synthetic_views(
    seed: int = 50303,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(
        seed
    )

    n = 24
    z = rng.normal(
        size=(
            n,
            3,
        )
    )

    views: dict[
        str,
        np.ndarray,
    ] = {}

    for view, p in zip(
        VIEW_KEYS,
        [
            18,
            24,
            15,
        ],
        strict=True,
    ):
        w = rng.normal(
            size=(
                3,
                p,
            )
        )

        x = (
            z @ w
            + rng.normal(
                scale=0.5,
                size=(
                    n,
                    p,
                ),
            )
        )

        views[
            view
        ] = x.astype(
            np.float64
        )

    views[
        "rna"
    ][2, 0] = np.nan

    views[
        "methylation"
    ][7, :] = np.nan

    views[
        "cnv"
    ][11, 3:6] = np.nan

    sample_ids = np.asarray(
        [
            f"SYNTH_{i:03d}"
            for i in range(
                n
            )
        ]
    )

    return (
        views,
        z.astype(
            np.float64
        ),
        sample_ids,
    )


def run_synthetic_icluster_test(
    r_env: dict[str, Any],
    views: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx = np.arange(
        next(
            iter(
                views.values()
            )
        ).shape[0],
        dtype=int,
    )

    adapted, _, _ = adapt_views(
        views,
        fit_indices=idx,
        output_indices=idx,
        dtype=np.float64,
    )

    seed = derive_seed(
        [
            "iCluster2b",
            "SYNTHETIC_PREAUTH",
            "dataset_scientific_tuning_and_fit",
        ],
        1103,
    )

    fit = run_icluster_fit(
        r_env,
        adapted,
        seed,
    )

    return {
        "latent_shape":
            list(
                fit[
                    "latent"
                ].shape
            ),
        "lambda":
            fit[
                "lambda"
            ],
        "finite":
            bool(
                np.isfinite(
                    fit[
                        "latent"
                    ]
                ).all()
            ),
    }


def run_synthetic_move_test(
    move_api: dict[str, Any],
    views: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx = np.arange(
        next(
            iter(
                views.values()
            )
        ).shape[0],
        dtype=int,
    )

    (
        adapted,
        masks,
        _,
    ) = adapt_views(
        views,
        fit_indices=idx,
        output_indices=idx,
        dtype=np.float32,
    )

    seed = derive_seed(
        [
            "MOVE",
            "SYNTHETIC_PREAUTH",
            "MOVE_H64_32",
            "full_data_stability",
        ],
        1103,
    )

    fit = fit_move(
        move_api,
        adapted,
        hidden=[
            64,
            32,
        ],
        seed=seed,
        epochs=2,
        kld_warmup_steps=[
            1,
            1,
            2,
            2,
        ],
    )

    sse = move_reconstruction_sse(
        fit[
            "model"
        ],
        adapted,
        masks,
    )

    shape = list(
        fit[
            "latent"
        ].shape
    )

    finite = bool(
        np.isfinite(
            fit[
                "latent"
            ]
        ).all()
    )

    del fit["model"]
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "latent_shape":
            shape,
        "finite":
            finite,
        "synthetic_reconstruction_sse":
            float(
                sse
            ),
    }


def run_synthetic_metric_test(
    linear: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(
        88017
    )

    truth = rng.normal(
        size=(
            40,
            3,
        )
    )

    rotation, _ = np.linalg.qr(
        rng.normal(
            size=(
                3,
                3,
            )
        )
    )

    estimated = truth @ rotation

    extra = rng.normal(
        size=(
            40,
            2,
        )
    )

    estimated5 = np.column_stack(
        [
            estimated,
            extra,
        ]
    )

    (
        subspace,
        replay,
    ) = paper5_shared_subspace_recovery(
        linear,
        estimated5,
        truth,
    )

    if subspace < 1.0 - 1e-10:
        raise RuntimeError(
            "Synthetic true-span subspace recovery "
            f"should be ~1; got {subspace}"
        )

    perm = np.array(
        [
            2,
            0,
            1,
        ]
    )

    signs = np.array(
        [
            -1.0,
            1.0,
            -1.0,
        ]
    )

    permuted = (
        truth[:, perm]
        * signs
    )

    permuted5 = np.column_stack(
        [
            permuted,
            rng.normal(
                size=(
                    40,
                    2,
                )
            ),
        ]
    )

    (
        factor,
        details,
    ) = factor_alignment_correlation(
        linear,
        permuted5,
        truth,
    )

    if factor < 1.0 - 1e-10:
        raise RuntimeError(
            "Synthetic sign/permutation factor alignment "
            f"should be ~1; got {factor}"
        )

    return {
        "subspace_recovery":
            float(
                subspace
            ),
        "subspace_helper_replay":
            replay,
        "factor_alignment":
            float(
                factor
            ),
        "factor_matches":
            details,
    }


def run_synthetic_selection_logic_test() -> dict[str, Any]:
    ids = np.asarray(
        [
            "S10",
            "S2",
            "S1",
            "S11",
            "S3",
            "S20",
            "S4",
            "S5",
            "S6",
            "S7",
        ]
    )

    folds1 = deterministic_sample_folds(
        ids
    )
    folds2 = deterministic_sample_folds(
        ids
    )

    if any(
        not np.array_equal(
            a,
            b,
        )
        for a, b in zip(
            folds1,
            folds2,
            strict=True,
        )
    ):
        raise RuntimeError(
            "Deterministic sample-ID folds are not deterministic."
        )

    means = {
        "A": 100.0,
        "B": 101.5,
        "C": 106.0,
    }

    best_se = 2.0

    retained = sorted(
        [
            k
            for k, v in means.items()
            if v
            <= means["A"]
            + best_se
        ]
    )

    if retained != [
        "A",
        "B",
    ]:
        raise RuntimeError(
            "Synthetic MOVE one-SE rule replay failed."
        )

    stability = [
        {
            "candidate_id": "A",
            "stability": 0.80,
            "params": 200,
        },
        {
            "candidate_id": "B",
            "stability": 0.84,
            "params": 300,
        },
    ]

    selected = sorted(
        stability,
        key=lambda r: (
            -r[
                "stability"
            ],
            r[
                "params"
            ],
            r[
                "candidate_id"
            ],
        ),
    )[0][
        "candidate_id"
    ]

    if selected != "B":
        raise RuntimeError(
            "Synthetic MOVE stage-2 selection rule replay failed."
        )

    return {
        "deterministic_folds":
            "PASS",
        "one_se_retained":
            retained,
        "stage2_selected":
            selected,
    }


def run_synthetic_npz_schema_test(linear: Any) -> dict[str, Any]:
    """Serialize fresh artificial v2-format NPZs and exercise the ACTUAL loaders.

    These files are unrelated to any frozen dataset values. The temporary NPZs
    include unused technical truth so that read tracing can prove its exclusion.
    """
    import io
    import zipfile
    from unittest.mock import patch

    rng = np.random.default_rng(6202608)
    n = 24
    cases = [
        ("all3", 4, True, True),
        ("partial_RNA_METH", 2, True, True),
        ("partial_RNA_CNV", 2, True, True),
        ("partial_METH_CNV", 2, True, True),
        ("mixed_support_and_pair_corruption", 4, True, True),
        ("rotational_subspace_only", 2, False, True),
        ("technical_only_biological_null", 0, False, False),
    ]
    positive: list[str] = []
    negative: list[str] = []
    original_load = np.load
    phase = "observed"
    reads: list[tuple[str, str]] = []

    class TracedNPZ:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.files = inner.files
        def __enter__(self) -> Any:
            return self
        def __exit__(self, *args: Any) -> None:
            self.inner.close()
        def __getitem__(self, key: str) -> Any:
            reads.append((phase, key))
            allowed = (set(NPZ_VIEW_KEYS.values()) | {"sample_index"}
                       if phase == "observed" else
                       {"Z_biological_shared", "biological_factor_ids", "row_source_rna"})
            if key not in allowed:
                raise RuntimeError(f"Synthetic payload access firewall failed: {phase}/{key}")
            return self.inner[key]

    def traced_load(*args: Any, **kwargs: Any) -> Any:
        return TracedNPZ(original_load(*args, **kwargs))

    def save(path: Path, data: dict[str, np.ndarray], compressed: bool = False) -> None:
        compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(path, "w", compression=compression) as archive:
            for key in sorted(data):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, data[key], version=(2, 0), allow_pickle=False)
                archive.writestr(key + ".npy", buffer.getvalue())

    def expect_stop(label: str, fn: Any) -> None:
        try:
            fn()
        except (RuntimeError, ValueError, KeyError):
            negative.append(label)
        else:
            raise RuntimeError(f"Synthetic NPZ negative test failed to stop: {label}")

    with tempfile.TemporaryDirectory(prefix="paper5_cmp3_v6_schema_") as td:
        root = Path(td)
        path = root / "fixture.npz"
        for case_number, (name, k, factor_flag, subspace_flag) in enumerate(cases):
            z = rng.normal(size=(n, k)).astype(np.float32)
            data = {key: rng.normal(size=(n, p)).astype(np.float32)
                    for key, p in zip(NPZ_VIEW_KEYS.values(), (12, 14, 8), strict=True)}
            data["X_methylation"][0, :] = np.nan
            data["X_cnv"][1, 0] = np.nan
            if "corruption" in name:
                data["X_methylation"] = data["X_methylation"][::-1].copy()
                data["X_cnv"] = np.roll(data["X_cnv"], 3, axis=0)
            support_lookup = {
                "partial_RNA_METH": [1, 1, 0],
                "partial_RNA_CNV": [1, 0, 1],
                "partial_METH_CNV": [0, 1, 1],
            }
            support = np.tile(np.asarray(support_lookup.get(name, [1, 1, 1]),
                                         dtype=np.uint8), (k, 1))
            if name == "mixed_support_and_pair_corruption":
                support = np.asarray([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
                                     dtype=np.uint8)
            data.update({
                "biological_support_mask": support,
                "row_source_methylation": (np.arange(n, dtype=np.int32)[::-1].copy()
                                           if "corruption" in name else np.arange(n, dtype=np.int32)),
                "row_source_cnv": (np.roll(np.arange(n, dtype=np.int32), 3)
                                   if "corruption" in name else np.arange(n, dtype=np.int32)),
                "sample_index": np.arange(n, dtype=np.int32),
                "Z_biological_shared": z,
                "biological_factor_ids": np.asarray(
                    [f"BIO_SHARED_{i+1:03d}" for i in range(k)], dtype="U32"),
                "row_source_rna": np.arange(n, dtype=np.int32),
                "Z_technical": rng.normal(size=(n, 2)).astype(np.float32),
                "Z_private_rna": rng.normal(size=(n, 2)).astype(np.float32),
            })
            save(path, data, compressed=bool(case_number % 2))
            row = pd.Series({
                "path": path.name, "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size, "n_samples": n,
                "n_features_rna": 12, "n_features_methylation": 14,
                "n_features_cnv": 8, "n_biological_shared_factors": k,
            })
            with patch.object(np, "load", side_effect=traced_load):
                phase = "observed"
                observed = load_scientific_npz(root, row)
                if set(observed) != {*VIEW_KEYS, "sample_index"}:
                    raise RuntimeError("Observed loader exposed non-fit arrays.")
                for view, key in NPZ_VIEW_KEYS.items():
                    if not np.array_equal(observed[view], data[key], equal_nan=True):
                        raise RuntimeError("Observed loader changed values/missingness/order.")
                phase = "truth"
                truth = load_scientific_truth(root, row, linear)
            if not np.array_equal(truth, z):
                raise RuntimeError("Biological truth was filtered/replaced/reordered.")
            latent = np.concatenate((z[:, ::-1] * -1.0,
                                     rng.normal(size=(n, 5-k))), axis=1)
            metrics = evaluate_truth_metrics(linear, latent, truth, factor_flag, subspace_flag)
            if k == 0:
                if metrics["shared_subspace_recovery_status"] != "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS":
                    raise RuntimeError("Null truth was not retained as NOT_APPLICABLE.")
            else:
                if not np.isclose(metrics["shared_subspace_recovery"], 1.0, rtol=0, atol=1e-10):
                    raise RuntimeError("Serialized subspace metric replay failed.")
                if factor_flag:
                    if not np.isclose(metrics["factor_alignment_correlation"], 1.0, rtol=0, atol=1e-10):
                        raise RuntimeError("Serialized factor-alignment metric replay failed.")
                elif metrics["factor_alignment_correlation"] is not None:
                    raise RuntimeError("Rotational fixture incorrectly scored factor alignment.")
            positive.append(name)

        # Deliberately invalid artificial files, never frozen scientific datasets.
        pristine = {key: value.copy() for key, value in data.items()}
        pristine_row = row.copy()
        def exercise_bad(label: str, changes: Any, truth_stage: bool = False) -> None:
            bad = {key: value.copy() for key, value in pristine.items()}
            changes(bad)
            save(path, bad)
            bad_row = pristine_row.copy()
            bad_row["size_bytes"] = path.stat().st_size
            bad_row["sha256"] = sha256_file(path)
            fn = (lambda: load_scientific_truth(root, bad_row, linear)) if truth_stage else (
                lambda: load_scientific_npz(root, bad_row))
            expect_stop(label, fn)
        exercise_bad("legacy_Z_shared_only_rejected",
                     lambda d: d.update({"Z_shared": d.pop("Z_biological_shared")}))
        exercise_bad("dual_truth_keys_rejected", lambda d: d.update({"Z_shared": d["Z_biological_shared"]}))
        exercise_bad("missing_biological_truth_rejected", lambda d: d.pop("Z_biological_shared"))
        exercise_bad("sample_index_reordering_rejected",
                     lambda d: d.update({"sample_index": d["sample_index"][::-1]}))
        exercise_bad("view_shape_mismatch_rejected", lambda d: d.update({"X_rna": d["X_rna"][:-1]}))
        exercise_bad("RNA_anchor_repair_forbidden",
                     lambda d: d.update({"row_source_rna": d["row_source_rna"][::-1]}), True)
        exercise_bad("biological_column_count_mismatch_rejected",
                     lambda d: d.update({"Z_biological_shared": np.zeros((n, 1), dtype=np.float32)}), True)
        save(path, pristine)
        row = pristine_row.copy()
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
        bad_size = row.copy(); bad_size["size_bytes"] += 1
        expect_stop("size_mismatch_rejected", lambda: load_scientific_npz(root, bad_size))
        bad_sha = row.copy(); bad_sha["sha256"] = "0" * 64
        expect_stop("SHA_mismatch_rejected", lambda: load_scientific_npz(root, bad_sha))

    return {
        "status": "PASS_SERIALIZED_V2_NPZ_LOADERS_AND_TRUTH_SEMANTICS",
        "positive_cases": positive, "negative_cases": negative,
        "observed_input_payloads_only": True,
        "truth_payload_read_only_in_explicit_truth_stage": True,
        "technical_and_private_truth_payloads_decoded": False,
        "biological_columns_filtered_by_view_support": False,
        "RNA_anchor_helper": "05c1.integrated_anchor_truth(arrays, 'biological')",
        "frozen_scientific_NPZ_opened": False,
        "tracked_payload_reads": len(reads),
    }


def loader_order(loader):
    ids, sizes = [], []
    for cat, con in loader:
        ids.extend(int(x) for x in con[:, 0].tolist())
        sizes.append(int(con.shape[0]))
    return ids, sizes

def exercise_orders(old_loader, candidate, api, np, torch):
    sizes = list(range(2, 101)) + [113, 129, 145, 161]
    cases = []
    for n in sizes:
        views = [np.arange(n, dtype=np.float32).reshape(n, 1),
                 np.ones((n, 2), dtype=np.float32), np.ones((n, 2), dtype=np.float32)]
        for seed in (1701, 2701):
            old = old_loader(api["make_dataloader"], views, min(16, n), True, seed)
            new = candidate(api["make_dataloader"], views, min(16, n), True, seed)
            for epoch in range(2):
                a, old_sizes = loader_order(old)
                b, new_sizes = loader_order(new)
                bn_require(a == b and sorted(b) == list(range(n)), "Row loss/duplication/order drift")
                bn_require(len(old_sizes) == len(new_sizes), "Optimizer step count changed")
                bn_require(all(2 <= size <= min(16, n) for size in new_sizes), "Singleton remains")
                bn_require(torch.equal(old.generator.get_state(), new.generator.get_state()),
                        "Loader generator state changed")
                if n % min(16, n) != 1:
                    bn_require(old_sizes == new_sizes, "Unaffected batch geometry changed")
            proj = candidate(api["make_dataloader"], views, min(16, n), False, seed)
            ids, proj_sizes = loader_order(proj)
            bn_require(ids == list(range(n)), "Projection order changed")
            cases.append({"n": n, "seed": seed, "epochs": 2,
                          "original_sizes": old_sizes, "candidate_sizes": new_sizes,
                          "projection_sizes": proj_sizes})
    return {"status": "PASS", "case_count": len(cases), "cases": cases}


def run_synthetic_move_batch_test(api: dict[str, Any]) -> dict[str, Any]:
    """Actual production loader/selection path, artificial data and 2-epoch fits only.

    FunctionType isolates synthetic budgets without changing scientific globals.
    No project runner, scientific NPZ, old metric or R result is loaded here.
    """
    order_report = exercise_orders(_move_loader_before_bn_amendment, move_loader, api, np, torch)
    rng = np.random.default_rng(740091)
    raw = {k: rng.normal(size=(62, p)) for k, p in zip(VIEW_KEYS, [500, 4000, 500])}
    ids = np.arange(62, dtype=np.int32)
    folds = deterministic_sample_folds(ids, 5)
    train = np.setdiff1d(ids, folds[0])
    bn_require(len(train) == 49, "Batch regression fixture first train size differs")
    adapted, _, _ = adapt_views(raw, train, train, np.float32)
    old_scope = dict(globals())
    old_scope["move_loader"] = _move_loader_before_bn_amendment
    old_fit = types.FunctionType(fit_move.__code__, old_scope, "synthetic_old_batch_fit",
                                 fit_move.__defaults__, fit_move.__closure__)
    baseline_error = None
    try:
        bad_fit = old_fit(api, adapted, [64, 32], 740093, 2, [1, 1, 2, 2])
        del bad_fit
    except ValueError as exc:
        bn_require("Expected more than 1 value per channel when training" in str(exc)
                   and "[1, 64]" in str(exc), "Unexpected baseline batch regression error")
        baseline_error = str(exc)
    bn_require(baseline_error is not None, "Original singleton error did not reproduce")
    gc.collect()
    torch.cuda.empty_cache()
    print("Original batch policy / artificial 49-row fit: expected singleton error reproduced", flush=True)
    calls: list[dict[str, Any]] = []
    def traced_fit(api_arg, fitted_views, hidden, seed, epochs, kld_warmup_steps):
        bn_require(epochs == 2 and list(kld_warmup_steps) == [1, 1, 2, 2],
                   "Synthetic batching regression must use the two-epoch budget only")
        value = fit_move(api_arg, fitted_views, hidden, seed, epochs, kld_warmup_steps)
        calls.append({"n_fit": len(fitted_views[0]), "hidden": list(hidden), "epochs": epochs,
                      "latent_shape": list(value["latent"].shape),
                      "finite": bool(np.isfinite(value["latent"]).all())})
        return value
    scope = dict(globals())
    scope.update({"MOVE_EPOCHS": 2, "MOVE_KLD_WARMUP_STEPS": [1, 1, 2, 2], "fit_move": traced_fit})
    selection = types.FunctionType(select_move_architecture.__code__, scope,
                                    "synthetic_batch_selection", select_move_architecture.__defaults__,
                                    select_move_architecture.__closure__)
    try:
        chosen = selection(api, raw, ids, "SYNTHETIC_ONLY_BN_TAIL_N62_740091")
        bn_require(len(chosen["stage1"]) == 3 and all(len(r["fold_sse"]) == 5 for r in chosen["stage1"]),
                   "Incomplete three-architecture/five-fold synthetic batch regression")
        bn_require([c["n_fit"] for c in calls[:15]] == [49, 49, 50, 50, 50] * 3,
                   "Unexpected synthetic CV training sizes")
        stability_count = len(calls) - 15
        bn_require(stability_count in (3, 6, 9) and all(c["n_fit"] == 62 for c in calls[15:]),
                   "Incomplete full-data stability fits")
        latent = np.asarray(chosen["canonical_latent"])
        bn_require(latent.shape == (62, 5) and np.isfinite(latent).all() and all(c["finite"] for c in calls),
                   "Invalid latent output in synthetic batch regression")
        bn_require(all(len(r["seed_runs"]) == 3 for r in chosen["stage2"]), "Stability seeds incomplete")
        bn_require(MOVE_EPOCHS == 300 and MOVE_KLD_WARMUP_STEPS == [1, 75, 150, 300],
                   "Scientific training constants changed during preflight")
        return {"status": "PASS_V7_ACTUAL_MOVE_SYNTHETIC_BATCHING", "policy_id": BN_AMENDMENT["policy"]["policy_id"],
                "baseline_singleton_error_reproduced": True, "baseline_error": baseline_error,
                "batch_order_tests": order_report, "cv_fit_count": 15,
                "stability_fit_count": stability_count, "fit_calls": calls,
                "canonical_latent_shape": [62, 5], "all_latents_finite": True,
                "synthetic_epochs": 2, "scientific_epochs_unchanged": MOVE_EPOCHS,
                "scientific_module_globals_modified": False, "frozen_scientific_npz_opened": False}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def verify_bn_preflight(report: dict[str, Any]) -> None:
    bn_require(report.get("batch_amendment", {}).get("evidence_sha256") == BN_EVIDENCE_SHA256,
               "Preflight does not bind the exact batch amendment and old history")
    test = report.get("tests", {}).get("singleton_batch_training_regression", {})
    bn_require(test.get("status") == "PASS_V7_ACTUAL_MOVE_SYNTHETIC_BATCHING"
               and test.get("cv_fit_count") == 15 and test.get("stability_fit_count") in (3, 6, 9)
               and test.get("canonical_latent_shape") == [62, 5]
               and test.get("all_latents_finite") is True
               and test.get("scientific_epochs_unchanged") == 300
               and test.get("batch_order_tests", {}).get("status") == "PASS"
               and test.get("batch_order_tests", {}).get("case_count") == 206
               and test.get("baseline_singleton_error_reproduced") is True
               and test.get("scientific_module_globals_modified") is False
               and test.get("frozen_scientific_npz_opened") is False,
               "New v7 batching regression preflight is missing or invalid")


def preauthorization_main(
    script_path: Path,
    paths: dict[str, Path],
    contract: dict[str, Any],
    registry: pd.DataFrame,
) -> None:
    print(
        "Execution mode: "
        "SYNTHETIC_ONLY_PREAUTHORIZATION"
    )
    print(
        "Frozen 360-dataset simulation NPZ access: FORBIDDEN"
    )
    print(
        "New v7 scientific comparator results: FORBIDDEN"
    )

    history_state = verify_bn_history(paths)
    print("Historical v6 shards: 82 verified unchanged; numeric metrics not decoded")

    result_root = paths[
        "result_root"
    ]

    if (
        result_root.exists()
        and any(
            p.is_file()
            for p in result_root.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "Scientific comparator result files exist "
            "before 05cmp3 source authorization."
        )

    if (
        paths[
            "result_lock_dir"
        ].exists()
        and any(
            p.is_file()
            for p in paths[
                "result_lock_dir"
            ].rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "05cmp4 result-lock files exist "
            "before 05cmp3 authorization."
        )

    preflight_path = paths[
        "preflight"
    ]

    runner_sha = sha256_file(
        script_path
    )

    contract_sha = sha256_file(
        paths[
            "cmp2_contract"
        ]
    )

    if preflight_path.exists():
        old = load_json(
            preflight_path
        )

        if (
            old.get(
                "status"
            )
            != PREAUTH_STATUS
            or old.get(
                "runner_sha256"
            )
            != runner_sha
            or old.get(
                "cmp2_contract_sha256"
            )
            != contract_sha
        ):
            raise RuntimeError(
                "Existing 05cmp3 preflight does not bind "
                "this exact runner/contract."
            )

        if old.get("tests", {}).get("serialized_v2_npz_schema_and_truth", {}).get("status") != (
            "PASS_SERIALIZED_V2_NPZ_LOADERS_AND_TRUTH_SEMANTICS"
        ):
            raise RuntimeError("Existing v7 preflight lacks serialized NPZ schema PASS.")
        verify_bn_preflight(old)
        print(
            "Existing exact synthetic preflight: PASS"
        )
        print(
            "No scientific NPZ opened; historical v6 shard bytes checked by SHA only."
        )
        print(
            "Next: freeze 05cmp3c source authorization; "
            "do not edit this runner."
        )
        return

    linear = import_linear_helper(
        paths[
            "linear_helper"
        ]
    )

    move_api = import_move_api()

    r_env = verify_r_library(
        paths[
            "paper5"
        ]
    )

    (
        views,
        truth,
        sample_ids,
    ) = synthetic_views()

    schema_test = run_synthetic_npz_schema_test(linear)

    metric_test = run_synthetic_metric_test(
        linear
    )

    selection_test = (
        run_synthetic_selection_logic_test()
    )

    move_test = run_synthetic_move_test(
        move_api,
        views,
    )

    icluster_test = run_synthetic_icluster_test(
        r_env,
        views,
    )

    batch_test = run_synthetic_move_batch_test(move_api)
    bn_require(verify_bn_history(paths) == history_state, "History changed during synthetic tests")

    report = {
        "batch_amendment": history_state,
        "status":
            PREAUTH_STATUS,
        "script_version":
            SCRIPT_VERSION,
        "runner_sha256":
            runner_sha,
        "cmp2_contract_sha256":
            contract_sha,
        "cmp2_registry_sha256":
            sha256_file(
                paths[
                    "cmp2_registry"
                ]
            ),
        "created_at_utc":
            utc_now_iso(),
        "execution_mode":
            "SYNTHETIC_ONLY_PREAUTHORIZATION",
        "linear_helper_source": {
            "path":
                relative_to_project(
                    paths[
                        "linear_helper"
                    ],
                    paths[
                        "project_root"
                    ],
                ),
            "sha256":
                sha256_file(
                    paths[
                        "linear_helper"
                    ]
                ),
        },
        "environment": {
            "move_version":
                move_api[
                    "move_version"
                ],
            "torch_version":
                torch.__version__,
            "cuda_build":
                str(
                    torch.version.cuda
                ),
            "gpu":
                torch.cuda.get_device_name(
                    0
                ),
            "rscript_sha256":
                sha256_file(
                    r_env[
                        "rscript"
                    ]
                ),
            "icluster_package_tree_sha256":
                r_env[
                    "package_tree_sha256"
                ],
        },
        "schema_amendment": {
            "historical_authorized_v5_preserved": True,
            "current_invocation_scientific_npz_values_decoded": False,
            "previous_v5_failure": "required Z_shared absent before array extraction or fitting",
            "truth_key": "Z_biological_shared",
            "truth_stage": "after each method fit/architecture selection completes",
            "history_bindings": verify_schema_amendment_inputs(paths),
        },
        "tests": {
            "singleton_batch_training_regression": batch_test,
            "serialized_v2_npz_schema_and_truth": schema_test,
            "frozen_upstream_metadata":
                "PASS",
            "deterministic_seed_derivation":
                (
                    "PASS"
                    if derive_seed(
                        [
                            "X",
                            "Y",
                        ],
                        1103,
                    )
                    == derive_seed(
                        [
                            "X",
                            "Y",
                        ],
                        1103,
                    )
                    else "FAIL"
                ),
            "adapter_feature_and_whole_view_missing":
                "PASS",
            "paper5_subspace_helper_replay":
                metric_test,
            "move_selection_logic":
                selection_test,
            "move_exact_api_synthetic_fit":
                move_test,
            "icluster2b_exact_tune_api_synthetic_fit":
                icluster_test,
        },
        "scientific_state": {
            "frozen_simulation_npz_opened":
                False,
            "frozen_simulation_truth_opened":
                False,
            "scientific_comparator_fit_performed":
                False,
            "TARGET_accessed":
                False,
            "outcomes_clinical_biology_accessed":
                False,
        },
    }

    verify_bn_preflight(report)
    write_json_atomic(
        report,
        preflight_path,
    )

    print(
        "Exact frozen 05cmp2/05cmp1 identities: PASS"
    )
    print(
        "Frozen Paper 5 subspace helper synthetic replay: PASS"
    )
    print(
        "Factor-alignment sign/permutation fixture: PASS"
    )
    print(
        "MOVE deterministic fold/one-SE/stability selection logic: PASS"
    )
    print(
        "MOVE exact CUDA API synthetic fit: PASS"
    )
    print(
        "iCluster2b exact tune.iCluster2b synthetic fit: PASS"
    )
    print("Serialized v2 NPZ observed/truth loader and semantics: PASS")
    print("MOVE singleton-tail production loader / 3 architectures x 5 folds / stability: PASS")
    print(
        "Frozen 360-dataset simulation NPZ opened: NO"
    )
    print(
        "Frozen simulation truth opened: NO"
    )
    print(
        "Scientific comparator fit performed: NO"
    )
    print(
        "TARGET/outcomes/clinical/biology accessed: NO"
    )
    print(
        "=" * 112
    )
    print(
        "Paper 5 05cmp3 scientific comparator runner "
        "synthetic preauthorization: PASS"
    )
    print(
        "=" * 112
    )
    print(
        "Preflight: method_contract/05cmp3_preflight/"
        "secondary_comparator_runner_synthetic_preflight_v7.json"
    )
    print(
        "Next: freeze the exact runner SHA with a separate "
        "05cmp3c source authorizer. Do not edit or run real "
        "scientific data before authorization."
    )
    print(
        "=" * 112
    )


# =============================================================================
# Authorized real execution
# =============================================================================

def shard_path(
    result_root: Path,
    method: str,
    dataset_id: str,
) -> Path:
    safe = (
        dataset_id
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
        .replace(
            ":",
            "_",
        )
    )

    return (
        result_root
        / "shards"
        / method
        / f"{safe}.json"
    )


def verify_existing_shard(
    path: Path,
    runner_sha: str,
    contract_sha: str,
    method: str,
    dataset_id: str,
) -> dict[str, Any]:
    shard = load_json(
        path
    )

    if shard.get(
        "runner_sha256"
    ) != runner_sha:
        raise RuntimeError(
            "Existing shard runner SHA mismatch: "
            f"{path}"
        )

    if shard.get(
        "cmp2_contract_sha256"
    ) != contract_sha:
        raise RuntimeError(
            "Existing shard contract SHA mismatch: "
            f"{path}"
        )

    if shard.get(
        "method"
    ) != method:
        raise RuntimeError(
            "Existing shard method mismatch: "
            f"{path}"
        )

    if shard.get(
        "comparator_dataset_id"
    ) != dataset_id:
        raise RuntimeError(
            "Existing shard dataset ID mismatch: "
            f"{path}"
        )

    if shard.get(
        "status"
    ) not in {
        "PASS",
        "FAILED_AFTER_IDENTICAL_RETRY",
    }:
        raise RuntimeError(
            "Existing shard has invalid status: "
            f"{path}"
        )

    return shard


def write_scientific_shard(
    result_root: Path,
    payload: dict[str, Any],
    runner_sha: str,
    contract_sha: str,
) -> Path:
    payload = {
        **payload,
        "runner_sha256":
            runner_sha,
        "cmp2_contract_sha256":
            contract_sha,
        "created_at_utc":
            utc_now_iso(),
    }

    path = shard_path(
        result_root,
        str(
            payload[
                "method"
            ]
        ),
        str(
            payload[
                "comparator_dataset_id"
            ]
        ),
    )

    if path.exists():
        raise RuntimeError(
            "Refusing to overwrite existing scientific shard: "
            f"{path}"
        )

    write_json_atomic(
        payload,
        path,
    )

    return path


def make_dataset_metrics_table(
    shards: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for s in shards:
        metrics = (
            s.get(
                "metrics",
                {},
            )
            if s.get(
                "status"
            )
            == "PASS"
            else {}
        )

        fit = (
            s.get(
                "fit",
                {},
            )
            if s.get(
                "status"
            )
            == "PASS"
            else {}
        )

        subspace = metrics.get(
            "shared_subspace_recovery"
        )

        rows.append(
            {
                "method":
                    s[
                        "method"
                    ],
                "comparator_dataset_id":
                    s[
                        "comparator_dataset_id"
                    ],
                "scenario_id":
                    s[
                        "scenario_id"
                    ],
                "replicate_index":
                    int(
                        s[
                            "replicate_index"
                        ]
                    ),
                "simulation_seed":
                    int(
                        s[
                            "simulation_seed"
                        ]
                    ),
                "scenario_family":
                    s.get(
                        "scenario_family",
                        "",
                    ),
                "missingness_mode":
                    s.get(
                        "missingness_mode",
                        "",
                    ),
                "is_missing_view_scenario":
                    bool(
                        s.get(
                            "is_missing_view_scenario",
                            False,
                        )
                    ),
                "factor_level_identifiable":
                    bool(
                        s.get(
                            "factor_level_identifiable",
                            False,
                        )
                    ),
                "subspace_level_identifiable":
                    bool(
                        s.get(
                            "subspace_level_identifiable",
                            False,
                        )
                    ),
                "status":
                    s[
                        "status"
                    ],
                "attempt_count":
                    int(
                        s.get(
                            "attempt_count",
                            0,
                        )
                    ),
                "shared_subspace_recovery":
                    subspace,
                "factor_alignment_correlation":
                    metrics.get(
                        "factor_alignment_correlation"
                    ),
                "missing_view_robustness_shared_subspace":
                    (
                        subspace
                        if bool(
                            s.get(
                                "is_missing_view_scenario",
                                False,
                            )
                        )
                        else None
                    ),
                "latent_subspace_stability":
                    (
                        fit.get(
                            "latent_subspace_stability"
                        )
                        if isinstance(
                            fit.get(
                                "latent_subspace_stability"
                            ),
                            (
                                int,
                                float,
                                np.integer,
                                np.floating,
                            ),
                        )
                        else None
                    ),
                "heldout_reconstruction":
                    fit.get(
                        "heldout_reconstruction"
                    ),
                "selected_candidate_id":
                    fit.get(
                        "selected_candidate_id",
                        "",
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def scenario_summary_table(
    dataset_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for (
        method,
        scenario,
    ), part in dataset_metrics.groupby(
        [
            "method",
            "scenario_id",
        ],
        sort=True,
    ):
        pass_part = part.loc[
            part[
                "status"
            ].eq(
                "PASS"
            )
        ]

        row: dict[str, Any] = {
            "method":
                method,
            "scenario_id":
                scenario,
            "n_datasets":
                int(
                    part.shape[0]
                ),
            "n_pass":
                int(
                    pass_part.shape[0]
                ),
            "n_failed":
                int(
                    part.shape[0]
                    - pass_part.shape[0]
                ),
            "is_missing_view_scenario":
                bool(
                    part[
                        "is_missing_view_scenario"
                    ].astype(
                        bool
                    ).any()
                ),
        }

        for metric in [
            "shared_subspace_recovery",
            "factor_alignment_correlation",
            "missing_view_robustness_shared_subspace",
            "latent_subspace_stability",
            "heldout_reconstruction",
        ]:
            values = (
                pd.to_numeric(
                    pass_part[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    float
                )
            )

            row[
                f"{metric}_n"
            ] = int(
                values.size
            )

            row[
                f"{metric}_mean"
            ] = (
                float(
                    np.mean(
                        values
                    )
                )
                if values.size
                else np.nan
            )

            row[
                f"{metric}_median"
            ] = (
                float(
                    np.median(
                        values
                    )
                )
                if values.size
                else np.nan
            )

            row[
                f"{metric}_sd"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if values.size > 1
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def overall_summary_table(
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for method, part in scenario_summary.groupby(
        "method",
        sort=True,
    ):
        row: dict[str, Any] = {
            "method":
                method,
            "n_scenarios":
                int(
                    part.shape[0]
                ),
            "n_dataset_failures":
                int(
                    part[
                        "n_failed"
                    ].sum()
                ),
        }

        for metric in [
            "shared_subspace_recovery",
            "factor_alignment_correlation",
            "missing_view_robustness_shared_subspace",
            "latent_subspace_stability",
            "heldout_reconstruction",
        ]:
            vals = (
                pd.to_numeric(
                    part[
                        f"{metric}_mean"
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    float
                )
            )

            row[
                f"scenario_balanced_{metric}_mean"
            ] = (
                float(
                    np.mean(
                        vals
                    )
                )
                if vals.size
                else np.nan
            )

            row[
                f"scenario_balanced_{metric}_n_scenarios"
            ] = int(
                vals.size
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def finalize_results(
    paths: dict[str, Path],
    registry: pd.DataFrame,
    runner_sha: str,
    contract_sha: str,
    auth_path: Path,
) -> None:
    history_state = verify_bn_history(paths)
    result_root = paths[
        "result_root"
    ]

    all_shards: list[
        dict[str, Any]
    ] = []

    shard_records: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in registry.itertuples(
        index=False
    ):
        dataset_id = str(
            row.comparator_dataset_id
        )

        for method in EXPECTED[
            "methods"
        ]:
            path = shard_path(
                result_root,
                method,
                dataset_id,
            )

            shard = verify_existing_shard(
                path,
                runner_sha,
                contract_sha,
                method,
                dataset_id,
            )

            all_shards.append(
                shard
            )

            key = (
                f"{method}/"
                f"{path.name}"
            )

            shard_records[
                key
            ] = {
                "path":
                    relative_to_project(
                        path,
                        paths[
                            "project_root"
                        ],
                    ),
                "sha256":
                    sha256_file(
                        path
                    ),
                "size_bytes":
                    path.stat().st_size,
                "status":
                    shard[
                        "status"
                    ],
            }

    if len(
        all_shards
    ) != EXPECTED[
        "n_datasets"
    ] * 2:
        raise RuntimeError(
            "Unexpected total scientific shard count."
        )

    dataset_metrics = (
        make_dataset_metrics_table(
            all_shards
        )
    )

    if dataset_metrics.shape[0] != 720:
        raise RuntimeError(
            "Dataset-metrics table does not contain "
            "720 method-dataset rows."
        )

    scenario_summary = (
        scenario_summary_table(
            dataset_metrics
        )
    )

    if scenario_summary.shape[0] != 72:
        raise RuntimeError(
            "Scenario summary does not contain "
            "72 method-scenario rows."
        )

    overall_summary = (
        overall_summary_table(
            scenario_summary
        )
    )

    if overall_summary.shape[0] != 2:
        raise RuntimeError(
            "Overall summary does not contain two methods."
        )

    tables_dir = (
        result_root / "tables"
    )

    if (
        tables_dir.exists()
        and any(
            p.is_file()
            for p in tables_dir.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "Final comparator tables already exist; "
            "refusing to overwrite."
        )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        tables_dir
        / "secondary_comparator_dataset_metrics.csv"
    )

    scenario_path = (
        tables_dir
        / "secondary_comparator_scenario_summary.csv"
    )

    overall_path = (
        tables_dir
        / "secondary_comparator_overall_summary.csv"
    )

    dataset_metrics.to_csv(
        dataset_path,
        index=False,
    )

    scenario_summary.to_csv(
        scenario_path,
        index=False,
    )

    overall_summary.to_csv(
        overall_path,
        index=False,
    )

    summary_payload = {
        "batch_amendment": history_state,
        "batch_policy": BN_AMENDMENT["policy"],
        "post_failure_amended_execution": True,
        "status":
            "05CMP3_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_COMPLETE",
        "script_version":
            SCRIPT_VERSION,
        "created_at_utc":
            utc_now_iso(),
        "runner_sha256":
            runner_sha,
        "cmp2_contract_sha256":
            contract_sha,
        "n_expected_datasets":
            360,
        "n_expected_method_dataset_shards":
            720,
        "method_status_counts":
            (
                dataset_metrics.groupby(
                    [
                        "method",
                        "status",
                    ]
                )
                .size()
                .rename(
                    "n"
                )
                .reset_index()
                .to_dict(
                    orient="records"
                )
            ),
        "overall_summary":
            overall_summary.to_dict(
                orient="records"
            ),
        "scientific_firewalls": {
            "TARGET_accessed":
                False,
            "outcomes_clinical_biology_accessed":
                False,
            "cross_family_winner_declared":
                False,
            "new_composite_score_created":
                False,
            "05e_reopened":
                False,
            "final_gate_changed":
                False,
        },
    }

    summary_path = (
        result_root
        / "secondary_comparator_confirmatory_summary.json"
    )

    write_json_atomic(
        summary_payload,
        summary_path,
    )

    result_lock_dir = paths[
        "result_lock_dir"
    ]

    if (
        result_lock_dir.exists()
        and any(
            p.is_file()
            for p in result_lock_dir.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "05cmp4 result-lock directory already contains files."
        )

    result_lock_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        dataset_path,
        scenario_path,
        overall_path,
        summary_path,
    ]

    lock_payload = {
        "batch_amendment": history_state,
        "batch_policy": BN_AMENDMENT["policy"],
        "historical_v6_shards": BN_AMENDMENT["historical_shards"],
        "historical_metrics_reused_or_transplanted": False,
        "status":
            RESULT_LOCK_STATUS,
        "script_version":
            SCRIPT_VERSION,
        "created_at_utc":
            utc_now_iso(),
        "runner_source": {
            "path":
                relative_to_project(
                    Path(__file__).resolve(),
                    paths["project_root"],
                ),
            "sha256":
                runner_sha,
        },
        "cmp2_contract": {
            "path":
                "paper5/method_contract/05cmp2/"
                "secondary_comparator_confirmatory_benchmark_contract.json",
            "sha256":
                contract_sha,
        },
        "cmp3c_batch_amendment_authorization": {
            "path":
                relative_to_project(
                    auth_path,
                    paths[
                        "project_root"
                    ],
                ),
            "sha256":
                sha256_file(
                    auth_path
                ),
        },
        "scientific_shards":
            shard_records,
        "aggregate_files": {
            p.name: {
                "path":
                    relative_to_project(
                        p,
                        paths[
                            "project_root"
                        ],
                    ),
                "sha256":
                    sha256_file(
                        p
                    ),
                "size_bytes":
                    p.stat().st_size,
            }
            for p in output_files
        },
        "rewrite_policy": (
            "Scientific comparator results are frozen. "
            "Never overwrite/delete/rerun in place. Any "
            "recovery or extension requires a new versioned stage."
        ),
    }

    lock_path = (
        result_lock_dir
        / "secondary_comparator_confirmatory_results_lock.json"
    )

    write_json_atomic(
        lock_payload,
        lock_path,
    )

    print(
        "-" * 112
    )
    print(
        "Secondary comparator confirmatory scientific completion"
    )
    print(
        "-" * 112
    )
    print(
        overall_summary.to_string(
            index=False
        )
    )
    print(
        "Cross-family winner declared: NO"
    )
    print(
        "New composite score created: NO"
    )
    print(
        "05e reopened: NO"
    )
    print(
        "Final gate changed: NO"
    )
    print(
        "TARGET/outcomes/clinical/biology accessed: NO"
    )
    print(
        "=" * 112
    )
    print(
        "Paper 5 05cmp3 secondary comparator benchmark: "
        "COMPLETE + FROZEN"
    )
    print(
        "=" * 112
    )
    print(
        "Results lock: method_contract/05cmp4_v7_batch_rebalance/"
        "secondary_comparator_confirmatory_results_lock.json"
    )
    print(
        "=" * 112
    )


def authorized_real_main(
    script_path: Path,
    paths: dict[str, Path],
    contract: dict[str, Any],
    registry: pd.DataFrame,
) -> None:
    auth = load_json(
        paths[
            "auth"
        ]
    )

    history_state = verify_bn_history(paths)
    bn_require(auth.get("batch_amendment") == history_state, "Authorization batch/history binding mismatch")
    bn_require(auth.get("batch_policy") == BN_AMENDMENT["policy"], "Authorization batch policy mismatch")
    bn_require(auth.get("new_result_root") == BN_AMENDMENT["policy"]["new_result_root"], "Authorization output scope mismatch")

    runner_sha = sha256_file(
        script_path
    )

    contract_sha = sha256_file(
        paths[
            "cmp2_contract"
        ]
    )

    if auth.get(
        "status"
    ) != AUTH_STATUS:
        raise RuntimeError(
            "05cmp3c source-authorization status mismatch."
        )

    if str(
        auth.get(
            "runner_source",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower() != runner_sha:
        raise RuntimeError(
            "05cmp3c does not authorize this exact runner SHA."
        )

    if str(
        auth.get(
            "cmp2_contract",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower() != contract_sha:
        raise RuntimeError(
            "05cmp3c does not bind this exact 05cmp2 contract SHA."
        )

    preflight = load_json(
        paths[
            "preflight"
        ]
    )

    verify_bn_preflight(preflight)
    if preflight.get("tests", {}).get("serialized_v2_npz_schema_and_truth", {}).get("status") != (
        "PASS_SERIALIZED_V2_NPZ_LOADERS_AND_TRUTH_SEMANTICS"
    ):
        raise RuntimeError("v7 authorization requires serialized NPZ schema preflight PASS.")
    if preflight.get(
        "status"
    ) != PREAUTH_STATUS:
        raise RuntimeError(
            "05cmp3 synthetic preflight status mismatch."
        )

    if preflight.get(
        "runner_sha256"
    ) != runner_sha:
        raise RuntimeError(
            "05cmp3 synthetic preflight runner SHA mismatch."
        )

    if preflight.get(
        "cmp2_contract_sha256"
    ) != contract_sha:
        raise RuntimeError(
            "05cmp3 synthetic preflight contract SHA mismatch."
        )

    if str(
        auth.get(
            "synthetic_preflight",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower() != sha256_file(
        paths[
            "preflight"
        ]
    ):
        raise RuntimeError(
            "05cmp3c synthetic-preflight hash binding mismatch."
        )

    if (
        paths[
            "result_lock_dir"
        ].exists()
        and any(
            p.is_file()
            for p in paths[
                "result_lock_dir"
            ].rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "05cmp4 frozen result lock already exists. "
            "Never rerun scientific comparator benchmark."
        )

    print(
        "Execution mode: "
        "AUTHORIZED_RESUMABLE_SCIENTIFIC_COMPARATOR_BENCHMARK"
    )
    print(
        "Scientific dataset bank: exact 360 frozen confirmatory simulations"
    )
    print(
        "Methods: iCluster2b + MOVE"
    )
    print(
        "TARGET/outcomes/clinical/biology access: FORBIDDEN"
    )
    print(
        "05e/final-gate consequence: NONE"
    )

    print("Batch amendment: MOVE_TRAIN_SINGLETON_TAIL_REBALANCE_V1")
    print("Historical v6 records: preserved, not reused in this execution")
    print("New result scope:", BN_AMENDMENT["policy"]["new_result_root"])

    linear = import_linear_helper(
        paths[
            "linear_helper"
        ]
    )

    if sha256_file(
        paths[
            "linear_helper"
        ]
    ) != str(
        auth.get(
            "linear_helper_source",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower():
        raise RuntimeError(
            "Frozen Paper 5 linear metric-helper SHA changed."
        )

    move_api = import_move_api()

    r_env = verify_r_library(
        paths[
            "paper5"
        ]
    )

    result_root = paths[
        "result_root"
    ]

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = 0
    skipped = 0

    total = (
        EXPECTED[
            "n_datasets"
        ]
        * 2
    )

    registry_work = registry.copy()

    registry_work[
        "replicate_index"
    ] = pd.to_numeric(
        registry_work[
            "replicate_index"
        ],
        errors="raise",
    ).astype(
        int
    )

    registry_work[
        "simulation_seed"
    ] = pd.to_numeric(
        registry_work[
            "simulation_seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    registry_work = (
        registry_work.sort_values(
            [
                "scenario_id",
                "replicate_index",
                "simulation_seed",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    for dataset_number, (
        _,
        row,
    ) in enumerate(
        registry_work.iterrows(),
        start=1,
    ):
        dataset_id = str(
            row[
                "comparator_dataset_id"
            ]
        )

        print(
            f"[{dataset_number:03d}/"
            f"{EXPECTED['n_datasets']}] "
            f"{dataset_id}"
        )

        needed_methods: list[
            str
        ] = []

        for method in EXPECTED[
            "methods"
        ]:
            path = shard_path(
                result_root,
                method,
                dataset_id,
            )

            if path.exists():
                verify_existing_shard(
                    path,
                    runner_sha,
                    contract_sha,
                    method,
                    dataset_id,
                )

                skipped += 1

                print(
                    f"  {method}: "
                    "verified_existing"
                )

            else:
                needed_methods.append(
                    method
                )

        if not needed_methods:
            continue

        # First authorized NPZ input read; truth payload is decoded only after each method fit.
        arrays = load_scientific_npz(
            paths[
                "project_root"
            ],
            row,
        )

        for method in needed_methods:
            started = time.perf_counter()

            if method == "iCluster2b":
                fn = (
                    lambda row=row, arrays=arrays:
                    run_icluster_dataset(
                        row,
                        arrays,
                        linear,
                        r_env,
                        paths["project_root"],
                    )
                )

            elif method == "MOVE":
                fn = (
                    lambda row=row, arrays=arrays:
                    run_move_dataset(
                        row,
                        arrays,
                        linear,
                        move_api,
                        paths["project_root"],
                    )
                )

            else:
                raise RuntimeError(
                    "Unexpected comparator method: "
                    f"{method}"
                )

            payload = (
                execute_with_one_identical_retry(
                    method,
                    fn,
                    row,
                )
            )

            payload[
                "wall_seconds_including_retry"
            ] = float(
                time.perf_counter()
                - started
            )

            out = write_scientific_shard(
                result_root,
                payload,
                runner_sha,
                contract_sha,
            )

            completed += 1

            print(
                f"  {method}: "
                f"{payload['status']} "
                f"attempts="
                f"{payload.get('attempt_count', 2)} "
                f"shard={out.name}"
            )

        del arrays
        gc.collect()
        torch.cuda.empty_cache()

        print(
            "  progress method-dataset shards: "
            f"new={completed} "
            f"existing={skipped} "
            f"total_expected={total}"
        )

    finalize_results(
        paths,
        registry=registry_work,
        runner_sha=runner_sha,
        contract_sha=contract_sha,
        auth_path=paths[
            "auth"
        ],
    )


# =============================================================================
# Main gate
# =============================================================================

def main() -> None:
    print(
        "=" * 112
    )
    print(
        "Paper 5 - post-05e scientific secondary-comparator "
        "confirmatory benchmark runner"
    )
    print(
        "=" * 112
    )
    print(
        "Script version:",
        SCRIPT_VERSION,
    )

    script_path = Path(
        __file__
    ).resolve()

    if script_path.name != RUNNER_FILENAME:
        raise RuntimeError("Run the reviewed v7 filename; do not rename a runner as an authorizer.")
    paths = build_paths(
        script_path
    )
    verify_schema_amendment_inputs(paths)
    verify_bn_history(paths)

    contract, registry = verify_upstream(
        paths
    )

    runner_sha = sha256_file(
        script_path
    )

    contract_sha = sha256_file(
        paths[
            "cmp2_contract"
        ]
    )

    print(
        "Runner SHA-256:",
        runner_sha,
    )

    print(
        "05cmp2 contract SHA-256:",
        contract_sha,
    )

    print(
        "Exact frozen 05cmp2/05cmp1 metadata: PASS"
    )

    if not paths[
        "auth"
    ].exists():
        preauthorization_main(
            script_path,
            paths,
            contract,
            registry,
        )
        return

    authorized_real_main(
        script_path,
        paths,
        contract,
        registry,
    )


if __name__ == "__main__":
    main()
