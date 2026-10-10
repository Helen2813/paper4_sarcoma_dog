#!/usr/bin/env python3
"""Paper 5: separate v7 batch implementation amendment and source authorization.
Standard library only. Never imports a runner, model, R or CUDA; opens no NPZ.
Requires a NEW v7 synthetic preflight. Hashes all 82 prior shard files without
parsing their numeric metric contents. Preserves the v6 failures AND successes.
Authorizes a distinct complete 360-dataset/two-method execution in a new output
namespace. This is an explicit post-failure amendment, not a pre-first-fit freeze.
Existing locks/results are never overwritten. Do not run concurrently.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import stat
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

SCRIPT_VERSION = '05cmp3c-freeze-secondary-comparator-runner-v7-batch-amendment-v1-no-cli'
AUTHOR_FILENAME = '05cmp3c_freeze_secondary_comparator_runner_v7_batch_amendment_v1.py'
RUNNER_REL = 'paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v7.py'
RUNNER_SHA = '7d434b5bb168ce2be62ae3c94d1ff93e48e5ed5e0aea21aae3238fdb4579c268'
RUNNER_VERSION = '05cmp3-run-secondary-comparator-confirmatory-benchmark-v7-singleton-tail-rebalance-preserved-v6-history-isolated-results-no-cli'
PREAUTH_STATUS = "05CMP3_SECONDARY_COMPARATOR_RUNNER_SYNTHETIC_PREAUTH_PASS"
AUTH_STATUS = '05CMP3C_SECONDARY_COMPARATOR_RUNNER_V7_BATCH_AMENDMENT_AUTHORIZED'
PREFLIGHT_REL = 'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v7.json'
AUTH_REL = 'paper5/method_contract/05cmp3c/secondary_comparator_runner_v7_batch_amendment_authorization_lock.json'
OLD_RUNNER_REL = "paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v5.py"
OLD_RUNNER_SHA = '802c131dfda75effb4e60dd38ebf05512fa5da6c1b3eec97c0c499b3c4e869f5'
OLD_PREFLIGHT_REL = 'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v5.json'
OLD_PREFLIGHT_SHA = '7ec4e687151a4af67b718a1ee13e521109a0071be91de2e751e9d0bb02b96b94'
HELPER_REL = "paper5/scripts/05c1_fit_linear_simulation_baselines.py"
HELPER_SHA = "c2c91b4ad274f78ccc369e6fca765102285cf3d9c8191f3c33f4a450b7426367"

OLD_AUTH_REL = "paper5/method_contract/05cmp3a/secondary_comparator_runner_source_authorization_lock.json"
OLD_AUTH_SHA = "5dbb075b3467fa865cb2b0570a79c2419e952a2c8c001eb04287d8ade5fba319"
OLD_AUTHOR_REL = "paper5/scripts/05cmp3a_freeze_secondary_comparator_runner_v5_source_authorization_v1.py"
OLD_AUTHOR_SHA = "526bf30c101dea38cb26d07c9acd625892febebcd697fb466be0c4c1fe653bfa"
GENERATOR_REL = "paper5/scripts/05b_generate_small_n_multiomics_simulations_v2.py"
GENERATOR_SHA = "b4adb1fcb32b65a998ee70e1b72d52e13490194c7ad8e5ddd802b95bf05a28db"
REVIEWED_EVIDENCE_ZIP_SHA = "0752c651b93a47505d3887f67630a421c31d279606472e3f018e7322ca5d9e1d"
REVIEWED_HISTORY = {
    OLD_RUNNER_REL: OLD_RUNNER_SHA, OLD_PREFLIGHT_REL: OLD_PREFLIGHT_SHA,
    OLD_AUTH_REL: OLD_AUTH_SHA, OLD_AUTHOR_REL: OLD_AUTHOR_SHA,
    GENERATOR_REL: GENERATOR_SHA, HELPER_REL: HELPER_SHA,
}
POSITIVE_SCHEMA_CASES = [
    "all3", "partial_RNA_METH", "partial_RNA_CNV", "partial_METH_CNV",
    "mixed_support_and_pair_corruption", "rotational_subspace_only",
    "technical_only_biological_null",
]
NEGATIVE_SCHEMA_CASES = [
    "legacy_Z_shared_only_rejected", "dual_truth_keys_rejected",
    "missing_biological_truth_rejected", "sample_index_reordering_rejected",
    "view_shape_mismatch_rejected", "RNA_anchor_repair_forbidden",
    "biological_column_count_mismatch_rejected", "size_mismatch_rejected", "SHA_mismatch_rejected",
]

RESULT_SCOPES = ('paper5/results/simulation_benchmark/05cmp_secondary_comparators/v7_batch_rebalance',
 'paper5/method_contract/05cmp4_v7_batch_rebalance')
SCIENTIFIC_STATE = {
    "frozen_simulation_npz_opened": False,
    "frozen_simulation_truth_opened": False,
    "scientific_comparator_fit_performed": False,
    "TARGET_accessed": False,
    "outcomes_clinical_biology_accessed": False,
}

FROZEN_INPUTS = {'cmp2_contract': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract.json',
                   '8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2'),
 'cmp2_lock': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract_lock.json',
               'a55d3e3498d9cc6a4acaaab64e5098007b093071b60999c82cf8e124a0301b5b'),
 'cmp2_registry': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_dataset_registry.csv',
                   '4b72b94c4df49dbbbd1a90d035855e628a4fa50eede0451dfe796227bda53314'),
 'cmp1_contract': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_contract.json',
                   '418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95'),
 'cmp1_lock': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_contract_lock.json',
               '37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648'),
 'cmp1_impl': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_registry.csv',
               '85dbc68289bae71b9cf4f8a01af34b09d50750fde97f0b57ef3b6c57ebf299b0'),
 'cmp1_adapter': ('paper5/method_contract/05cmp1/secondary_comparator_scientific_adapter_registry.csv',
                  'f30c73d19627d357073786b75f7b8141a757becf9d2413491518948cc099ea48'),
 'cmp1_seed': ('paper5/method_contract/05cmp1/secondary_comparator_seed_registry.csv',
               '2c70c516ad07496a71fe513139b357b92f46fd746bea5258f1fb237028c4e87c'),
 'cmp1_metric': ('paper5/method_contract/05cmp1/secondary_comparator_metric_eligibility_registry.csv',
                 '8802d8d4bba20cfd4acc3263b12f0a119bcef4e4107164e6f944d0a01d9bef46'),
 'manifest05b': ('paper5/method_contract/05b_v2/architecture_screening_data_manifest.csv',
                 'f3980e88aff408b65cbfce96aca50bad4fc08ecece2bc42bbc04f365c35d9c87')}

ENVIRONMENT_EXPECTED = {'move_version': '1.5.0',
 'torch_version': '2.12.1+cu130',
 'cuda_build': '13.0',
 'rscript_sha256': 'd829bcf7e9fa1d7e3e828c565c3cdbb1ed416f551f4fa6fd4dfcdf231e33e5e8',
 'icluster_package_tree_sha256': 'cad0c995150e778026fb227ac533a974d77106e431c65a3802aeda4254f6fd70'}



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

class AuthorizationError(RuntimeError):
    """Fail-closed authorizer error; no fallback or input repair is permitted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            require(key not in out, f"Duplicate JSON key in {label}: {key}")
            out[key] = value
        return out

    def reject_constant(value: str) -> None:
        raise AuthorizationError(f"Non-finite JSON constant in {label}: {value}")

    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                       parse_constant=reject_constant)
    require(isinstance(value, dict), f"Expected JSON object: {label}")
    return value


def project_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts,
            f"Non-project-relative path rejected: {relative}")
    path = root / rel
    # Static, reviewed paths only. Refuse links/reparse points rather than
    # following them to scientific data or a different project location.
    current = root
    for part in rel.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        linked = stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        require(not linked, f"Symlink/reparse path rejected: {current}")
    require(path.resolve().is_relative_to(root.resolve()),
            f"Path escapes the project: {path}")
    return path


def read_snapshot(root: Path, relative: str, expected: str | None = None
                  ) -> tuple[bytes, dict[str, Any]]:
    path = project_path(root, relative)
    require(path.is_file(), f"Required file missing: {path}")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if expected is not None:
        require(digest == expected,
                f"SHA-256 mismatch for {relative}: {digest} != {expected}. "
                "Do not edit hashes or rewrite the input to bypass this check.")
    return data, {"path": relative, "sha256": digest, "size_bytes": len(data)}


def assert_empty_results(root: Path) -> None:
    # Directory metadata only: never read, hash, import or parse a result file.
    for relative in RESULT_SCOPES:
        path = project_path(root, relative)
        if not path.exists():
            continue
        require(path.is_dir(), f"Unexpected non-directory result scope: {path}")

        def walk_error(error: OSError) -> None:
            raise AuthorizationError(f"Cannot inspect result directory: {error}")

        for directory, dirs, files in os.walk(path, followlinks=False, onerror=walk_error):
            require(not files, f"Pre-authorization result files already exist in {directory}. "
                    "Do not delete them or create a replacement authorization.")
            for name in dirs:
                project_path(root, (Path(directory) / name).relative_to(root).as_posix())


def assert_no_authorization(root: Path) -> Path:
    path = project_path(root, AUTH_REL)
    require(not path.exists(), f"Authorization already exists: {path}. "
            "It was not changed. Do not rerun the authorizer or remove the lock.")
    return path


def top_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    return {node.targets[0].id: node.value for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)}


def validate_runner(data: bytes) -> None:
    compile(data, RUNNER_REL, "exec", dont_inherit=True)  # Compilation only.
    tree = ast.parse(data.decode("utf-8-sig"), feature_version=(3, 10))
    assignments = top_assignments(tree)
    for key, expected in (("SCRIPT_VERSION", RUNNER_VERSION),
                          ("PREAUTH_STATUS", PREAUTH_STATUS),
                          ("AUTH_STATUS", AUTH_STATUS)):
        require(key in assignments and ast.literal_eval(assignments[key]) == expected,
                f"Runner constant mismatch: {key}")
    require(ast.literal_eval(assignments["RUNNER_FILENAME"]) == Path(RUNNER_REL).name,
            "Runner filename binding mismatch.")
    require(ast.literal_eval(assignments["AMENDMENT_REVIEWED_INPUTS"]) == REVIEWED_HISTORY,
            "Runner reviewed-history bindings mismatch.")
    node = assignments.get("EXPECTED")
    require(isinstance(node, ast.Dict), "Runner EXPECTED is not a literal dictionary.")
    fields = {ast.literal_eval(key): value for key, value in zip(node.keys, node.values)}
    for key, (_, digest) in FROZEN_INPUTS.items():
        require(ast.literal_eval(fields[key + "_sha256"]) == digest,
                f"Authorizer/runner frozen binding mismatch: {key}")
    for key, value in (("n_datasets", 360), ("n_scenarios", 36),
                       ("methods", ["iCluster2b", "MOVE"]), ("latent_dim", 5)):
        require(ast.literal_eval(fields[key]) == value, f"Runner scope mismatch: {key}")
    require(ast.literal_eval(assignments["BN_AMENDMENT"]) == BN_AMENDMENT
            and ast.literal_eval(assignments["BN_EVIDENCE_SHA256"]) == BN_EVIDENCE_SHA256,
            "Runner batch amendment evidence mismatch.")


def integer(text: str, label: str) -> int:
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise AuthorizationError(f"Invalid integer metadata: {label}={text!r}") from exc
    require(value.is_finite() and value == value.to_integral_value(),
            f"Non-integer metadata: {label}={text!r}")
    return int(value)


def validate_registry(data: bytes) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    needed = {"comparator_dataset_id", "scenario_id", "replicate_index", "simulation_seed"}
    require(needed.issubset(fields), f"Registry is missing columns: {sorted(needed-set(fields))}")
    require(len(fields) == len(set(fields)), "Duplicate registry column names.")
    rows = list(reader)
    require(len(rows) == 360, f"Registry row count: {len(rows)} != 360")
    ids: set[str] = set()
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        require(None not in row and all(value is not None for value in row.values()),
                "Malformed CSV row in frozen registry.")
        dataset_id = row["comparator_dataset_id"]
        scenario = row["scenario_id"]
        require(bool(dataset_id.strip()) and dataset_id not in ids,
                "Empty or duplicate comparator_dataset_id.")
        require(bool(scenario.strip()), "Empty scenario_id.")
        ids.add(dataset_id)
        grouped[scenario].append(integer(row["replicate_index"], "replicate_index"))
        integer(row["simulation_seed"], "simulation_seed")
    require(len(grouped) == 36, f"Registry scenario count: {len(grouped)} != 36")
    required_replicates = list(range(11, 21))
    for scenario, replicates in grouped.items():
        require(sorted(replicates) == required_replicates,
                f"Incomplete/duplicate confirmatory replicates for {scenario}.")
    return {"n_datasets": 360, "n_scenarios": 36,
            "replicate_indices": required_replicates, "methods": ["iCluster2b", "MOVE"],
            "expected_method_dataset_shards": 720,
            "simulation_npz_paths_followed": False}


def mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"Expected object: {label}")
    return value


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value),
            f"Non-finite or nonnumeric value: {label}")
    return float(value)


def validate_preflight(report: dict[str, Any]) -> None:
    for key, expected in (
        ("status", PREAUTH_STATUS), ("execution_mode", "SYNTHETIC_ONLY_PREAUTHORIZATION"),
        ("script_version", RUNNER_VERSION), ("runner_sha256", RUNNER_SHA),
        ("cmp2_contract_sha256", FROZEN_INPUTS["cmp2_contract"][1]),
        ("cmp2_registry_sha256", FROZEN_INPUTS["cmp2_registry"][1]),
    ):
        require(report.get(key) == expected, f"v7 preflight mismatch: {key}")
    helper = mapping(report.get("linear_helper_source"), "linear_helper_source")
    require(helper.get("path") == HELPER_REL and helper.get("sha256") == HELPER_SHA,
            "v7 preflight does not bind the reviewed linear helper.")
    state = mapping(report.get("scientific_state"), "scientific_state")
    for key in SCIENTIFIC_STATE:
        require(state.get(key) is False, f"Preflight scientific firewall is not false: {key}")
    environment = mapping(report.get("environment"), "environment")
    for key, value in ENVIRONMENT_EXPECTED.items():
        require(environment.get(key) == value, f"Preflight environment mismatch: {key}")
    require(isinstance(environment.get("gpu"), str) and bool(environment["gpu"].strip()),
            "Preflight GPU evidence is missing.")
    timestamp = datetime.fromisoformat(str(report.get("created_at_utc", "")).replace("Z", "+00:00"))
    require(timestamp.tzinfo is not None, "Preflight timestamp lacks a timezone.")
    tests = mapping(report.get("tests"), "tests")
    for key in ("adapter_feature_and_whole_view_missing", "deterministic_seed_derivation",
                "frozen_upstream_metadata"):
        require(tests.get(key) == "PASS", f"Synthetic test not PASS: {key}")
    for key in ("move_exact_api_synthetic_fit", "icluster2b_exact_tune_api_synthetic_fit"):
        fit = mapping(tests.get(key), key)
        require(fit.get("finite") is True and fit.get("latent_shape") == [24, 5],
                f"Synthetic fit is not a finite 24-by-5 result: {key}")
    move = tests["move_exact_api_synthetic_fit"]
    require(finite_number(move.get("synthetic_reconstruction_sse"), "synthetic SSE") >= 0,
            "Negative synthetic SSE.")
    lambdas = tests["icluster2b_exact_tune_api_synthetic_fit"].get("lambda")
    require(isinstance(lambdas, list) and len(lambdas) == 3, "Synthetic lambda shape mismatch.")
    for value in lambdas:
        require(10 <= finite_number(value, "synthetic lambda") <= 500,
                "Synthetic lambda outside the unchanged frozen interval.")
    selection = mapping(tests.get("move_selection_logic"), "move_selection_logic")
    require(selection.get("deterministic_folds") == "PASS"
            and selection.get("one_se_retained") == ["A", "B"]
            and selection.get("stage2_selected") == "B", "MOVE selection fixture failed.")
    metric = mapping(tests.get("paper5_subspace_helper_replay"), "metric replay")
    for key in ("factor_alignment", "subspace_recovery"):
        require(abs(finite_number(metric.get(key), key) - 1.0) <= 1e-10,
                f"Exact synthetic metric fixture failed: {key}")
    replay = mapping(metric.get("subspace_helper_replay"), "subspace_helper_replay")
    require(replay.get("status") == "PASS_FROZEN_HELPER_NAMED_SCALAR"
            and replay.get("helper_scalar_path") == "mean_canonical_correlation",
            "Frozen named metric-helper replay mismatch.")


    schema = mapping(tests.get("serialized_v2_npz_schema_and_truth"), "serialized schema test")
    require(schema.get("status") == "PASS_SERIALIZED_V2_NPZ_LOADERS_AND_TRUTH_SEMANTICS",
            "Actual serialized v2 NPZ loaders have not passed.")
    require(schema.get("positive_cases") == POSITIVE_SCHEMA_CASES
            and schema.get("negative_cases") == NEGATIVE_SCHEMA_CASES,
            "Serialized schema positive/negative coverage mismatch.")
    for key in ("observed_input_payloads_only", "truth_payload_read_only_in_explicit_truth_stage"):
        require(schema.get(key) is True, f"Schema read firewall failed: {key}")
    for key in ("technical_and_private_truth_payloads_decoded",
                "biological_columns_filtered_by_view_support", "frozen_scientific_NPZ_opened"):
        require(schema.get(key) is False, f"Schema firewall failed: {key}")
    require(schema.get("RNA_anchor_helper") == "05c1.integrated_anchor_truth(arrays, 'biological')",
            "Integrated biological anchor helper mismatch.")
    require(schema.get("tracked_payload_reads") == 49, "Payload tracing coverage mismatch.")
    amendment = mapping(report.get("schema_amendment"), "schema_amendment")
    require(amendment.get("historical_authorized_v5_preserved") is True
            and amendment.get("current_invocation_scientific_npz_values_decoded") is False
            and amendment.get("truth_key") == "Z_biological_shared"
            and amendment.get("truth_stage") == "after each method fit/architecture selection completes",
            "Schema-amendment state mismatch.")
    history = mapping(amendment.get("history_bindings"), "history_bindings")
    require(set(history) == set(REVIEWED_HISTORY), "Incomplete preflight history bindings.")
    for relative, digest in REVIEWED_HISTORY.items():
        require(mapping(history[relative], relative).get("sha256") == digest,
                f"Preflight history hash mismatch: {relative}")

    batch = mapping(report.get("batch_amendment"), "batch_amendment")
    expected_batch = {"status": "PASS_EXACT_V6_HISTORY_PRESERVED", "evidence_sha256": BN_EVIDENCE_SHA256,
                      "old_shard_count": 82, "old_iCluster2b_PASS": 41, "old_MOVE_singleton_failures": 41,
                      "old_shard_bytes_hashed": True, "old_metric_values_decoded": False,
                      "old_records_transplanted_into_new_results": False}
    require(batch == expected_batch, "Preflight old-result history binding mismatch.")
    test = mapping(tests.get("singleton_batch_training_regression"), "batch regression")
    require(test.get("status") == "PASS_V7_ACTUAL_MOVE_SYNTHETIC_BATCHING"
            and test.get("policy_id") == BN_AMENDMENT["policy"]["policy_id"]
            and test.get("baseline_singleton_error_reproduced") is True,
            "Original failure / amended batch regression has not passed.")
    require(test.get("cv_fit_count") == 15 and test.get("stability_fit_count") in (3, 6, 9)
            and test.get("canonical_latent_shape") == [62, 5]
            and test.get("all_latents_finite") is True
            and test.get("synthetic_epochs") == 2 and test.get("scientific_epochs_unchanged") == 300
            and test.get("scientific_module_globals_modified") is False
            and test.get("frozen_scientific_npz_opened") is False,
            "Batch regression budget/shape/access checks failed.")
    calls = test.get("fit_calls")
    require(isinstance(calls, list) and len(calls) == 15 + test["stability_fit_count"],
            "Incomplete actual synthetic fit records.")
    require([c.get("n_fit") for c in calls[:15]] == [49, 49, 50, 50, 50] * 3,
            "Synthetic fold-size coverage differs.")
    require([c.get("hidden") for c in calls[:15]] == [[64, 32]]*5 + [[128, 64]]*5 + [[256, 128]]*5,
            "Synthetic architecture coverage differs.")
    for c in calls:
        require(c.get("finite") is True and c.get("epochs") == 2
                and c.get("latent_shape") == [c.get("n_fit"), 5], "Synthetic fit invalid.")
    require(all(c.get("n_fit") == 62 for c in calls[15:]), "Full-data stability size mismatch.")
    order = mapping(test.get("batch_order_tests"), "batch_order_tests")
    require(order.get("status") == "PASS" and order.get("case_count") == 206,
            "Batch sample/order coverage missing.")
    cases = order.get("cases")
    require(isinstance(cases, list) and len(cases) == 206, "Batch case details missing.")
    wanted = {(n, seed) for n in list(range(2, 101))+[113,129,145,161] for seed in [1701,2701]}
    require({(c.get("n"), c.get("seed")) for c in cases} == wanted, "Batch coverage grid differs.")
    for c in cases:
        n = c["n"]; b = min(16, n)
        expected = [b]*(n//b) + ([n%b] if n%b else [])
        original = list(expected)
        if expected[-1] == 1:
            expected[-2] -= 1; expected[-1] = 2
        require(c.get("epochs") == 2 and c.get("original_sizes") == original
                and c.get("projection_sizes") == original and c.get("candidate_sizes") == expected,
                "Unexpected batch geometry in preflight case.")


def atomic_publish(path: Path, content: bytes, recheck: Callable[[], None]) -> None:
    """Publish complete bytes without overwrite. No weaker fallback is used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".05cmp3c_pending_", suffix=".tmp", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        require(pending.read_bytes() == content, "Pending authorization verification failed.")
        recheck()
        # Same directory/volume. A pre-existing destination makes this fail.
        # On filesystems without hard-link support we stop, not overwrite.
        os.link(pending, path)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def authorize(project_root: Path, authorizer_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    authorizer_path = authorizer_path.resolve()
    expected_self = root / "paper5" / "scripts" / AUTHOR_FILENAME
    require(authorizer_path == expected_self, "Place the authorizer under paper5/scripts "
            f"with the exact filename {AUTHOR_FILENAME}")
    auth_path = assert_no_authorization(root)
    assert_empty_results(root)
    history_state = verify_bn_history({"project_root": root})
    print("All 82 historical v6 shards: exact bytes preserved; numeric metrics not decoded")
    snapshots: dict[str, dict[str, Any]] = {}

    def capture(relative: str, expected: str | None = None) -> bytes:
        data, record = read_snapshot(root, relative, expected)
        snapshots[relative] = record
        return data

    capture(authorizer_path.relative_to(root).as_posix())
    runner = capture(RUNNER_REL, RUNNER_SHA)
    validate_runner(runner)
    print("Exact v7 runner SHA/version and source bindings: PASS")
    capture(OLD_RUNNER_REL, OLD_RUNNER_SHA)
    capture(OLD_PREFLIGHT_REL, OLD_PREFLIGHT_SHA)
    old_auth = strict_json(capture(OLD_AUTH_REL, OLD_AUTH_SHA), "historical v5 authorization")
    require(old_auth.get("runner_source", {}).get("sha256") == OLD_RUNNER_SHA,
            "Historical authorization runner binding mismatch.")
    capture(OLD_AUTHOR_REL, OLD_AUTHOR_SHA)
    capture(GENERATOR_REL, GENERATOR_SHA)
    print("Historical v5 source/preflight/authorization and generator: PASS")
    blobs = {key: capture(relative, digest) for key, (relative, digest) in FROZEN_INPUTS.items()}
    contract = strict_json(blobs["cmp2_contract"], "05cmp2 contract")
    contract_lock = strict_json(blobs["cmp2_lock"], "05cmp2 lock")
    require(contract.get("status") ==
            "FROZEN_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_BEFORE_SCIENTIFIC_FITS",
            "Frozen 05cmp2 contract status mismatch.")
    require(contract_lock.get("status") ==
            "05CMP2_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_CONTRACT_FROZEN",
            "Frozen 05cmp2 lock status mismatch.")
    scope = validate_registry(blobs["cmp2_registry"])
    print("Ten exact frozen upstream files and 360-row registry: PASS")
    capture(HELPER_REL, HELPER_SHA)
    print("Frozen linear metric-helper SHA: PASS")
    preflight_bytes = capture(PREFLIGHT_REL)
    preflight = strict_json(preflight_bytes, "NEW v7 preflight")
    validate_preflight(preflight)
    print("NEW v7 synthetic preflight and all scientific firewalls: PASS")

    def record(relative: str) -> dict[str, Any]:
        return dict(snapshots[relative])

    payload: dict[str, Any] = {
        "batch_amendment": history_state,
        "batch_policy": BN_AMENDMENT["policy"],
        "batch_amendment_evidence": BN_AMENDMENT,
        "new_result_root": BN_AMENDMENT["policy"]["new_result_root"],
        "new_result_lock_dir": BN_AMENDMENT["policy"]["new_result_lock_dir"],
        "post_failure_amendment_not_pre_first_fit_freeze": True,
        "status": AUTH_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_source": {**record(RUNNER_REL), "script_version": RUNNER_VERSION},
        "cmp2_contract": record(FROZEN_INPUTS["cmp2_contract"][0]),
        "synthetic_preflight": record(PREFLIGHT_REL),
        "linear_helper_source": record(HELPER_REL),
        "authorizer_source": record(authorizer_path.relative_to(root).as_posix()),
        "frozen_upstream_files": {key: record(relative)
                                  for key, (relative, _) in FROZEN_INPUTS.items()},
        "source_history": {
            "v5_source": record(OLD_RUNNER_REL),
            "v5_original_successful_preflight": record(OLD_PREFLIGHT_REL),
            "v5_authorization": record(OLD_AUTH_REL),
            "v5_authorizer": record(OLD_AUTHOR_REL),
            "generator_source": record(GENERATOR_REL),
            "reviewed_evidence_zip_sha256": REVIEWED_EVIDENCE_ZIP_SHA,
            "schema_changes_retained_from_v6": [
                "strict generator-v2 NPZ observed-input loader; no legacy aliases",
                "biological truth decoded only after fit/selection via frozen integrated_anchor_truth",
                "synthetic serialized NPZ integration regression and guard cases",
                "separate v6 preflight and additive 05cmp3b authorization gate",
                "exact retained v5/generator/helper source bindings",
                "retained schema interface; v7 final lock identifies cmp3c batch amendment explicitly",
            ],
            "model_selection_and_metric_formula_function_bodies_changed": False,
            "training_batch_composition_changed": True,
            "input_validation_and_truth_loading_code_changed": True,
            "historical_preflight_relabelled_as_v7": False,
            "v5_is_preserved_as_failed_schema_execution_not_a_scientific_method_failure": True,
        },
        "execution_scope": scope,
        "environment_evidence_from_v7_preflight": preflight["environment"],
        "preauthorization_scientific_state": dict(SCIENTIFIC_STATE),
        "historical_access_disclosure": {
            "source": "reviewed schema evidence and v5 source/traceback; not new NPZ access here",
            "first_dataset_id": "A01_N62_STANDARD__R11__S1956851480",
            "previous_authorized_v5_npz_bytes_hashed_and_container_opened": True,
            "previous_v5_array_values_decoded_before_missing_key_error": False,
            "followup_diagnostic_read_npy_headers": True,
            "followup_diagnostic_decoded_array_values": False,
            "scientific_fit_started_in_failed_v5_attempt": False,
        },
        "authorizer_actions": {
            "runner_or_helper_imported": False, "R_or_CUDA_started": False,
            "model_fit_performed": False, "simulation_npz_read": False,
            "historical_scientific_result_bytes_hashed": True,
            "historical_numeric_metric_values_decoded": False,
            "new_scientific_result_contents_read": False,
            "result_directory_metadata_checked": True,
            "existing_sources_preflights_contracts_modified": False,
        },
        "scope_limits": {
            "TARGET_outcomes_clinical_biology_access_authorized": False,
            "05e_final_gate_or_TARGET_selection_changes_authorized": False,
            "model_selection_or_metric_formula_changes_authorized": False,
            "singleton_tail_batch_composition_amendment_authorized": True,
            "full_uniform_two_method_rerun_in_new_scope_authorized": True,
            "data_interface_and_truth_access_order_correction_authorized": True,
            "fallback_authorized": False,
            "benchmark_started_by_authorizer": False,
            "full_scientific_success_guaranteed": False,
        },
        "rewrite_policy": "Do not overwrite/delete/rebind this lock. Changes require a separate "
                          "versioned review and successful matching preflight before scientific use.",
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(strict_json(encoded, "authorization payload") == payload,
            "Authorization serialization round-trip failed.")

    def recheck() -> None:
        assert_no_authorization(root)
        assert_empty_results(root)
        require(verify_bn_history({"project_root": root}) == history_state, "v6 history changed during authorization")
        for relative, previous in snapshots.items():
            _, current = read_snapshot(root, relative, previous["sha256"])
            require(current == previous, f"Input changed before publication: {relative}")

    recheck()
    atomic_publish(auth_path, encoded, recheck)
    require(auth_path.read_bytes() == encoded,
            "Authorization post-write verification failed. Preserve the lock and report the error.")
    print("Final read-only input recheck before publication: PASS")
    print("Paper 5 05cmp3c v7 batch amendment and source authorization: FROZEN")
    print(f"Authorization: {AUTH_REL}")
    print(f"Authorization SHA-256: {sha256_bytes(encoded)}")
    print(f"Runner SHA-256: {RUNNER_SHA}")
    print(f"Synthetic preflight SHA-256: {snapshots[PREFLIGHT_REL]['sha256']}")
    print("Scientific benchmark started: NO")
    print("STOP HERE. A subsequent v7 runner invocation will enter scientific mode.")
    return payload


def main() -> int:
    print("=" * 104)
    print("Paper 5 - 05cmp3c exact v7 batch amendment authorizer (no scientific execution)")
    print(f"Script version: {SCRIPT_VERSION}")
    print("=" * 104)
    try:
        require(len(sys.argv) == 1, "No command-line arguments are accepted.")
        require(sys.version_info >= (3, 10), "Python 3.10 or newer is required.")
        path = Path(__file__).resolve()
        require(path.name == AUTHOR_FILENAME and path.parent.name == "scripts"
                and path.parent.parent.name == "paper5",
                "Incorrect authorizer filename/location; expected paper5/scripts/" + AUTHOR_FILENAME)
        authorize(path.parents[2], path)
    except (AuthorizationError, OSError, ValueError, TypeError, KeyError, SyntaxError, RuntimeError) as exc:
        print(f"AUTHORIZATION STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Benchmark was not started. Do not edit hashes, delete evidence or bypass the error.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
