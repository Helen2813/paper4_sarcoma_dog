#!/usr/bin/env python3
"""Paper 5: post-result, read-only interpretation refinements (no model fitting).

Save this single file as p5refine.py in Downloads; run from the workspace root.
Reads exact TARGET 06b tables/source and archived inner JSONs, and optionally
completed SARC/KIRC result records. Writes only a new, separately named report
folder and a <=5 MiB evidence JSON. No project imports, checkpoint loading,
network calls, clinical data, threshold changes, or manuscript overwriting.
The original 06b inner shards were NOT individually bound by its result lock;
this audit records their current hashes and replays the pinned selection table,
without claiming that such replay authenticates every historical component.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import platform
import stat
import statistics
import sys
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
VERSION = 'p5-refine-postresult-target-components-selection-ties-external-seed-audit-v1-no-fit'
MAX_FILE = 4 * 1024 * 1024
MAX_EVIDENCE = 5 * 1024 * 1024
MAX_TOTAL_READ = 40 * 1024 * 1024
ATOL = 2e-12
RTOL = 2e-10
TIE_ATOL = 1e-12
MODEL_IDS = ('BLOCK_PCA','MOFA_K5','MOFA_K10','MOFA_K15','MOFA_K20',
             'V0_BALANCED_AE','V1_SHARED_PRIVATE_AE','V2_CONTRASTIVE_SPAE',
             'V3_DROPOUT_SPAE','V4_SABLE_MO')
V1, V3, V4 = 'V1_SHARED_PRIVATE_AE', 'V3_DROPOUT_SPAE', 'V4_SABLE_MO'
COMPLEXITY = {m: i + 1 for i, m in enumerate(MODEL_IDS)}
WEIGHTS = {'heldout_reconstruction_score': .30,
           'random_adjusted_cross_view_mrr': .30,
           'missing_view_reconstruction_score': .25,
           'modality_balance_score': .15}
METRICS = tuple(WEIGHTS) + ('utility',)
OUTER_SEEDS, INNER_SEEDS = (1103, 2207, 3301), (1103, 2207)
TARGET_ROOT = 'paper5/results/target_representation/06b_primary_benchmark'
SOURCE_REL = 'paper5/scripts/06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py'
OUTER_REL = TARGET_ROOT + '/target_outer_fixed_model_benchmark.csv'
SELECTION_REL = TARGET_ROOT + '/target_inner_one_se_selection_by_outer_split.csv'
SUMMARY_REL = TARGET_ROOT + '/target_outcome_blind_benchmark_summary.json'
LOCK_REL = 'paper5/method_contract/06b/target_outcome_blind_benchmark_results_lock.json'
LOSS_REL = 'paper5/method_contract/05d1/neural_loss_and_architecture_contract.json'
SARC_HELPER = 'paper5_parallel_sarc/sarc5.py'
SARC_HELPER_SHA = 'dd16e547972d8d298491e2d3c5003dea16760e969e9ea846fda5b0f83020f96f'
PINS = {
    SOURCE_REL: '4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab',
    OUTER_REL: 'b3c04b163392df043f78c8d377ff9f802afea2d7e95dcbff21fc5c87e504f2c5',
    SELECTION_REL: '8d6f17311cc98c3312619d3410f804e6581f152ce7811616b87ab9774a3e68c1',
    SUMMARY_REL: 'c2b350f8d0e957cf1a1005d0aa5f490d631a30d9e6607c12a90659c8037ca2e5',
    LOCK_REL: '2730c783f1a74f7aa58030d1ee03f324f1d50c4830c72e4229a15e41b60dc121',
    LOSS_REL: 'b90ded1d011ee3210154c2102389df0b4e0c8fcf668b70d7e2b8e14184e3f6b1',
}
EXTERNAL = {
    'SARC': {'base': 'paper5_parallel_sarc', 'n': 250,
             'protocol_sha': 'f3a63188ecbaff4a5164d999d933a5f6481ff7d0560b999022faa38f7a0cfff8',
             'split_sha': '78a702ac43d55dd37f8b194b7b7933fb547f09e04cf6333662b57221160377f0',
             'completion_sha': 'ca6f6931de8ef223792488b0dd43b8b66b3a29e010be395466973da080698fc1',
             'status': 'LIMITED_SARC_FIXED_MODEL_BENCHMARK_COMPLETE_FROZEN'},
    'KIRC': {'base': 'paper5_parallel_kirc', 'n': 304,
             'protocol_sha': '6ac98e5fe49065ba450a54a744b33fbc45197a382dd4a522ec784e494fc232e1',
             'split_sha': 'f7b73461fe420d3ded1f5450aef3402e0c48ac07c1ef570aa0eddff1e3c7661b',
             'completion_sha': '09679c7f6ad778631f7b5173b4cae11cdbaea181d12aa1f557ef2f821470d54a',
             'status': 'LIMITED_KIRC_FIXED_MODEL_BENCHMARK_COMPLETE_FROZEN'},
}


class AuditError(RuntimeError):
    pass


def need(ok: Any, message: str) -> None:
    if not bool(ok):
        raise AuditError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(obj: Any, compact: bool = False) -> bytes:
    options = dict(ensure_ascii=True, sort_keys=True, allow_nan=False)
    if compact:
        options['separators'] = (',', ':')
    else:
        options['indent'] = 2
    return (json.dumps(obj, **options) + '\n').encode('utf-8')


def decode(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'Duplicate JSON key: ' + k)
            result[k] = v
        return result
    def invalid(x):
        raise AuditError('Non-finite JSON token: ' + x)
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)


def plain(path: Path) -> None:
    for p in (path, *path.parents):
        if p.exists() or p.is_symlink():
            st = p.lstat()
            need(not stat.S_ISLNK(st.st_mode) and not (getattr(st, 'st_file_attributes', 0) & 0x400),
                 'Symlink/reparse path not allowed: ' + str(p))


def safe(root: Path, rel: str) -> Path:
    need(isinstance(rel, str) and bool(rel), 'Empty/non-string relative path')
    parts = rel.split('/')
    need(not PurePosixPath(rel).is_absolute() and '\\' not in rel and ':' not in rel
         and all(x not in ('', '.', '..') for x in parts), 'Unsafe relative path: ' + rel)
    p = root.joinpath(*parts)
    plain(p)
    return p


def normalize_record_path(rel: str) -> str:
    need(isinstance(rel, str), 'Invalid path record')
    return rel.replace('\\', '/')


class Reader:
    """Bounded file reader; hashes all inputs and rechecks them before publication."""
    def __init__(self, root: Path):
        self.root = root.absolute()
        self.records: dict[str, dict[str, Any]] = {}
        self.raw: dict[str, bytes] = {}
        self.total = 0

    def get(self, rel: str, sha: str | None = None, size: int | None = None,
            role: str = 'current_hash_not_historically_pinned') -> bytes:
        p = safe(self.root, rel)
        need(p.is_file(), 'Required file missing: ' + str(p))
        need(p.stat().st_size <= MAX_FILE, 'Unexpectedly large input: ' + rel)
        with p.open('rb') as f:
            raw = f.read(MAX_FILE + 1)
        need(len(raw) <= MAX_FILE, 'Input grew above size limit: ' + rel)
        h = digest(raw)
        if sha is not None:
            need(h == sha.lower(), 'SHA mismatch: ' + rel + '; observed=' + h)
        if size is not None:
            need(len(raw) == size, 'Size mismatch: ' + rel)
        if rel in self.records:
            need(self.records[rel]['sha256'] == h, 'Input changed during audit: ' + rel)
        else:
            self.total += len(raw)
            need(self.total <= MAX_TOTAL_READ, 'Bounded input budget exceeded')
            self.records[rel] = {'path': rel, 'sha256': h, 'size_bytes': len(raw), 'verification': role}
            self.raw[rel] = raw
        return raw

    def json(self, rel: str, sha: str | None = None, role: str = 'current_hash_not_historically_pinned') -> Any:
        return decode(self.get(rel, sha, role=role))

    def recheck(self) -> None:
        for rel, rec in list(self.records.items()):
            self.get(rel, rec['sha256'], rec['size_bytes'])


def publish(path: Path, raw: bytes) -> None:
    """Exclusive create: never replaces an existing file, even on repeat runs."""
    plain(path)
    need(not path.exists(), 'Existing output preserved: ' + str(path))
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def csv_read(raw: bytes) -> list[dict[str, str]]:
    f = io.StringIO(raw.decode('utf-8-sig'), newline='')
    reader = csv.DictReader(f)
    head = reader.fieldnames
    need(head and len(head) == len(set(head)), 'CSV header missing or duplicated')
    out = []
    for row in reader:
        need(None not in row and all(v is not None for v in row.values()), 'Malformed CSV row width')
        out.append(row)
    return out


def csv_encode(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b''
    fields = list(rows[0])
    need(all(set(r) == set(fields) for r in rows), 'Inconsistent derived CSV columns')
    h = io.StringIO(newline='')
    w = csv.DictWriter(h, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return h.getvalue().encode('utf-8')


def number(x: Any, name: str) -> float:
    need(not isinstance(x, bool), 'Boolean in numeric field: ' + name)
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise AuditError('Invalid numeric field: ' + name) from exc
    need(math.isfinite(v), 'Non-finite numeric field: ' + name)
    return v


def integer(x: Any, name: str) -> int:
    v = number(x, name)
    need(v.is_integer(), 'Non-integer field: ' + name)
    return int(v)


def boolean(x: Any, name: str) -> bool:
    if type(x) is bool:
        return x
    if x in ('True', 'true', '1', 1):
        return True
    if x in ('False', 'false', '0', 0):
        return False
    raise AuditError('Unrecognized boolean in ' + name)


def close(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=ATOL)


def mean(vals) -> float:
    seq = list(vals)
    need(bool(seq), 'Cannot average empty set')
    return math.fsum(seq) / len(seq)


def sd(vals) -> float:
    seq = list(vals)
    return statistics.stdev(seq) if len(seq) > 1 else 0.0


def split_id(repeat: int, fold: int) -> str:
    return f'R{repeat:02d}_F{fold:02d}'


def normalize_metric_row(row: dict[str, Any], stage: str) -> dict[str, Any]:
    required = set(METRICS) | {'model_id','seed','inner_fold','outer_fold','repeat_index',
                              'split_id','stage','script_sha256','failed','outcome_access',
                              'biology_access','q10_retention_applied'}
    need(required <= set(row), 'Metric row missing keys: ' + str(sorted(required - set(row))))
    need(not boolean(row['failed'], 'failed'),
         'Archived failed row found: preserve it; no success-only averaging is performed')
    for k in ('outcome_access','biology_access','q10_retention_applied'):
        need(not boolean(row[k], k), 'Historical boundary differs: ' + k)
    need(row['script_sha256'] == PINS[SOURCE_REL], 'Metric row binds another source')
    need(row['stage'] == stage, 'Wrong evaluation stage')
    m = row['model_id']; need(m in MODEL_IDS, 'Unexpected model: ' + str(m))
    r, f, i, s = (integer(row[k], k) for k in ('repeat_index','outer_fold','inner_fold','seed'))
    need(1 <= r <= 5 and 1 <= f <= 5 and row['split_id'] == split_id(r, f), 'Split identity mismatch')
    seeds = OUTER_SEEDS if stage == 'outer_evaluation' else INNER_SEEDS
    need(s in ((0,) if m == 'BLOCK_PCA' else seeds), 'Wrong seed schedule')
    need((i == 0) if stage == 'outer_evaluation' else (1 <= i <= 4), 'Wrong inner fold')
    out = {'repeat_index': r, 'outer_fold': f, 'split_id': row['split_id'], 'inner_fold': i,
           'model_id': m, 'seed': s, 'stage': stage}
    for k in METRICS:
        out[k] = number(row[k], k)
        need(-ATOL <= out[k] <= 1 + ATOL, 'Historical score outside [0,1]: ' + k)
    recomputed = math.fsum(WEIGHTS[k] * out[k] for k in WEIGHTS)
    need(close(recomputed, out['utility']), 'Recorded utility does not equal frozen weighted components')
    e = row.get('selected_epoch')
    missing = e is None or str(e).strip().lower() in ('', 'nan', 'none', 'null')
    if m.startswith('V'):
        need(not missing, 'Neural selected_epoch missing')
        out['selected_epoch'] = integer(e, 'selected_epoch')
        need(1 <= out['selected_epoch'] <= 400, 'Unexpected neural epoch')
    else:
        need(missing, 'Unexpected epoch for non-neural model')
        out['selected_epoch'] = None
    if stage == 'outer_evaluation':
        need('selected_by_inner_cv' in row, 'Outer selection flag missing')
        out['selected_by_inner_cv'] = boolean(row['selected_by_inner_cv'], 'selected_by_inner_cv')
    return out


def validate_schedule(rows: list[dict[str, Any]], stage: str) -> None:
    inner = stage == 'inner_model_selection'
    expected = {(r, f, i, m, s) for r in range(1, 6) for f in range(1, 6)
                for i in (range(1, 5) if inner else (0,)) for m in MODEL_IDS
                for s in ((0,) if m == 'BLOCK_PCA' else (INNER_SEEDS if inner else OUTER_SEEDS))}
    keys = [(x['repeat_index'],x['outer_fold'],x['inner_fold'],x['model_id'],x['seed']) for x in rows]
    need(len(keys) == len(set(keys)), 'Duplicated model/split/seed rows')
    need(set(keys) == expected, 'Incomplete/extra model-split-seed schedule: ' + stage)


def read_source_spec(raw: bytes) -> dict[str, Any]:
    """Read literals and source excerpts; never import/execute the historical script."""
    text = raw.decode('utf-8-sig'); tree = ast.parse(text)
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('PRIMARY_WEIGHTS','INNER_MODEL_SEEDS','OUTER_MODEL_SEEDS','EXPECTED_ELIGIBLE_MODELS'):
                literals[key] = ast.literal_eval(node.value)
    need(literals.get('PRIMARY_WEIGHTS') == WEIGHTS and tuple(literals.get('INNER_MODEL_SEEDS', ())) == INNER_SEEDS
         and tuple(literals.get('OUTER_MODEL_SEEDS', ())) == OUTER_SEEDS
         and tuple(literals.get('EXPECTED_ELIGIBLE_MODELS', ())) == MODEL_IDS,
         'Pinned source does not contain expected fixed weights/seeds/models')
    names = {'one_se_select','utility_from_components','reconstruction_score','r2_score','cross_view_mrr'}
    return {'literals': literals, 'functions': {
        n.name: {'start_line': n.lineno, 'end_line': n.end_lineno, 'source': ast.get_source_segment(text, n)}
        for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}}


def verify_target(reader: Reader) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = {p: reader.get(p, h, role='reviewed_historical_sha') for p, h in PINS.items()}
    lock, summary = decode(raw[LOCK_REL]), decode(raw[SUMMARY_REL])
    need(lock['status'] == '06B_TARGET_OUTCOME_BLIND_REPRESENTATION_BENCHMARK_RESULTS_FROZEN', 'TARGET result lock incomplete')
    need(summary['status'] == '06B_TARGET_OUTCOME_BLIND_REPRESENTATION_BENCHMARK_COMPLETE', 'TARGET summary incomplete')
    need(summary['n_cases'] == 78 and summary['primary_utility_weights'] == WEIGHTS,
         'TARGET cohort or utility weights differ')
    need(summary['eligible_models'] == list(MODEL_IDS) and summary['nested_cv'] == {
        'outer_repeats': 5, 'outer_folds': 5, 'inner_folds': 4}, 'TARGET frozen design differs')
    for field, p in [('runner_source', SOURCE_REL), ('summary', SUMMARY_REL),
                     ('inner_selection', SELECTION_REL), ('outer_benchmark', OUTER_REL)]:
        rec = lock[field]
        need(normalize_record_path(rec['path']) == p and rec['sha256'] == PINS[p], 'TARGET lock binding differs: ' + field)
    auth = lock['06b2_authorization_lock']
    auth_rel = normalize_record_path(auth['path'])
    need(auth_rel == 'paper5/method_contract/06b2/target_runner_v4_source_authorization_lock.json', 'Unexpected authorization path')
    reader.get(auth_rel, auth['sha256'], role='linked_from_pinned_TARGET_result_lock')
    need(all(v is False for v in summary['guardrails'].values()), 'TARGET guardrail state differs')
    spec = read_source_spec(raw[SOURCE_REL])
    variants = decode(raw[LOSS_REL])['ablation_components']
    component_keys = ('block_balanced_reconstruction','shared_private_decomposition','paired_contrastive_alignment',
                      'modality_dropout','missing_view_reconstruction','variational_latent')
    need(all(variants[V3][k] == variants[V4][k] for k in component_keys)
         and variants[V3]['stability_gate'] is False and variants[V4]['stability_gate'] is True,
         'V3/V4 specification no longer matches the reviewed distinction')
    outer = [normalize_metric_row(x, 'outer_evaluation') for x in csv_read(raw[OUTER_REL])]
    validate_schedule(outer, 'outer_evaluation')
    selection = csv_read(raw[SELECTION_REL])
    need(len(selection) == 25, 'Need 25 inner-selection records')
    lookup = {}
    for x in selection:
        r, f = integer(x['repeat_index'], 'repeat_index'), integer(x['outer_fold'], 'outer_fold')
        need(x['split_id'] == split_id(r,f) and 1 <= r <= 5 and 1 <= f <= 5, 'Selection split mismatch')
        need(x['split_id'] not in lookup and x['selected_model_id'] in MODEL_IDS
             and x['best_model_id'] in MODEL_IDS, 'Invalid/duplicated selection row')
        lookup[x['split_id']] = x
    for x in outer:
        need(x['selected_by_inner_cv'] == (x['model_id'] == lookup[x['split_id']]['selected_model_id']),
             'Outer annotation disagrees with historical inner selection')
    counts = Counter(x['selected_model_id'] for x in selection)
    ranked = sorted(MODEL_IDS, key=lambda m: (-counts[m], COMPLEXITY[m], m))
    chosen = [ranked[0]] if counts[ranked[0]] / 25 >= .4 else ranked[:2]
    need(summary['final_full_data_method_freeze']['primary_model_ids'] == chosen, 'Pinned final choice disagrees with selection records')
    spec['historical_primary_models'] = chosen
    return outer, selection, spec


def inner_files(reader: Reader) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    paths = [f'{TARGET_ROOT}/shards/inner__{split_id(r,f)}__I{i:02d}__{m}__S{s}.json'
             for r in range(1,6) for f in range(1,6) for i in range(1,5) for m in MODEL_IDS
             for s in ((0,) if m == 'BLOCK_PCA' else INNER_SEEDS)]
    missing = [p for p in paths if not safe(reader.root,p).is_file()]
    info = {'expected_shards': 1900, 'present_shards': 1900-len(missing), 'missing_paths': missing,
            'historically_individually_hashed_by_06b_lock': False,
            'scope': 'Current inner-shard hashes plus numerical replay against pinned selection; not proof of all original inner component bytes.'}
    if missing:
        info['status'] = 'UNAVAILABLE_NO_PARTIAL_INNER_AVERAGE'
        return None, info
    rows = [normalize_metric_row(reader.json(p), 'inner_model_selection') for p in paths]
    validate_schedule(rows, 'inner_model_selection')
    info['status'] = 'CURRENT_SHARDS_READ_SELECTION_REPLAY_REQUIRED'
    return rows, info


def model_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for x in rows:
        groups[(x['repeat_index'],x['outer_fold'],x['model_id'])].append(x)
    result = []
    for (r,f,m), vals in sorted(groups.items()):
        out = {'repeat_index': r, 'outer_fold': f, 'split_id': split_id(r,f), 'model_id': m,
               'stage': vals[0]['stage'], 'n_records': len(vals)}
        # Inner design has equal two seeds per four inner folds; preserve fold-first averaging.
        byinner = defaultdict(list)
        for v in vals:
            byinner[v['inner_fold']].append(v)
        for k in METRICS:
            out[k] = mean(mean(y[k] for y in block) for _, block in sorted(byinner.items()))
        result.append(out)
    return result


def pair_rows(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    a = {(x['repeat_index'],x['outer_fold'],x['inner_fold'],x['seed']): x for x in rows if x['model_id'] == left}
    b = {(x['repeat_index'],x['outer_fold'],x['inner_fold'],x['seed']): x for x in rows if x['model_id'] == right}
    need(a and set(a) == set(b), 'Paired evaluations do not match by split and seed')
    out = []
    for key in sorted(a):
        r,f,i,s = key; x,y = a[key], b[key]
        d = {'repeat_index':r,'outer_fold':f,'split_id':split_id(r,f),'inner_fold':i,'seed':s,'left':left,'right':right}
        for k in METRICS:
            d['delta_'+k] = x[k]-y[k]
        for k,w in WEIGHTS.items():
            d['weighted_'+k] = w*d['delta_'+k]
        d['weighted_component_sum'] = math.fsum(d['weighted_'+k] for k in WEIGHTS)
        d['utility_replay_residual'] = d['delta_utility']-d['weighted_component_sum']
        need(close(d['delta_utility'], d['weighted_component_sum']), 'Paired utility decomposition failed')
        out.append(d)
    return out


def aggregate_pairs(rows: list[dict[str, Any]], by_repeat: bool = False) -> list[dict[str, Any]]:
    fields = [k for k in rows[0] if k.startswith(('delta_', 'weighted_')) or k == 'utility_replay_residual']
    groups = defaultdict(list)
    for x in rows:
        groups[(x['repeat_index'],) if by_repeat else (x['repeat_index'],x['outer_fold'])].append(x)
    out = []
    for key, vals in sorted(groups.items()):
        d = {'repeat_index':key[0]}
        if not by_repeat:
            d.update(outer_fold=key[1],split_id=split_id(*key))
        d.update(n_averaged_records=len(vals),left=vals[0]['left'],right=vals[0]['right'])
        for k in fields:
            d[k] = mean(x[k] for x in vals)
        out.append(d)
    return out


def contrast_summary(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for k in METRICS:
        vals = [x['delta_'+k] for x in split_rows]
        out.append({'metric':k,'direction':'higher historical TARGET score is better','left':V4,'right':V1,
                    'mean_delta':mean(vals),'descriptive_split_sd':sd(vals),'min_delta':min(vals),'max_delta':max(vals),
                    'positive_splits':sum(x>TIE_ATOL for x in vals),'negative_splits':sum(x < -TIE_ATOL for x in vals),
                    'tied_splits':sum(abs(x)<=TIE_ATOL for x in vals),'n_splits':len(vals),
                    'frozen_weight':WEIGHTS.get(k),
                    'mean_weighted_contribution':mean(x['weighted_'+k] for x in split_rows) if k in WEIGHTS else None,
                    'scope':'post-result descriptive; correlated splits; no p-values or inferential CI'})
    return out


def equality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def index(m):
        return {(x['repeat_index'],x['outer_fold'],x['inner_fold'],x['seed']):x for x in rows if x['model_id']==m}
    a,b=index(V3),index(V4);need(a and set(a)==set(b),'V3/V4 equality schedule differs')
    out=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        d={'stage':x['stage'],'split_id':x['split_id'],'inner_fold':key[2],'seed':key[3]}
        for k in METRICS:
            d[k+'_abs_delta']=abs(x[k]-y[k])
        d['all_metric_values_exactly_equal']=all(x[k]==y[k] for k in METRICS)
        d['all_metric_values_within_tolerance']=all(close(x[k],y[k]) for k in METRICS)
        d['selected_epochs_equal']=x['selected_epoch']==y['selected_epoch']
        d['scope']='scalar result equality, not checkpoint/latent byte identity'
        out.append(d)
    return out


def one_se_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries=[]
    for m in MODEL_IDS:
        vals=[x for x in rows if x['model_id']==m]
        fold_means=[mean(x['utility'] for x in vals if x['inner_fold']==f) for f in range(1,5)]
        summaries.append({'model_id':m,'mean':mean(fold_means),'se':sd(fold_means)/2})
    best=max(x['mean'] for x in summaries)
    co_best=[x for x in summaries if abs(x['mean']-best)<=TIE_ATOL]
    chosen_best=min(co_best,key=lambda x:(COMPLEXITY[x['model_id']],x['model_id']))
    cutoff=best-chosen_best['se']
    within=sorted([x['model_id'] for x in summaries if x['mean']>=cutoff-TIE_ATOL], key=lambda m:(COMPLEXITY[m],m))
    return {'best_mean':best,'best_se':chosen_best['se'],'one_se_cutoff':cutoff,
            'best_model_id':chosen_best['model_id'],'selected_model_id':within[0],
            'within_one_se':within,'tied_numerical_maxima':sorted(x['model_id'] for x in co_best),
            'model_statistics':summaries}


def verify_selection(inner: list[dict[str, Any]], selection: list[dict[str, Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    lookup={x['split_id']:x for x in selection};output=[];stats=[]
    for r in range(1,6):
        for f in range(1,6):
            sid=split_id(r,f);hist=lookup[sid]
            replay=one_se_replay([x for x in inner if x['split_id']==sid])
            for k in ('best_model_id','selected_model_id'):
                need(replay[k]==hist[k], 'One-SE categorical replay differs: '+sid+' '+k)
            for k in ('best_mean','best_se','one_se_cutoff'):
                need(close(replay[k],number(hist[k],k)), 'One-SE numeric replay differs: '+sid+' '+k)
            need(replay['within_one_se']==hist['within_one_se'].split(';'), 'One-SE eligible set differs: '+sid)
            d={'repeat_index':r,'outer_fold':f,'split_id':sid,'historical_selected':hist['selected_model_id'],
               'historical_best_label':hist['best_model_id'],'replayed_best_mean':replay['best_mean'],
               'replayed_best_se':replay['best_se'],'replayed_one_se_cutoff':replay['one_se_cutoff'],
               'all_tied_numerical_maxima':';'.join(replay['tied_numerical_maxima']),
               'within_one_se':';'.join(replay['within_one_se']),
               'V4_is_tied_maximum':V4 in replay['tied_numerical_maxima'],
               'selection_replay':'PASS','historical_selection_changed':False}
            output.append(d)
            for s in replay['model_statistics']:
                stats.append({'repeat_index':r,'outer_fold':f,'split_id':sid,**s,
                              'tied_numerical_maximum':s['model_id'] in replay['tied_numerical_maxima'],
                              'historical_best_label':s['model_id']==hist['best_model_id'],
                              'historical_selected':s['model_id']==hist['selected_model_id']})
    return output,stats


def subspace_agreement(x: Any, y: Any) -> dict[str, Any]:
    """Same definition as frozen sarc5; not resampling stability or a q10 test."""
    import numpy as np
    def basis(a):
        a = np.asarray(a, dtype=np.float64)
        need(a.ndim == 2 and np.isfinite(a).all(), 'Invalid latent agreement input')
        a = a - a.mean(axis=0)
        u,s,_ = np.linalg.svd(a, full_matrices=False)
        tol = max(a.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
        return u[:,s>tol]
    need(len(x)==len(y), 'Agreement row count differs')
    a,b=basis(x),basis(y)
    if min(a.shape[1],b.shape[1])==0:
        return {'status':'UNDEFINED_ZERO_RANK','value':None,'rank_left':a.shape[1],'rank_right':b.shape[1]}
    c=np.clip(np.linalg.svd(a.T@b, compute_uv=False),0.0,1.0)
    return {'status':'DESCRIPTIVE_SEED_ONLY','value':float(c.mean()),'rank_left':a.shape[1],'rank_right':b.shape[1]}


def external_audit(reader: Reader, cohort: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    spec=EXTERNAL[cohort];base=spec['base']+'/limited_model_v1';root=base+'/scientific_results'
    required={base+'/fixed_protocol.json':spec['protocol_sha'],base+'/split_registry.json':spec['split_sha'],
              root+'/completion_lock.json':spec['completion_sha']}
    missing=[p for p in required if not safe(reader.root,p).is_file()]
    if missing:
        return {'status':'UNAVAILABLE_NOT_RECONSTRUCTED','missing':missing},{}
    objects={p:reader.json(p,h,role='reviewed_external_sha') for p,h in required.items()}
    plan=objects[base+'/fixed_protocol.json'];split=objects[base+'/split_registry.json'];lock=objects[root+'/completion_lock.json']
    need(lock['status']==spec['status'] and lock['n_evaluations']==25 and lock['n_failed_evaluations']==0,
         cohort+': incomplete result lock')
    ids=split['case_ids'];need(len(ids)==len(set(ids))==spec['n'],'External patient count/order invalid')
    need(len(split['splits'])==5 and sorted(i for sp in split['splits'] for i in sp['test'])==list(range(len(ids))),
         cohort+': test partitions incomplete')
    models=('BLOCK_PCA_BUDGETED_K10',V1,V4)
    need(plan['models']==list(models) and plan['CV']['neural_fit_seeds']==list(INNER_SEEDS), 'External protocol changed')
    records={x['path']:x for x in lock['files']}
    jobs=[(f,m,s) for f in range(1,6) for m in models for s in ((0,) if m==models[0] else INNER_SEEDS)]
    expected={f'shards/F{f:02d}__{m}__S{s}/{fn}' for f,m,s in jobs for fn in ('result.json','receipt.json')}
    expected|={f'preprocessing/fold_{f:02d}.json' for f in range(1,6)}
    expected|={'summaries/'+n for n in ('fold_metrics.csv','model_summary.csv','paired_descriptive_contrasts.csv','seed_only_agreement.csv','summary.json')}
    expected.add('execution_start.json')
    need(len(records)==len(lock['files'])==61 and set(records)==expected, 'External completion file set differs')
    for rel,rec in sorted(records.items()):
        reader.get(root+'/'+rel,rec['sha256'],rec['size_bytes'],role='bound_by_reviewed_external_completion')
    reader.get(SARC_HELPER,SARC_HELPER_SHA,role='reviewed_numerical_source_read_not_executed')
    data={};epochs=[];geometries=[]
    for sp in split['splits']:
        f=sp['fold'];sets={k:set(sp[k]) for k in ('fit','early','train','test')}
        need(sets['fit'].isdisjoint(sets['early']) and sets['fit']|sets['early']==sets['train']
             and sets['train'].isdisjoint(sets['test']) and sets['train']|sets['test']==set(range(len(ids))),
             'External fit/test partition error')
        geometries.append({'cohort':cohort,'fold':f,**{k+'_n':len(sp[k]) for k in sets},
                           'assessment_rows':'held-out outer-test','seed_pair':'1103;2207'})
    for f,m,s in jobs:
        rel=f'shards/F{f:02d}__{m}__S{s}/result.json';raw=reader.raw[root+'/'+rel];d=decode(raw)
        receipt=decode(reader.raw[root+f'/shards/F{f:02d}__{m}__S{s}/receipt.json'])
        rr=receipt['result'];need(rr['path']=='result.json' and rr['sha256']==digest(raw) and rr['size_bytes']==len(raw),
                                'External receipt mismatch')
        need(d['bindings']==lock['bindings']==receipt['bindings'],'External result binding mismatch')
        sp=split['splits'][f-1]
        need(d['fold']==f and d['test_row_indices']==sp['test']
             and d['test_case_ids']==[ids[i] for i in sp['test']], 'External latent row order mismatch')
        e=d['evaluation'];need(e['status']=='PASS' and e['model_id']==m and e['seed']==s,'External job identity differs')
        need(len(e['latent'])==len(sp['test']) and all(len(row)==(10 if s==0 else 5) for row in e['latent']),
             'External latent shape differs')
        data[(f,m,s)]=e
        if s:
            a=e['fit_audits'];need(len(a)==2 and a[0]['selected_epoch']==e['selected_epoch']==a[1]['epochs_run'],
                                 'External selected/refit epoch mismatch')
            epochs.append({'cohort':cohort,'fold':f,'model_id':m,'seed':s,
                           'epoch_selection_epochs_run':a[0]['epochs_run'],'selected_epoch':e['selected_epoch'],
                           'refit_epochs_run':a[1]['epochs_run'],'at_400_cap':a[0]['epochs_run']==400,
                           'learning_curve_replayed':False,'terminal_slope':None,
                           'reason':'Full per-epoch curves are not in the allowlisted result records; best/final alone are not a slope.'})
    stored=csv_read(reader.raw[root+'/summaries/seed_only_agreement.csv'])
    need(len(stored)==10,'External seed-agreement table incomplete')
    old={(integer(x['fold'],'fold'),x['model_id']):x for x in stored}
    agreement=[]
    for f in range(1,6):
        for m in (V1,V4):
            value=subspace_agreement(data[(f,m,1103)]['latent'],data[(f,m,2207)]['latent'])
            row=old[(f,m)];v=value['value']
            if v is not None:
                need(close(v,number(row['value'],'seed agreement')),'External agreement replay differs')
            else:
                need(row.get('value','').lower() in ('','none','nan'),'Undefined rank previously had numeric agreement')
            agreement.append({'cohort':cohort,'fold':f,'model_id':m,'seed_left':1103,'seed_right':2207,
                              'train_n':len(split['splits'][f-1]['train']),'assessment_n':len(split['splits'][f-1]['test']),
                              'assessment_rows':'held-out outer-test','statistic':'centered_mean_principal_angle_cosine',
                              **value,'interpretation':'same-training-fold initialization agreement; no resampling/q10/biological truth'})
    avg=[]
    for m in (V1,V4):
        vals=[x['value'] for x in agreement if x['model_id']==m]
        avg.append({'cohort':cohort,'model_id':m,'planned_folds':5,'defined_folds':sum(v is not None for v in vals),
                    'mean_only_if_all_folds_defined':mean(vals) if all(v is not None for v in vals) else None,
                    'pooled_across_cohorts':False,'sample_size_effect_established':False})
    info={'status':'PINNED_RESULTS_SEED_AND_EPOCH_REPLAY_COMPLETE','files_verified':61,
          'small_test_latents_read':True,'models_fitted':0,'protocol_provenance_boundary':plan.get('provenance_boundary'),
          'full_learning_curves_present_in_read_records':False,'TARGET_06e4_statistic_not_assumed_identical':True}
    return info,{'seed_agreement_by_fold':agreement,'seed_agreement_summary':avg,'epoch_cap_audit':epochs,'split_geometries':geometries}


def target_06e4_inventory(reader: Reader) -> dict[str, Any]:
    """Discover only 06e4-named diagnostic sources/outputs; do not interpret unknown latent schemas."""
    paths=[];sources={};unread=[]
    script_dir=safe(reader.root,'paper5/scripts')
    if script_dir.is_dir():
        for p in sorted(script_dir.iterdir()):
            if p.name.startswith('06e4') and p.suffix=='.py':
                plain(p);need(p.is_file(),'Unexpected 06e4 source object')
                rel=p.relative_to(reader.root).as_posix()
                raw=reader.get(rel,role='current_06e4_source_for_later_schema_review_not_executed')
                need(len(raw)<=300000,'Unexpectedly large 06e4 source')
                sources[rel]=raw.decode('utf-8-sig')
                paths.append({'path':rel,'size_bytes':len(raw),'sha256':digest(raw),'action':'source copied, not executed'})
    for parent in ('paper5/results/target_representation','paper5/method_contract'):
        directory=safe(reader.root,parent)
        if not directory.is_dir():
            continue
        candidates=[p for p in directory.iterdir() if p.name.startswith('06e4')]
        need(len(candidates)<=30,'Unexpected 06e4 directory count')
        for top in sorted(candidates):
            plain(top)
            stack=[(top,0)]
            while stack:
                p,depth=stack.pop();plain(p)
                need(len(paths)<500,'06e4 inventory limit exceeded')
                if p.is_dir():
                    if depth>=3:
                        unread.append(p.relative_to(reader.root).as_posix());continue
                    stack.extend((q,depth+1) for q in sorted(p.iterdir(),reverse=True))
                elif p.is_file():
                    rel=p.relative_to(reader.root).as_posix()
                    # Names and sizes only: no unreviewed coordinate/biological arrays decoded.
                    paths.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':None,'action':'filename/size inventory only; values not read'})
    return {'status':'INVENTORY_ONLY_ADAPTER_REQUIRES_REVIEW' if paths else 'NO_06E4_ARTIFACTS_FOUND_IN_NAMED_LOCATIONS',
            'files':paths,'source_texts':sources,'depth_limited_directories_not_scanned':unread,
            'full_data_TARGET_not_substituted_for_heldout':True,'TARGET_subspace_recomputed':False,
            'cross_cohort_curve_by_n_created':False,
            'scope':'The verified 06b metrics-only output does not provide outer-fold latent matrices; unknown 06e4 schemas are not guessed.'}


def metric_boundary_rows() -> list[dict[str,Any]]:
    return [
        {'item':'ordinary reconstruction','TARGET':'clipped [0,1] score; neural PCA coordinates; per-view training-null normalization',
         'SARC_KIRC':'observed molecular targets after inverse PCA; un-clipped NMSE; training-null normalization',
         'comparison':'not a common numeric endpoint; do not convert TARGET score to external NMSE'},
        {'item':'omitted-view reconstruction','TARGET':'mean clipped [0,1] test-centered R2-style score in neural PCA coordinates',
         'SARC_KIRC':'observed molecular targets; training-normalized omitted-view NMSE',
         'comparison':'target space, normalization and clipping differ; no direct numerical cross-cohort scale'},
        {'item':'retrieval','TARGET':'six ordered pairs; harmonic-null adjusted MRR; actual test n',
         'SARC_KIRC':'same basic retrieval definition; actual 50 or 60/61 candidates',
         'comparison':'candidate counts differ; comparable definition does not isolate an n effect'},
        {'item':'decision','TARGET':'inner-CV utility plus one-SE/complexity rule; outer scores excluded from selection',
         'SARC_KIRC':'report all three fixed procedures; no cohort-specific model selection',
         'comparison':'outer decomposition cannot alone explain the historical inner selection'},
        {'item':'initialization reproducibility','TARGET':'06e4 source and fit/assessment scope must be reviewed; no substitution of full-data for held-out',
         'SARC_KIRC':'same-training-fold two-seed agreement of held-out test representations',
         'comparison':'no common three-cohort stability curve or q10 validation is claimed'},
        {'item':'sample size','TARGET':'78 matched cases, different cohort and nested design',
         'SARC_KIRC':'250/304 cases, different diseases and fixed-procedure folds',
         'comparison':'cohort-context comparison; no isolated causal/sample-size effect'},
    ]


def manuscript_notes(info: dict[str,Any], outer_summary: list[dict[str,Any]]) -> str:
    vals={r['metric']:r for r in outer_summary}
    rows='\n'.join(f"| `{k}` | {v['mean_delta']:+.6f} | {v['mean_weighted_contribution']:+.6f} |"
                   for k,v in vals.items() if k in WEIGHTS)
    inner=info['inner']['status']
    return f'''# Paper 5: proposed interpretation edits (not applied to the manuscript)

## Status and provenance
This is a post-result descriptive audit of existing outputs, not a new prespecified
experiment. The historical selection, utility weights, models, seeds, thresholds
and frozen result files have not been changed. No new model fitting, p-values,
inferential confidence intervals or cohort pooling were performed.

## Suggested short-paper title (author decision still required)
Latent-Factor Stability Is Not Sufficient: A Simulation Study and Multi-Omics
Applications of SABLE-MO

## TARGET component decomposition
The frozen TARGET utility is U = 0.30 R + 0.30 Q + 0.25 M + 0.15 B.
Reported V4-minus-V1 differences were paired by repeat, outer fold and seed.
For the primary outer description the three original seeds were averaged within
each split; the 25 split means were summarized with equal weights. The separate
1103/2207 table is a post-result common-seed sensitivity, not a replacement.

| TARGET component | Mean V4 minus V1 | Contribution to delta U |
|---|---:|---:|
{rows}

The recorded mean outer utility difference is {vals['utility']['mean_delta']:+.6f}.
These contributions explain the arithmetic of the fixed utility; they do not
estimate a causal contribution of any training component. Inner and outer
analyses remain separate. Inner component/replay status: `{inner}`.
The historical one-SE rule uses inner-fold means and its standard-error rule;
that rule is replayed for provenance, not offered as a new inferential test.

## V3 and V4
V3 and V4 share the specified representation-training components. The V4 post-fit
retention layer was not applied during TARGET cross-family selection or during
the limited SARC/KIRC experiments. Numerical row equality is reported separately
from checkpoint/latent identity. If numerical maxima are tied, a label resolved
by complexity is not evidence that the tied method never reached the maximum.
Use `target_v3_v4_equality.csv` and, when available, `target_inner_selection_replay.csv`.

## Cross-cohort interpretation
Describe task- and cohort-context-dependent behavior, not a threshold sample size
at which V4 becomes effective. TARGET reconstruction components are clipped PCA-
space scores, whereas SARC/KIRC use observed molecular-target NMSE. Do not pool
these numbers into a common reconstruction ranking or convert 1-score to NMSE.
SARC/KIRC use cohort-specific refitting, not zero-shot application of TARGET
weights. The TARGET V1 selection remains unchanged.

## Representation versus retention
Retention operating properties were evaluated in simulations. The real-cohort
applications evaluate representations before the retention layer. Agreement of
two seeds is initialization reproducibility in its specified fit/assessment
setting, not subsampling stability, a q10 test or biological factor validation.
A Boolean variant specification alone does not demonstrate execution of a gate.

## Study-method coverage
TARGET: historical block PCA, MOFA K5/K10/K15/K20, and the specified neural models.
SARC and KIRC: only budgeted block PCA K10, V1, and pre-retention V4.
MOVE and iCluster2b: a separate confirmatory simulation branch. Do not say that
all these comparators were evaluated in all three real cohorts.

## Seed-dependent biological alignment
Keep the previously fixed primary seed. Report any existing seed-specific M34/M40
rank changes descriptively without selecting a more favorable seed. A larger
absolute correlation is not automatically stronger matched-random specificity.
This script does not reread biological scores, recompute matched-random tests,
or verify the quoted 06e4 M40/M34 values; insert those from their audited source.
Preselecting a seed prevents favorable-run selection but does not establish stability.

## KIRC chronology
The KIRC protocol was frozen after inspection of SARC results and technical KIRC
QC, but before KIRC model-performance evaluation. It is a fixed-procedure
replication, not an untouched preregistration predating both external studies.

## Training budget and dimension
Report the 400-epoch cap and actual selected epochs. Convergence is not established.
Best and final objectives cannot substitute for a full learning curve or a terminal
slope. Do not extend training solely for this descriptive audit. Fixing the PCA
rank in development documents provenance, not a sensitivity analysis of that rank.

## Code/Data Availability: replacement scope for editing
Paper 5 uses matched RNA expression, methylation and copy-number data from
TARGET-OS, TCGA-SARC and TCGA-KIRC with cohort-specific manifests and preprocessing.
Describe exact code, environment, input manifests and access procedures actually
released. Add only a verified repository URL, release identifier and license.
Do not claim that large matrices or data are in Git unless they are actually
included. Remove any copied Availability statement that identifies this study as
Molecular Transport Audit or identifies canine accessions as its primary multi-omics
cohorts. Canine program references elsewhere may be legitimate and must not be
blindly deleted. Repository/deposition details remain an author-supplied item.

## Submission checklist (manual manuscript edits; no source file was changed)
Check title/abstract scope, unfilled numeric placeholders, figure cross-references,
metric directions and denominators; preserve D05 and correlated-private limitations.
Neither completion of these edits nor a checksum audit guarantees publication.
'''


def build_assets(reader: Reader, outer: list[dict[str,Any]], selection: list[dict[str,Any]],
                 spec: dict[str,Any]) -> tuple[dict[str,bytes],dict[str,Any]]:
    files={};info={'version':VERSION,'post_result_descriptive_analysis':True,'new_fits':0,
                   'weights_changed':False,'historical_selection_changed':False,'clinical_outcomes_read':False,
                   'molecular_arrays_read':False,'project_python_modules_imported':False,
                   'new_p_values_or_inferential_intervals':False,'historical_primary_models':spec['historical_primary_models']}
    files['source_metric_spec.json']=encode(spec)
    files['metric_comparability.csv']=csv_encode(metric_boundary_rows())
    pairs=pair_rows(outer,V4,V1);split=aggregate_pairs(pairs);repeats=aggregate_pairs(split,by_repeat=True)
    summary=contrast_summary(split)
    files['target_outer_model_split_means.csv']=csv_encode(model_means(outer))
    files['target_outer_v4_v1_by_seed.csv']=csv_encode(pairs)
    files['target_outer_v4_v1_by_split.csv']=csv_encode(split)
    files['target_outer_v4_v1_by_repeat.csv']=csv_encode(repeats)
    files['target_outer_component_summary.csv']=csv_encode(summary)
    common=[x for x in pairs if x['seed'] in INNER_SEEDS]
    files['target_outer_common_two_seed_sensitivity.csv']=csv_encode(aggregate_pairs(common))
    dispersion=[]
    for r in range(1,6):
        for f in range(1,6):
            for m in (V1,V4):
                vals=[x for x in outer if x['split_id']==split_id(r,f) and x['model_id']==m]
                dispersion.append({'split_id':split_id(r,f),'model_id':m,'n_seeds':3,
                                   **{k+'_seed_sd':sd(x[k] for x in vals) for k in METRICS}})
    files['target_outer_within_split_seed_sd.csv']=csv_encode(dispersion)
    files['target_outer_seed_sd_summary.csv']=csv_encode([
        {'model_id':m,'metric':k,'mean_within_split_seed_sd':mean(x[k+'_seed_sd'] for x in dispersion if x['model_id']==m),
         'scope':'descriptive numerical initialization dispersion, not uncertainty of a population mean'}
        for m in (V1,V4) for k in METRICS])
    equality=equality_rows(outer)
    inner,inner_info=inner_files(reader);info['inner']=inner_info
    if inner is not None:
        replay,stats=verify_selection(inner,selection)
        inner_info['status']='ALL_1900_CURRENT_SHARDS_SELECTION_REPLAY_PASS'
        equality.extend(equality_rows(inner))
        ip=pair_rows(inner,V4,V1);isp=aggregate_pairs(ip)
        files['target_inner_v4_v1_by_seed_fold.csv']=csv_encode(ip)
        files['target_inner_v4_v1_by_outer_split.csv']=csv_encode(isp)
        files['target_inner_component_summary.csv']=csv_encode(contrast_summary(isp))
        files['target_inner_model_split_means.csv']=csv_encode(model_means(inner))
        files['target_inner_selection_replay.csv']=csv_encode(replay)
        files['target_inner_best_and_selected_counts.csv']=csv_encode([
            {'model_id':m,'tied_numerical_maximum_splits':sum(x['tied_numerical_maximum'] for x in stats if x['model_id']==m),
             'historical_best_label_splits':sum(x['historical_best_label'] for x in stats if x['model_id']==m),
             'historical_selected_splits':sum(x['historical_selected'] for x in stats if x['model_id']==m),
             'n_splits':25,'scope':'ties counted separately from complexity-resolved label'} for m in MODEL_IDS])
        # Explicitly normalized extracts: original per-shard hashes remain in audit.json.
        files['target_inner_normalized_metrics.json']=encode(inner,compact=True)
    files['target_v3_v4_equality.csv']=csv_encode(equality)
    info['outer']={'rows_verified':700,'paired_V4_V1_rows':75,'outer_split_pairs':25,
                   'repeat_means':5,'primary_outer_seeds':list(OUTER_SEEDS),
                   'inner_model_selection_recomputed_from_outer':False}
    info['V3_V4']={'n_paired_rows':len(equality),
                   'exact_metric_equality_rows':sum(x['all_metric_values_exactly_equal'] for x in equality),
                   'tolerance_metric_equality_rows':sum(x['all_metric_values_within_tolerance'] for x in equality),
                   'equal_selected_epoch_rows':sum(x['selected_epochs_equal'] for x in equality),
                   'checkpoint_or_latent_byte_identity_tested':False}
    info['external']={};aggregate=defaultdict(list)
    for cohort in EXTERNAL:
        state,tables=external_audit(reader,cohort);info['external'][cohort]=state
        print(cohort+' result diagnostic: '+state['status'],flush=True)
        for name,rows in tables.items():aggregate[name].extend(rows)
    for name,rows in aggregate.items():files['external_'+name+'.csv']=csv_encode(rows)
    inv=target_06e4_inventory(reader)
    for i,(rel,text) in enumerate(sorted(inv.pop('source_texts').items()),1):
        filename=f'TARGET_06e4_source_{i:02d}.txt';files[filename]=text.encode('utf-8')
        inv.setdefault('source_copy_map',{})[filename]=rel
    info['TARGET_06e4']=inv
    info['pending_scope_items']=[
        'TARGET 06e4 mathematical/schema/fit-scope review: no cross-cohort latent recomputation is claimed.',
        'Actual manuscript edits and repository/deposition URL: proposed text supplied, no manuscript overwritten.',
        'No component scale conversion, sample-size causal claim, or new model/weight selection.']
    if inner is None:info['pending_scope_items'].append('Inner decomposition unavailable; no partial inner averaging.')
    files['manuscript_edits_EN.md']=manuscript_notes(info,summary).encode('utf-8')
    # Include exact small TARGET anchors to make later review possible without another upload.
    for rel in (OUTER_REL,SELECTION_REL,SUMMARY_REL,LOCK_REL,SOURCE_REL,LOSS_REL):
        files['source_'+Path(rel).name]=reader.raw[rel]
    files['README_RU.md']=(
        '# Paper 5: описательные доработки без переобучения\n\n'
        'Все разности направлены V4 минус V1. В TARGET четыре score и utility: больше — лучше.\n'
        'Основной outer-анализ сохраняет seeds 1103/2207/3301; отдельная двух-seed таблица — sensitivity.\n'
        'Inner и outer не смешиваются. При отсутствии любого из 1900 inner-файлов inner-средние не строятся.\n'
        'Inner-файлы не перечислены по индивидуальным SHA в историческом 06b lock: текущие SHA и replay\n'
        'таблицы выбора не доказывают побайтную неизменность всех исторических inner-компонентов.\n'
        'V3/V4: сравниваются скалярные метрики и эпохи, не веса или все latent-матрицы.\n'
        'SARC/KIRC: перепроверено согласие test-представлений двух seeds одной формулой из sarc5;\n'
        '06e4 пока только инвентаризирован, статистика TARGET не подменяется другой задачей.\n'
        'Из best/final objective не вычисляется наклон learning curve.\n'
        'Тексты находятся в manuscript_edits_EN.md. Старые результаты, lock и рукопись не изменены.\n'
        'Все это post-result descriptive audit, не новый confirmatory эксперимент.\n'
    ).encode('utf-8')
    return files,info


def plan_document() -> dict[str,Any]:
    return {'version':VERSION,'analysis_role':'POST_RESULT_DESCRIPTIVE_INTERPRETATION_REFINEMENT',
            'historical_weights':WEIGHTS,'primary_outer_seeds':list(OUTER_SEEDS),'inner_seeds':list(INNER_SEEDS),
            'comparison':'V4 minus V1, original scales','inner_outer_separation':True,
            'secondary_common_seed_table':'1103/2207, separate sensitivity, no replacement of three-seed primary',
            'tie_absolute_tolerance':TIE_ATOL,'arithmetic_tolerances':{'absolute':ATOL,'relative':RTOL},
            'replay_one_se_without_reselection':True,'missing_inner_policy':'skip entire inner component/replay module; report every missing path',
            'failed_or_inconsistent_present_input_policy':'stop, retain evidence; do not drop rows',
            'external_seed_metric':'frozen sarc5 centered mean principal-angle cosine, no q10 decision',
            'TARGET06e4_policy':'bounded filename/size inventory and source copy only; unknown schemas not guessed',
            'prohibited':['model fitting','source edits','changes to weights/seeds/thresholds','clinical/outcome access',
                          'molecular/checkpoint loading','inferential p-values/CI','cross-cohort pooling',
                          'sample-size curve from different cohort protocols','automatic manuscript replacement']}


def run(project: Path, script: Path) -> Path:
    project=project.absolute();script=script.absolute();plain(project);plain(script)
    need((project/'paper5/scripts').is_dir(), 'Run from the workspace root containing paper5/scripts')
    need(script.name=='p5refine.py','Save the file as p5refine.py (not .txt or a duplicate (1) name)')
    print('Paper 5 - READ-ONLY TARGET components and manuscript refinements; NO FIT',flush=True)
    print('Version: '+VERSION,flush=True)
    print('GPU/R/network/molecular matrices/project imports: NOT USED',flush=True)
    parent=safe(project,'paper5/interpretation_audit_v1');parent.mkdir(exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=parent/('r_'+tag);out.mkdir(exist_ok=False)
    reader=Reader(project)
    source_raw=script.read_bytes();source_sha=digest(source_raw)
    publish(out/'p5refine_source.py',source_raw)
    publish(out/'analysis_plan.json',encode(plan_document()))
    try:
        outer,selection,spec=verify_target(reader)
        print('Pinned TARGET source / result lock / 700 rows / 25 selections: PASS',flush=True)
        print('Frozen utility recomputed; weights unchanged: PASS',flush=True)
        files,info=build_assets(reader,outer,selection,spec)
        print('TARGET inner module: '+info['inner']['status'],flush=True)
        print('V3/V4 scalar equality audit: '+str(info['V3_V4']['n_paired_rows'])+' pairs checked',flush=True)
        print('TARGET 06e4: '+info['TARGET_06e4']['status'],flush=True)
        reader.recheck();need(digest(script.read_bytes())==source_sha,'This audit source changed while running')
        all_numerical_inputs = (info['inner']['status'] == 'ALL_1900_CURRENT_SHARDS_SELECTION_REPLAY_PASS'
                                and all(x['status'] == 'PINNED_RESULTS_SEED_AND_EPOCH_REPLAY_COMPLETE'
                                        for x in info['external'].values()))
        completion_status = ('P5_READONLY_REFINEMENTS_COMPLETE_NO_NEW_FIT' if all_numerical_inputs
                             else 'P5_READONLY_REFINEMENTS_PARTIAL_INPUTS_NO_NEW_FIT')
        info.update(status=completion_status,
                    inputs_rechecked_unchanged=True,source_sha256=source_sha,
                    runtime={'python':sys.version,'platform':platform.platform()},
                    inputs=list(reader.records.values()))
        files['audit.json']=encode(info)
        for name,raw in sorted(files.items()):publish(safe(out,name),raw)
        manifest={name:{'sha256':digest(raw),'size_bytes':len(raw)} for name,raw in sorted(files.items())}
        manifest['analysis_plan.json']={'sha256':digest(encode(plan_document())),'size_bytes':len(encode(plan_document()))}
        manifest['p5refine_source.py']={'sha256':source_sha,'size_bytes':len(source_raw)}
        complete={'status':info['status'],'version':VERSION,'source_sha256':source_sha,'files':manifest,
                  'new_fits':0,'historical_files_changed':False,'scope_items_remaining':info['pending_scope_items']}
        # Prepare and size-check the compact packet before writing a completion marker.
        packet_files = {**files, 'analysis_plan.json': encode(plan_document()), 'p5refine_source.py': source_raw}
        packet={'schema':'paper5-readonly-refinements-evidence-v1','status':info['status'],'archive_required':False,
                'new_fits':0,'completion':complete,'files':{
                    name:{**manifest[name],'text':raw.decode('utf-8')} for name,raw in sorted(packet_files.items())}}
        report_bytes=encode(packet,compact=True)
        if len(report_bytes)>MAX_EVIDENCE:
            publish(out/'export_issue.json',encode({'status':'ASSETS_WRITTEN_EVIDENCE_TOO_LARGE',
                   'bytes':len(report_bytes),'limit':MAX_EVIDENCE,'no_sources_modified':True}))
            raise AuditError('Evidence exceeds 5 MiB; local assets preserved, do not rerun training')
        # A second final read catches changes made while outputs were written.
        reader.recheck()
        publish(out/'completion.json',encode(complete))
        dest=parent/('p5r_'+tag+'.json');publish(dest,report_bytes)
        print('Source results rechecked unchanged: PASS',flush=True)
        print('Status: '+info['status'],flush=True)
        print('Assets directory: '+str(out),flush=True)
        print('Evidence JSON (send this FILE; no ZIP): '+str(dest),flush=True)
        print('Evidence bytes: '+str(len(report_bytes)),flush=True)
        print('Evidence SHA-256: '+digest(report_bytes),flush=True)
        print('No new fits, historical selection, source authorization or manuscript overwrite.',flush=True)
        return dest
    except Exception as exc:
        failure={'status':'P5_REFINEMENT_AUDIT_STOPPED','error_type':type(exc).__name__,'error':str(exc),
                 'traceback':traceback.format_exc(),'source_sha256':source_sha,
                 'inputs_checked_before_stop':list(reader.records.values()),'new_fits':0,
                 'instruction':'Preserve old inputs and this report. Do not edit hashes or rerun training.'}
        path=out/'failure.json'
        if not path.exists():publish(path,encode(failure))
        print('Failure JSON: '+str(path),flush=True)
        raise


def main() -> None:
    need(len(sys.argv)==1,'No command-line options; use the existing workspace root')
    need(sys.version_info >= (3,10),'Python 3.10 or later is required')
    run(Path.cwd(),Path(__file__))


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Original results remain unchanged; keep the incomplete audit folder.')
        raise SystemExit(130)
    except Exception as exc:
        print('P5 REFINEMENT STOPPED: '+type(exc).__name__+': '+str(exc),flush=True)
        traceback.print_exc()
        raise SystemExit(2)
