#!/usr/bin/env python3
"""Paper 5: V0 external addendum protocol and ARTIFICIAL CUDA preflight only.

Save as p5v0prep3.py in Downloads; launch with the existing Python from the
workspace root. No command-line options. No downloads or real-cohort fitting.
Historical source, protocols, folds, matrices and result locks are not edited.
A separate exact-source authorization is required for a later V0 executor.

The frozen sarc5 evaluator intentionally accepts only V1/V4, not V0. Therefore
this NEW, explicitly recorded adapter calls its unchanged low-level training,
projection and endpoint functions. It does not monkeypatch sarc5's allowlist.
Artificial V1 regression comparisons check adapter parity with sarc5.

Revision 2 is an infrastructure-only amendment for the reported Windows
platform._syscmd_ver -> subprocess.Popen failure. A bounded exception permits
only the standard-library VER query through the absolute Windows system shell;
network and all other subprocess/file restrictions stay in effect. This is an
accidental-access audit, not an OS-level security sandbox. No interpreter,
package, numerical function, scientific setting or previous file is patched.
The previous failed v1 directory is verified and preserved. New output goes to
paper5/v0_external_v3. This entry point still cannot authorize real fitting.

Revision 3 repairs a same-process Windows byte-range-lock conflict: checks use
only the handle that owns a lock, never reopen a locked marker for reading.
Existing v1/v2 markers are opened without creation, truncation or initialization.
Their directories and all numerical settings remain unchanged. v2 is accepted
only in the reported lock-only state (no committed protocol, report or fits).
The unchanged v2 source must remain beside this file in Downloads for verification.
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
import ntpath
import os
import platform
import stat
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

VERSION = 'p5-v0-external-addendum-protocol-and-artificial-cuda-preflight-v3-owner-handle-lock-fix-no-real-fit'
LEGACY_VERSION = 'p5-v0-external-addendum-protocol-and-artificial-cuda-preflight-v1-no-real-fit'
LEGACY_BASE = 'paper5/v0_external_v1'
LEGACY_SOURCE_SHA = 'd590287d3cf95f8e0ab7c5371cfcc39e2483c0913d223c88c175fae875aa7383'
KNOWN_FAILURE = 'V0 PREFLIGHT FIREWALL: Network/subprocess not allowed: subprocess.Popen'
FINAL_STATUS = 'P5_V0_PROTOCOL_AND_SYNTHETIC_PREFLIGHT_PASS_NOT_AUTHORIZED'
V0 = 'V0_BALANCED_AE'
V1 = 'V1_SHARED_PRIVATE_AE'
V4 = 'V4_SABLE_MO'
VIEWS = ('RNA', 'METHYLATION', 'CNV')
SEEDS = (1103, 2207)
BUDGETS = {'RNA': 5000, 'METHYLATION': 10000, 'CNV': 5000}
PC_RANKS = {v: 10 for v in VIEWS}
BASE = 'paper5/v0_external_v3'
FAILED_V2_BASE = 'paper5/v0_external_v2'
FAILED_V2_SOURCE_SHA = 'ab552a236465b9ede78ab9a9e0f465ccf0a185133f5ae1b5fa3541c74b79d3d1'
K4_REL = 'paper5_parallel_kirc/limited_model_v1/kirc4_source.py'
K4_SHA = '0d03eebd11e67325b319360e7ee4b9ee030bcdd2741b8d7b08e15204cdf33e41'
S5_REL = 'paper5_parallel_sarc/sarc5.py'
S5_SHA = 'dd16e547972d8d298491e2d3c5003dea16760e969e9ea846fda5b0f83020f96f'
SOURCE_PINS = {
    'paper5/scripts/06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py':
        '4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab',
    'paper5/scripts/05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py':
        '05ae6923ce588c137bc6f4aa934262181dd99f4ebe6bec3250e4b7cd54ed1ff3',
    'paper5/scripts/05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py':
        '8ab3b84f1d7bcd475b88b5f06d49e661c2d6230ea3598406e66970d792c74a5c',
    'paper5/method_contract/05d1/neural_architecture_regularization_profile_grid.csv':
        '118b2fd4c16751f0b83ac3053f726f9421db26949ab98a8a531ff584c194ee32',
    'paper5/method_contract/05d1/neural_loss_and_architecture_contract.json':
        'b90ded1d011ee3210154c2102389df0b4e0c8fcf668b70d7e2b8e14184e3f6b1',
    S5_REL: S5_SHA,
    K4_REL: K4_SHA,
}
COHORTS = {
    'SARC': {
        'base': 'paper5_parallel_sarc', 'n': 250,
        'protocol_sha': 'f3a63188ecbaff4a5164d999d933a5f6481ff7d0560b999022faa38f7a0cfff8',
        'split_sha': '78a702ac43d55dd37f8b194b7b7933fb547f09e04cf6333662b57221160377f0',
        'results_sha': 'ca6f6931de8ef223792488b0dd43b8b66b3a29e010be395466973da080698fc1',
        'input_rel': 'pointwise_inputs_v1/sarc4_20261010T015241Z_dbf619ed',
        'input_sha': '81a0c365e9b7d62af957161350f7fabc4de700785285aab16d8436d89e863265',
        'patient_sha': 'e279425a714649c30a409d091fd9c819b86e2912784e9413a53e21d7ccfdde3f',
        'geometry': [[160,40,200,50]] * 5,
        'completed_status': 'LIMITED_SARC_FIXED_MODEL_BENCHMARK_COMPLETE_FROZEN',
        'artificial_seed': 2026101101,
    },
    'KIRC': {
        'base': 'paper5_parallel_kirc', 'n': 304,
        'protocol_sha': '6ac98e5fe49065ba450a54a744b33fbc45197a382dd4a522ec784e494fc232e1',
        'split_sha': 'f7b73461fe420d3ded1f5450aef3402e0c48ac07c1ef570aa0eddff1e3c7661b',
        'results_sha': '09679c7f6ad778631f7b5173b4cae11cdbaea181d12aa1f557ef2f821470d54a',
        'input_rel': 'pointwise_inputs_v1/kirc3_20261010T232651Z_5cc5b59c',
        'input_sha': 'd00cc7ce1714658b9c8d5b5e8b7e47d251c69a4ae9d02c06605405025726f9e4',
        'patient_sha': '35be88d1a151449da976210a924f76cd1a3da20278d6c4774eb3a6d5b344dbea',
        'geometry': [[194,49,243,61]] * 4 + [[195,49,244,60]],
        'completed_status': 'LIMITED_KIRC_FIXED_MODEL_BENCHMARK_COMPLETE_FROZEN',
        'artificial_seed': 2026101102,
    },
}
SMALL_LIMIT = 5 * 1024 * 1024
MAX_EVIDENCE = 5 * 1024 * 1024
ATOL, RTOL = 2e-12, 2e-10


def need(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(obj: Any, compact: bool = False) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       indent=None if compact else 2,
                       separators=(',', ':') if compact else None) + '\n').encode('utf-8')


def decode(data: bytes) -> Any:
    def pairs(items):
        out = {}
        for k,v in items:
            need(k not in out, 'Duplicate JSON key: ' + str(k)); out[k] = v
        return out
    def invalid(x):
        raise ValueError('Non-finite JSON token: ' + x)
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)


def plain(path: Path) -> None:
    for p in (path, *path.parents):
        if p.exists() or p.is_symlink():
            s = p.lstat()
            need(not stat.S_ISLNK(s.st_mode) and not (getattr(s, 'st_file_attributes', 0) & 0x400),
                 'Symlink/reparse path is not allowed: ' + str(p))


def child(root: Path, rel: str) -> Path:
    need(type(rel) is str and bool(rel), 'Empty relative path')
    pp = PurePosixPath(rel)
    need(not pp.is_absolute() and '\\' not in rel and ':' not in rel and
         all(p not in ('', '.', '..') for p in rel.split('/')), 'Unsafe relative path: ' + rel)
    p = root.joinpath(*pp.parts); plain(p)
    need(p.absolute().is_relative_to(root.absolute()), 'Path escapes workspace')
    return p


def read_file(path: Path) -> bytes:
    plain(path)
    need(path.is_file() and path.stat().st_size <= SMALL_LIMIT, 'Required small file missing/too large: ' + str(path))
    with path.open('rb') as f:
        b = f.read(SMALL_LIMIT + 1)
    need(len(b) <= SMALL_LIMIT, 'File grew past limit: ' + str(path))
    return b


def create(path: Path, data: bytes) -> None:
    plain(path); need(not path.exists(), 'Existing file is preserved: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / ('.writing_' + uuid.uuid4().hex + '.tmp')
    with temp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if os.name == 'nt':
        os.rename(temp, path)  # Windows rejects an existing destination.
    else:
        os.link(temp, path); temp.unlink()  # Exclusive publication on POSIX.


def pin(data: bytes) -> dict[str, Any]:
    return {'sha256': digest(data), 'size_bytes': len(data)}


class Reader:
    def __init__(self, project: Path):
        self.project = project
        self.records: dict[str, dict[str, Any]] = {}

    def get(self, rel: str, sha: str | None = None, size: int | None = None) -> bytes:
        need(Path(rel).suffix.lower() in ('.json', '.py', '.csv', '.tsv'),
             'This stage reads only allowlisted metadata/source/result bytes: ' + rel)
        b = read_file(child(self.project, rel))
        need(sha is None or digest(b) == sha, 'Historical SHA mismatch: ' + rel)
        need(size is None or len(b) == size, 'Historical size mismatch: ' + rel)
        rec = {'path': rel, **pin(b)}
        if rel in self.records:
            need(self.records[rel] == rec, 'Input changed during preparation: ' + rel)
        self.records[rel] = rec
        return b

    def recheck(self) -> None:
        for rec in list(self.records.values()):
            self.get(rec['path'], rec['sha256'], rec['size_bytes'])


def read_owned_process_lock(handle: Any, path: Path) -> bytes:
    """Read a locked marker ONLY through its owner, preserving its file offset.

    Windows byte-range locks belong to a file handle. A separately opened handle
    is denied even in the same process. This helper must stay out of Reader:
    Reader deliberately reopens ordinary scientific/source files for hashing.
    """
    need(not handle.closed, 'Process-lock owner is closed: ' + str(path))
    need(Path(handle.name).absolute() == path.absolute(),
         'Wrong owner handle for process lock: ' + str(path))
    position = handle.tell()
    try:
        handle.seek(0)
        need(os.fstat(handle.fileno()).st_size == 1,
             'Unexpected process-lock size; preserve the file: ' + str(path))
        data = handle.read(2)
        need(data == b'0', 'Unexpected process-lock contents; preserve the file: ' + str(path))
        return data
    finally:
        handle.seek(position)


@contextlib.contextmanager
def process_lock(path: Path, *, create_if_missing: bool = True):
    """Nonblocking OS lock; yield the ONLY handle permitted to read the marker.

    Historical markers must already exist and are never rewritten. New markers
    are initialized only after their lock is acquired. No unlink/force-unlock,
    append mode, replacement or privilege escalation is used.
    """
    plain(path)
    created_here = False
    if create_if_missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            f = path.open('x+b', buffering=0)
            created_here = True
        except FileExistsError:
            f = path.open('r+b', buffering=0)
    else:
        need(path.is_file(), 'Required historical process lock is missing: ' + str(path))
        f = path.open('r+b', buffering=0)
    acquired = False
    try:
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as e:
            raise RuntimeError('Cannot acquire process lock: ' + str(path) +
                               '. Another run may hold it; do not delete it or change permissions.') from e
        if created_here:
            need(os.fstat(f.fileno()).st_size == 0, 'New lock changed before acquisition: ' + str(path))
            need(f.write(b'0') == 1, 'Could not initialize new process-lock marker')
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
        read_owned_process_lock(f, path)
        yield f
    finally:
        try:
            if acquired:
                f.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def windows_ver_launch_matches(args: tuple[Any, ...], system_shell: str) -> bool:
    """Exact CPython Windows audit-event shapes; never a basename/prefix match."""
    if len(args) != 4 or not isinstance(system_shell, str):
        return False
    executable, command, cwd, env = args
    if not ntpath.isabs(system_shell) or ntpath.basename(system_shell).lower() != 'cmd.exe':
        return False
    # platform._syscmd_ver uses inherited cwd/env and an exact built-in command.
    if cwd is not None or env is not None or not isinstance(command, str):
        return False
    if executable is not None:
        if not isinstance(executable, str) or not ntpath.isabs(executable):
            return False
        if ntpath.normcase(ntpath.normpath(executable)) != ntpath.normcase(ntpath.normpath(system_shell)):
            return False
    allowed = (system_shell + ' /c "ver"', '"' + system_shell + '" /c "ver"')
    return command.casefold() in {x.casefold() for x in allowed}


def windows_probe_spec() -> dict[str, Any] | None:
    """Identify the existing CPython stdlib and OS shell, without invoking either."""
    if os.name != 'nt' or sys.platform != 'win32':
        return None
    need(sys.version_info[:2] == (3, 10), 'Use the existing Python 3.10 environment')
    root = os.environ.get('SystemRoot', '')
    need(bool(root) and ntpath.isabs(root), 'Missing absolute Windows SystemRoot')
    shell = ntpath.join(root, 'System32', 'cmd.exe')
    configured = os.environ.get('COMSPEC', '')
    need(bool(configured) and ntpath.isabs(configured) and
         ntpath.normcase(ntpath.normpath(configured)) == ntpath.normcase(ntpath.normpath(shell)),
         'COMSPEC is not the absolute System32/cmd.exe; do not change it blindly')
    # Preserve configured case, because it is used verbatim by subprocess.py.
    shell = configured
    plain(Path(shell)); need(Path(shell).is_file(), 'Windows system shell is missing')
    path = Path(platform.__file__).absolute()
    expected = (Path(sys.base_prefix) / 'Lib' / 'platform.py').absolute()
    need(path == expected, 'Unexpected platform module; no OS-command exception granted')
    raw = read_file(path)
    fn = getattr(platform, '_syscmd_ver', None)
    need(callable(fn) and hasattr(fn, '__code__') and
         Path(fn.__code__.co_filename).absolute() == path and fn.__globals__ is platform.__dict__,
         'Unexpected platform._syscmd_ver implementation')
    return {'shell': shell, 'code': fn.__code__, 'globals': platform.__dict__,
            'module_path': str(path), 'module_sha256': digest(raw),
            'module_size_bytes': len(raw)}


def on_platform_ver_stack(spec: dict[str, Any]) -> bool:
    """The allowed event must originate in the actual stdlib VER branch."""
    frame = sys._getframe(1)
    try:
        for _ in range(64):
            if frame is None:
                break
            if frame.f_code is spec['code'] and frame.f_globals is spec['globals']:
                return frame.f_locals.get('cmd') == 'ver'
            frame = frame.f_back
        return False
    finally:
        del frame


class PreflightFirewall:
    """An accidental real-array read or historical file write makes PASS impossible."""
    def __init__(self, project: Path, output: Path):
        self.root = os.path.normcase(os.path.abspath(project))
        self.out = os.path.normcase(os.path.abspath(output))
        self.active = False
        self.blocked: list[str] = []
        self.windows_spec = windows_probe_spec()
        self.windows_version_queries: list[dict[str, Any]] = []

    @staticmethod
    def under(p: str, base: str) -> bool:
        return p == base or p.startswith(base + os.sep)

    def stop(self, message: str) -> None:
        self.blocked.append(message)
        raise RuntimeError('V0 PREFLIGHT FIREWALL: ' + message)

    def hook(self, event, args):
        if not self.active: return
        if event == 'subprocess.Popen':
            spec = self.windows_spec
            if (spec is not None and len(self.windows_version_queries) < 8 and
                    windows_ver_launch_matches(args, spec['shell']) and on_platform_ver_stack(spec)):
                self.windows_version_queries.append({
                    'event': event, 'executable': args[0], 'command_line': args[1],
                    'purpose': 'CPython platform._syscmd_ver: local Windows version metadata only'})
                return
            self.stop('Network/subprocess not allowed: ' + event)
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
            self.stop('Network/subprocess not allowed: ' + event)
        if event == 'open' and args and not isinstance(args[0], int):
            try: p = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            except (TypeError,ValueError): return
            if self.under(p,self.root):
                mode = args[1] if len(args)>1 else None
                flags = args[2] if len(args)>2 and isinstance(args[2],int) else 0
                writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                if writing and not self.under(p,self.out): self.stop('Historical write: ' + p)
                if not writing and not self.under(p,self.out):
                    if p.endswith(('.npy','.npz','.pt','.pth','.pkl','.pickle','.rds','.rdata')):
                        self.stop('Real matrix/checkpoint read: ' + p)
                    forbidden = ('paper5/model_inputs','paper5/data','paper5/simulations','paper5/clinical','paper5/outcomes',
                                 'paper5_parallel_sarc/acquisition_v1','paper5_parallel_sarc/raw_staging_v1',
                                 'paper5_parallel_kirc/acquisition_v1','paper5_parallel_kirc/raw_staging_v1')
                    if any(self.under(p,os.path.normcase(os.path.join(self.root,*x.split('/')))) for x in forbidden):
                        self.stop('Raw/clinical/simulation tree read: ' + p)
        if event in ('os.remove','os.rmdir','os.rename','os.link','os.mkdir') and args:
            count = 2 if event in ('os.rename','os.link') else 1
            for obj in args[:count]:
                if isinstance(obj,(str,bytes,os.PathLike)):
                    p = os.path.normcase(os.path.abspath(os.fsdecode(obj)))
                    if self.under(p,self.root) and not self.under(p,self.out):
                        self.stop('Historical filesystem change: ' + p)

    def __enter__(self):
        sys.addaudithook(self.hook); self.active=True; return self

    def __exit__(self,*args): self.active=False


def validate_splits(reg: dict[str,Any], cohort: str) -> None:
    spec=COHORTS[cohort];n=spec['n'];ids=reg['case_ids']
    need(len(ids)==len(set(ids))==n and ids==sorted(ids), 'Wrong patient registry: '+cohort)
    need(len(reg['splits'])==5, 'Expected 5 folds: '+cohort)
    for fold,sp in enumerate(reg['splits'],1):
        need(sp['fold']==fold, 'Fold numbering differs')
        seq=[sp[k] for k in ('fit','early','train','test')]
        need([len(x) for x in seq]==spec['geometry'][fold-1], 'Fold geometry differs: '+cohort)
        need(all(x==sorted(set(x)) and all(type(i) is int and 0<=i<n for i in x) for x in seq), 'Invalid split rows')
        fit,early,train,test=map(set,seq)
        need(not fit&early and fit|early==train and not train&test and train|test==set(range(n)), 'Split overlap/omissions')
    need(sorted(i for sp in reg['splits'] for i in sp['test'])==list(range(n)), 'Test partition is incomplete')


def expected_historical_paths() -> set[str]:
    models=('BLOCK_PCA_BUDGETED_K10',V1,V4)
    paths={'execution_start.json'}
    paths.update('preprocessing/fold_%02d.json'%f for f in range(1,6))
    paths.update('summaries/'+x for x in ('fold_metrics.csv','model_summary.csv','paired_descriptive_contrasts.csv','seed_only_agreement.csv','summary.json'))
    for f in range(1,6):
        for m in models:
            for s in ((0,) if m==models[0] else SEEDS):
                for name in ('result.json','receipt.json'):
                    paths.add('shards/F%02d__%s__S%d/%s'%(f,m,s,name))
    need(len(paths)==61,'Internal historical file schedule error')
    return paths


def verify_historical(reader: Reader) -> dict[str,Any]:
    source_raw={rel:reader.get(rel,sha) for rel,sha in SOURCE_PINS.items()}
    arch=decode(source_raw['paper5/method_contract/05d1/neural_loss_and_architecture_contract.json'])
    expected_v0={'block_balanced_reconstruction':True,'shared_private_decomposition':False,
                 'paired_contrastive_alignment':False,'modality_dropout':False,
                 'missing_view_reconstruction':False,'stability_gate':False,'variational_latent':False}
    v0=arch['ablation_components'][V0]
    need(all(v0[k] is v for k,v in expected_v0.items()),'V0 definition differs')
    states={}
    for cohort,spec in COHORTS.items():
        base=spec['base'];limited=base+'/limited_model_v1'
        pr=reader.get(limited+'/fixed_protocol.json',spec['protocol_sha'])
        sr=reader.get(limited+'/split_registry.json',spec['split_sha'])
        rr=reader.get(limited+'/scientific_results/completion_lock.json',spec['results_sha'])
        cr=reader.get(base+'/'+spec['input_rel']+'/input_complete.json',spec['input_sha'])
        plan,reg,lock,complete=map(decode,(pr,sr,rr,cr))
        need(plan['models']==['BLOCK_PCA_BUDGETED_K10',V1,V4] and plan['CV']['neural_fit_seeds']==list(SEEDS), 'Historical method schedule changed')
        need(plan['preprocessing']['feature_budgets']==BUDGETS and plan['preprocessing']['neural_PCA_per_view']==PC_RANKS,
             'Historical feature/PCA budgets changed')
        need(plan['input_complete_sha256']==spec['input_sha'],'Input completion binding differs')
        need(plan['neural_profile']['maximum_epochs']==400 and plan['neural_profile']['patience']==30 and
             plan['neural_profile']['min_delta']==.0001, 'Historical epoch policy changed')
        validate_splits(reg,cohort)
        need(lock['status']==spec['completed_status'] and lock['n_evaluations']==25 and
             lock['n_failed_evaluations']==0 and lock['n_patients']==spec['n'] and lock['n_folds']==5,
             'Historical real benchmark not complete: '+cohort)
        recs=lock['files'];need(type(recs) is list and len(recs)==61,'Historical manifest schema differs')
        need({x['path'] for x in recs}==expected_historical_paths(),'Unexpected historical manifest files')
        for rec in recs:
            reader.get(limited+'/scientific_results/'+rec['path'],rec['sha256'],rec['size_bytes'])
        # Read only patient identity/order metadata, never molecular values.
        if cohort=='SARC':
            pts=reader.get(base+'/'+spec['input_rel']+'/patients.tsv',spec['patient_sha'])
            rows=list(csv.DictReader(io.StringIO(pts.decode('utf-8-sig')),delimiter='\t'))
            need([r['case_submitter_id'] for r in rows]==reg['case_ids'] and
                 all(int(r['row_index'])==i for i,r in enumerate(rows)),'SARC patient row order differs')
            inputfiles={r['path']:r for r in complete['files']}
        else:
            # Exact split registry is already bound to patient.tsv in the completed KIRC protocol.
            need(plan['patient_sha256']==spec['patient_sha'],'KIRC patient binding differs')
            inputfiles=complete['files']
        if cohort=='KIRC':
            need(complete['patients']==spec['n'] and complete['status']=='KIRC_RAW_AND_POINTWISE_INPUTS_COMPLETE_NOT_MODEL_AUTHORIZED', 'Prepared KIRC count/status differs')
        else:
            need(complete['status']=='POINTWISE_INPUTS_COMPLETE_NOT_MODEL_AUTHORIZED', 'Prepared SARC status differs')
        for view,(rel,sha,width) in plan['input_matrices'].items():
            need(view in VIEWS and inputfiles[rel]['sha256']==sha and width=={'RNA':60660,'METHYLATION':482421,'CNV':60623}[view],
                 'Prepared matrix metadata differs')
        states[cohort]={'plan':plan,'split':reg,'completion':complete,
            'protocol_bytes':pr,'split_bytes':sr,'result_lock':lock,'input_files':inputfiles}
    need(states['SARC']['plan']['neural_profile']==states['KIRC']['plan']['neural_profile'],
         'Cohort profiles are not the same')
    return {'cohorts':states,'architecture':arch,'source_bytes':source_raw}


def interpretation_policy() -> dict[str,Any]:
    return {
        'contrast':'V1 minus V0, each endpoint and each cohort separately',
        'directions':{'reconstruction_nmse':'lower','omitted_view_nmse':'lower','adjusted_cross_view_mrr':'higher'},
        'all_three_endpoints_reported':True,
        'aggregation':'paired same-fold/same-seed deltas; average the two seeds within fold, then five equally weighted folds; show all seed and fold values',
        'no_cross_cohort_pooling':True,
        'five_of_five':'descriptively consistent direction in this cohort, not significance or independent replication',
        'mixed_fold_signs':'show the mean, median, range and all five deltas; describe heterogeneous fold-level differences',
        'discordant_tasks':'report task-dependent trade-off; no universal winner',
        'discordant_cohorts':'report cohort-specific results; no general architectural superiority',
        'small_differences':'report exact observed magnitudes; no equivalence/noninferiority claim or new practical margin',
        'sign_count':'count raw numerical signs; no 4/5 or 5/5 success gate; display zeros separately',
        'technical_failure':'stop and preserve error; no seed/parameter rescue and no successful-only primary average',
        'collapse_flags':'retain every completed evaluation and flag; no exclusion/rescue',
        'scope':'the complete frozen V0-to-V1 architecture/objective transition; private heads, decoder inputs and alignment/orthogonality/private-separation terms differ; not equal-capacity causal decomposition',
        'no_new_p_values_or_inferential_CI':True,
        'no_new_model_selection':True,
    }


def protocol(state: dict[str,Any], source_sha: str) -> dict[str,Any]:
    profile=state['cohorts']['SARC']['plan']['neural_profile']
    cohorts={}
    for c,sp in COHORTS.items():
        cohorts[c]={'n':sp['n'],'historical_protocol':sp['base']+'/limited_model_v1/fixed_protocol.json',
            'historical_protocol_sha256':sp['protocol_sha'],
            'split_path':sp['base']+'/limited_model_v1/split_registry.json','split_sha256':sp['split_sha'],
            'completed_results_sha256':sp['results_sha'],'input_root':sp['base']+'/'+sp['input_rel'],
            'input_complete_sha256':sp['input_sha'],'input_matrices':state['cohorts'][c]['plan']['input_matrices'],
            'fold_geometry_fit_early_refit_test':sp['geometry'],
            'endpoints_copied_without_changes':state['cohorts'][c]['plan']['endpoints']}
    return {'version':VERSION,'status':'FIXED_V0_ADDENDUM_BEFORE_EXTERNAL_V0_FITS_NOT_AUTHORIZATION',
        'source_sha256':source_sha,'numerical_sources':SOURCE_PINS,
        'provenance':{'historical_TARGET_SARC_KIRC_results_seen':True,
            'planned_after_external_V1_V4_results':True,'prospective_only_with_respect_to_new_external_V0_fits':True,
            'not_a_pre_original_study_registration':True,'historical_contracts_rewritten':False},
        'new_scientific_model':V0,'historical_comparison_model':V1,
        'new_real_evaluations':20,'new_real_neural_training_stages':40,'real_V1_V4_PCA_refits':0,
        'seeds':list(SEEDS),'five_folds_one_repeat':True,'cohorts':cohorts,
        'neural_profile':profile,'variant':{'variant_id':V0,**state['architecture']['ablation_components'][V0]},
        'geometry':{'shared_dim':5,'nominal_profile_private_dim':2,'V0_actual_private_dim':0,
                    'V1_actual_private_dim':2,'hidden_dim':32,'per_view_input_PCs':10},
        'preprocessing':{'budgets':BUDGETS,'PCs':PC_RANKS,'rules':'unchanged sarc5.prepared_views; no case filtering',
            'epoch_selection':'fit rows only; early rows select epoch only',
            'refit':'refit preprocessing on all outer-training rows; same-seed reinitialize; selected epoch count',
            'future_real_digest_check':'both preliminary/refit preprocessing fingerprints must match the historical same-fold preprocessing record before V0 fitting'},
        'adapter':{'old_sarc5_evaluator_accepts_V0':False,'old_module_monkeypatched':False,
            'implementation':'new evaluate_extension_model wrapper calling unchanged 05d4 train/encode/predict and sarc5 observed_nmse/prepared_views; synthetic V1 parity control',
            'scientific_allowlist':[V0],'synthetic_parity_control_only':[V1]},
        'interpretation_precommit':interpretation_policy(),
        'diagnostics':'unchanged test-latent collapse and within-fold two-seed agreement, not q10/resampling/biological validation',
        'synthetic':{'n_by_cohort':{'SARC':250,'KIRC':304},'rows_use_existing_split_indices_not_real_values':True,
            'V0_evaluations':20,'V0_optimization_stages':40,'V1_regression_evaluations':6,
            'V1_regression_optimization_stages':12,'epochs_per_training_stage':2,
            'parity_geometries':[['SARC',1],['KIRC',1],['KIRC',5]],
            'widths':{'RNA':48,'METHYLATION':64,'CNV':40},
            'budget_fixture_widths':{'RNA':5016,'METHYLATION':10016,'CNV':5016},
            'parity_tolerance':{'absolute':ATOL,'relative':RTOL}},
        'future_real_output':BASE+'/scientific_results',
        'real_execution_requires':'separate exact-source authorizer and a separate executor after this CUDA preflight report is reviewed',
        'prohibited':['real V0 fit in this script entry point','any real V1/V4/PCA refit','changes to old results/locks',
            'new cohorts','clinical/outcome access','new retention threshold','larger epoch cap','automatic paper edit'],
    }


def load_runtime(project: Path) -> tuple[Any,dict[str,Any]]:
    path=child(project,K4_REL);raw=read_file(path);need(digest(raw)==K4_SHA,'Runtime loader source differs')
    name='p5v0_exact_kirc4_loader';spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None,'Cannot create source module')
    m=importlib.util.module_from_spec(spec);prior=sys.modules.get(name);sys.modules[name]=m
    try:exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__)
    except BaseException:
        if prior is None:sys.modules.pop(name,None)
        else:sys.modules[name]=prior
        raise
    s5,runtime=m.load_runtime(project)
    need(V0 in runtime['variants'] and s5.NEURAL_MODELS==(V1,V4),'Variant availability/old allowlist changed')
    need(not runtime['variants'][V0].shared_private_decomposition,'V0 incorrectly has private branch')
    return s5,runtime


def environment(s5:Any) -> dict[str,Any]:
    need(sys.version_info[:2]==(3,10),'Use existing Python 3.10 environment; do not reinstall blindly')
    env=s5.cuda_environment()
    need(env['pandas']=='2.3.3','pandas differs from the completed external runs')
    return env


def evaluate_extension_model(s5:Any,runtime:dict[str,Any],p:dict[str,Any],model_id:str,
                             seed:int,smoke:bool=False) -> dict[str,Any]:
    """Candidate future numerical adapter; entry point calls only smoke=True.

    Scientific callers must enforce separate authorization before entering here.
    V1 is accepted exclusively for artificial regression tests, never real refit.
    The legacy sarc5 numeric functions and model source are not modified.
    """
    need(model_id==V0 or (smoke and model_id==V1),'This addendum permits real fitting of V0 only')
    need(seed in SEEDS,'Unexpected seed')
    np=s5.np;ab=runtime['ablation'];target=runtime['target'];model=None
    start=time.perf_counter()
    profile=replace(runtime['profile'],maximum_epochs=2,patience=2) if smoke else runtime['profile']
    ntest=len(next(iter(p['test_pc'].values())));ntrain=len(next(iter(p['train_pc'].values())))
    try:
        af=np.ones((len(next(iter(p['fit_pc'].values()))),3),dtype=bool)
        ae=np.ones((len(next(iter(p['early_pc'].values()))),3),dtype=bool)
        epoch,ea=ab.train_with_early_stopping(runtime['neural'],profile,runtime['variants'][model_id],
                                            p['fit_pc'],af,p['early_pc'],ae,int(seed))
        need(type(epoch) is int and 1<=epoch<=profile.maximum_epochs,'Invalid selected epoch')
        at=np.ones((ntrain,3),dtype=bool);av=np.ones((ntest,3),dtype=bool)
        model,ra=ab.train_fixed_epochs(runtime['neural'],profile,runtime['variants'][model_id],
                                     p['train_pc'],at,int(seed),epoch)
        audits=[ea,ra]
        for audit in audits:
            for key,value in audit.items():
                if isinstance(value,(float,np.floating)):need(math.isfinite(float(value)),'Nonfinite audit: '+key)
        train_enc=ab.encode_numpy(model,p['train_pc'],at);test_enc=ab.encode_numpy(model,p['test_pc'],av)
        latent=test_enc['patient_shared']
        pred=ab.predict_reconstruction_numpy(model,p['test_pc'],av)
        recon={v:p['refit']['pcas'][v].inverse(pred[v]) for v in VIEWS}
        omitted={};invariance={}
        for v in VIEWS:
            pred,valid=ab.predict_missing_view_numpy(model,p['test_pc'],av,v)
            need(np.asarray(valid).shape==(ntest,) and np.asarray(valid).all(),'Invalid omitted-view projection')
            omitted[v]=p['refit']['pcas'][v].inverse(pred)
            if smoke:
                changed=dict(p['test_pc']);changed[v]=np.full_like(changed[v],12345.0)
                pred2,valid2=ab.predict_missing_view_numpy(model,changed,av,v)
                need(np.array_equal(pred,pred2) and np.array_equal(valid,valid2),'Omitted target affected prediction')
            invariance[v]=True
        need(np.asarray(latent).shape==(ntest,5) and np.isfinite(latent).all(),'Invalid shared latent geometry')
        rec={v:s5.observed_nmse(p['raw_train_std'][v],p['raw_test_std'][v],recon[v]) for v in VIEWS}
        miss={v:s5.observed_nmse(p['raw_train_std'][v],p['raw_test_std'][v],omitted[v]) for v in VIEWS}
        mrr,adjusted=target.cross_view_mrr(train_enc['shared_by_view'],test_enc['shared_by_view'])
        rank=s5.effective_rank(latent);var=float(np.median(np.var(latent,axis=0,ddof=1)))
        result={'model_id':model_id,'seed':int(seed),'status':'PASS',
            'scope':'ARTIFICIAL_SELFTEST' if smoke else 'V0_EXTERNAL_ADDENDUM',
            'selected_epoch':epoch,'reconstruction':rec,'omitted_view':miss,
            'reconstruction_nmse':float(np.mean([x['nmse'] for x in rec.values()])),
            'omitted_view_nmse':float(np.mean([x['nmse'] for x in miss.values()])),
            'raw_cross_view_mrr':float(mrr),'adjusted_cross_view_mrr':float(adjusted),
            'latent_effective_rank':rank,'latent_median_variance':var,
            'neural_heldout_collapse':bool(rank<2 or var<1e-4),'fit_audits':audits,
            'latent':np.asarray(latent).tolist(),'omission_invariance_checked':invariance if smoke else None,
            'elapsed_seconds':time.perf_counter()-start,
            'actual_private_dim':int(model.private_dim),'parameter_count':int(sum(x.numel() for x in model.parameters()))}
        encode(result);return result
    finally:
        if model is not None:del model
        gc.collect()
        import torch
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def close_number(a:Any,b:Any) -> bool:
    return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=ATOL,rel_tol=RTOL)


def check_result(s5:Any,r:dict[str,Any],n:int,model_id:str,seed:int) -> None:
    np=s5.np
    need(r['status']=='PASS' and r['scope']=='ARTIFICIAL_SELFTEST' and r['model_id']==model_id and r['seed']==seed,
         'Artificial result identity differs')
    need(r['actual_private_dim']==(0 if model_id==V0 else 2),'Wrong effective private dimension')
    z=np.asarray(r['latent']);need(z.shape==(n,5) and np.isfinite(z).all(),'Wrong latent shape')
    for fam,metric in (('reconstruction','reconstruction_nmse'),('omitted_view','omitted_view_nmse')):
        need(set(r[fam])==set(VIEWS),'Missing endpoint view')
        for v in VIEWS:
            d=r[fam][v];ss=d['test_row_sse'];cc=d['test_row_counts']
            need(len(ss)==len(cc)==n and all(type(c) is int and c>0 for c in cc),'Invalid observed counts')
            need(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in ss),'Invalid SSE')
            need(sum(cc)==d['n_observed_test_entries'] and close_number(d['mse'],math.fsum(ss)/sum(cc)), 'SSE/count identity differs')
            need(close_number(d['nmse'],d['mse']/max(d['train_null_mse'],1e-12)),'NMSE normalization differs')
        need(close_number(r[metric],math.fsum(r[fam][v]['nmse'] for v in VIEWS)/3),'View weighting differs')
    for v in VIEWS:
        need(r['reconstruction'][v]['test_row_counts']==r['omitted_view'][v]['test_row_counts'] and
             r['reconstruction'][v]['train_null_mse']==r['omitted_view'][v]['train_null_mse'],'Task targets differ')
    raw=r['raw_cross_view_mrr'];null=math.fsum(1/i for i in range(1,n+1))/n
    need(0<=raw<=1 and close_number(r['adjusted_cross_view_mrr'],min(1,max(0,(raw-null)/(1-null)))),'Actual-candidate MRR differs')
    e=r['selected_epoch'];a=r['fit_audits']
    need(type(e) is int and 1<=e<=2 and len(a)==2,'Artificial epoch schedule differs')
    for x in a:
        need(x['device']=='cuda:0' and x['variant_id']==model_id and x['fit_seed']==seed and
             x['profile_id']=='ARCH_COMPACT__REG_WEAK' and x['fit_status']=='PASS','Training audit identity differs')
    need(a[0]['fit_stage']=='confirmatory_early_stop' and a[0]['maximum_epochs']==2 and a[0]['patience']==2 and
         a[0]['min_delta']==.0001 and a[0]['selected_epoch']==e and e<=a[0]['epochs_run']<=2,'Early stopping mismatch')
    need(a[1]['fit_stage']=='confirmatory_refit_full_training_fold' and a[1]['selected_epoch']==e and a[1]['epochs_run']==e,'Refit epochs mismatch')
    if model_id==V0:
        for x,prefix in ((a[0],'final_monitor_'),(a[1],'final_training_')):
            for component in ('shared_alignment','contrastive','orthogonality','private_separation','cross_reconstruction','kl'):
                need(x[prefix+component]==0.0,'V0 unexpectedly used extra objective: '+component)
    need(r['omission_invariance_checked']==dict.fromkeys(VIEWS,True),'Missing omission test')
    need(close_number(r['latent_effective_rank'],s5.effective_rank(z)),'Effective-rank mismatch')
    var=float(np.median(np.var(z,axis=0,ddof=1)))
    need(close_number(r['latent_median_variance'],var) and r['neural_heldout_collapse']==bool(r['latent_effective_rank']<2 or var<1e-4),'Collapse diagnostic mismatch')


def numeric_parity(old:Any,new:Any,path:str='') -> float:
    """Compare the entire legacy payload except clocks and declared extra metadata."""
    if type(old) is dict:
        need(type(new) is dict and set(old)<=set(new),'Parity keys differ at '+path)
        return max([numeric_parity(v,new[k],path+'/'+k) for k,v in old.items()
                    if k not in ('elapsed_seconds','wall_seconds')]+[0.0])
    if type(old) is list:
        need(type(new) is list and len(old)==len(new),'Parity list differs at '+path)
        return max([numeric_parity(a,b,path+f'/{i}') for i,(a,b) in enumerate(zip(old,new))]+[0.0])
    if type(old) in (int,float):
        need(close_number(old,new),'Numerical parity failed at '+path)
        return abs(old-new)
    need(old==new,'Parity metadata differs at '+path)
    return 0.0


def synthetic_data(np:Any,cohort:str,big:bool=False) -> dict[str,Any]:
    spec=COHORTS[cohort];n=spec['n'];rng=np.random.default_rng(spec['artificial_seed']+(10 if big else 0))
    dims=(5016,10016,5016) if big else (48,64,40)
    z=rng.normal(size=(n,5));out={}
    for v,d in zip(VIEWS,dims):
        x=(rng.normal(size=(n,d)) if big else z@rng.normal(size=(5,d))+.6*rng.normal(size=(n,d))).astype(np.float32)
        x[:,0]=0;x[:,1]=np.nan
        if not big and v=='METHYLATION':
            x[rng.random((n,d))<.08]=np.nan;x[:int(n*.4),2]=np.nan
        out[v]=x
    return out


def synthetic_preflight(s5:Any,runtime:dict[str,Any],state:dict[str,Any]) -> dict[str,Any]:
    np=s5.np;results=[];parity=[];agreements=[];budget=[];invariance=[]
    initial_profile=asdict(runtime['profile']);initial_allowlist=tuple(s5.NEURAL_MODELS)
    for cohort,spec in COHORTS.items():
        splits=state['cohorts'][cohort]['split']['splits'];n=spec['n']
        original=synthetic_data(np,cohort)
        with tempfile.TemporaryDirectory(prefix='p5v0_ARTIFICIAL_ONLY_') as td:
            root=Path(td);specs={};matrices={}
            for v in VIEWS:
                path=root/(v+'.npy')
                with path.open('xb') as f:np.save(f,original[v],allow_pickle=False)
                specs[v]=(path.name,digest(path.read_bytes()),original[v].shape[1])
            matrices=s5.load_pointwise_inputs(root,specs,n)
            try:
                need(all(np.array_equal(original[v],matrices[v],equal_nan=True) for v in VIEWS),'Artificial loader mismatch')
                big=synthetic_data(np,cohort,big=True)
                budget_folds=(1,) if cohort=='SARC' else (1,5)
                for fold in budget_folds:
                    bp=s5.prepared_views(runtime,big,splits[fold-1])
                    selected={stage:{v:len(bp[stage]['transforms'][v].selected_indices) for v in VIEWS}
                              for stage in ('prelim','refit')}
                    need(all(x==BUDGETS for x in selected.values()),'Primary budgets not reached')
                    budget.append({'cohort':cohort,'fold':fold,'geometry':spec['geometry'][fold-1],
                        'selected':selected,'status':'PASS'})
                    del bp;gc.collect()
                del big;gc.collect()
                print(cohort+' artificial primary-budget preprocessing: PASS',flush=True)
                for sp in splits:
                    fold=sp['fold'];p=s5.prepared_views(runtime,matrices,sp)
                    if fold in budget_folds:
                        altered={v:a.copy() for v,a in original.items()}
                        for v in VIEWS:altered[v][sp['test'],3:]=9999.0
                        q=s5.prepared_views(runtime,altered,sp)
                        need(s5.preprocessor_digest(p['prelim'])==s5.preprocessor_digest(q['prelim']) and
                             s5.preprocessor_digest(p['refit'])==s5.preprocessor_digest(q['refit']),'Test data affected preprocessing')
                        del q
                        for v in VIEWS:altered[v][sp['early'],3:]=-7777.0
                        q=s5.prepared_views(runtime,altered,sp)
                        need(s5.preprocessor_digest(p['prelim'])==s5.preprocessor_digest(q['prelim']),'Early data affected fit preprocessing')
                        del q,altered
                        invariance.append({'cohort':cohort,'fold':fold,'heldout':'PASS','early_excluded_from_prelim':'PASS'})
                        # Same input, profile and seed; compare old and new wrappers, not real V1 models.
                        old=s5.evaluate_fixed_model(runtime,p,V1,1103,smoke=True)
                        new=evaluate_extension_model(s5,runtime,p,V1,1103,smoke=True)
                        check_result(s5,new,len(sp['test']),V1,1103)
                        delta=numeric_parity(old,new)
                        parity.append({'cohort':cohort,'fold':fold,'status':'PASS','maximum_absolute_numeric_difference':delta,
                            'new_V1_parameter_count':new['parameter_count'],'artificial_evaluations':2,'training_stages':4})
                        print(cohort+' V1 legacy/new adapter parity at fold '+str(fold)+': PASS',flush=True)
                        del old,new
                    pair=[]
                    for seed in SEEDS:
                        r=evaluate_extension_model(s5,runtime,p,V0,seed,smoke=True)
                        check_result(s5,r,len(sp['test']),V0,seed)
                        r.update(cohort=cohort,fold=fold,assessment_rows='artificial, existing index geometry',n_test=len(sp['test']))
                        results.append(r);pair.append(np.asarray(r['latent']))
                        print('%s synthetic V0 fold %d/5 seed=%d: PASS'%(cohort,fold,seed),flush=True)
                    agreements.append({'cohort':cohort,'fold':fold,**s5.subspace_agreement(*pair)})
                    del p,pair;gc.collect()
                for v,(rel,sha,width) in specs.items():need(digest((root/rel).read_bytes())==sha,'Artificial file changed')
            finally:s5.close_inputs(matrices)
    need(len(results)==20 and len(parity)==3 and len(agreements)==10,'Incomplete artificial schedule')
    need({(x['cohort'],x['fold'],x['seed']) for x in results}=={(c,f,s) for c in COHORTS for f in range(1,6) for s in SEEDS},'Missing/duplicate artificial V0 results')
    need(asdict(runtime['profile'])==initial_profile and tuple(s5.NEURAL_MODELS)==initial_allowlist==(V1,V4),
         'Profile or historical allowlist was modified')
    need(len({r['parameter_count'] for r in results})==1,'V0 parameter counts differ')
    return {'status':'V0_ARTIFICIAL_20_EVALUATIONS_AND_THREE_V1_PARITY_CHECKS_PASS',
        'V0_evaluation_records':20,'V0_optimization_stages':40,'V1_regression_evaluation_records':6,
        'V1_regression_optimization_stages':12,'total_artificial_evaluations':26,
        'total_artificial_optimization_stages':52,'maximum_epochs_per_stage':2,
        'real_cohort_fits':0,'real_molecular_values_used':False,
        'serialized_production_loader':'PASS','scientific_profile_unchanged':True,
        'old_sarc5_allowlist_unchanged':True,'budget_checks':budget,'preprocessing_invariance':invariance,
        'V1_wrapper_parity':parity,'V0_results':results,'V0_seed_only_agreement':agreements,
        'V0_parameter_count':results[0]['parameter_count'],
        'V1_parameter_count':parity[0]['new_V1_parameter_count'],
        'no_positive_scientific_result_required_for_PASS':True}


def verify_completed(out:Path,identity:dict[str,Any]) -> Path:
    lock=decode(read_file(out/'preflight_lock.json'))
    need(lock['status']==FINAL_STATUS and lock['identity']==identity and lock['real_fit_authorization_created'] is False,
         'Existing preflight has different identity')
    for name,rec in lock['files'].items():
        b=read_file(child(out,name));need(pin(b)==rec,'Completed preflight artifact differs: '+name)
    packet=child(out,lock['evidence']['path']);need(pin(read_file(packet))=={k:lock['evidence'][k] for k in ('sha256','size_bytes')},
                                            'Completed evidence differs')
    return packet


def verify_previous_failed_preflight(reader: Reader, state: dict[str, Any],
                                    snapshot: dict[str, Any], legacy_lock_handle: Any) -> dict[str, Any]:
    """Permit only the reported import-time failure, never a generic marker bypass."""
    root = child(reader.project, LEGACY_BASE)
    names = {'fixed_protocol.json', 'input_snapshot.json', 'p5v0prep_source.py',
             'SARC_split_registry.json', 'KIRC_split_registry.json',
             'SARC_historical_protocol.json', 'KIRC_historical_protocol.json',
             'started.json', 'failure.json', '.preflight.lock'}
    need(root.is_dir() and {p.name for p in root.iterdir()} == names,
         'Previous V0 directory is not exactly the reported failed preflight; preserve it for review')
    for p in root.iterdir():
        plain(p); need(p.is_file(), 'Unexpected object in previous V0 preflight')
    read_owned_process_lock(legacy_lock_handle, root / '.preflight.lock')
    previous_source = reader.get(LEGACY_BASE + '/p5v0prep_source.py', LEGACY_SOURCE_SHA)
    prior_plan = protocol(state, LEGACY_SOURCE_SHA)
    prior_plan['version'] = LEGACY_VERSION
    prior_plan['future_real_output'] = LEGACY_BASE + '/scientific_results'
    expected = {'fixed_protocol.json': encode(prior_plan),
                'input_snapshot.json': encode(snapshot), 'p5v0prep_source.py': previous_source}
    for cohort, st in state['cohorts'].items():
        expected[cohort + '_split_registry.json'] = st['split_bytes']
        expected[cohort + '_historical_protocol.json'] = st['protocol_bytes']
    identity = {k: digest(v) for k, v in expected.items()}
    for name, raw in expected.items():
        need(reader.get(LEGACY_BASE + '/' + name) == raw,
             'Previous precommitted bytes differ: ' + name)
    failure = decode(reader.get(LEGACY_BASE + '/failure.json'))
    started = decode(reader.get(LEGACY_BASE + '/started.json'))
    need(started.get('identity') == identity and started.get('real_cohort_fits') == 0 and
         started.get('stage') == 'protocol committed before artificial models',
         'Previous V0 start record differs')
    need(failure.get('status') == 'V0_PREFLIGHT_INCOMPLETE_NOT_AUTHORIZED' and
         failure.get('identity') == identity and failure.get('real_cohort_fits') == 0 and
         failure.get('error_type') == 'RuntimeError' and failure.get('error') == KNOWN_FAILURE and
         failure.get('blocked_attempts') == ['Network/subprocess not allowed: subprocess.Popen'],
         'Previous failure is not the reviewed Windows platform query failure')
    tb = failure.get('traceback', '')
    need(isinstance(tb, str) and all(x in tb for x in
         ('load_runtime', 'import torch', '_load_dll_libraries', 'platform.py',
          '_syscmd_ver', 'subprocess.py', 'subprocess.Popen')) and
         'synthetic_preflight(' not in tb,
         'Previous traceback does not match the reviewed pre-training import path')
    return {'status': 'EXACT_V1_PRECOMMIT_AND_KNOWN_WINDOWS_IMPORT_FAILURE_PRESERVED',
            'previous_directory': LEGACY_BASE, 'previous_source_sha256': LEGACY_SOURCE_SHA,
            'previous_identity': identity, 'previous_failure': failure,
            'previous_files': [r for r in reader.records.values() if r['path'].startswith(LEGACY_BASE + '/')],
            'process_lock_bytes_sha256': digest(b'0'),
            'scientific_plan_change': 'none; only source/version/output identities differ',
            'failure_marker_deleted_or_overridden': False,
            'old_models_or_sources_executed': False,
            'scope': 'Checks recorded bytes and reported failure stage, not operating-system access history'}


def verify_v2_lock_only_state(project: Path, lock_handle: Any,
                              previous_source: bytes) -> dict[str, Any]:
    """Verify the state produced by v2 stopping BEFORE protocol publication.

    The posted PermissionError was caught only by v2.main, not by its later
    failure.json writer. Therefore no fictitious failure.json is required or
    generated for v2. Any evidence of progress beyond the lone process marker
    requires review instead of a generic continuation.
    """
    need(digest(previous_source) == FAILED_V2_SOURCE_SHA, 'Previous v2 source SHA differs')
    root = child(project, FAILED_V2_BASE)
    need(root.is_dir() and {p.name for p in root.iterdir()} == {'.preflight.lock'},
         'Previous v2 is not in the reported lock-only startup state; preserve it for review')
    marker = root / '.preflight.lock'
    plain(marker)
    data = read_owned_process_lock(lock_handle, marker)
    return {'status': 'V2_LOCK_ONLY_STARTUP_STATE_VERIFIED_AND_PRESERVED',
            'directory': FAILED_V2_BASE, 'previous_source': pin(previous_source),
            'process_lock_marker': pin(data),
            'marker_read': 'same owning handle; no second open',
            'files_present': ['.preflight.lock'],
            'failure_json_expected': False,
            'reason': 'reported v2 error preceded its protocol/start/failure-writing section',
            'scope': 'current directory and supplied source verified; console failure not reconstructed',
            'historical_files_modified': False, 'model_authorization_created': False}


def windows_probe_report(firewall: PreflightFirewall) -> dict[str, Any]:
    spec = firewall.windows_spec
    return {'policy': 'at most 8 exact stdlib Windows VER queries; all other subprocesses blocked',
            'platform_source': None if spec is None else {
                'path': spec['module_path'], 'sha256': spec['module_sha256'],
                'size_bytes': spec['module_size_bytes']},
            'system_shell': None if spec is None else spec['shell'],
            'allowed_query_count': len(firewall.windows_version_queries),
            'allowed_queries': list(firewall.windows_version_queries),
            'network_exception': False, 'general_subprocess_exception': False,
            'real_data_or_historical_write_exception': False,
            'stdlib_and_packages_patched': False,
            'audit_hook_is_not_an_OS_security_sandbox': True}


def run(project:Path,script:Path) -> Path:
    project=project.absolute();script=script.absolute();plain(project);plain(script)
    need((project/'paper5/scripts').is_dir(),'Run from workspace root containing paper5/scripts')
    need(script.name=='p5v0prep3.py','Save this file as p5v0prep3.py, not .txt or (1).py')
    source=read_file(script);out=child(project,BASE)
    v2_source_path=script.with_name('p5v0prep2.py')
    v2_source=read_file(v2_source_path)
    need(digest(v2_source)==FAILED_V2_SOURCE_SHA,'Keep the unchanged p5v0prep2.py beside this file')
    legacy_lock=child(project,LEGACY_BASE+'/.preflight.lock')
    need(legacy_lock.is_file(),'The previous failed V0 evidence is required; do not delete or recreate it')
    v2_lock=child(project,FAILED_V2_BASE+'/.preflight.lock')
    need(v2_lock.is_file(),'The v2 startup marker is missing; preserve existing folders for review')
    out.mkdir(parents=True,exist_ok=True)
    with process_lock(out/'.preflight.lock') as output_handle, \
            process_lock(legacy_lock,create_if_missing=False) as legacy_handle, \
            process_lock(v2_lock,create_if_missing=False) as v2_handle, \
            PreflightFirewall(project,out) as firewall:
        print('Process locks: owner-handle checks PASS; no second-handle reads',flush=True)
        v2_review=verify_v2_lock_only_state(project,v2_handle,v2_source)
        reader=Reader(project);state=verify_historical(reader)
        plan=protocol(state,digest(source))
        snapshot={'files':list(reader.records.values()),'real_matrix_bytes_read':False,
                  'historical_completed_result_files_hashed':122,'historical_results_not_refitted':True}
        previous=verify_previous_failed_preflight(reader,state,snapshot,legacy_handle)
        assets={'fixed_protocol.json':encode(plan),'input_snapshot.json':encode(snapshot),
                'p5v0prep3_source.py':source,'v1_failure_review.json':encode(previous),
                'v2_startup_review.json':encode(v2_review)}
        for c,st in state['cohorts'].items():
            assets[c+'_split_registry.json']=st['split_bytes']
            assets[c+'_historical_protocol.json']=st['protocol_bytes']
        identity={k:digest(v) for k,v in assets.items()}
        if (out/'preflight_lock.json').exists():
            packet=verify_completed(out,identity);reader.recheck()
            need(read_file(script)==source and read_file(v2_source_path)==v2_source,'Audit source changed')
            verify_v2_lock_only_state(project,v2_handle,v2_source)
            read_owned_process_lock(legacy_handle,legacy_lock)
            read_owned_process_lock(output_handle,out/'.preflight.lock')
            print('Existing exact V0 preflight: PASS; no synthetic fits repeated',flush=True)
            print('Status: '+FINAL_STATUS,flush=True)
            print('Evidence JSON (send this FILE; no ZIP): '+str(packet),flush=True)
            return packet
        need(not (out/'scientific_results').exists() and not (out/'authorization_v1').exists(),
             'V0 authorization/scientific output already exists; do not restart preparation')
        need(not (out/'failure.json').exists(),'Earlier V0 preflight failed; preserve the report and request review')
        need(not (out/'started.json').exists(),'An interrupted preflight exists; preserve it and request review')
        for name,raw in assets.items():
            path=child(out,name)
            if path.exists():need(read_file(path)==raw,'Precommitted file differs: '+name)
            else:create(path,raw)
        create(out/'started.json',encode({'identity':identity,'started_at_utc':now(),
            'real_cohort_fits':0,'stage':'protocol committed before artificial models'}))
        print('Exact sources / old protocols / split registries / 122 historical result files: PASS',flush=True)
        print('Previous v1 evidence and v2 lock-only startup preserved; scientific plan unchanged',flush=True)
        print('V0-only addendum and interpretation policy saved BEFORE synthetic fits',flush=True)
        print('Windows runtime exception: exact stdlib VER query only; all other subprocesses/network blocked',flush=True)
        print('Scientific V0 plan: 20 evaluations / 40 training stages; no real fit authorized',flush=True)
        print('Artificial check: 20 V0 evaluations + 6 V1 parity evaluations; max 2 epochs per stage',flush=True)
        try:
            s5,runtime=load_runtime(project)
            need(asdict(runtime['profile'])==plan['neural_profile'] and asdict(runtime['variants'][V0])==plan['variant'],
                 'Loaded runtime does not match precommitted V0 protocol')
            env=environment(s5)
            print('Exact runtime loaded; allowed Windows VER queries: '+str(len(firewall.windows_version_queries)),flush=True)
            synthetic=synthetic_preflight(s5,runtime,state)
            reader.recheck()
            need(read_file(script)==source and read_file(v2_source_path)==v2_source and not firewall.blocked,
                 'Source changed or prohibited operation attempted')
            need(verify_v2_lock_only_state(project,v2_handle,v2_source)==v2_review,'v2 state changed')
            read_owned_process_lock(legacy_handle,legacy_lock)
            read_owned_process_lock(output_handle,out/'.preflight.lock')
            for name,raw in assets.items():need(read_file(child(out,name))==raw,'Protocol changed during preflight')
            report={'version':VERSION,'status':FINAL_STATUS,'identity':identity,'environment':env,
                'synthetic':synthetic,'finished_at_utc':now(),'input_source_recheck':'PASS',
                'real_cohort_fits':0,'real_molecular_arrays_opened':False,'historical_files_modified':False,
                'real_fit_authorization_created':False,'firewall_blocked_attempts':firewall.blocked,
                'windows_platform_query_audit':windows_probe_report(firewall),
                'previous_failed_v1_preserved':True,'previous_v2_lock_only_preserved':True,
                'process_lock_validation':'same owning handles; existing markers never rewritten',
                'next_stage':'review this report, then create a separate exact-source V0 authorizer and executor'}
            report_bytes=encode(report)
            files={**assets,'report.json':report_bytes}
            packet={'schema':'p5-v0-protocol-artificial-evidence-v1','status':FINAL_STATUS,'archive_required':False,
                'new_real_fits':0,'real_execution_authorized':False,
                'files':{name:{**pin(raw),'text':raw.decode('utf-8')} for name,raw in files.items()}}
            packet_bytes=encode(packet,compact=True);need(len(packet_bytes)<=MAX_EVIDENCE,'Evidence exceeds 5 MiB; do not zip the folder')
            tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
            packet_path=out/('v0p3_'+tag+'.json')
            create(out/'report.json',report_bytes);create(packet_path,packet_bytes)
            create(out/'preflight_lock.json',encode({'status':FINAL_STATUS,'identity':identity,
                'files':{name:pin(raw) for name,raw in files.items()},
                'real_fit_authorization_created':False,'real_cohort_fits':0,
                'evidence':{'path':packet_path.name,**pin(packet_bytes)}}))
        except (Exception,KeyboardInterrupt) as e:
            if not (out/'failure.json').exists():
                create(out/'failure.json',encode({'status':'V0_PREFLIGHT_INCOMPLETE_NOT_AUTHORIZED','error':str(e),
                    'error_type':type(e).__name__,'traceback':traceback.format_exc(),'real_cohort_fits':0,
                    'identity':identity,'blocked_attempts':firewall.blocked,
                    'windows_platform_query_audit':windows_probe_report(firewall)}))
            raise
        print('Original source/results and new protocol rechecked unchanged: PASS',flush=True)
        print('Status: '+FINAL_STATUS,flush=True)
        print('Evidence JSON (send this FILE; no ZIP): '+str(packet_path),flush=True)
        print('Evidence bytes: '+str(len(packet_bytes)),flush=True)
        print('Evidence SHA-256: '+digest(packet_bytes),flush=True)
        print('STOP. This script cannot start the real V0 benchmark; separate authorization is required.',flush=True)
        return packet_path


def main() -> int:
    print('Paper 5 V0 - fixed external addendum + ARTIFICIAL preflight; NO real cohort fitting',flush=True)
    print('Version: '+VERSION,flush=True)
    try:
        need(len(sys.argv)==1,'No command-line options; run from the workspace root')
        run(Path.cwd(),Path(__file__))
        return 0
    except (Exception,KeyboardInterrupt) as e:
        print('P5 V0 PREFLIGHT STOPPED: '+type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        print('Preserve all files. Do not edit hashes, overwrite a lock, or run historical models.',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
