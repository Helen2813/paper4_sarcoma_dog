#!/usr/bin/env python3
"""Paper 5 V0 addendum: authorized, resumable real SARC/KIRC execution.

Save p5v0run.py and p5v0auth.py in Downloads; run from the workspace root.
Run the separate authorizer FIRST. This entry point never grants authorization.
Only V0 is fitted. Its numerical implementation is the exact p5v0prep3 source
reviewed after the user's Windows/CUDA synthetic run. Existing V1 results are
read only for paired descriptive comparisons; no historical model is refitted.
No packages, thresholds, folds, profiles, source files or old results are edited.
All imports here are standard library until exact authorization has been checked.
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
import os
import stat
import statistics
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
VERSION = 'p5-v0-authorized-sarc-kirc-fixed-addendum-v1-no-cli'
AUTH_VERSION = 'p5-v0-exact-source-authorizer-v1-no-fit'
AUTH_STATUS = 'P5_V0_EXACT_SOURCE_AUTHORIZED'
COMPLETE_STATUS = 'P5_V0_EXTERNAL_ADDENDUM_COMPLETE_FROZEN'
PREFLIGHT_STATUS = 'P5_V0_PROTOCOL_AND_SYNTHETIC_PREFLIGHT_PASS_NOT_AUTHORIZED'
BASE = 'paper5/v0_external_v3'
PREP_REL = BASE + '/p5v0prep3_source.py'
AUTH_REL = BASE + '/authorization_v1'
RESULT_REL = BASE + '/scientific_results'
V0 = 'V0_BALANCED_AE'
V1 = 'V1_SHARED_PRIVATE_AE'
VIEWS = ('RNA', 'METHYLATION', 'CNV')
COHORTS = ('SARC','KIRC')
SEEDS = (1103,2207)
METRICS = ('reconstruction_nmse','omitted_view_nmse','raw_cross_view_mrr','adjusted_cross_view_mrr')
PRIMARY_METRICS = ('reconstruction_nmse','omitted_view_nmse','adjusted_cross_view_mrr')
MAX_SMALL = 5 * 1024 * 1024
MAX_PACKET = 5 * 1024 * 1024
ATOL, RTOL = 2e-12, 2e-10

PREP_SHA = 'f47c19a6e99423d4dd1ab91e19e26cb55216d34c7a3c52f50521aaa31b7bf8c2'

PREFLIGHT_PINS = {'KIRC_historical_protocol.json': {'sha256': '6ac98e5fe49065ba450a54a744b33fbc45197a382dd4a522ec784e494fc232e1',
                                   'size_bytes': 11973},
 'KIRC_split_registry.json': {'sha256': 'f7b73461fe420d3ded1f5450aef3402e0c48ac07c1ef570aa0eddff1e3c7661b',
                              'size_bytes': 41383},
 'SARC_historical_protocol.json': {'sha256': 'f3a63188ecbaff4a5164d999d933a5f6481ff7d0560b999022faa38f7a0cfff8',
                                   'size_bytes': 8066},
 'SARC_split_registry.json': {'sha256': '78a702ac43d55dd37f8b194b7b7933fb547f09e04cf6333662b57221160377f0',
                              'size_bytes': 33985},
 'fixed_protocol.json': {'sha256': '9d8f2eadc23165e2dbd2c744974b033d9b72a1b37187774cb730ed65d3d54709',
                         'size_bytes': 12924},
 'input_snapshot.json': {'sha256': '55d68de4041ff324acfd65539a2b4c65239782e04af1bb68afc5aa2f2bfe00a5',
                         'size_bytes': 32941},
 'p5v0prep3_source.py': {'sha256': 'f47c19a6e99423d4dd1ab91e19e26cb55216d34c7a3c52f50521aaa31b7bf8c2',
                         'size_bytes': 62656},
 'report.json': {'sha256': '753a4a694590cef75883a2fbd24c1be34ba8e6b94c86364826b8a715cbad8934',
                 'size_bytes': 653224},
 'v1_failure_review.json': {'sha256': 'cd59eac50b7226f6263dc13406045639cb5f1ae6b2ca3921894a0bfbdd38cf0f',
                            'size_bytes': 7079},
 'v2_startup_review.json': {'sha256': '665cb493b95c5287fe1e3d233fb8b749ee9b90649621616c00c0fbcfdce79c83',
                            'size_bytes': 772}}

PRELOCK_PIN = {'sha256': '55dbd2298ce7475c6278b702248ae5c6db1b5a402388327c1e90dae02684f32b', 'size_bytes': 2737}

EXPECTED_ENV = {'AMP': False,
 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
 'GPU': 'NVIDIA GeForce RTX 4080 Laptop GPU',
 'TF32': False,
 'cpu_threads': 2,
 'cuda': '13.0',
 'numpy': '1.26.4',
 'pandas': '2.3.3',
 'torch': '2.12.1+cu130',
 'torch_compile': False}

SOURCE_PINS = {'paper5/method_contract/05d1/neural_architecture_regularization_profile_grid.csv': '118b2fd4c16751f0b83ac3053f726f9421db26949ab98a8a531ff584c194ee32',
 'paper5/method_contract/05d1/neural_loss_and_architecture_contract.json': 'b90ded1d011ee3210154c2102389df0b4e0c8fcf668b70d7e2b8e14184e3f6b1',
 'paper5/scripts/05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py': '05ae6923ce588c137bc6f4aa934262181dd99f4ebe6bec3250e4b7cd54ed1ff3',
 'paper5/scripts/05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py': '8ab3b84f1d7bcd475b88b5f06d49e661c2d6230ea3598406e66970d792c74a5c',
 'paper5/scripts/06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py': '4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab',
 'paper5_parallel_kirc/limited_model_v1/kirc4_source.py': '0d03eebd11e67325b319360e7ee4b9ee030bcdd2741b8d7b08e15204cdf33e41',
 'paper5_parallel_sarc/sarc5.py': 'dd16e547972d8d298491e2d3c5003dea16760e969e9ea846fda5b0f83020f96f'}


def need(ok: Any, message: str) -> None:
    if not bool(ok):
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(obj: Any, compact: bool=False) -> bytes:
    return (json.dumps(obj,sort_keys=True,ensure_ascii=True,allow_nan=False,
                       indent=None if compact else 2,
                       separators=(',',':') if compact else None)+'\n').encode('utf-8')


def decode(raw: bytes) -> Any:
    def pairs(items):
        d={}
        for k,v in items:
            need(k not in d,'Duplicate JSON key: '+str(k));d[k]=v
        return d
    def bad(x):raise ValueError('Nonfinite JSON token: '+x)
    return json.loads(raw.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=bad)


def plain(path: Path) -> None:
    for p in (path,*path.parents):
        if p.exists() or p.is_symlink():
            s=p.lstat()
            need(not stat.S_ISLNK(s.st_mode) and not(getattr(s,'st_file_attributes',0)&0x400),
                 'Symlink/reparse path forbidden: '+str(p))


def child(root: Path, rel: str) -> Path:
    need(type(rel) is str and bool(rel),'Empty/non-string path')
    pp=PurePosixPath(rel)
    need(not pp.is_absolute() and '\\' not in rel and ':' not in rel and
         all(x not in ('','.','..') for x in rel.split('/')),'Unsafe relative path: '+rel)
    p=root.joinpath(*pp.parts);plain(p)
    need(p.absolute().is_relative_to(root.absolute()),'Path escapes root')
    return p


def read_small(path: Path) -> bytes:
    plain(path);need(path.is_file() and path.stat().st_size<=MAX_SMALL,'Small file missing/oversized: '+str(path))
    with path.open('rb') as f:b=f.read(MAX_SMALL+1)
    need(len(b)<=MAX_SMALL,'File grew past limit: '+str(path))
    return b


def sha_file(path: Path) -> str:
    plain(path);need(path.is_file(),'File missing: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(raw: bytes) -> dict[str,Any]:
    return {'sha256':digest(raw),'size_bytes':len(raw)}


def checked(path: Path, expected: dict[str,Any]) -> bytes:
    b=read_small(path)
    need(pin(b)=={'sha256':expected['sha256'],'size_bytes':expected['size_bytes']},'Size/SHA mismatch: '+str(path))
    return b


def create(path: Path, raw: bytes) -> None:
    plain(path);need(not path.exists(),'Existing file preserved: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.parent/('.writing_'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if os.name=='nt':os.rename(tmp,path)
    else:os.link(tmp,path);tmp.unlink()


def keep_or_create(path: Path, raw: bytes) -> None:
    if path.exists():need(read_small(path)==raw,'Existing derived file differs: '+str(path))
    else:create(path,raw)


@contextlib.contextmanager
def execution_lock(path: Path):
    """Reuse the existing preflight marker, without modifying it.

    Only its owning handle reads the locked byte. There is no second open,
    marker reinitialization, force-unlock, unlink or privilege escalation.
    """
    plain(path);need(path.is_file(),'Existing preflight process-lock marker missing')
    f=path.open('r+b',buffering=0);acquired=False
    try:
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
        except OSError as e:
            raise RuntimeError('Cannot acquire V0 lock; another process may be running. Do not delete it.') from e
        need(os.fstat(f.fileno()).st_size==1 and f.read(2)==b'0','Unexpected process-lock contents')
        f.seek(0)
        yield
    finally:
        try:
            if acquired:
                f.seek(0)
                if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        finally:f.close()


def close_number(a: Any,b: Any) -> bool:
    return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=ATOL,rel_tol=RTOL)


def jobs() -> list[tuple[str,int,int]]:
    return [(c,f,s) for c in COHORTS for f in range(1,6) for s in SEEDS]


def job_name(j: tuple[str,int,int]) -> str:
    need(j in jobs(),'Unexpected scientific job')
    return '%s_F%02d_S%d'%j


def split_check(reg: dict[str,Any], c: str) -> None:
    n=250 if c=='SARC' else 304
    ids=reg['case_ids'];need(len(ids)==len(set(ids))==n and ids==sorted(ids),'Patient registry mismatch')
    need(len(reg['splits'])==5,'Five folds required')
    for f,s in enumerate(reg['splits'],1):
        geom=[160,40,200,50] if c=='SARC' else ([194,49,243,61] if f<5 else [195,49,244,60])
        sets={k:set(s[k]) for k in ('fit','early','train','test')}
        need(s['fold']==f and [len(s[k]) for k in sets]==geom,'Fold geometry mismatch')
        need(all(s[k]==sorted(sets[k]) and all(type(i) is int and 0<=i<n for i in s[k]) for k in sets),'Duplicate/noninteger/out-of-range split index')
        need(not sets['fit']&sets['early'] and sets['fit']|sets['early']==sets['train'] and
             not sets['train']&sets['test'] and sets['train']|sets['test']==set(range(n)),'Split overlap')
    need(sorted(i for s in reg['splits'] for i in s['test'])==list(range(n)),'Test partition mismatch')


def validate_evaluation(r: dict[str,Any],n: int,seed: int,*,smoke: bool=False,model: str=V0) -> None:
    need(r['status']=='PASS' and r['model_id']==model and type(r['seed']) is int and r['seed']==seed,'Result identity mismatch')
    need(seed in SEEDS and model in (V0,V1),'Unrecognized seed/model')
    if model==V0:
        need(r['scope']==('ARTIFICIAL_SELFTEST' if smoke else 'V0_EXTERNAL_ADDENDUM'),'Unexpected V0 scope')
        need(r['actual_private_dim']==0 and r['parameter_count']==3501,'V0 geometry/parameter count changed')
    z=r['latent'];need(len(z)==n and all(len(row)==5 and all(type(x) in (int,float) and math.isfinite(x) for x in row) for row in z),'Invalid test latent')
    for family,metric in (('reconstruction','reconstruction_nmse'),('omitted_view','omitted_view_nmse')):
        need(set(r[family])==set(VIEWS),'Missing modality')
        for v in VIEWS:
            d=r[family][v];ss=d['test_row_sse'];cc=d['test_row_counts']
            need(len(ss)==len(cc)==n and all(type(x) is int and x>0 for x in cc),'Invalid observed counts')
            need(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in ss),'Invalid squared errors')
            null=d['train_null_mse'];need(type(null) in (int,float) and math.isfinite(null) and null>=0,'Invalid normalizer')
            need(sum(cc)==d['n_observed_test_entries'] and close_number(d['mse'],math.fsum(ss)/sum(cc)) and
                 close_number(d['nmse'],d['mse']/max(null,1e-12)),'NMSE replay mismatch')
        need(close_number(r[metric],math.fsum(r[family][v]['nmse'] for v in VIEWS)/3),'Equal-view aggregation mismatch')
    for v in VIEWS:
        need(r['reconstruction'][v]['test_row_counts']==r['omitted_view'][v]['test_row_counts'] and
             r['reconstruction'][v]['train_null_mse']==r['omitted_view'][v]['train_null_mse'],'Task target mask/null differs')
    raw=r['raw_cross_view_mrr'];null=math.fsum(1/i for i in range(1,n+1))/n
    need(type(raw) in (int,float) and 0<=raw<=1 and close_number(r['adjusted_cross_view_mrr'],max(0,min(1,(raw-null)/(1-null)))),'MRR adjustment mismatch')
    for k in METRICS+('latent_effective_rank','latent_median_variance'):
        need(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>=0,'Invalid metric: '+k)
    need(type(r['neural_heldout_collapse']) is bool and r['neural_heldout_collapse']==bool(r['latent_effective_rank']<2 or r['latent_median_variance']<1e-4),'Collapse flag mismatch')
    e=r['selected_epoch'];a=r['fit_audits'];cap=2 if smoke else 400
    need(type(e) is int and 1<=e<=cap and len(a)==2,'Epoch schedule mismatch')
    for x in a:
        need(x['device']=='cuda:0' and x['fit_status']=='PASS' and x['fit_seed']==seed and
             x['variant_id']==model and x['profile_id']=='ARCH_COMPACT__REG_WEAK','Training identity mismatch')
        need(all(not isinstance(v,float) or math.isfinite(v) for v in x.values()),'Nonfinite fit audit')
    need(a[0]['fit_stage']=='confirmatory_early_stop' and a[0]['maximum_epochs']==cap and
         a[0]['patience']==(2 if smoke else 30) and a[0]['min_delta']==.0001 and
         a[0]['selected_epoch']==e and e<=a[0]['epochs_run']<=cap,'Early stopping mismatch')
    need(a[1]['fit_stage']=='confirmatory_refit_full_training_fold' and a[1]['selected_epoch']==e and a[1]['epochs_run']==e,'Refit epoch mismatch')
    if model==V0:
        for x,prefix in ((a[0],'final_monitor_'),(a[1],'final_training_')):
            for k in ('shared_alignment','contrastive','orthogonality','private_separation','cross_reconstruction','kl'):
                need(x[prefix+k]==0.0,'Non-V0 objective used: '+k)
        need(r['omission_invariance_checked']==(dict.fromkeys(VIEWS,True) if smoke else None),'Synthetic/scientific omission metadata mismatch')
    encode(r)


def verify_preflight_report(r: dict[str,Any]) -> None:
    need(r['status']==PREFLIGHT_STATUS and r['input_source_recheck']=='PASS' and r['real_cohort_fits']==0,'Preflight incomplete')
    for k in ('real_molecular_arrays_opened','historical_files_modified','real_fit_authorization_created'):
        need(r[k] is False,'Preflight boundary violated: '+k)
    need(r['firewall_blocked_attempts']==[],'Forbidden operation attempted during preflight')
    syn=r['synthetic'];need(syn['status']=='V0_ARTIFICIAL_20_EVALUATIONS_AND_THREE_V1_PARITY_CHECKS_PASS','Synthetic status differs')
    for k,n in {'V0_evaluation_records':20,'V0_optimization_stages':40,'V1_regression_evaluation_records':6,
                'V1_regression_optimization_stages':12,'total_artificial_evaluations':26,'total_artificial_optimization_stages':52,
                'maximum_epochs_per_stage':2,'V0_parameter_count':3501,'V1_parameter_count':3891,'real_cohort_fits':0}.items():
        need(syn[k]==n,'Synthetic schedule mismatch: '+k)
    need(syn['old_sarc5_allowlist_unchanged'] and syn['scientific_profile_unchanged'] and not syn['real_molecular_values_used'] and syn['serialized_production_loader']=='PASS','Synthetic scope mismatch')
    rows=syn['V0_results'];need(len(rows)==20 and {(x['cohort'],x['fold'],x['seed']) for x in rows}==set(jobs()),'Synthetic jobs missing/duplicated')
    for x in rows:
        n=50 if x['cohort']=='SARC' else (61 if x['fold']<5 else 60)
        validate_evaluation(x,n,x['seed'],smoke=True)
    par=syn['V1_wrapper_parity'];need(len(par)==3 and {(x['cohort'],x['fold']) for x in par}=={('SARC',1),('KIRC',1),('KIRC',5)},'V1 parity geometry mismatch')
    need(all(x['status']=='PASS' and x['maximum_absolute_numeric_difference']==0.0 and x['training_stages']==4 and x['artificial_evaluations']==2 for x in par),'V1 parity failed')
    for x in syn['budget_checks']:
        need(x['status']=='PASS' and all(y=={'RNA':5000,'METHYLATION':10000,'CNV':5000} for y in x['selected'].values()),'Budget check failed')
    need(len(syn['budget_checks'])==3 and len(syn['preprocessing_invariance'])==3 and
         all(x['heldout']=='PASS' and x['early_excluded_from_prelim']=='PASS' for x in syn['preprocessing_invariance']),'Preprocessing invariance missing')
    for k,v in EXPECTED_ENV.items():need(r['environment'][k]==v,'Reviewed environment mismatch: '+k)


def verify_inputs(project: Path) -> dict[str,Any]:
    """No numerical imports, array reads or project-code execution."""
    base=child(project,BASE);raws={}
    for name,rec in PREFLIGHT_PINS.items():raws[name]=checked(child(base,name),rec)
    lock=decode(checked(base/'preflight_lock.json',PRELOCK_PIN));r=decode(raws['report.json'])
    need(lock['status']==PREFLIGHT_STATUS and lock['files']==PREFLIGHT_PINS and lock['identity']==r['identity'],'Preflight lock bindings differ')
    need(lock['real_fit_authorization_created'] is False and lock['real_cohort_fits']==0,'Unexpected preflight authorization')
    need(r['identity']=={k:v['sha256'] for k,v in PREFLIGHT_PINS.items() if k!='report.json'},'Preflight identity mismatch')
    checked(child(base,lock['evidence']['path']),lock['evidence'])
    verify_preflight_report(r)
    plan=decode(raws['fixed_protocol.json']);snap=decode(raws['input_snapshot.json']);entries=snap['files']
    need(len(entries)==138 and len({x['path'] for x in entries})==138,'Historical snapshot changed')
    historical={}
    for x in entries:
        need(Path(x['path']).suffix in ('.json','.py','.csv','.tsv'),'Unexpected historical file class')
        historical[x['path']]=checked(child(project,x['path']),x)
    need(plan['new_scientific_model']==V0 and plan['historical_comparison_model']==V1 and plan['seeds']==list(SEEDS) and
         plan['new_real_evaluations']==20 and plan['new_real_neural_training_stages']==40 and plan['real_V1_V4_PCA_refits']==0 and
         plan['future_real_output']==RESULT_REL and plan['numerical_sources']==SOURCE_PINS,'V0 plan mismatch')
    prof=plan['neural_profile'];need(prof['maximum_epochs']==400 and prof['patience']==30 and prof['min_delta']==.0001,'Scientific profile changed')
    cohorts={};matrices={}
    for c in COHORTS:
        cp=plan['cohorts'][c];reg=decode(raws[c+'_split_registry.json']);split_check(reg,c)
        oldplan=decode(raws[c+'_historical_protocol.json'])
        need(historical[cp['historical_protocol']]==raws[c+'_historical_protocol.json'] and
             historical[cp['split_path']]==raws[c+'_split_registry.json'],'Copied historical plan/split mismatch')
        oldroot=str(PurePosixPath(cp['historical_protocol']).parent/'scientific_results')
        oldlock=decode(historical[oldroot+'/completion_lock.json'])
        need(oldlock['n_evaluations']==25 and oldlock['n_failed_evaluations']==0 and oldlock['n_patients']==cp['n'],'Incomplete historical benchmark')
        need(len(oldlock['files'])==61 and len({x['path'] for x in oldlock['files']})==61,'Historical manifest not 61 distinct files')
        for rec in oldlock['files']:
            b=historical[oldroot+'/'+rec['path']];need(pin(b)=={'sha256':rec['sha256'],'size_bytes':rec['size_bytes']},'Historical lock/snapshot disagreement')
        complete=decode(historical[cp['input_root']+'/input_complete.json'])
        infiles={x['path']:x for x in complete['files']} if type(complete['files']) is list else complete['files']
        specs=cp['input_matrices'];need(set(specs)==set(VIEWS) and specs==oldplan['input_matrices'],'Input specifications differ')
        for v,(rel,sha,width) in specs.items():
            expected_name = {'RNA':'RNA_log2tpm.npy','METHYLATION':'METHYLATION_Mvalue.npy','CNV':'CNV_log2ratio.npy'}[v] if c=='SARC' else v+'.npy'
            need(rel=='matrices/'+expected_name and sha==infiles[rel]['sha256'] and width=={'RNA':60660,'METHYLATION':482421,'CNV':60623}[v],'Matrix identity differs')
            matrices[cp['input_root']+'/'+rel]={'sha256':sha,'size_bytes':infiles[rel]['size_bytes'],'cohort':c,'view':v,'n':cp['n'],'width':width}
        refs={};pres={}
        for f in range(1,6):
            pres[f]=decode(historical[oldroot+'/preprocessing/fold_%02d.json'%f])
            for s in SEEDS:
                rel=oldroot+'/shards/F%02d__%s__S%d/result.json'%(f,V1,s)
                ref=decode(historical[rel]);sp=reg['splits'][f-1]
                need(ref['test_row_indices']==sp['test'] and ref['test_case_ids']==[reg['case_ids'][i] for i in sp['test']],'Historical V1 test alignment differs')
                validate_evaluation(ref['evaluation'],len(sp['test']),s,model=V1)
                refs[(f,s)]={'data':ref,'path':rel,**pin(historical[rel])}
        cohorts[c]={'plan':cp,'registry':reg,'historical_root':oldroot,'reference':refs,'preprocessing':pres}
    return {'plan':plan,'report':r,'lock':lock,'snapshot':entries,'raws':raws,'historical':historical,'cohorts':cohorts,'matrices':matrices}


def import_exact(path: Path,sha: str,name: str) -> Any:
    b=read_small(path);need(digest(b)==sha,'Module source mismatch: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);need(spec is not None,'Cannot build source module')
    module=importlib.util.module_from_spec(spec);prior=sys.modules.get(name);sys.modules[name]=module
    try:exec(compile(b,str(path),'exec',dont_inherit=True),module.__dict__)
    except BaseException:
        if prior is None:sys.modules.pop(name,None)
        else:sys.modules[name]=prior
        raise
    return module


def authorize(project: Path, authorizer: Path, executor: Path, expected_executor_sha: str) -> Path:
    need(executor.name=='p5v0run.py' and authorizer.name=='p5v0auth.py','Keep exact Downloads filenames')
    rb=read_small(executor);ab=read_small(authorizer)
    need(digest(rb)==expected_executor_sha,'Downloaded executor hash mismatch')
    with execution_lock(child(project,BASE+'/.preflight.lock')):
        st=verify_inputs(project);out=child(project,AUTH_REL);results=child(project,RESULT_REL)
        data={'schema':'p5-v0-exact-source-authorization-v1','status':AUTH_STATUS,'version':VERSION,'authorizer_version':AUTH_VERSION,
              'executor':{'path':'p5v0run_source.py',**pin(rb)},'authorizer':{'path':'p5v0auth_source.py',**pin(ab)},
              'preflight_lock':PRELOCK_PIN,'preflight_files':PREFLIGHT_PINS,'historical_inputs':st['snapshot'],
              'expected_environment':EXPECTED_ENV,'numerical_adapter':{'path':PREP_REL,'sha256':PREP_SHA},
              'scientific_results_path':RESULT_REL,'only_new_model':V0,'new_evaluations':20,'new_training_stages':40,
              'old_model_refits':0,'scientific_benchmark_started':False,'real_molecular_values_opened':False,'historical_files_modified':False}
        wanted={'p5v0run_source.py':rb,'p5v0auth_source.py':ab,'authorization.json':encode(data)}
        if out.exists():
            need({x.name for x in out.iterdir()}==set(wanted),'Partial/unexpected authorization directory; preserve it')
            for k,b in wanted.items():need(read_small(out/k)==b,'Existing authorization differs: '+k)
        else:
            need(not results.exists(),'Scientific result directory exists before authorization')
            pending=child(project,BASE)/('.authorizing_'+uuid.uuid4().hex)
            pending.mkdir(exist_ok=False)
            for k,b in wanted.items():create(pending/k,b)
            verify_inputs(project)
            need(read_small(executor)==rb and read_small(authorizer)==ab,'Downloads source changed during authorizing')
            os.rename(pending,out)
        print('Exact V0 preflight / fixed protocol / historical inputs: PASS',flush=True)
        print('P5 V0 source authorization: FROZEN',flush=True)
        print('Authorization: '+str(out/'authorization.json'),flush=True)
        print('Authorization SHA-256: '+digest(encode(data)),flush=True)
        print('Scientific benchmark started by this command: NO',flush=True)
        return out/'authorization.json'


def verify_auth(project: Path, executor: Path, st: dict[str,Any]) -> dict[str,Any]:
    out=child(project,AUTH_REL);a=decode(read_small(out/'authorization.json'))
    need(a['schema']=='p5-v0-exact-source-authorization-v1' and a['status']==AUTH_STATUS and a['version']==VERSION and a['authorizer_version']==AUTH_VERSION,'Exact V0 source is not authorized')
    need(a['executor']['path']=='p5v0run_source.py' and a['authorizer']['path']=='p5v0auth_source.py','Authorization source path differs')
    rb=checked(out/'p5v0run_source.py',a['executor']);checked(out/'p5v0auth_source.py',a['authorizer'])
    need(read_small(executor)==rb,'Executor is not the authorized source')
    need(a['preflight_lock']==PRELOCK_PIN and a['preflight_files']==PREFLIGHT_PINS and a['historical_inputs']==st['snapshot'] and
         a['expected_environment']==EXPECTED_ENV and a['numerical_adapter']=={'path':PREP_REL,'sha256':PREP_SHA},'Authorization input bindings changed')
    need(a['scientific_results_path']==RESULT_REL and a['only_new_model']==V0 and a['new_evaluations']==20 and a['new_training_stages']==40 and a['old_model_refits']==0,'Authorization scope changed')
    need(a['scientific_benchmark_started'] is False and a['real_molecular_values_opened'] is False and a['historical_files_modified'] is False,'Invalid source-authorization boundary')
    return a


def bindings(project: Path,a: dict[str,Any]) -> dict[str,Any]:
    return {'executor_sha256':a['executor']['sha256'],'executor_version':VERSION,
            'authorization_sha256':sha_file(child(project,AUTH_REL+'/authorization.json')),
            'preflight_lock_sha256':PRELOCK_PIN['sha256'],'adapter_sha256':PREP_SHA,
            'protocol_sha256':PREFLIGHT_PINS['fixed_protocol.json']['sha256']}


def runtime_firewall(prep: Any, project: Path, output: Path, matrix_paths: dict[str,Any]) -> Any:
    """Exact six-file read exception, installed only AFTER source authorization.

    Inherits the reviewed Windows VER handling. Does not change the preflight
    module or its source. Python audit hooks are accidental-access checks, not
    an OS sandbox. No second-handle access to process locks occurs here.
    """
    permitted={os.path.normcase(os.path.abspath(child(project,p))) for p in matrix_paths}
    class ScientificFirewall(prep.PreflightFirewall):
        def __init__(self):
            super().__init__(project,output);self.matrix_read_paths=set()
        def hook(self,event,args):
            if self.active and event=='open' and args and not isinstance(args[0],int):
                try:p=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
                except (ValueError,TypeError):return super().hook(event,args)
                if p in permitted:
                    mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
                    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                    if writing:self.stop('Authorized matrices are read-only: '+p)
                    self.matrix_read_paths.add(p);return
            return super().hook(event,args)
    return ScientificFirewall()


def check_preprocessing(s5: Any,p: dict[str,Any],old: dict[str,Any],sp: dict[str,Any]) -> dict[str,Any]:
    need(old['fold']==sp['fold'] and old['test_used_for_preprocessing'] is False,'Historical preprocessing scope changed')
    for k in ('fit','early','train','test'):need(old[k+'_rows']==sp[k],'Historical preprocessing rows differ')
    np=s5.np;actual={}
    def arr(a):
        x=np.ascontiguousarray(a)
        return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':digest(x.tobytes())}
    for stage in ('prelim','refit'):
        b=p[stage];st={'combined_digest':s5.preprocessor_digest(b),'views':{}}
        for v in VIEWS:
            t=b['transforms'][v];pc=b['pcas'][v]
            st['views'][v]={'selected_count':len(t.selected_indices),'selected_indices':arr(t.selected_indices),
                            'medians':arr(t.medians),'means':arr(t.means),'scales':arr(t.scales),
                            'PCA_mean':arr(pc.mean),'PCA_components':arr(pc.components)}
        need(st==old['stages'][stage],'Preprocessing does not match the frozen same-fold reference: '+stage)
        actual[stage]=st
    return actual


def reference_targets_match(v0: dict[str,Any],v1: dict[str,Any]) -> None:
    for family in ('reconstruction','omitted_view'):
        for v in VIEWS:
            a=v0[family][v];b=v1[family][v]
            need(a['test_row_counts']==b['test_row_counts'] and a['n_observed_test_entries']==b['n_observed_test_entries'] and
                 close_number(a['train_null_mse'],b['train_null_mse']),'Historical V1 target-count/normalizer mismatch: '+family+'/'+v)


def inventory(root: Path) -> None:
    if not root.exists():return
    allowed={'execution_start.json','completion_lock.json','technical_stop.json','preprocessing','shards','summaries','sessions'}
    need(all(p.name in allowed for p in root.iterdir()),'Unexpected scientific output; preserve partial files')
    for p in root.rglob('*'):
        plain(p);need(not p.name.startswith(('.writing_','.pending_')),'Uncommitted artifact found; preserve it for review')
    if (root/'shards').exists():
        names={job_name(j) for j in jobs()}
        for p in (root/'shards').iterdir():
            need(p.is_dir() and p.name in names and {q.name for q in p.iterdir()}=={'result.json','receipt.json'},'Unexpected/incomplete shard: '+str(p))
    if (root/'preprocessing').exists():
        names={'%s_F%02d.json'%(c,f) for c in COHORTS for f in range(1,6)}
        need(all(p.is_file() and p.name in names for p in (root/'preprocessing').iterdir()),'Unexpected preprocessing output')


def read_shard(root: Path,j: tuple[str,int,int],binds: dict[str,Any],st: dict[str,Any]) -> dict[str,Any]:
    c,f,s=j;path=root/'shards'/job_name(j)
    receipt=decode(read_small(path/'receipt.json'));need(receipt['result']['path']=='result.json' and receipt['bindings']==binds,'Shard receipt binding mismatch')
    d=decode(checked(path/'result.json',receipt['result']));sp=st['cohorts'][c]['registry']['splits'][f-1]
    need(d['bindings']==binds and [d['cohort'],d['fold'],d['seed']]==list(j),'Shard identity mismatch')
    need(d['test_row_indices']==sp['test'] and d['test_case_ids']==[st['cohorts'][c]['registry']['case_ids'][i] for i in sp['test']],'Shard test alignment mismatch')
    need(d['preprocessing']['path']=='preprocessing/%s_F%02d.json'%(c,f),'Unexpected shard preprocessing path')
    checked(child(root,d['preprocessing']['path']),d['preprocessing'])
    ref=st['cohorts'][c]['reference'][(f,s)]
    need(d['historical_V1_reference']=={k:ref[k] for k in ('path','sha256','size_bytes')},'Reference result binding mismatch')
    validate_evaluation(d['evaluation'],len(sp['test']),s)
    reference_targets_match(d['evaluation'],ref['data']['evaluation'])
    return d


def save_shard(root: Path,j: tuple[str,int,int],data: dict[str,Any]) -> None:
    parent=root/'shards';parent.mkdir(exist_ok=True);final=parent/job_name(j)
    need(not final.exists(),'Completed shard may not be replaced')
    temp=parent/('.pending_'+uuid.uuid4().hex);temp.mkdir(exist_ok=False)
    raw=encode(data);create(temp/'result.json',raw)
    create(temp/'receipt.json',encode({'bindings':data['bindings'],'result':{'path':'result.json',**pin(raw)}}))
    need(not final.exists(),'Concurrent shard collision')
    os.rename(temp,final)


def csv_bytes(rows: list[dict[str,Any]]) -> bytes:
    need(bool(rows),'Refusing to create an empty summary')
    h=io.StringIO(newline='');w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return h.getvalue().encode()


def summarize(root: Path, st: dict[str,Any], binds: dict[str,Any], s5: Any) -> dict[str,bytes]:
    records={j:read_shard(root,j,binds,st) for j in jobs()}
    seedrows=[];foldrows=[];summary=[];agreement=[];epochrows=[];viewrows=[]
    for c in COHORTS:
        for f in range(1,6):
            for m in (V0,V1):
                rr=[records[(c,f,s)]['evaluation'] if m==V0 else st['cohorts'][c]['reference'][(f,s)]['data']['evaluation'] for s in SEEDS]
                ag=s5.subspace_agreement(s5.np.asarray(rr[0]['latent']),s5.np.asarray(rr[1]['latent']))
                agreement.append({'cohort':c,'fold':f,'model_id':m,'seed_pair':'1103;2207',**ag})
            for s in SEEDS:
                a=records[(c,f,s)]['evaluation'];b=st['cohorts'][c]['reference'][(f,s)]['data']['evaluation']
                for k in METRICS:
                    seedrows.append({'cohort':c,'fold':f,'seed':s,'metric':k,'V0':a[k],'V1_historical':b[k],'delta_V1_minus_V0':b[k]-a[k]})
                for fam in ('reconstruction','omitted_view'):
                    for v in VIEWS:
                        viewrows.append({'cohort':c,'fold':f,'seed':s,'task':fam,'view':v,
                                         'V0_nmse':a[fam][v]['nmse'],'V1_historical_nmse':b[fam][v]['nmse'],
                                         'delta_V1_minus_V0':b[fam][v]['nmse']-a[fam][v]['nmse']})
                epochrows.append({'cohort':c,'fold':f,'seed':s,'model_id':V0,'selected_epoch':a['selected_epoch'],
                                  'epoch_selection_epochs_run':a['fit_audits'][0]['epochs_run'],
                                  'reached_400_cap':a['fit_audits'][0]['epochs_run']==400,
                                  'collapse_flag':a['neural_heldout_collapse'],'parameter_count':a['parameter_count']})
            for k in METRICS:
                pair=[x for x in seedrows if x['cohort']==c and x['fold']==f and x['metric']==k]
                foldrows.append({'cohort':c,'fold':f,'metric':k,'V0':statistics.mean(x['V0'] for x in pair),
                                 'V1_historical':statistics.mean(x['V1_historical'] for x in pair),
                                 'delta_V1_minus_V0':statistics.mean(x['delta_V1_minus_V0'] for x in pair)})
        for k in METRICS:
            a=[x for x in foldrows if x['cohort']==c and x['metric']==k];d=[x['delta_V1_minus_V0'] for x in a]
            lower=k in ('reconstruction_nmse','omitted_view_nmse')
            summary.append({'cohort':c,'metric':k,'role':'primary_endpoint' if k in PRIMARY_METRICS else 'raw_retrieval_context',
                'V0_mean':statistics.mean(x['V0'] for x in a),'V1_historical_mean':statistics.mean(x['V1_historical'] for x in a),
                'mean_delta_V1_minus_V0':statistics.mean(d),'median_delta':statistics.median(d),'fold_SD_descriptive':statistics.stdev(d),
                'min_delta':min(d),'max_delta':max(d),'folds_favor_V1':sum(x<0 if lower else x>0 for x in d),
                'folds_favor_V0':sum(x>0 if lower else x<0 for x in d),'exact_zero_folds':sum(x==0 for x in d),'n_folds':5})
    info={'status':COMPLETE_STATUS,'new_V0_evaluations':20,'old_model_refits':0,'primary_contrast':'V1 minus V0',
          'aggregation':'two seeds within each fold, then five equally weighted folds; cohorts not pooled',
          'interpretation_precommit':st['plan']['interpretation_precommit'],'results':summary,
          'seed_only_agreement':agreement,'epochs':epochrows,'no_new_p_values_or_CI':True}
    text=['# V0 external addendum: fixed-procedure Results','',
          'This addendum was specified after the original external V1/V4 results, before the new external V0 fits.',
          'Only V0 was trained. V1 results were reused without modification. This compares the entire fixed architecture/objective transition, not an equal-capacity causal effect.',
          'Folds and seeds are not independent external replications. Differences and fold SDs are descriptive; no inference or new selection is performed.','',
          '| Cohort | Metric | V0 | V1 (historical) | V1 minus V0 | Folds favor V1 |','|---|---|---:|---:|---:|---:|']
    for x in summary:
        if x['role']=='primary_endpoint':text.append('| {cohort} | {metric} | {V0_mean:.6f} | {V1_historical_mean:.6f} | {mean_delta_V1_minus_V0:+.6f} | {folds_favor_V1}/5 |'.format(**x))
    text+=['','NMSE: lower is better; adjusted MRR: higher is better. All endpoints/folds are retained, irrespective of direction or collapse flags.',
           'Two-seed subspace agreement is not resampling stability or retention-gate validation. The 400-epoch cap does not establish convergence. No historic claim or manuscript file was automatically changed.','']
    return {'paired_by_seed.csv':csv_bytes(seedrows),'paired_by_fold.csv':csv_bytes(foldrows),
            'V1_minus_V0_summary.csv':csv_bytes(summary),'per_view_by_seed.csv':csv_bytes(viewrows),
            'seed_only_agreement.csv':csv_bytes(agreement),'V0_epoch_audit.csv':csv_bytes(epochrows),
            'summary.json':encode(info),'MANUSCRIPT_RESULTS_EN.md':'\n'.join(text).encode()}


def verify_complete(root: Path, binds: dict[str,Any]) -> dict[str,Any]:
    data=decode(read_small(root/'completion_lock.json'))
    need(data['status']==COMPLETE_STATUS and data['bindings']==binds and data['new_V0_evaluations']==20 and data['old_model_refits']==0,'Invalid V0 completion')
    manifest=data['files'];need(len(manifest)==len({x['path'] for x in manifest}),'Duplicate manifest entries')
    expected={'execution_start.json'}|{'preprocessing/%s_F%02d.json'%(c,f) for c in COHORTS for f in range(1,6)}
    expected|={'shards/'+job_name(j)+'/'+x for j in jobs() for x in ('result.json','receipt.json')}
    expected|={'summaries/'+x for x in ('paired_by_seed.csv','paired_by_fold.csv','V1_minus_V0_summary.csv','per_view_by_seed.csv','seed_only_agreement.csv','V0_epoch_audit.csv','summary.json','MANUSCRIPT_RESULTS_EN.md')}
    need({x['path'] for x in manifest}==expected,'Incomplete completion manifest')
    for rec in manifest:checked(child(root,rec['path']),rec)
    packet=child(root.parent,data['evidence']['path']);checked(packet,data['evidence'])
    return data


def finalize(project: Path,root: Path,st: dict[str,Any],binds: dict[str,Any],s5: Any,environment: dict[str,Any],firewall: Any) -> Path:
    assets=summarize(root,st,binds,s5)
    for k,b in assets.items():keep_or_create(root/'summaries'/k,b)
    manifest=[];files={}
    paths=[root/'execution_start.json']
    paths+=sorted((root/'preprocessing').glob('*.json'))
    paths+=[root/'shards'/job_name(j)/k for j in jobs() for k in ('result.json','receipt.json')]
    paths+=sorted((root/'summaries').iterdir())
    for p in paths:
        rel=p.relative_to(root).as_posix();b=read_small(p);manifest.append({'path':rel,**pin(b)});files['scientific_results/'+rel]={'text':b.decode(),**pin(b)}
    for k,b in st['raws'].items():files['preflight/'+k]={'text':b.decode(),**pin(b)}
    for k in ('authorization.json','p5v0run_source.py','p5v0auth_source.py'):
        b=read_small(child(project,AUTH_REL+'/'+k));files['authorization/'+k]={'text':b.decode(),**pin(b)}
    for c in COHORTS:
        for (f,s),ref in st['cohorts'][c]['reference'].items():
            b=st['historical'][ref['path']];files['historical_references/%s_F%02d_S%d.json'%(c,f,s)]={'text':b.decode(),**pin(b),'original_path':ref['path']}
    completion={'status':COMPLETE_STATUS,'bindings':binds,'new_V0_evaluations':20,'new_V0_training_stages':40,
                'old_model_refits':0,'historical_files_modified':False,'sources_inputs_rechecked':True,
                'files':manifest,'environment':environment,
                'runtime_access_audit':{'matrix_paths':sorted(firewall.matrix_read_paths),'blocked_attempts':list(firewall.blocked),
                    'allowed_windows_ver_queries':list(firewall.windows_version_queries),'not_OS_security_sandbox':True}}
    packet={'schema':'p5-v0-real-results-evidence-v1','status':COMPLETE_STATUS,'archive_required':False,
            'molecular_matrices_included':False,'completion':completion,'files':files}
    b=encode(packet,compact=True);need(len(b)<=MAX_PACKET,'Evidence exceeds 5 MiB; preserve outputs for review')
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output=root.parent/('v0r_'+tag+'.json');create(output,b)
    completion['evidence']={'path':output.name,**pin(b)}
    create(root/'completion_lock.json',encode(completion));verify_complete(root,binds)
    return output


def run_science(project: Path,executor: Path,st: dict[str,Any],a: dict[str,Any]) -> Path:
    root=child(project,RESULT_REL);binds=bindings(project,a);inventory(root)
    need(not (root/'technical_stop.json').exists(),'Prior technical_stop exists; preserve it and request review')
    if (root/'completion_lock.json').exists():
        complete=verify_complete(root,binds)
        print('Existing exact V0 completion: PASS; no model fits repeated',flush=True)
        return root.parent/complete['evidence']['path']
    root.mkdir(exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    session=root/'sessions'/tag;session.mkdir(parents=True,exist_ok=False)
    log={'status':'AUTHORIZED_START','started_at_utc':now(),'bindings':binds,'current_job':None,'new_jobs':[],
         'verified_existing_jobs':[],'real_matrix_values_opened':False}
    create(session/'start.json',encode(log));prep=None;s5=None;matrices={};firewall=None
    try:
        # The exact adapter is imported only after authorization; its main/run
        # synthetic entry points are never invoked.
        prep=import_exact(child(project,PREP_REL),PREP_SHA,'p5v0run_exact_prep3')
        with runtime_firewall(prep,project,root,st['matrices']) as firewall:
            log['current_job']={'stage':'load_exact_runtime'}
            s5,runtime=prep.load_runtime(project)
            environment=prep.environment(s5)
            for k,v in EXPECTED_ENV.items():need(environment[k]==v,'Current runtime differs from reviewed CUDA run: '+k)
            need(prep.asdict(runtime['profile'])==st['plan']['neural_profile'] and
                 prep.asdict(runtime['variants'][V0])==st['plan']['variant'],'Loaded numerical profile/variant differs')
            need(s5.NEURAL_MODELS==(V1,'V4_SABLE_MO'),'Historical helper allowlist was changed')
            snapshot={'bindings':binds,'protocol_sha256':PREFLIGHT_PINS['fixed_protocol.json']['sha256'],
                      'new_model':V0,'schedule':[list(j) for j in jobs()],'old_model_refits':0}
            keep_or_create(root/'execution_start.json',encode(snapshot))
            print('Exact runtime: PASS; only V0 may be fitted',flush=True)
            for c in COHORTS:
                cs=st['cohorts'][c];spec=cs['plan'];registry=cs['registry']
                pending=[]
                for j in (x for x in jobs() if x[0]==c):
                    if (root/'shards'/job_name(j)).exists():
                        read_shard(root,j,binds,st);log['verified_existing_jobs'].append(job_name(j))
                        print(job_name(j)+': VERIFIED_EXISTING',flush=True)
                    else:pending.append(j)
                if not pending:continue
                log['current_job']={'cohort':c,'stage':'hash_bound_matrix_load'}
                print(c+': loading the exact read-only matrices ...',flush=True)
                for rel,p in st['matrices'].items():
                    if p['cohort']==c:
                        path=child(project,rel);need(path.is_file() and path.stat().st_size==p['size_bytes'],'Matrix size mismatch: '+rel)
                log['real_matrix_values_opened']=True
                matrices=s5.load_pointwise_inputs(child(project,spec['input_root']),spec['input_matrices'],spec['n'])
                try:
                    for sp in registry['splits']:
                        f=sp['fold'];needed=[j for j in pending if j[1]==f]
                        if not needed:continue
                        log['current_job']={'cohort':c,'fold':f,'stage':'preprocessing_and_historical_digest_match'}
                        print('%s fold %d/5: train-only preprocessing and historical fingerprint check ...'%(c,f),flush=True)
                        p=s5.prepared_views(runtime,matrices,sp)
                        stages=check_preprocessing(s5,p,cs['preprocessing'][f],sp)
                        oldrel=cs['historical_root']+'/preprocessing/fold_%02d.json'%f
                        audit={'cohort':c,'fold':f,'bindings':binds,'historical_preprocessing':{'path':oldrel,**pin(st['historical'][oldrel])},
                               'both_preprocessing_stages_match':True,'stages':stages,'test_used_for_preprocessing':False}
                        rel='preprocessing/%s_F%02d.json'%(c,f);keep_or_create(root/rel,encode(audit))
                        print('%s fold %d/5: exact historical preprocessing PASS'%(c,f),flush=True)
                        for j in needed:
                            seed=j[2];log['current_job']={'cohort':c,'fold':f,'seed':seed,'stage':'V0_fit'}
                            print(job_name(j)+': START full frozen budget (V0 only)',flush=True)
                            result=prep.evaluate_extension_model(s5,runtime,p,V0,seed,smoke=False)
                            validate_evaluation(result,len(sp['test']),seed)
                            ref=cs['reference'][(f,seed)];reference_targets_match(result,ref['data']['evaluation'])
                            need(not firewall.blocked and sha_file(executor)==binds['executor_sha256'] and
                                 sha_file(child(project,PREP_REL))==PREP_SHA,'Forbidden operation/source change during fit')
                            data={'cohort':c,'fold':f,'seed':seed,'bindings':binds,
                                  'preprocessing':{'path':rel,**pin(read_small(root/rel))},
                                  'historical_V1_reference':{k:ref[k] for k in ('path','sha256','size_bytes')},
                                  'test_row_indices':sp['test'],'test_case_ids':[registry['case_ids'][i] for i in sp['test']],
                                  'evaluation':result}
                            save_shard(root,j,data);read_shard(root,j,binds,st);log['new_jobs'].append(job_name(j))
                            print(job_name(j)+': PASS selected_epoch=%d | completed=%d/20'%(result['selected_epoch'],len(log['new_jobs'])+len(log['verified_existing_jobs'])),flush=True)
                        del p;gc.collect()
                finally:
                    s5.close_inputs(matrices);matrices={};gc.collect()
            log['current_job']={'stage':'final_input_and_source_recheck'}
            after=verify_inputs(project);need(after['snapshot']==st['snapshot'],'Historical inputs changed')
            aa=verify_auth(project,executor,after);need(bindings(project,aa)==binds,'Authorization changed')
            for rel,pin_data in st['matrices'].items():
                path=child(project,rel)
                need(path.stat().st_size==pin_data['size_bytes'] and sha_file(path)==pin_data['sha256'],'Real input changed: '+rel)
            need(not firewall.blocked,'A prohibited access attempt occurred')
            # Evidence is written beside scientific_results. Exit the runtime
            # firewall first; all model work is over and all source hashes checked.
        output=finalize(project,root,st,binds,s5,environment,firewall)
        log['status']=COMPLETE_STATUS;log['finished_at_utc']=now();create(session/'end.json',encode(log))
        print('All 20 V0 evaluations complete; historical inputs/results rechecked unchanged: PASS',flush=True)
        return output
    except KeyboardInterrupt:
        log.update(status='USER_INTERRUPTED_NO_AUTOMATIC_RETRY',finished_at_utc=now(),traceback=traceback.format_exc())
        create(session/'interrupted.json',encode(log))
        print('Interrupted. Committed shards are preserved. The unchanged runner may verify/resume them; incomplete publications require review.',file=sys.stderr)
        raise
    except Exception as e:
        log.update(status='V0_TECHNICAL_STOP',finished_at_utc=now(),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        if firewall is not None:log['firewall_blocked_attempts']=list(firewall.blocked)
        if not(root/'technical_stop.json').exists():create(root/'technical_stop.json',encode(log))
        create(session/'failure.json',encode(log));raise
    finally:
        if s5 is not None and matrices:s5.close_inputs(matrices)


def main() -> int:
    print('Paper 5 V0 - exact-source authorized SARC/KIRC addendum',flush=True)
    print('Version: '+VERSION,flush=True)
    try:
        need(len(sys.argv)==1,'No command-line switches')
        project=Path.cwd().absolute();executor=Path(__file__).absolute()
        need(executor.name=='p5v0run.py','Save as p5v0run.py in Downloads')
        need((project/'paper5/scripts').is_dir(),'Run from workspace root containing paper5/scripts')
        # Missing authorization is rejected before project imports, GPU or arrays.
        need(child(project,AUTH_REL+'/authorization.json').is_file(),'Run the separate p5v0auth.py first; no fitting was started')
        with execution_lock(child(project,BASE+'/.preflight.lock')):
            st=verify_inputs(project);auth=verify_auth(project,executor,st)
            print('Exact V0 source/protocol/synthetic report/authorization: PASS',flush=True)
            print('Execution mode: AUTHORIZED_V0_ONLY_SARC_KIRC_ADDENDUM',flush=True)
            print('New evaluations: 20 | neural stages: 40 | real V1/V4/PCA refits: 0',flush=True)
            output=run_science(project,executor,st,auth)
        print('Status: '+COMPLETE_STATUS,flush=True)
        print('Evidence JSON (send this FILE; no ZIP): '+str(output),flush=True)
        print('Evidence bytes: '+str(output.stat().st_size),flush=True)
        print('Evidence SHA-256: '+sha_file(output),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as e:
        print('P5 V0 EXECUTION STOPPED: '+type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        print('Preserve all files; do not edit hashes, delete technical_stop or rerun historical models.',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
