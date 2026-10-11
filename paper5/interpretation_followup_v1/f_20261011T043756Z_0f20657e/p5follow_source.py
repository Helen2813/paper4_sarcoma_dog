#!/usr/bin/env python3
"""Paper 5: descriptive utility-weight sensitivity and exact 06e4 result replay.

Save as p5follow.py in Windows Downloads; execute from the workspace root.
Never imports project Python modules or trains/refits a model. Reads the exact
reviewed p5refine evidence, its live TARGET scalar inputs, and the completed
06e4/06e5 analysis. The only NPY files read are the three pinned 78-by-5 V1
representations, not molecular matrices. Program scores are the existing frozen
four-program scores. No new program, null panel, p/q value, gate or selection
lock is created. Weight changes are hypothetical sensitivity, not reselection.
All writes use a new folder and a bounded JSON, never replace old results.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import stat
import sys
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import numpy as np

sys.dont_write_bytecode = True
VERSION = 'p5-follow-weight-sensitivity-completed-06e4-replay-v1-no-fit'
INPUT_REL = 'paper5/interpretation_audit_v1/p5r_20261011T040217Z_10bb7868.json'
INPUT_SHA = 'b4dc49fe2e71f26f0585b9f9bb033dcc191a849adba5cb0a5903391d187585aa'
INPUT_BYTES = 2293141
INPUT_VERSION = 'p5-refine-postresult-target-components-selection-ties-external-seed-audit-v1-no-fit'
TARGET = 'paper5/results/target_representation/06b_primary_benchmark'
SOURCE = 'paper5/scripts/06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py'
SOURCE_SHA = '4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab'
MODEL_IDS = ('BLOCK_PCA','MOFA_K5','MOFA_K10','MOFA_K15','MOFA_K20','V0_BALANCED_AE',
             'V1_SHARED_PRIVATE_AE','V2_CONTRASTIVE_SPAE','V3_DROPOUT_SPAE','V4_SABLE_MO')
V1, V4 = 6, 9
METRICS = ('heldout_reconstruction_score','random_adjusted_cross_view_mrr',
           'missing_view_reconstruction_score','modality_balance_score')
SHORT = ('R','Q','M','B')
W0 = np.array([.30,.30,.25,.15],dtype=np.float64)
TIE_ATOL = 1e-12
ATOL = 2e-12
RTOL = 2e-10
MAX_FILE = 5*1024*1024
MAX_READ_TOTAL = 24*1024*1024
MAX_REPORT = 5*1024*1024
BIO_DIR = 'paper5/results/target_representation/06e4_seed_subspace_sensitivity_v1'
BIO_RUNNER = 'paper5/scripts/06e4_run_v1_seed_subspace_biology_sensitivity_v1.py'
BIO_RUNNER_SHA = '97a8af048e38c16000f27a078ea976af1c894a21850ac2924a05e3b8c16cee86'
BIO_AUTHOR = 'paper5/scripts/06e4a_freeze_v1_seed_subspace_runner_source_authorization_v1.py'
BIO_AUTHOR_SHA = 'b73871417541b60eb3fae95e9e15ac905d6331f949260031a85ae685dfa91f63'
BIO_CONTRACT = 'paper5/method_contract/06e3/v1_seed_subspace_biology_sensitivity_contract.json'
BIO_CONTRACT_SHA = 'cbac52ec634e566ccee070ec0116577388a23d4ceefd881fbe761b55a39e40a6'
BIO_CONTRACT_LOCK = 'paper5/method_contract/06e3/06e3_v1_seed_subspace_biology_sensitivity_contract_lock.json'
BIO_PREFLIGHT = 'paper5/method_contract/06e4_preflight/v1_seed_subspace_runner_synthetic_selftest_v1.json'
BIO_AUTH = 'paper5/method_contract/06e4a/v1_seed_subspace_sensitivity_runner_source_authorization_lock.json'
BIO_LOCK = 'paper5/method_contract/06e5/v1_seed_subspace_biology_sensitivity_results_lock.json'
BIO_STATUS = '06E5_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_FROZEN'
BIO_SUMMARY_STATUS = '06E4_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_COMPLETE'
PROGRAMS = ('M34','M40','M11','M24')
SEEDS = (1103,2207,3301)
SEED_PAIRS = ((1103,2207),(1103,3301),(2207,3301))
BIO_FILES = (
 'v1_program_coordinate_correlations_by_seed.csv',
 'v1_program_alignment_summary_by_seed.csv',
 'v1_program_raw_seed_sensitivity_summary.csv',
 'v1_pairwise_seed_subspace_summary.csv',
 'v1_pairwise_seed_canonical_correlations.csv',
 'v1_pairwise_seed_hungarian_matches.csv',
 'v1_program_procrustes_aligned_correlations.csv',
 'v1_program_procrustes_aligned_summary.csv',
 'v1_seed_subspace_biology_sensitivity_summary.json')
FULL_DIR = 'paper5/results/target_representation/06c_full_data_v1'
CASE_REL = FULL_DIR+'/target_v1_full_data_case_order.csv'
CASE_SHA = 'f259778ecfe8a19e83d366615734d0318b14738267af2162ff45f0d240b26f96'
LATENT_SHA = {1103:'c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f',
              2207:'08c10fb0c5c27b19955b735e02abee513002439a892ed4d642178afc2cbf2989',
              3301:'f0c4b28797f1ed4dd4fc06cbdfff1dfc39139bf060cba34a2c18edd139ce4c06'}
PRIMARY_DIR = 'paper5/results/target_representation/06e1_biological_alignment_v3'
PRIMARY_PINS = {
 'v1_frozen_program_scores_78.csv':'7129ad9719d0f01ed12b3cf854c1a4dff2e7295094986e09f08c7344134b00dc',
 'v1_program_coordinate_correlations.csv':'58cadf6ce02603d8d9a05cf06068c9f23e43a0a2dafe8835729407374d89c7f8',
 'v1_program_alignment_summary.csv':'a35e24eda531f40d1685f24feea6ae0eecd60f6bc9978feb1c73f5ca0a12b30f',
 'v1_biological_alignment_summary.json':'e4e68f5f21b2a54db4d0524fdaba265c42972c0fb6654036d9116f5ed3ca8370'}

class FollowError(RuntimeError):
    pass

def need(ok: Any, message: str) -> None:
    if not bool(ok): raise FollowError(message)

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def encode(obj: Any, compact: bool=False) -> bytes:
    kw=dict(sort_keys=True,ensure_ascii=True,allow_nan=False)
    kw.update(separators=(',',':')) if compact else kw.update(indent=2)
    return (json.dumps(obj,**kw)+'\n').encode('utf-8')

def decode(raw: bytes) -> Any:
    def pairs(items):
        obj={}
        for k,v in items:
            need(k not in obj,'Duplicate JSON key: '+k);obj[k]=v
        return obj
    def bad(x):raise FollowError('Nonfinite JSON token: '+x)
    return json.loads(raw.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=bad)

def plain(p: Path) -> None:
    for q in (p,*p.parents):
        if q.exists() or q.is_symlink():
            st=q.lstat()
            need(not stat.S_ISLNK(st.st_mode) and not (getattr(st,'st_file_attributes',0)&0x400),
                 'Symlink/reparse path forbidden: '+str(q))

def safe(root: Path, rel: str) -> Path:
    need(isinstance(rel,str) and bool(rel),'Empty relative path')
    need(not PurePosixPath(rel).is_absolute() and '\\' not in rel and ':' not in rel and
         all(x not in ('','.','..') for x in rel.split('/')),'Unsafe path: '+rel)
    p=root.joinpath(*rel.split('/'));plain(p);return p

class Reader:
    def __init__(self,root: Path):self.root=root;self.raw={};self.records={};self.total=0
    def get(self,rel: str,pin: str|None=None,size: int|None=None,role: str='first_received_current_chain') -> bytes:
        p=safe(self.root,rel);need(p.is_file(),'Required file missing: '+str(p))
        need(p.stat().st_size<=MAX_FILE,'Oversized input: '+rel)
        with p.open('rb') as f:b=f.read(MAX_FILE+1)
        need(len(b)<=MAX_FILE,'Input grew beyond limit: '+rel);h=sha(b)
        if pin is not None:need(h==pin,'SHA mismatch: '+rel+'; observed='+h)
        if size is not None:need(len(b)==size,'Size mismatch: '+rel)
        if rel in self.records:need(h==self.records[rel]['sha256'],'Input changed during read: '+rel)
        else:
            self.total+=len(b);need(self.total<=MAX_READ_TOTAL,'Bounded read budget exceeded')
            self.records[rel]={'path':rel,'sha256':h,'size_bytes':len(b),'scope':role};self.raw[rel]=b
        return b
    def json(self,rel: str,pin: str|None=None) -> Any:return decode(self.get(rel,pin))
    def recheck(self) -> None:
        for rel,r in list(self.records.items()):self.get(rel,r['sha256'],r['size_bytes'])

def publish(p: Path,raw: bytes) -> None:
    plain(p);need(not p.exists(),'Existing output preserved: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def csv_read(raw: bytes) -> list[dict[str,str]]:
    h=io.StringIO(raw.decode('utf-8-sig'),newline='');r=csv.DictReader(h)
    need(r.fieldnames is not None and len(r.fieldnames)==len(set(r.fieldnames)),'Missing/duplicate CSV header')
    rows=list(r);need(all(None not in x and all(v is not None for v in x.values()) for x in rows),'Ragged CSV rows')
    return rows

def csv_write(rows: list[dict[str,Any]]) -> bytes:
    need(bool(rows),'Empty table');h=io.StringIO(newline='');w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows);return h.getvalue().encode('utf-8')

def num(x: Any) -> float:
    need(not isinstance(x,bool),'Boolean in numeric field');v=float(x);need(math.isfinite(v),'Nonfinite number');return v

def integer(x: Any) -> int:
    v=num(x);need(v==int(v),'Nonintegral index');return int(v)

def near(a: Any,b: Any,atol: float=ATOL) -> bool:
    return math.isclose(num(a),num(b),rel_tol=RTOL,abs_tol=atol)

def subset_equal(row: dict, keys: tuple[str,...], expected: dict) -> None:
    for k in keys:need(row[k]==expected[k],'Identity differs: '+k)

def sid(r: int,f: int) -> str:return f'R{r:02d}_F{f:02d}'

def verify_packet(raw: bytes,expected_sha: str|None=None) -> tuple[dict,dict[str,bytes]]:
    if expected_sha is None:expected_sha=INPUT_SHA
    need(sha(raw)==expected_sha,'Reviewed p5refine evidence SHA mismatch')
    p=decode(raw);need(p['status']=='P5_READONLY_REFINEMENTS_COMPLETE_NO_NEW_FIT' and p['new_fits']==0,'p5refine is not complete')
    c=p['completion'];need(c['version']==INPUT_VERSION and c['historical_files_changed'] is False,'Wrong refinement provenance')
    need(set(p['files'])==set(c['files']) and len(p['files'])==35,'Incomplete refinement packet')
    fs={}
    for n,f in p['files'].items():
        need('/' not in n and '\\' not in n and n not in ('.','..'),'Invalid packet filename')
        b=f['text'].encode('utf-8');pin={k:f[k] for k in ('sha256','size_bytes')}
        need(sha(b)==pin['sha256'] and len(b)==pin['size_bytes'] and c['files'][n]==pin,'Embedded artifact mismatch: '+n);fs[n]=b
    return p,fs

def metric_array(rows: list[dict],stage: str) -> tuple[np.ndarray,np.ndarray]:
    inner=stage=='inner_model_selection';seeds=(1103,2207) if inner else SEEDS
    ni=4 if inner else 1
    groups={};keyset=set()
    for r in rows:
        m=r['model_id'];need(m in MODEL_IDS,'Unknown model');j=MODEL_IDS.index(m)
        rep,fold,iff,s=(integer(r[k]) for k in ('repeat_index','outer_fold','inner_fold','seed'))
        need(1<=rep<=5 and 1<=fold<=5 and r['split_id']==sid(rep,fold),'Wrong split')
        need(r['stage']==stage and (1<=iff<=4 if inner else iff==0),'Wrong stage/inner fold')
        need(s in ((0,) if j==0 else seeds),'Wrong seed schedule')
        k=((rep-1)*5+fold-1,j,(iff-1) if inner else 0,s);need(k not in keyset,'Duplicate metric row');keyset.add(k)
        v=np.array([num(r[n]) for n in METRICS]);need(np.all((v>=-ATOL)&(v<=1+ATOL)),'Historical score outside [0,1]')
        need(near(float(v@W0),r['utility']),'Frozen utility arithmetic mismatch')
        groups[k]=v
    expected={(i,j,f,s) for i in range(25) for j in range(10) for f in range(ni) for s in ((0,) if j==0 else seeds)}
    need(keyset==expected,'Incomplete metric schedule')
    out=np.empty((25,10,ni,4))
    for i,j,f in itertools.product(range(25),range(10),range(ni)):
        ss=(0,) if j==0 else seeds;out[i,j,f]=np.mean([groups[(i,j,f,s)] for s in ss],axis=0)
    return out, np.array([len(rows)],dtype=int)

def one_se(fold_components: np.ndarray,w: np.ndarray) -> tuple[int,int,np.ndarray,np.ndarray,float,np.ndarray]:
    # Shape (10 models, 4 inner folds, 4 metrics). The source heuristic SE is
    # reproduced; it is not an inferential standard error for independent trials.
    vals=fold_components@w;means=np.mean(vals,axis=1);ses=np.std(vals,axis=1,ddof=1)/math.sqrt(vals.shape[1])
    best=float(np.max(means));ties=np.abs(means-best)<=TIE_ATOL
    best_id=int(np.flatnonzero(ties)[0]);cut=best-float(ses[best_id]);within=means>=cut-TIE_ATOL
    return int(np.flatnonzero(within)[0]),best_id,means,ses,cut,ties

def verify_historical_selection(tensor: np.ndarray,rows: list[dict]) -> list[dict]:
    need(len(rows)==25,'Expected 25 selections');seen=set();out=[]
    for r in rows:
        i=(integer(r['repeat_index'])-1)*5+integer(r['outer_fold'])-1
        need(0<=i<25 and i not in seen,'Duplicate/invalid selected split');seen.add(i)
        chosen,best,mu,se,cut,ties=one_se(tensor[i],W0)
        need(MODEL_IDS[chosen]==r['selected_model_id'] and MODEL_IDS[best]==r['best_model_id'],'Historical one-SE choice mismatch')
        need(near(mu.max(),r['best_mean']) and near(se[best],r['best_se']) and near(cut,r['one_se_cutoff']),'Historical one-SE summary mismatch')
        within=[MODEL_IDS[j] for j in range(10) if mu[j]>=cut-TIE_ATOL]
        need(within==r['within_one_se'].split(';'),'Historical one-SE eligibility differs')
        out.append({'split_id':r['split_id'],'historical_selected':MODEL_IDS[chosen],'historical_best_label':MODEL_IDS[best],
                    'numeric_maxima':';'.join(MODEL_IDS[j] for j in np.flatnonzero(ties)),'status':'PASS'})
    return sorted(out,key=lambda r:r['split_id'])

def read_target(reader: Reader) -> tuple[dict[str,bytes],np.ndarray,np.ndarray,list[dict],dict]:
    raw=reader.get(INPUT_REL,INPUT_SHA,INPUT_BYTES,role='reviewed_refinement_snapshot')
    p,fs=verify_packet(raw);a=decode(fs['audit.json'])
    need(a['inner']['status']=='ALL_1900_CURRENT_SHARDS_SELECTION_REPLAY_PASS','All inner rows required')
    pins={r['path']:r for r in a['inputs']};need(len(pins)==len(a['inputs']),'Duplicate input fingerprint')
    for name in ('target_outer_fixed_model_benchmark.csv','target_inner_one_se_selection_by_outer_split.csv','target_outcome_blind_benchmark_summary.json'):
        rel=TARGET+'/'+name;rec=pins[rel];b=reader.get(rel,rec['sha256'],rec['size_bytes'],role='historical_pinned_summary')
        need(b==fs['source_'+name],'Current TARGET file differs from snapshot')
    for rel in (SOURCE,'paper5/method_contract/06b/target_outcome_blind_benchmark_results_lock.json',
                'paper5/method_contract/05d1/neural_loss_and_architecture_contract.json'):
        rec=pins[rel];reader.get(rel,rec['sha256'],rec['size_bytes'],role='reviewed_source_or_lock')
    need(sha(reader.raw[SOURCE])==SOURCE_SHA,'Historical source mismatch')
    inn=decode(fs['target_inner_normalized_metrics.json']);current=[]
    for row in inn:
        rel=TARGET+'/shards/inner__'+row['split_id']+f"__I{row['inner_fold']:02d}__{row['model_id']}__S{row['seed']}.json"
        rec=pins[rel];b=reader.get(rel,rec['sha256'],rec['size_bytes'],role='unchanged_since_p5refine_NOT_original_06b_individual_lock')
        rr=decode(b)
        for flag in ('failed','biology_access','outcome_access','q10_retention_applied'):need(rr[flag] is False,'Inner scientific boundary differs')
        need(rr['script_sha256']==SOURCE_SHA,'Inner source binding differs')
        for k,v in row.items():
            if k in METRICS or k=='utility':need(near(rr[k],v),'Normalized inner component differs')
            elif k=='selected_epoch' and v is None:need(rr.get(k) is None,'Unexpected epoch')
            else:need(rr[k]==v,'Normalized inner identity differs: '+k)
        current.append(rr)
    need(len(current)==1900,'Expected 1900 inner records')
    it,_=metric_array(current,'inner_model_selection')
    ot,_=metric_array(csv_read(fs['source_target_outer_fixed_model_benchmark.csv']),'outer_evaluation')
    hs=csv_read(fs['source_target_inner_one_se_selection_by_outer_split.csv']);rep=verify_historical_selection(it,hs)
    return fs,it,ot[:,:,0,:],rep,{'input_packet_sha256':sha(raw),'embedded_files':35,'inner_rows':1900,'outer_rows':700,
       'historical_selection_replayed':25,'inner_provenance':'Current inner bytes matched p5refine snapshot; no original per-shard 06b lock is invented.'}

def plan_document() -> dict:
    return {'version':VERSION,'analysis_role':'POST_RESULT_DESCRIPTIVE_SENSITIVITY_NO_RESELECTION',
      'source_components_already_seen':True,'weight_grid_outputs_not_used_to_modify_domain':True,
      'metric_order':list(METRICS),'historical_weights':W0.tolist(),
      'global_domain':{'definition':'all nonnegative weights summing to one','step':.05,'grid_points':1771},
      'local_domain':{'definition':'each weight within +/-0.05 of historical value; sum one','step':.01,
                      'label':'arithmetic neighborhood, NOT asserted scientifically plausible prior'},
      'selection_sensitivity':'all ten frozen candidates; recalculate fold utility, mean, SE, best-label ties and one-SE at every weight',
      'paired_boundary':'minimum global-simplex L1 distance to equality of V4/V1 mean; not distance to one-SE selection boundary',
      'no_true_weight_probability':'grid counts depend on grid measure and are not probabilities of scientific superiority',
      'outer_policy':'three seeds averaged within split; 25 split means; NOT used for hypothetical selection',
      'inner_policy':'two seeds averaged within inner fold; four fold values reproduce historical one-SE heuristic',
      'bio_policy':'replay completed 06e4/06e5 statistics on exact three full-data V1 latents and four frozen program scores',
      'bioscope':'same 78 training cases, three initializations, not independent cohorts or held-out validation',
      'primary_seed':1103,'no_new_p_or_q':True,'no_rotation_invariant_program_biology_endpoint':True,
      'historical_sources_results_and_decisions_modified':False,'models_fitted':0}

def grids() -> list[tuple[str,np.ndarray]]:
    global_w=np.array([(a,b,c,20-a-b-c) for a in range(21) for b in range(21-a) for c in range(21-a-b)],float)/20
    local_w=np.array([(30+a,30+b,25+c,15-a-b-c) for a,b,c in itertools.product(range(-5,6),repeat=3) if -5<=-a-b-c<=5],float)/100
    need(len(global_w)==1771 and np.allclose(global_w.sum(axis=1),1),'Global simplex construction failed')
    need(np.allclose(local_w.sum(axis=1),1) and np.max(np.abs(local_w-W0))<=.05+1e-12,'Local domain construction failed')
    return [('global_step_005',global_w),('local_pm005_step001',local_w)]

def minimal_l1_equality(d: np.ndarray,w: np.ndarray) -> dict:
    """Exact continuous-simplex mass-transfer solution for a pairwise linear mean.

    To increase d.w to zero, move mass from the smallest d donors to the
    largest d recipient (reverse for a decrease). Each transferred unit costs
    L1=2, so descending gain per unit is optimal. This is NOT a one-SE margin.
    """
    d=np.asarray(d,float);w=np.asarray(w,float);v=float(d@w)
    need(d.shape==w.shape==(4,) and np.isfinite(d).all() and np.isfinite(w).all(),'Invalid boundary geometry')
    need(np.min(w)>=0 and near(w.sum(),1),'Invalid starting weights')
    if abs(v)<=1e-14:return {'reachable':True,'l1_distance':0.,'weights':w.tolist(),'residual':v,'strict_opposite_sign_possible':bool(d.min()<0<d.max())}
    oriented=d if v<0 else -d;recv=int(np.argmax(oriented));need_gain=abs(v)
    if oriented[recv]<0:return {'reachable':False,'l1_distance':None,'weights':None,'residual':None,'strict_opposite_sign_possible':False}
    out=w.copy();mass=0.
    for donor in np.argsort(oriented,kind='stable'):
        donor=int(donor)
        if donor==recv:continue
        gain=float(oriented[recv]-oriented[donor])
        if gain<=0:continue
        take=min(float(out[donor]),need_gain/gain)
        out[donor]-=take;out[recv]+=take;mass+=take;need_gain-=take*gain
        if need_gain<=1e-14:break
    need(need_gain<=1e-12 and abs(float(d@out))<=1e-12,'Pairwise boundary solver failed')
    return {'reachable':True,'l1_distance':2*mass,'weights':out.tolist(),'residual':float(d@out),
            'strict_opposite_sign_possible':bool(oriented[recv]>0)}

def weight_analysis(it: np.ndarray,ot: np.ndarray) -> tuple[dict,dict[str,bytes]]:
    need(it.shape==(25,10,4,4) and ot.shape==(25,10,4),'Unexpected weight-analysis tensor')
    deltas={'inner':it.mean(axis=2)[:,V4,:]-it.mean(axis=2)[:,V1,:],
            'outer':ot[:,V4,:]-ot[:,V1,:]}
    info={'domains':{},'interpretation':'Counterfactual arithmetic of frozen scalar components; no historical reselection or new fits.',
          'mean_comparison_not_selection':True,'outer_used_for_hypothetical_selection':False}
    assets={};all_boundary=[];base=np.array([one_se(it[i],W0)[0] for i in range(25)])
    for domain,ww in grids():
        aggregate=[];by_split=[];by_best=[];by_ties=[]
        for idx,w in enumerate(ww):
            chosen=[];best=[];co_max=[]
            for i in range(25):
                ch,be,mu,se,cut,ties=one_se(it[i],w);chosen.append(ch);best.append(be);co_max.append(ties)
            co_max=np.array(co_max);dc={k:a@w for k,a in deltas.items()}
            row={'grid_index':idx,**dict(zip(SHORT,map(float,w))),'l1_from_historical':float(np.sum(abs(w-W0))),
                 'changed_selected_splits':int(np.sum(np.array(chosen)!=base)),
                 'inner_mean_V4_minus_V1':float(dc['inner'].mean()),'outer_mean_V4_minus_V1':float(dc['outer'].mean()),
                 'inner_pair_V4_higher_splits':int(sum(dc['inner']>TIE_ATOL)),
                 'outer_pair_V4_higher_splits':int(sum(dc['outer']>TIE_ATOL))}
            for j in range(10):row['selected_M%02d'%j]=chosen.count(j)
            for j in range(10):row['best_label_M%02d'%j]=best.count(j)
            for j in range(10):row['co_max_M%02d'%j]=int(co_max[:,j].sum())
            aggregate.append(row)
            by_split.append({'grid_index':idx,**{sid(i//5+1,i%5+1):v for i,v in enumerate(chosen)}})
            by_best.append({'grid_index':idx,**{sid(i//5+1,i%5+1):v for i,v in enumerate(best)}})
            by_ties.append({'grid_index':idx,**{sid(i//5+1,i%5+1):int(sum((1<<j) for j in np.flatnonzero(co_max[i]))) for i in range(25)}})
        info['domains'][domain]={'grid_points':len(ww),'inner_pair_mean_V4_higher_grid_points':sum(x['inner_mean_V4_minus_V1']>TIE_ATOL for x in aggregate),
          'outer_pair_mean_V4_higher_grid_points':sum(x['outer_mean_V4_minus_V1']>TIE_ATOL for x in aggregate),
          'minimum_grid_L1_any_split_selection_change':min((x['l1_from_historical'] for x in aggregate if x['changed_selected_splits']),default=None),
          'grid_points_all_25_historical_choices_unchanged':sum(x['changed_selected_splits']==0 for x in aggregate),
          'selected_split_grid_counts':{m:sum(x['selected_M%02d'%j] for x in aggregate) for j,m in enumerate(MODEL_IDS)},
          'modal_hypothetical_selected_model_counts':dict(Counter(MODEL_IDS[max(range(10),key=lambda j:x['selected_M%02d'%j])] for x in aggregate)),
          'V1_selected_split_count_range':[min(x['selected_M06'] for x in aggregate),max(x['selected_M06'] for x in aggregate)],
          'no_probability_interpretation':True}
        assets[domain+'_summary.csv']=csv_write(aggregate)
        assets[domain+'_hypothetical_selected.csv']=csv_write(by_split)
        assets[domain+'_best_labels.csv']=csv_write(by_best)
        assets[domain+'_co_max_bitmasks.csv']=csv_write(by_ties)
    for scope,dd in deltas.items():
        for name,d in [('MEAN_OF_25_SPLITS',dd.mean(axis=0))]+[(sid(i//5+1,i%5+1),d) for i,d in enumerate(dd)]:
            b=minimal_l1_equality(d,W0)
            all_boundary.append({'scope':scope,'split':name,**{'delta_'+k:float(v) for k,v in zip(SHORT,d)},
              'historical_delta_utility':float(d@W0),'pairwise_mean_equality_reachable':b['reachable'],
              'min_global_L1_to_pairwise_mean_equality':b['l1_distance'],
              **{'boundary_'+k:(None if b['weights'] is None else b['weights'][i]) for i,k in enumerate(SHORT)},
              'strict_opposite_sign_possible':b['strict_opposite_sign_possible'],
              'not_a_one_SE_selection_margin':True})
    assets['pairwise_mean_continuous_boundaries.csv']=csv_write(all_boundary)
    assets['model_codebook.csv']=csv_write([{'integer_code':i,'model_id':m,'complexity_rank':i+1,'co_max_bit':1<<i} for i,m in enumerate(MODEL_IDS)])
    info['aggregate_pairwise_mean_boundaries']=[r for r in all_boundary if r['split']=='MEAN_OF_25_SPLITS']
    info['original_selected_counts']=dict(Counter(MODEL_IDS[x] for x in base))
    return info,assets

def ranks(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float);need(x.ndim==1 and np.isfinite(x).all(),'Invalid Spearman values')
    ix=np.argsort(x,kind='stable');out=np.empty(len(x));a=0
    while a<len(x):
        b=a+1
        while b<len(x) and x[ix[b]]==x[ix[a]]:b+=1
        out[ix[a:b]]=(a+b-1)/2+1;a=b
    return out

def rho(x: np.ndarray,y: np.ndarray) -> float:
    r=ranks(x);s=ranks(y);need(r.std()>0 and s.std()>0,'Constant correlation vector')
    return float(np.corrcoef(r,s)[0,1])

def basis(x: np.ndarray) -> np.ndarray:
    u,s,_=np.linalg.svd(x,full_matrices=False);tol=float(s[0])*max(x.shape)*np.finfo(np.float64).eps
    rank=int(sum(s>tol));need(rank==5,'Frozen TARGET latent rank differs');return u[:,:rank]

def numeric_map(rows: list[dict],keys: tuple[str,...],n: int) -> dict:
    d={tuple(str(r[k]) for k in keys):r for r in rows};need(len(d)==len(rows)==n,'Wrong/duplicate biological table geometry');return d

def biological_replay(tables: dict[str,list[dict]],summary: dict,z: dict[int,np.ndarray],scores: np.ndarray,
                      primary: dict) -> tuple[dict,dict[str,bytes]]:
    """Reproduce only statistics already defined and executed in 06e4."""
    need(set(z)==set(SEEDS) and scores.shape==(78,4) and np.isfinite(scores).all(),'Unexpected frozen biology shape')
    for s in SEEDS:need(z[s].shape==(78,5) and np.isfinite(z[s]).all(),'Wrong latent shape')
    rc=numeric_map(tables[BIO_FILES[0]],('program_id','seed','coordinate_id'),60)
    rs=numeric_map(tables[BIO_FILES[1]],('program_id','seed'),12)
    across=numeric_map(tables[BIO_FILES[2]],('program_id',),4)
    ps=numeric_map(tables[BIO_FILES[3]],('reference_seed','moving_seed'),3)
    can=numeric_map(tables[BIO_FILES[4]],('reference_seed','moving_seed','canonical_index'),15)
    hm=numeric_map(tables[BIO_FILES[5]],('reference_seed','moving_seed','reference_coordinate_id'),15)
    ac=numeric_map(tables[BIO_FILES[6]],('program_id','seed','aligned_coordinate_id'),40)
    ass=numeric_map(tables[BIO_FILES[7]],('program_id','seed'),8)
    need(summary['status']==BIO_SUMMARY_STATUS and summary['primary_seed']==1103 and summary['n_cases']==78 and summary['shared_dim']==5,'06e4 summary identity mismatch')
    need(summary['programs']==list(PROGRAMS) and summary['reproducibility_only_seeds']==[2207,3301],'Wrong program/seed scope')
    need(summary['primary_seed_replay']=='PASS_20_OF_20' and all(v is False for v in summary['firewalls'].values()),'Historical biology boundary violated')
    pr={r['program_id']:r for r in primary['program_results']};need(set(pr)==set(PROGRAMS) and primary['primary_seed']==1103,'Primary null table identity differs')
    centered={s:z[s]-z[s].mean(axis=0,keepdims=True) for s in SEEDS};raw={};program_table=[];spectra=[];pairs=[]
    for g,p in enumerate(PROGRAMS):
        row={'program_id':p,'scope':'fixed four programs; full-data V1; initialization sensitivity; primary-null only at seed1103'};vals=[]
        for s in SEEDS:
            rr=np.array([rho(scores[:,g],z[s][:,j]) for j in range(5)]);b=int(np.argmax(abs(rr)));val=float(abs(rr[b]));raw[(p,s)]=val;vals.append(val)
            for j in range(5):
                r=rc[(p,str(s),f'V1_SHARED_{j+1:02d}')];need(near(r['spearman_rho'],rr[j]) and near(r['absolute_spearman_rho'],abs(rr[j])),'Raw correlation replay mismatch')
            r=rs[(p,str(s))];need(near(r['observed_max_abs_spearman'],val) and near(r['best_signed_spearman'],rr[b]) and r['best_coordinate_id']==f'V1_SHARED_{b+1:02d}','Raw maximum replay mismatch')
            row[f'seed{s}_max_abs_rho']=val;row[f'seed{s}_best_coordinate']=r['best_coordinate_id'];row[f'seed{s}_signed_rho']=float(rr[b])
        expected={'primary_seed_1103_max_abs_spearman':vals[0],'seed2207_max_abs_spearman':vals[1],'seed3301_max_abs_spearman':vals[2],
                  'min_raw_max_abs_spearman':min(vals),'median_raw_max_abs_spearman':float(np.median(vals)),
                  'max_raw_max_abs_spearman':max(vals),'range_raw_max_abs_spearman':max(vals)-min(vals)}
        for k,v in expected.items():need(near(across[(p,)][k],v),'Across-seed maximum arithmetic differs')
        need(near(pr[p]['observed_max_abs_spearman'],vals[0]),'Primary program statistic changed')
        row.update(min_max_abs_rho=min(vals),max_max_abs_rho=max(vals),range_max_abs_rho=max(vals)-min(vals),
                   primary_seed1103_BH_q_recorded=num(pr[p]['bh_q_4']),primary_seed1103_matched_null_supported=bool(pr[p]['matched_random_support_q_le_0_05']),
                   no_new_seed_specific_p_or_q=True,same_factor_reproducibility_not_established=True)
        program_table.append(row)
    for ref,mov in SEED_PAIRS:
        a,b=centered[ref],centered[mov];qa,qb=basis(a),basis(b)
        cc=np.clip(np.linalg.svd(qb.T@qa,compute_uv=False),0,1);angles=np.degrees(np.arccos(cc))
        u,_,vt=np.linalg.svd(b.T@a,full_matrices=False);aligned=b@(u@vt);res=float(np.linalg.norm(aligned-a,'fro')/np.linalg.norm(a,'fro'))
        pair=ps[(str(ref),str(mov))]
        statistics_expected={'mean_canonical_correlation':float(cc.mean()),'median_canonical_correlation':float(np.median(cc)),
                             'minimum_canonical_correlation':float(cc.min()),'maximum_principal_angle_degrees':float(angles.max()),'normalized_procrustes_residual':res}
        for k,v in statistics_expected.items():need(near(pair[k],v,atol=2e-6 if 'angle' in k else 1e-9),'Subspace replay differs: '+k)
        corr=np.array([[np.corrcoef(aligned[:,j],a[:,k])[0,1] for k in range(5)] for j in range(5)])
        matched=[];moving_ids=[]
        for j in range(5):
            c=can[(str(ref),str(mov),str(j+1))];need(near(c['canonical_correlation'],cc[j],1e-9) and near(c['principal_angle_degrees'],angles[j],2e-6),'Canonical spectrum replay differs')
            spectra.append({'reference_seed':ref,'moving_seed':mov,'canonical_index':j+1,'canonical_correlation':float(cc[j]),'principal_angle_degrees':float(angles[j])})
            h=hm[(str(ref),str(mov),f'V1_SHARED_{j+1:02d}')];mi=int(h['aligned_moving_coordinate_id'].split('_')[-1])-1
            need(0<=mi<5 and near(h['matched_signed_pearson'],corr[mi,j],1e-9) and near(h['matched_absolute_pearson'],abs(corr[mi,j]),1e-9),'Hungarian correlation cell differs')
            matched.append(abs(corr[mi,j]));moving_ids.append(mi)
        need(sorted(moving_ids)==list(range(5)),'Hungarian assignment not a permutation')
        optimum=max(sum(abs(corr[perm[j],j]) for j in range(5)) for perm in itertools.permutations(range(5)))
        need(near(sum(matched),optimum,1e-9),'Recorded Hungarian assignment is not optimal')
        need(near(pair['mean_abs_hungarian_matched_correlation'],np.mean(matched),1e-9) and near(pair['minimum_abs_hungarian_matched_correlation'],min(matched),1e-9),'Hungarian summary differs')
        pairs.append({'reference_seed':ref,'moving_seed':mov,**statistics_expected,
                      'mean_abs_hungarian_matched_correlation':float(np.mean(matched)),
                      'minimum_abs_hungarian_matched_correlation':float(min(matched)),
                      'scope':'78 full-data training cases; not held-out or subsample refits; no PASS/FAIL stability threshold'})
        if ref==1103:
            for g,p in enumerate(PROGRAMS):
                rr=np.array([rho(scores[:,g],aligned[:,j]) for j in range(5)]);best=int(np.argmax(abs(rr)))
                for j in range(5):
                    v=ac[(p,str(mov),f'ALIGNED_TO_1103_SHARED_{j+1:02d}')];need(near(v['spearman_rho'],rr[j]) and near(v['absolute_spearman_rho'],abs(rr[j])),'Aligned correlation replay mismatch')
                v=ass[(p,str(mov))];need(near(v['aligned_max_abs_spearman'],abs(rr[best])) and near(v['best_aligned_signed_spearman'],rr[best]) and v['best_aligned_coordinate_id']==f'ALIGNED_TO_1103_SHARED_{best+1:02d}','Aligned maximum replay mismatch')
    smp=numeric_map(summary['pairwise_subspace_summary'],('reference_seed','moving_seed'),3)
    for row in pairs:
        for k,v in row.items():
            if k not in ('reference_seed','moving_seed','scope'):need(near(smp[(str(row['reference_seed']),str(row['moving_seed']))][k],v,2e-6 if 'angle' in k else 1e-9),'JSON/CSV pair summary differs')
    sma=numeric_map(summary['program_raw_seed_sensitivity'],('program_id',),4)
    for key,row in across.items():
        for k,v in row.items():
            if k not in ('program_id','analysis_role'):need(near(sma[key][k],v),'JSON/CSV program summary differs')
    need(near(summary['05a_compatible_seed_aligned_subspace_similarity'],np.median([p['mean_canonical_correlation'] for p in pairs]),1e-9),'05a compatible summary differs')
    info={'status':'COMPLETED_06E4_STATISTICS_REPLAYED_NO_NEW_ENDPOINT','programs':4,'seeds':3,'raw_correlations_replayed':60,
          'aligned_correlations_replayed':40,'canonical_correlations_replayed':15,'seed_pairs':3,'Hungarian_cells_checked':15,
          'primary_null_results_recomputed':False,'new_p_values':False,'new_program_subspace_statistic':False,
          'molecular_matrices_read':False,'three_frozen_78_by_5_latents_read':True,'full_data_not_heldout':True,
          'cross_cohort_stability_curve':False,'primary_seed_changed':False}
    assets={'program_by_seed.csv':csv_write(program_table),'TARGET_full_canonical_spectra.csv':csv_write(spectra),
            'TARGET_full_data_pair_diagnostics.csv':csv_write(pairs),
            'TARGET_existing_aligned_program_summary.csv':csv_write(tables[BIO_FILES[7]])}
    assets['TARGET_frozen_small_latents_and_scores.json']=encode({'seeds':list(SEEDS),'programs':list(PROGRAMS),'latents':{str(s):z[s].tolist() for s in SEEDS},
         'program_scores':scores.tolist(),'row_order':'same exact hash-bound 78-case order, identifiers not duplicated','scope':'replay only; not new patient data or biological program selection'},compact=True)
    return info,assets

def read_biological(reader: Reader) -> tuple[dict,dict[str,bytes]]:
    reader.get(BIO_RUNNER,BIO_RUNNER_SHA,role='reviewed_exact_06e4_source_NOT_executed')
    reader.get(BIO_AUTHOR,BIO_AUTHOR_SHA,role='reviewed_exact_06e4a_source_NOT_executed')
    contract=reader.json(BIO_CONTRACT,BIO_CONTRACT_SHA);cl=reader.json(BIO_CONTRACT_LOCK)
    need(contract['status']=='FROZEN_BEFORE_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY' and cl['status']=='06E3_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_CONTRACT_FROZEN','06e3 status differs')
    need(cl['contract']['sha256']==BIO_CONTRACT_SHA,'06e3 contract binding differs')
    auth=reader.json(BIO_AUTH);pre=reader.json(BIO_PREFLIGHT);lock=reader.json(BIO_LOCK)
    need(auth['status']=='06E4A_V1_SEED_SUBSPACE_RUNNER_SOURCE_AUTHORIZED' and auth['runner_source']['sha256']==BIO_RUNNER_SHA,'06e4 authorization identity differs')
    need(auth['06e3_contract']['sha256']==BIO_CONTRACT_SHA and auth['synthetic_preflight']['sha256']==sha(reader.raw[BIO_PREFLIGHT]),'06e4 authorization chain differs')
    need(pre['status']=='06E4_V1_SEED_SUBSPACE_RUNNER_SYNTHETIC_SELFTEST_PASS' and pre['runner_sha256']==BIO_RUNNER_SHA and pre['contract_sha256']==BIO_CONTRACT_SHA,'06e4 preflight differs')
    need(lock['status']==BIO_STATUS and lock['runner_source']['sha256']==BIO_RUNNER_SHA and lock['06e3_contract']['sha256']==BIO_CONTRACT_SHA and lock['06e4a_authorization']['sha256']==sha(reader.raw[BIO_AUTH]),'06e5 result binding differs')
    need(set(lock['result_files'])==set(BIO_FILES),'Wrong completed 06e4 file set')
    tables={};assets={}
    for name,rec in lock['result_files'].items():
        rel=BIO_DIR+'/'+name
        need(rec['path'].replace('\\','/')=='/'.join(rel.split('/')[1:]),'Unexpected 06e4 result path')
        raw=reader.get(rel,rec['sha256'],rec['size_bytes'],role='bound_by_first_received_06e5_results_lock')
        assets['source_06e4_'+name]=raw
        if name.endswith('.csv'):tables[name]=csv_read(raw)
        else:summary=decode(raw)
    need(summary['runner_sha256']==BIO_RUNNER_SHA,'06e4 summary binds wrong runner')
    score_raw=None
    for name,pin in PRIMARY_PINS.items():
        b=reader.get(PRIMARY_DIR+'/'+name,pin,role='previously_frozen_primary_biology')
        if name=='v1_frozen_program_scores_78.csv':score_raw=b
        elif name=='v1_biological_alignment_summary.json':primary=decode(b);assets['source_primary_biology_summary.json']=b
        else:assets['source_primary_'+name]=b
    ids=csv_read(reader.get(CASE_REL,CASE_SHA,role='pinned_full_data_case_order'));ss=csv_read(score_raw)
    need(len(ids)==len(ss)==78 and [integer(r['case_index']) for r in ids]==list(range(78)),'Bad case order')
    need([r['case_id'] for r in ids]==[r['case_id'] for r in ss] and len({r['case_id'] for r in ids})==78,'Program/latent patient order differs')
    scores=np.array([[num(row[p+'__strict_signed_mean_z']) for p in PROGRAMS] for row in ss])
    z={}
    for s in SEEDS:
        rel=FULL_DIR+f'/target_v1_patient_shared_seed{s}.npy';raw=reader.get(rel,LATENT_SHA[s],role='only_small_78_by_5_frozen_latent')
        z[s]=np.asarray(np.load(io.BytesIO(raw),allow_pickle=False),dtype=np.float64)
    info,more=biological_replay(tables,summary,z,scores,primary);assets.update(more)
    # Compare 20 primary cells to their separately pinned original table as well.
    prim=csv_read(reader.raw[PRIMARY_DIR+'/v1_program_coordinate_correlations.csv'])
    need(len(prim)==20,'Primary correlation table has wrong row count')
    for row in prim:
        p=row['program_id'];j=int(row['coordinate_id'].split('_')[-1])-1
        need(p in PROGRAMS and 0<=j<5 and near(rho(scores[:,PROGRAMS.index(p)],z[1103][:,j]),row['spearman_rho']),'Original primary 20-cell replay differs')
    info['primary_correlations_replayed']=20
    info['results_lock_sha256']=sha(reader.raw[BIO_LOCK])
    info['results_lock_provenance']='First received now; internal chain and pinned source/latent identities checked; not falsely described as a previously externally pinned result lock.'
    for name,rel in [('source_06e5_lock.json',BIO_LOCK),('source_06e4a_authorization.json',BIO_AUTH),('source_06e3_contract.json',BIO_CONTRACT)]:assets[name]=reader.raw[rel]
    return info,assets

def notes(w: dict,b: dict) -> str:
    return '''# Paper 5 follow-up: scope and use

This is post-result descriptive sensitivity and replay, not a new confirmatory
experiment. Historical TARGET primary selection remains unchanged. No models
were fitted and no old source, result, threshold or authorization was rewritten.

## Weight sensitivity
The global grid uses step 0.05 on the complete nonnegative four-weight simplex.
The local grid uses step 0.01 within +/-0.05 of each historical weight, with sum
one. The local box is an explicitly chosen arithmetic neighborhood, not a claim
about plausible scientific preferences. All ten frozen candidates are retained.
At each weight, two seeds are averaged within each inner fold, then four inner
folds supply the original mean/SE heuristic and complexity-resolved one-SE choice.
Outer values never select a weight or model. Grid counts are not probabilities.
The continuous L1 margins concern V4/V1 mean equality only, not one-SE selection.
Integer labels in selected/best tables use model_codebook.csv. Co-max tables use
bitmasks (bit j identifies model j), so V3/V4 ties remain explicit.
A hypothetical choice is not a new authorized primary representation. No outer
performance of a post-hoc selected weight is presented as unbiased validation.

## Completed 06e4 replay
The 78-case full-data V1 representations and existing four-program score table
are used to reproduce only the completed 06e4 statistics. The complete 5-value
canonical spectra and all four programs/all three seeds are shown. Association
maxima can be reached on different axes: repeated max-|rho| magnitudes do not
establish identity of a factor. A narrow range can also arise around zero.
Primary matched-null support stays at seed 1103; it is not copied to other seeds.
No new seed-specific p/q, program selection or program-to-subspace biology
statistic is computed. Procrustes residual is unscaled, with the original
reference-norm denominator; it is not a pure rotation-invariant angle measure.
Full-data TARGET diagnostics are not merged with held-out SARC/KIRC diagnostics.
Three seed pairs from the same cases are not independent replications. No
association of subspace agreement with biological reproducibility is tested.

## Interpretation
TARGET reconstruction scores are clipped PCA-space scores, unlike SARC/KIRC
molecular-target NMSE. Do not put them on a single error scale or infer an
isolated sample-size effect. Manuscript and README edits require author review.
'''

def run(project: Path,script: Path) -> Path:
    project=project.absolute();script=script.absolute();plain(project);plain(script)
    need((project/'paper5/scripts').is_dir(),'Run from workspace root containing paper5/scripts')
    need(script.name=='p5follow.py','Save as p5follow.py in Downloads, not .txt or a (1) duplicate')
    print('Paper 5 - weight sensitivity + completed 06e4 replay; NO FIT',flush=True)
    print('Version: '+VERSION,flush=True)
    parent=safe(project,'paper5/interpretation_followup_v1');parent.mkdir(exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];out=parent/('f_'+tag);out.mkdir(exist_ok=False)
    source=script.read_bytes();reader=Reader(project)
    publish(out/'analysis_plan.json',encode(plan_document()));publish(out/'p5follow_source.py',source)
    try:
        fs,it,ot,replay,inputs=read_target(reader)
        print('Reviewed p5refine packet / 700 outer / 1900 current inner / 25 historical selections: PASS',flush=True)
        # Both exact old results and source scopes are validated before the sensitivity.
        bio,bfiles=read_biological(reader)
        print('Frozen 06e4: 60 raw + 40 aligned correlations / 15 canonical values / 20 primary cells: PASS',flush=True)
        w,wfiles=weight_analysis(it,ot)
        for key,rec in w['domains'].items():print('Weight domain '+key+': '+str(rec['grid_points'])+' points, hypothetical one-SE replay: PASS',flush=True)
        assets={**wfiles,**bfiles,'historical_selection_replay.csv':csv_write(replay),
                'weight_summary.json':encode(w),'bio_replay_summary.json':encode(bio),
                'NOTES_EN.md':notes(w,bio).encode('utf-8'),
                'source_target_inner_normalized_metrics.json':fs['target_inner_normalized_metrics.json'],
                'source_target_outer_fixed_model_benchmark.csv':fs['source_target_outer_fixed_model_benchmark.csv']}
        reader.recheck();need(sha(script.read_bytes())==sha(source),'Follow-up source changed during run')
        report={'status':'P5_WEIGHT_SENSITIVITY_AND_06E4_REPLAY_COMPLETE_NO_NEW_FIT','version':VERSION,
                'inputs':inputs,'weights':w,'biology':bio,'input_fingerprints':list(reader.records.values()),
                'input_recheck':'PASS','models_fitted':0,'historical_selection_changed':False,'new_p_or_q':False,
                'runtime':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}}
        assets['audit.json']=encode(report)
        all_assets={**assets,'analysis_plan.json':encode(plan_document()),'p5follow_source.py':source}
        manifest={n:{'sha256':sha(b),'size_bytes':len(b)} for n,b in all_assets.items()}
        packet={'schema':'paper5-weight-seed-followup-evidence-v1','status':report['status'],'archive_required':False,
                'new_fits':0,'completion':{'files':manifest,'source_sha256':sha(source),'input_sha256':INPUT_SHA,'historical_files_changed':False},
                'files':{n:{**manifest[n],'text':b.decode('utf-8')} for n,b in all_assets.items()}}
        raw=encode(packet,compact=True);need(len(raw)<=MAX_REPORT,'Evidence exceeds 5 MiB; source results preserved')
        for n,b in sorted(assets.items()):publish(safe(out,n),b)
        reader.recheck();publish(out/'completion.json',encode(packet['completion']))
        dest=parent/('p5f_'+tag+'.json');publish(dest,raw)
        print('Original inputs and result locks rechecked unchanged: PASS',flush=True)
        print('Status: '+report['status'],flush=True)
        print('Evidence JSON (send this FILE; no ZIP): '+str(dest),flush=True)
        print('Evidence bytes: '+str(len(raw)),flush=True);print('Evidence SHA-256: '+sha(raw),flush=True)
        return dest
    except Exception as exc:
        f={'status':'P5_FOLLOWUP_STOPPED','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
           'source_sha256':sha(source),'inputs_read':list(reader.records.values()),'new_fits':0,
           'instruction':'Preserve old results; do not change hashes or rerun model training.'}
        publish(out/'failure.json',encode(f))
        print('P5 FOLLOW-UP STOPPED: '+str(exc),file=sys.stderr,flush=True)
        print('Failure report: '+str(out/'failure.json'),file=sys.stderr,flush=True)
        raise

if __name__=='__main__':
    try:run(Path.cwd(),Path(__file__))
    except Exception:sys.exit(1)
