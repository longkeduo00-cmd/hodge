from __future__ import annotations

"""Joint ER/WS/RGG mechanism discrimination over network size N and mean degree kappa.

Scientific contract
-------------------
* Full factorial cells over mechanism_state x N x target_mean_degree.
* Every (state, N, kappa, repeat) graph is independently sampled through the
  existing stable synthetic graph cache identity; no trajectory coupling.
* ER: p = kappa/(N-1).
* WS: degree = kappa, beta defines the mechanism state.
* RGG: radius is recomputed for every (N,kappa) using the finite-square
  expected-degree equation already used by the matched-density experiments.
* Same-cell noise uses unordered realization pairs; different cells use all
  cross-realization pairs.
* Hodge mathematics and all baseline semantics are unchanged.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any
import hashlib, json, math, shutil

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers
from hodge_vnext.models import synthetic_graph_record
from hodge_vnext.utils import stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import DEFAULT_METHODS, MethodManager
from .model_discrimination import (
    FINE_STATE_ORDER, _coarse_label, _graph_observables, _params_for_state,
    _state_from_beta, _tie_key,
)
from .model_joint_plotting import plot_joint_discrimination

JOINT_SCHEMA = "er-ws-rgg-joint-N-kappa-v1"


def build_joint_cells(network_sizes: list[int], target_degrees: list[float], ws_betas: list[float]) -> pd.DataFrame:
    sizes=[int(x) for x in network_sizes]
    kappas=[float(x) for x in target_degrees]
    if not sizes or len(sizes)!=len(set(sizes)) or any(b<=a for a,b in zip(sizes[:-1],sizes[1:])):
        raise ValueError('network_sizes must be non-empty, unique, strictly increasing')
    if not kappas or len(kappas)!=len(set(kappas)) or any(b<=a for a,b in zip(kappas[:-1],kappas[1:])):
        raise ValueError('target_degrees must be non-empty, unique, strictly increasing')
    states=['ER','RGG']+[_state_from_beta(float(b)) for b in ws_betas]
    if tuple(states)!=FINE_STATE_ORDER:
        raise ValueError('formal protocol requires WS betas [1e-3,1e-2,1e-1,1]')
    rows=[]; cell=0
    for state in states:
        for n in sizes:
            for kappa in kappas:
                model,params,derived=_params_for_state(state,n,kappa)
                rows.append({
                    'cell_index':cell,
                    'cell_label':f'{state}-N{n}-k{format(kappa,".6g")}',
                    'mechanism_state':state,
                    'coarse_mechanism':_coarse_label(state),
                    'model':model,
                    'network_size':n,
                    'target_mean_degree':kappa,
                    'er_p':derived['er_p'],'ws_degree':derived['ws_degree'],'ws_beta':derived['ws_beta'],
                    'rgg_radius':derived['rgg_radius'],
                    'generator_params_json':json.dumps(params,sort_keys=True,separators=(',',':')),
                }); cell+=1
    return pd.DataFrame(rows)


def _run_stem(exp:dict[str,Any], method:dict[str,Any], baselines:dict[str,Any])->str:
    identity={'schema':JOINT_SCHEMA,'network_sizes':exp['network_sizes'],'target_degrees':exp['target_degrees'],
              'repeats':exp['repeats'],'base_seed':exp.get('base_seed',22),'ws_betas':exp['ws_betas'],
              'methods':exp.get('methods',DEFAULT_METHODS),'method':method,'baselines':baselines}
    return f"joint_N{len(exp['network_sizes'])}_k{len(exp['target_degrees'])}_R{int(exp['repeats'])}_{stable_hash(identity,10)}"


def _protocol_pair_ok(ni:int,ki:float,nj:int,kj:float,protocol:str)->bool:
    same_n=(ni==nj); same_k=math.isclose(float(ki),float(kj),abs_tol=1e-12)
    if protocol=='matched_N_k': return same_n and same_k
    if protocol=='cross_N_only': return (not same_n) and same_k
    if protocol=='cross_k_only': return same_n and (not same_k)
    if protocol=='cross_N_k': return (not same_n) and (not same_k)
    if protocol=='global': return True
    raise ValueError(protocol)


def _retrieval_metrics_joint(D:np.ndarray, ids:np.ndarray, labels:np.ndarray, ns:np.ndarray, ks:np.ndarray,
                             protocol:str, tie_seed:int)->tuple[dict[str,float|int],pd.DataFrame]:
    n=len(labels); preds=[]; rows=[]; pair_y=[]; pair_score=[]
    for i in range(n):
        for j in range(i+1,n):
            if _protocol_pair_ok(int(ns[i]),float(ks[i]),int(ns[j]),float(ks[j]),protocol):
                pair_y.append(int(labels[i]==labels[j])); pair_score.append(float(-D[i,j]))
    for i in range(n):
        cand=[j for j in range(n) if j!=i and _protocol_pair_ok(int(ns[i]),float(ks[i]),int(ns[j]),float(ks[j]),protocol)]
        if not cand:
            continue
        ranked=sorted(cand,key=lambda j:(float(D[i,j]),_tie_key(str(ids[i]),str(ids[j]),tie_seed),str(ids[j])))
        pred=str(labels[ranked[0]]); preds.append((i,pred))
        rel=np.asarray([labels[j]==labels[i] for j in ranked],int); total=int(rel.sum()); cum=np.cumsum(rel)
        prec=cum/np.arange(1,len(rel)+1); ap=float(np.dot(prec,rel)/total) if total else float('nan')
        k5=min(5,len(rel)); disc=1/np.log2(np.arange(1,k5+1)+1); ideal=min(5,total)
        idcg=float(np.sum(1/np.log2(np.arange(1,ideal+1)+1))) if ideal else 0.0
        ndcg=float(np.dot(rel[:k5],disc)/idcg) if idcg else float('nan')
        rows.append({'query_index':i,'graph_id':str(ids[i]),'true_label':str(labels[i]),'predicted_label':pred,
                     'nearest_graph_id':str(ids[ranked[0]]),'nearest_distance':float(D[i,ranked[0]]),
                     'gallery_size':len(ranked),'relevant_total':total,'average_precision':ap,'ndcg_at_5':ndcg})
    if not rows:
        return {'n_queries':0,'balanced_accuracy':float('nan'),'macro_f1':float('nan'),'same_class_auroc':float('nan'),
                'mean_average_precision':float('nan'),'ndcg_at_5':float('nan')},pd.DataFrame()
    qdf=pd.DataFrame(rows); idx=np.asarray([i for i,_ in preds],int); yp=np.asarray([p for _,p in preds],str); yt=labels[idx]
    stats={'n_queries':len(qdf),'balanced_accuracy':float(balanced_accuracy_score(yt,yp)),
           'macro_f1':float(f1_score(yt,yp,average='macro',zero_division=0)),
           'same_class_auroc':float(roc_auc_score(pair_y,pair_score)) if len(set(pair_y))==2 else float('nan'),
           'mean_average_precision':float(np.nanmean(qdf.average_precision)),
           'ndcg_at_5':float(np.nanmean(qdf.ndcg_at_5))}
    return stats,qdf


def _cell_stats(D:np.ndarray, graph_cells:np.ndarray, num_cells:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    means=np.zeros((num_cells,num_cells),float); stds=np.zeros_like(means); counts=np.zeros((num_cells,num_cells),int)
    for a in range(num_cells):
        ia=np.flatnonzero(graph_cells==a)
        vals=D[np.ix_(ia,ia)][np.triu_indices(len(ia),1)]
        means[a,a]=float(vals.mean()); stds[a,a]=float(vals.std(ddof=1)) if len(vals)>1 else 0.; counts[a,a]=len(vals)
        for b in range(a+1,num_cells):
            ib=np.flatnonzero(graph_cells==b); vals=D[np.ix_(ia,ib)].ravel()
            means[a,b]=means[b,a]=float(vals.mean()); stds[a,b]=stds[b,a]=float(vals.std(ddof=1)) if len(vals)>1 else 0.; counts[a,b]=counts[b,a]=len(vals)
    return means,stds,counts


def run_joint_discrimination(payload:dict[str,Any], config_path:Path, *, force:bool=False)->Path:
    cfg=method_from_payload(payload,config_path); exp=dict(payload['validation'])
    sizes=[int(x) for x in exp['network_sizes']]; kappas=[float(x) for x in exp['target_degrees']]; repeats=int(exp['repeats'])
    if repeats<2: raise ValueError('repeats must be >=2')
    methods=MethodManager.validate_methods(list(exp.get('methods',DEFAULT_METHODS)))
    manager=MethodManager.from_config(cfg,payload.get('baselines',{})); tie_seed=int(exp.get('tie_seed',20260922)); base_seed=int(exp.get('base_seed',22))
    cells=build_joint_cells(sizes,kappas,[float(x) for x in exp['ws_betas']])
    root=resolve_path(exp.get('results_root','results/validation/model_joint_discrimination'),config_path)
    run=root/_run_stem(exp,payload['method'],payload.get('baselines',{})); complete=run/'RUN_COMPLETE.json'
    if complete.exists() and not force:
        plot_joint_discrimination(run); return run
    run.mkdir(parents=True,exist_ok=True)
    if force:
        for name in ('tables','arrays','plots'):
            p=run/name
            if p.exists(): shutil.rmtree(p)
    for name in ('tables','arrays','plots'): (run/name).mkdir(parents=True,exist_ok=True)
    eff=json.loads(json.dumps(payload)); eff['cache_schema']=CACHE_SCHEMA; eff['resolved_protocol']={
        'schema':JOINT_SCHEMA,'design':'full factorial mechanism x N x target mean degree',
        'sampling':'independent for every (state,N,kappa,repeat)',
        'ER':'p=kappa/(N-1)','WS':'degree=kappa; beta defines state',
        'RGG':'finite-square radius recomputed at every (N,kappa)',
        'pairing':'all cross-realization pairs between cells; unordered realization pairs within a cell'}
    write_json(run/'effective_config.json',eff); cells.to_csv(run/'tables'/'parameter_cells.csv',index=False)

    records=[]; rows=[]
    for c in cells.itertuples(index=False):
        state=str(c.mechanism_state); n=int(c.network_size); k=float(c.target_mean_degree); model,params,_=_params_for_state(state,n,k)
        print(f'[CELL] {state} N={n} kappa={k:g}',flush=True)
        for repeat in range(1,repeats+1):
            rec=synthetic_graph_record(cfg.cache.root,model,n,params,base_seed,repeat); records.append(rec)
            obs=_graph_observables(rec.graph); layers=enumerate_maximal_clique_layers(rec.graph,ArtifactStore(rec.cache_root,cfg.cache),progress=None)
            rows.append({'graph_index':len(rows),'cell_index':int(c.cell_index),'cell_label':str(c.cell_label),'mechanism_state':state,
                         'coarse_mechanism':_coarse_label(state),'model':model,'network_size':n,'target_mean_degree':k,'repeat':repeat,
                         'graph_hash':rec.graph_hash,'graph_seed':int(rec.source['graph_seed']),'num_maximal_cliques':sum(len(v) for v in layers.values()),
                         'maximal_q':max(layers) if layers else -1,**obs})
    gdf=pd.DataFrame(rows); gdf.to_csv(run/'tables'/'graph_index.csv',index=False)
    audit=gdf.groupby(['mechanism_state','network_size','target_mean_degree'],sort=False,as_index=False).agg(
        graph_count=('graph_hash','count'),realized_mean_degree_mean=('realized_mean_degree','mean'),realized_mean_degree_std=('realized_mean_degree','std'),
        average_clustering_mean=('average_clustering','mean'),largest_component_fraction_mean=('largest_component_fraction','mean'),
        num_maximal_cliques_mean=('num_maximal_cliques','mean'),maximal_q_mean=('maximal_q','mean'))
    audit.to_csv(run/'tables'/'structural_audit.csv',index=False)

    ids=gdf.graph_hash.astype(str).to_numpy(); labels=gdf.mechanism_state.astype(str).to_numpy(); ns=gdf.network_size.to_numpy(int); ks=gdf.target_mean_degree.to_numpy(float); cell_idx=gdf.cell_index.to_numpy(int)
    metrics=[]; runtime=[]; persist_graph=bool(exp.get('persist_graph_distance_matrices',True))
    protocols=['matched_N_k','cross_N_only','cross_k_only','cross_N_k','global']
    for method in methods:
        print(f'[METHOD] {method}',flush=True); reps=[]
        for gi,rec in enumerate(records):
            rep,elapsed,hit=manager.representation(method,rec); reps.append(rep); runtime.append({'method':method,'graph_index':gi,'seconds':elapsed,'cache_hit':hit})
        m=len(records); D=np.zeros((m,m),dtype=np.float32)
        for i in range(m):
            if i%250==0: print(f'  pairwise {method}: {i}/{m}',flush=True)
            for j in range(i+1,m):
                d=manager.distance(method,reps[i],reps[j],records[i].graph.number_of_nodes(),records[j].graph.number_of_nodes()); D[i,j]=D[j,i]=d
        if persist_graph: np.save(run/'arrays'/f'graph_distance_{method}.npy',D)
        mean,std,count=_cell_stats(D,cell_idx,len(cells)); np.save(run/'arrays'/f'cell_distance_{method}.npy',mean); np.save(run/'arrays'/f'cell_distance_std_{method}.npy',std); np.save(run/'arrays'/f'cell_pair_count_{method}.npy',count)
        for protocol in protocols:
            st,qdf=_retrieval_metrics_joint(D,ids,labels,ns,ks,protocol,tie_seed); metrics.append({'method':method,'protocol':protocol,**st})
            qdf.to_csv(run/'tables'/f'query_metrics_{protocol}_{method}.csv',index=False)
        del D
    pd.DataFrame(metrics).to_csv(run/'tables'/'metrics_all_methods.csv',index=False); pd.DataFrame(runtime).to_csv(run/'tables'/'runtime_per_graph.csv',index=False)
    plot_joint_discrimination(run)
    write_json(run/'run_manifest.json',{'schema':JOINT_SCHEMA,'network_sizes':sizes,'target_degrees':kappas,'repeats':repeats,'methods':methods,'num_cells':len(cells),'num_graphs':len(gdf),'method':asdict(cfg)})
    write_json(run/'RUN_COMPLETE.json',{'completed':True,'schema':JOINT_SCHEMA}); print(f'[DONE] {run}',flush=True); return run
