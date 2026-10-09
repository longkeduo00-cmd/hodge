from __future__ import annotations

"""ER/WS/RGG mechanism comparison while varying network size N.

The target mean degree is held fixed.  ER p and RGG radius are recomputed for
all N; WS ring degree remains fixed.  Every (state, N, repeat) graph is sampled
independently.  This isolates network-size sensitivity from the earlier
mean-degree scan.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from scipy.stats import spearmanr

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers
from hodge_vnext.models import synthetic_graph_record
from hodge_vnext.utils import stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import DEFAULT_METHODS, MethodManager
from .model_discrimination import FINE_STATE_ORDER, _coarse_label, _graph_observables, _params_for_state, _cell_distance_matrix
from .model_size_plotting import plot_size_discrimination

SIZE_SCHEMA = 'er-ws-rgg-size-discrimination-v1'


def _tie_key(a: str, b: str, seed: int) -> str:
    raw = json.dumps([int(seed), str(a), str(b)], separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def build_size_cells(network_sizes: list[int], target_mean_degree: float, ws_betas: list[float]) -> pd.DataFrame:
    sizes = [int(x) for x in network_sizes]
    if len(sizes) < 2 or len(set(sizes)) != len(sizes) or any(b <= a for a,b in zip(sizes[:-1], sizes[1:])):
        raise ValueError('network_sizes must contain at least two unique strictly increasing values')
    states = ['ER','RGG'] + [f'WS-beta{token}' for token in ('1e-3','1e-2','1e-1','1')]
    expected_betas = [1e-3,1e-2,1e-1,1.0]
    if len(ws_betas) != 4 or any(abs(float(a)-b)>1e-14 for a,b in zip(ws_betas, expected_betas)):
        raise ValueError('size protocol currently requires ws_betas=[0.001,0.01,0.1,1.0]')
    rows=[]; cell=0
    for state in states:
        for n in sizes:
            model, params, derived = _params_for_state(state, n, float(target_mean_degree))
            rows.append({
                'cell_index': cell,
                'cell_label': f'{state}-N{n}',
                'mechanism_state': state,
                'coarse_mechanism': _coarse_label(state),
                'model': model,
                'network_size': n,
                'target_mean_degree': float(target_mean_degree),
                'er_p': derived.get('er_p'),
                'ws_degree': derived.get('ws_degree'),
                'ws_beta': derived.get('ws_beta'),
                'rgg_radius': derived.get('rgg_radius'),
                'generator_params_json': json.dumps(params, sort_keys=True, separators=(',',':')),
            })
            cell += 1
    return pd.DataFrame(rows)


def _eligible_mask(sizes: np.ndarray, i: int, protocol: str) -> np.ndarray:
    mask = np.arange(len(sizes)) != i
    if protocol == 'matched_size':
        mask &= sizes == sizes[i]
    elif protocol == 'cross_size':
        mask &= sizes != sizes[i]
    elif protocol == 'global':
        pass
    else:
        raise ValueError(protocol)
    return mask


def _retrieval_metrics(D: np.ndarray, graph_ids: np.ndarray, labels: np.ndarray, sizes: np.ndarray, protocol: str, tie_seed: int):
    labels = np.asarray(labels, dtype=str)
    sizes = np.asarray(sizes, dtype=int)
    pair_y=[]; pair_score=[]; preds=[]; qrows=[]
    n=len(labels)
    for i in range(n):
        for j in range(i+1,n):
            eligible = protocol == 'global' or (protocol == 'matched_size' and sizes[i] == sizes[j]) or (protocol == 'cross_size' and sizes[i] != sizes[j])
            if eligible:
                pair_y.append(int(labels[i] == labels[j])); pair_score.append(float(-D[i,j]))
    for i in range(n):
        cand = np.flatnonzero(_eligible_mask(sizes,i,protocol)).tolist()
        if not cand:
            raise ValueError(f'no candidates for query {i} under {protocol}')
        ranked = sorted(cand, key=lambda j:(float(D[i,j]), _tie_key(str(graph_ids[i]),str(graph_ids[j]),tie_seed),str(graph_ids[j])))
        pred=str(labels[ranked[0]]); preds.append(pred)
        rel=np.asarray([labels[j]==labels[i] for j in ranked],dtype=int)
        total=int(rel.sum()); cum=np.cumsum(rel); prec=cum/np.arange(1,len(rel)+1)
        ap=float(np.dot(prec,rel)/total) if total else float('nan')
        k5=min(5,len(rel)); disc=1.0/np.log2(np.arange(1,k5+1)+1.0); ideal=min(5,total)
        idcg=float(np.sum(1.0/np.log2(np.arange(1,ideal+1)+1.0))) if ideal else 0.0
        ndcg=float(np.dot(rel[:k5],disc)/idcg) if idcg else float('nan')
        qrows.append({'query_index':i,'graph_id':str(graph_ids[i]),'true_label':str(labels[i]),'predicted_label':pred,
                      'query_size':int(sizes[i]),'nearest_graph_id':str(graph_ids[ranked[0]]),'nearest_size':int(sizes[ranked[0]]),
                      'nearest_distance':float(D[i,ranked[0]]),'gallery_size':len(ranked),'relevant_total':total,
                      'average_precision':ap,'ndcg_at_5':ndcg})
    preds=np.asarray(preds,dtype=str); qdf=pd.DataFrame(qrows)
    stats={
        'n_queries':n,
        'accuracy':float(np.mean(preds==labels)),
        'balanced_accuracy':float(balanced_accuracy_score(labels,preds)),
        'macro_f1':float(f1_score(labels,preds,average='macro',zero_division=0)),
        'same_class_auroc':float(roc_auc_score(pair_y,pair_score)) if len(set(pair_y))==2 else float('nan'),
        'mean_average_precision':float(np.nanmean(qdf.average_precision)),
        'ndcg_at_5':float(np.nanmean(qdf.ndcg_at_5)),
    }
    classes=sorted(set(labels.tolist()))
    confusion=pd.crosstab(pd.Series(labels,name='true_label'),pd.Series(preds,name='predicted_label')).reindex(index=classes,columns=classes,fill_value=0)
    return stats,qdf,confusion


def _size_sensitivity(D: np.ndarray, states: np.ndarray, sizes: np.ndarray, method: str) -> list[dict[str,Any]]:
    rows=[]
    unique_sizes=sorted(set(int(x) for x in sizes))
    for state in FINE_STATE_ORDER:
        for ia, na in enumerate(unique_sizes):
            idx_a=np.flatnonzero((states==state)&(sizes==na))
            noise_a=D[np.ix_(idx_a,idx_a)][np.triu_indices(len(idx_a),1)]
            for nb in unique_sizes[ia+1:]:
                idx_b=np.flatnonzero((states==state)&(sizes==nb))
                noise_b=D[np.ix_(idx_b,idx_b)][np.triu_indices(len(idx_b),1)]
                between=D[np.ix_(idx_a,idx_b)].ravel()
                noise=0.5*(float(noise_a.mean())+float(noise_b.mean()))
                rows.append({'method':method,'mechanism_state':state,'size_a':na,'size_b':nb,
                             'adjacent_size_pair':bool(unique_sizes.index(nb)==ia+1),
                             'log_size_ratio':float(abs(math.log(nb/na))),
                             'between_mean':float(between.mean()),'within_noise_mean':noise,
                             'normalized_excess':float((between.mean()-noise)/(noise+1e-15)) if noise>0 else float('nan'),
                             'between_pair_count':int(len(between))})
    return rows


def _run_stem(exp: dict[str,Any], method: dict[str,Any], baselines: dict[str,Any]) -> str:
    identity={'schema':SIZE_SCHEMA,'network_sizes':exp['network_sizes'],'target_mean_degree':exp['target_mean_degree'],
              'repeats':exp['repeats'],'base_seed':exp.get('base_seed',22),'ws_betas':exp['ws_betas'],'method':method,'baselines':baselines}
    sizes='-'.join(str(int(x)) for x in exp['network_sizes'])
    return f"size_N{sizes}_k{float(exp['target_mean_degree']):g}_{stable_hash(identity,10)}"


def run_model_size_discrimination(payload: dict[str,Any], config_path: Path, *, force: bool=False) -> Path:
    cfg=method_from_payload(payload,config_path); exp=dict(payload['validation'])
    sizes=[int(x) for x in exp['network_sizes']]; kappa=float(exp['target_mean_degree']); repeats=int(exp['repeats'])
    if repeats<2: raise ValueError('repeats must be >=2')
    methods=MethodManager.validate_methods(list(exp.get('methods',DEFAULT_METHODS)))
    manager=MethodManager.from_config(cfg,payload.get('baselines',{})); tie_seed=int(exp.get('tie_seed',20260921)); base_seed=int(exp.get('base_seed',22))
    cells=build_size_cells(sizes,kappa,[float(x) for x in exp['ws_betas']])
    root=resolve_path(exp.get('results_root','results/validation/model_size_discrimination'),config_path)
    run=root/_run_stem(exp,payload['method'],payload.get('baselines',{})); complete=run/'RUN_COMPLETE.json'
    if complete.exists() and not force:
        plot_size_discrimination(run); return run
    run.mkdir(parents=True,exist_ok=True)
    if force:
        for name in ('tables','arrays','plots'):
            p=run/name
            if p.exists(): shutil.rmtree(p)
    for name in ('tables','arrays','plots'): (run/name).mkdir(parents=True,exist_ok=True)
    effective=json.loads(json.dumps(payload)); effective['resolved_protocol']={
        'schema':SIZE_SCHEMA,'scientific_axis':'network size N','held_fixed':'target mean degree kappa',
        'ER':'p=kappa/(N-1)','WS':'degree=kappa; beta fixed by mechanism state','RGG':'radius recomputed at every N from finite-square expected-degree equation',
        'sampling':'every (state,N,repeat) is independent'}
    effective['cache_schema']=CACHE_SCHEMA; write_json(run/'effective_config.json',effective); cells.to_csv(run/'tables'/'parameter_cells.csv',index=False)

    records=[]; graph_rows=[]
    for c in cells.itertuples(index=False):
        state=str(c.mechanism_state); n=int(c.network_size); model,params,_=_params_for_state(state,n,kappa)
        print(f'[CELL] {state} N={n} kappa={kappa:g}',flush=True)
        for repeat in range(1,repeats+1):
            rec=synthetic_graph_record(cfg.cache.root,model,n,params,base_seed,repeat); records.append(rec)
            obs=_graph_observables(rec.graph); layers=enumerate_maximal_clique_layers(rec.graph,ArtifactStore(rec.cache_root,cfg.cache),progress=None)
            graph_rows.append({'graph_index':len(graph_rows),'cell_index':int(c.cell_index),'cell_label':str(c.cell_label),'mechanism_state':state,
                               'coarse_mechanism':_coarse_label(state),'model':model,'network_size':n,'target_mean_degree':kappa,'repeat':repeat,
                               'graph_hash':rec.graph_hash,'graph_seed':int(rec.source['graph_seed']),'num_maximal_cliques':sum(len(v) for v in layers.values()),
                               'maximal_q':max(layers) if layers else -1,**obs})
    graph_df=pd.DataFrame(graph_rows); graph_df.to_csv(run/'tables'/'graph_index.csv',index=False)
    audit=graph_df.groupby(['mechanism_state','network_size'],sort=False,as_index=False).agg(
        graph_count=('graph_hash','count'),target_mean_degree=('target_mean_degree','first'),realized_mean_degree_mean=('realized_mean_degree','mean'),
        realized_mean_degree_std=('realized_mean_degree','std'),average_clustering_mean=('average_clustering','mean'),average_clustering_std=('average_clustering','std'),
        largest_component_fraction_mean=('largest_component_fraction','mean'),num_maximal_cliques_mean=('num_maximal_cliques','mean'),maximal_q_mean=('maximal_q','mean'))
    audit.to_csv(run/'tables'/'structural_audit.csv',index=False)

    states=graph_df.mechanism_state.astype(str).to_numpy(); coarse=graph_df.coarse_mechanism.astype(str).to_numpy(); ns=graph_df.network_size.to_numpy(int); ids=graph_df.graph_hash.astype(str).to_numpy(); cell_idx=graph_df.cell_index.to_numpy(int)
    metrics=[]; runtime=[]; sens=[]
    for method in methods:
        print(f'[METHOD] {method}',flush=True); reps=[]
        for gi,rec in enumerate(records):
            rep,elapsed,hit=manager.representation(method,rec); reps.append(rep); runtime.append({'method':method,'graph_index':gi,'seconds':elapsed,'cache_hit':hit})
        m=len(records); D=np.zeros((m,m),float)
        for i in range(m):
            for j in range(i+1,m):
                D[i,j]=D[j,i]=manager.distance(method,reps[i],reps[j],records[i].graph.number_of_nodes(),records[j].graph.number_of_nodes())
        np.save(run/'arrays'/f'graph_distance_{method}.npy',D); cell_D,counts=_cell_distance_matrix(D,cell_idx,len(cells)); np.save(run/'arrays'/f'cell_distance_{method}.npy',cell_D); np.save(run/'arrays'/f'cell_pair_count_{method}.npy',counts)
        sens.extend(_size_sensitivity(D,states,ns,method))
        for level,labels in [('fine',states),('coarse',coarse)]:
            for protocol in ['matched_size','cross_size','global']:
                stats,qdf,conf=_retrieval_metrics(D,ids,labels,ns,protocol,tie_seed); metrics.append({'method':method,'label_level':level,'protocol':protocol,**stats})
                qdf.to_csv(run/'tables'/f'query_metrics_{protocol}_{level}_{method}.csv',index=False)
                if level=='fine' and protocol in {'matched_size','cross_size'}: conf.to_csv(run/'tables'/f'confusion_{protocol}_{method}.csv')
    pd.DataFrame(metrics).to_csv(run/'tables'/'metrics_all_methods.csv',index=False); pd.DataFrame(runtime).to_csv(run/'tables'/'runtime_per_graph.csv',index=False)
    sens_df=pd.DataFrame(sens); sens_df.to_csv(run/'tables'/'size_sensitivity.csv',index=False)
    corr_rows=[]
    for (method,state),sub in sens_df.groupby(['method','mechanism_state'],sort=False):
        if len(sub)>1 and not np.allclose(sub.log_size_ratio,sub.log_size_ratio.iloc[0]) and not np.allclose(sub.between_mean,sub.between_mean.iloc[0]):
            rho,p=spearmanr(sub.log_size_ratio,sub.between_mean)
        else: rho,p=float('nan'),float('nan')
        corr_rows.append({'method':method,'mechanism_state':state,'spearman_size_gap_vs_distance':rho,'p_value':p})
    pd.DataFrame(corr_rows).to_csv(run/'tables'/'size_gap_distance_spearman.csv',index=False)
    plot_size_discrimination(run)
    write_json(run/'run_manifest.json',{'schema':SIZE_SCHEMA,'network_sizes':sizes,'target_mean_degree':kappa,'repeats':repeats,'methods':methods,'method':asdict(cfg),'num_graphs':len(graph_df)})
    write_json(run/'RUN_COMPLETE.json',{'completed':True,'schema':SIZE_SCHEMA}); print(f'[DONE] {run}',flush=True); return run
