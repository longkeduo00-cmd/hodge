from __future__ import annotations

"""MCID-style real-network validation for Hodge VNext.

Grouping/evaluation is explicit:
  * unsupervised average-linkage hierarchical clustering on the raw distance matrix;
  * optimal leaf ordering for the clustering heatmap;
  * Kruskal minimum spanning tree on the complete weighted distance graph;
  * a separate supervised domain-block heatmap only for descriptive inspection;
  * same-domain AUROC, 1-NN classification, retrieval, and cluster/MST metrics.
"""

from pathlib import Path
from typing import Any, Callable
import json
import math
import shutil
import time

import numpy as np
import pandas as pd

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.config import MethodConfig
from hodge_vnext.graph_io import canonicalize_graph, create_real_record
from hodge_vnext.utils import safe_token, stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .real_metrics import global_metrics, hierarchical_clustering, retrieval_metrics


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _run_stem(n: int, cfg: MethodConfig, benchmark_id: str | None = None) -> str:
    S="all" if cfg.max_clique_size is None else str(cfg.max_clique_size)
    Q="all" if cfg.max_hodge_order is None else str(cfg.max_hodge_order)
    token=stable_hash({"n":n,"benchmark_id":benchmark_id,"method":cfg.to_dict()},12)
    return f"n{n}_{cfg.solver}_S{S}_Q{Q}_qmin{cfg.min_layer_q}_norm-{cfg.normalization}_{token}"


def _resolve_run_dir(results_root: Path, n: int, cfg: MethodConfig, benchmark_id: str | None, run_name: str | None = None) -> Path:
    if run_name is None:
        return results_root / _run_stem(n,cfg,benchmark_id)
    name=str(run_name).strip()
    if name in {"", ".", ".."} or safe_token(name) != name:
        raise ValueError("run_name must be a safe single directory name")
    return results_root / name


def _clear_run_outputs(run: Path) -> None:
    for name in ("tables","arrays","plots"):
        p=run/name
        if p.exists(): shutil.rmtree(p)
    for name in ("RUN_COMPLETE.json","run_manifest.json"):
        p=run/name
        if p.exists(): p.unlink()


def _prepare_manifest(payload: dict[str,Any], config_path: Path) -> pd.DataFrame:
    exp=payload["real_validation"]
    manifest_path=resolve_path(exp["manifest"],config_path)
    df=pd.read_csv(manifest_path)
    required={"label","domain","path"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if "nodes_path" not in df.columns: df["nodes_path"]=""
    root=resolve_path(".",config_path)
    def res(v):
        if pd.isna(v) or str(v).strip()=="": return ""
        p=Path(str(v)); return str(p.resolve() if p.is_absolute() else (root/p).resolve())
    df["path"]=df["path"].map(res); df["nodes_path"]=df["nodes_path"].map(res)
    # Optional filtering follows MCID's real pilot pattern.  Older manifests
    # may contain only label/domain/path.  When a size filter is requested,
    # compute N/E once from the graph files instead of silently ignoring it.
    max_nodes=exp.get("max_nodes"); max_edges=exp.get("max_edges"); max_per_domain=exp.get("max_per_domain")
    if "N" not in df.columns and "nodes" in df.columns:
        df["N"] = df["nodes"]
    if "E" not in df.columns and "edges" in df.columns:
        df["E"] = df["edges"]
    need_size_metadata = (max_nodes is not None or max_edges is not None or max_per_domain is not None) and ("N" not in df.columns or "E" not in df.columns)
    if need_size_metadata:
        Ns=[]; Es=[]
        for _, row in df.iterrows():
            G = canonicalize_graph(row["path"], row["nodes_path"] or None)
            Ns.append(int(G.number_of_nodes())); Es.append(int(G.number_of_edges()))
        df["N"] = Ns; df["E"] = Es
    if max_nodes is not None: df=df[df.N<=int(max_nodes)]
    if max_edges is not None: df=df[df.E<=int(max_edges)]
    if max_per_domain is not None:
        df=df.sort_values(["domain","N","label"])
        df=df.groupby("domain",sort=False,as_index=False).head(int(max_per_domain))
    requested=exp.get("labels")
    if requested:
        order={x:i for i,x in enumerate(requested)}; df=df[df.label.isin(requested)].copy(); df["__o"]=df.label.map(order); df=df.sort_values("__o").drop(columns="__o")
    return df.reset_index(drop=True)


def run_real_domain_validation(
    payload: dict[str,Any],
    config_path: Path,
    *, force: bool=False,
    progress: Callable[[str],None]|None=None,
) -> Path:
    progress=progress or _progress
    cfg=method_from_payload(payload,config_path)
    exp=dict(payload["real_validation"])
    df=_prepare_manifest(payload,config_path)
    if len(df)<2: raise ValueError("real validation needs at least two networks")
    methods=MethodManager.validate_methods(list(exp.get("methods",[PRIMARY_METHOD])))
    manager=MethodManager.from_config(cfg,payload.get("baselines",{}))
    results_root=resolve_path(exp.get("results_root","results/validation/real"),config_path)
    benchmark_id=str(exp.get("benchmark_id","")).strip() or None
    run=_resolve_run_dir(results_root,len(df),cfg,benchmark_id,run_name=exp.get("run_name"))
    complete=run/"RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed run exists: {run}"); return run
    run.mkdir(parents=True,exist_ok=True)
    if force: _clear_run_outputs(run)
    for d in ("tables","arrays","plots"): (run/d).mkdir(parents=True,exist_ok=True)
    df.to_csv(run/"manifest_snapshot.csv",index=False,encoding="utf-8-sig")
    effective=json.loads(json.dumps(payload)); effective["cache_schema"]=CACHE_SCHEMA; effective["selected_networks"]=df[["label","domain","path","nodes_path"]].to_dict("records")
    write_json(run/"effective_config.json",effective)

    records=[]
    for _,row in df.iterrows():
        rec=create_real_record(row.path,cfg.cache.root,str(row.label),row.nodes_path or None)
        records.append(rec)
        progress(f"[REAL] {row.domain}/{row.label}: N={rec.graph.number_of_nodes()} E={rec.graph.number_of_edges()}")

    labels=df.label.astype(str).tolist(); domains=df.domain.astype(str).tolist(); n=len(records)
    metric_rows=[]; runtime_rows=[]; dataset_metric_frames=[]; domain_metric_rows=[]
    linkage_method=str(exp.get("linkage","average")); mst_permutations=int(exp.get("mst_permutations",5000)); seed=int(exp.get("seed",20260826))
    plot_options=dict(exp.get("plotting", {}))

    for method in methods:
        progress(f"[METHOD] {method}")
        reps=[]
        for i,rec in enumerate(records):
            t0=time.perf_counter(); rep,elapsed,hit=manager.representation(method,rec); reps.append(rep)
            runtime_rows.append({"method":method,"label":labels[i],"elapsed_sec":float(elapsed),"cache_hit":bool(hit),"wall_sec":float(time.perf_counter()-t0)})
        D=np.zeros((n,n),dtype=float)
        for i in range(n):
            for j in range(i+1,n):
                d=manager.distance(method,reps[i],reps[j],records[i].graph.number_of_nodes(),records[j].graph.number_of_nodes())
                D[i,j]=D[j,i]=d
            progress(f"[DIST] {method} row {i+1}/{n}")
        np.save(run/"arrays"/f"distance_matrix_{method}.npy",D)
        matrix_df=pd.DataFrame(D,index=labels,columns=labels)
        matrix_df.to_csv(run/"tables"/f"distance_matrix_{method}.csv",encoding="utf-8-sig")
        if method==PRIMARY_METHOD:
            np.save(run/"arrays"/"distance_matrix.npy",D)
            matrix_df.to_csv(run/"tables"/"distance_matrix.csv",encoding="utf-8-sig")

        clustering=hierarchical_clustering(D,len(set(domains)),method=linkage_method)
        retrieval,per_query,ranks=retrieval_metrics(D,labels,domains)
        cluster_metrics=global_metrics(D,domains,clustering.cluster_labels,mst_permutations,seed)
        summary={"method":method,**retrieval,**cluster_metrics}
        metric_rows.append(summary)
        per_query.insert(0,"method",method); ranks.insert(0,"method",method)
        dataset_metric_frames.append(per_query.copy())
        aggregate_cols=[
            "nearest_neighbor_correct","average_precision","ndcg_at_7",
            "precision_at_1","precision_at_3","precision_at_5","precision_at_7",
            "recall_at_1","recall_at_3","recall_at_5","recall_at_7",
            "first_relevant_rank","nearest_same_domain_distance",
            "mean_same_domain_distance","mean_other_domain_distance","query_between_within_ratio",
        ]
        for domain, grp in per_query.groupby("domain", sort=False):
            row={"method":method,"domain":str(domain),"num_queries":int(len(grp))}
            for col in aggregate_cols:
                row[col if col != "nearest_neighbor_correct" else "nn1_accuracy"] = float(grp[col].mean())
            row["mAP"]=row.pop("average_precision")
            domain_metric_rows.append(row)
        per_query.to_csv(run/"tables"/f"retrieval_per_query_{method}.csv",index=False,encoding="utf-8-sig")
        ranks.to_csv(run/"tables"/f"retrieval_ranking_{method}.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame({"label":labels,"domain":domains,"cluster":clustering.cluster_labels,"leaf_order_position":np.argsort(clustering.leaf_order)}).to_csv(run/"tables"/f"cluster_assignments_{method}.csv",index=False,encoding="utf-8-sig")

        from .real_plotting import plot_cluster_heatmap, plot_domain_block_heatmap, plot_mst, plot_within_domain_panels, plot_mds
        pdir=run/"plots"/method
        plot_cluster_heatmap(
            D, labels, clustering, pdir/"global_cluster_heatmap", f"{method}: all real networks",
            plot_options=plot_options,
        )
        plot_domain_block_heatmap(
            D, labels, domains, pdir/"global_domain_block_heatmap", f"{method}: all real networks",
            plot_options=plot_options,
        )
        plot_mst(D,labels,domains,pdir/"global_mst",f"{method}: all real networks",seed=seed)
        plot_within_domain_panels(
            D, labels, domains, pdir/"four_domain_heatmap_mst", f"{method}: within-domain structure",
            seed=seed, plot_options=plot_options,
        )
        plot_mds(D,labels,domains,pdir/"global_mds_2d",f"{method}: all real networks")

    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(run/"tables"/"real_metrics_summary.csv",index=False,encoding="utf-8-sig")
    dataset_metrics=pd.concat(dataset_metric_frames,ignore_index=True) if dataset_metric_frames else pd.DataFrame()
    domain_metrics=pd.DataFrame(domain_metric_rows)
    if not dataset_metrics.empty:
        dataset_metrics.to_csv(run/"tables"/"dataset_metrics_all_methods.csv",index=False,encoding="utf-8-sig")
    if not domain_metrics.empty:
        domain_metrics.to_csv(run/"tables"/"domain_metrics_all_methods.csv",index=False,encoding="utf-8-sig")
    runtime_df=pd.DataFrame(runtime_rows)
    runtime_df.to_csv(run/"tables"/"runtime_by_graph.csv",index=False,encoding="utf-8-sig")
    from .real_plotting import (
        plot_retrieval_summary, plot_core_metrics_comparison,
        plot_domain_metrics_comparison, plot_runtime_comparison,
    )
    hrow=metrics[metrics.method==PRIMARY_METHOD]
    if not hrow.empty: plot_retrieval_summary(hrow,run/"plots"/PRIMARY_METHOD/"retrieval_summary")
    if not metrics.empty:
        plot_core_metrics_comparison(metrics,run/"plots"/"core_metrics_all_methods",plot_options=plot_options)
    if not domain_metrics.empty:
        plot_domain_metrics_comparison(domain_metrics,run/"plots"/"core_metrics_by_domain",plot_options=plot_options)
    if not runtime_df.empty:
        plot_runtime_comparison(runtime_df,run/"plots"/"runtime_all_methods")

    write_json(run/"run_manifest.json",{
        "architecture":"MCID-style separation: method core / validation / benchmarks",
        "method_core_changed":False,"cache_schema":CACHE_SCHEMA,"benchmark_id":benchmark_id,"num_networks":n,
        "domains":list(dict.fromkeys(domains)),"methods":methods,
        "grouping":{
            "hierarchical_clustering":"average linkage on raw pairwise distance; optimal leaf ordering for heatmap",
            "minimum_spanning_tree":"Kruskal MST on complete weighted raw-distance graph",
            "domain_block_heatmap":"supervised descriptive display only; domain blocks fixed by truth, neighborhood sorted within each block",
        },
        "evaluation":["same-domain AUROC","1-NN Accuracy","Macro-F1","P@1/3/5/7","Recall@1/3/5/7","mAP","NDCG@7","ARI","NMI","silhouette","within/between","MST same-domain permutation","per-dataset first-relevant-rank and distance separation","per-domain retrieval summaries"],
        "method":cfg.to_dict(),
        "baseline_settings":{
            m: manager.baseline.method_settings(m)
            for m in methods if m != PRIMARY_METHOD
        },
    })
    write_json(complete,{"status":"complete","run_dir":str(run),"method_core_changed":False})
    progress(f"[COMPLETE] {run}")
    return run
