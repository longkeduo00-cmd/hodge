from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from itertools import permutations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.cluster.hierarchy import dendrogram

from .real_metrics import ClusterResult, classical_mds, minimum_spanning_tree, validate_distance_matrix
from .style import (
    DMEASURE_CMAP, FIG_DPI, LABEL_SIZE, LEGEND_SIZE, METHOD_COLORS, MST_EDGE_WIDTH, MST_LABEL_SIZE,
    MST_NODE_SIZE, MST_NODE_SIZE_GLOBAL, SUPTITLE_SIZE, TICK_SIZE, TITLE_SIZE, apply_style,
)

apply_style()


@dataclass(frozen=True)
class SeriationResult:
    order: np.ndarray
    adjacent_distances: np.ndarray
    path_cost: float
    method: str


MAX_REAL_PNG_PIXELS = 36_000_000
MIN_REAL_PNG_DPI = 150


def _effective_png_dpi(fig) -> int:
    """Bound raster export memory while keeping the vector PDF unchanged.

    The expanded 50-network figures can exceed 30 inches in width. Rendering
    those canvases at the global 300-DPI setting allocates hundreds of MB per
    figure and can make later figures swap heavily. The adaptive cap changes
    only PNG raster density; geometry, labels, distances and PDF vectors are
    identical.
    """
    width, height = (float(x) for x in fig.get_size_inches())
    area = max(width * height, 1e-12)
    requested_pixels = area * float(FIG_DPI) ** 2
    if requested_pixels <= MAX_REAL_PNG_PIXELS:
        return int(FIG_DPI)
    capped = int(math.floor(math.sqrt(MAX_REAL_PNG_PIXELS / area)))
    return max(MIN_REAL_PNG_DPI, min(int(FIG_DPI), capped))


def _save(fig, base: Path, *, tight_bbox: bool = True) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight_bbox else None
    png_dpi = _effective_png_dpi(fig)
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi, bbox_inches=bbox, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches=bbox, facecolor="white")
    # Large Matplotlib renderers can otherwise remain resident in cyclic object
    # graphs long enough to make the next 50-network figure swap heavily.
    fig.clear()
    plt.close(fig)
    gc.collect()


DEFAULT_REAL_PLOT_OPTIONS = {
    # 32-network figures are intended for PPT/paper export; labels must remain readable.
    "heatmap_label_fontsize": 15,
    "heatmap_x_label_fontsize": 14,
    "heatmap_rotation": 90,
    "cluster_figure_width": 26.0,
    "cluster_figure_height": 27.0,
    "cluster_dendrogram_height_ratio": 2.25,
    "cluster_heatmap_height_ratio": 8.0,
    "cluster_dendrogram_linewidth": 2.2,
    "cluster_linkage_tick_fontsize": 14,
    # Explicit visual leaf-to-cell connectors.  These do not alter subplot
    # geometry: each dendrogram leaf is connected to the horizontal centre of
    # its corresponding heatmap column at the matrix top boundary.
    "cluster_leaf_connector": True,
    "cluster_leaf_connector_linewidth": 1.25,
    "cluster_leaf_connector_alpha": 0.72,
    "cluster_leaf_connector_color": "0.42",
    "domain_block_figure_width": 25.0,
    "domain_block_figure_height": 23.0,
    "within_domain_label_fontsize": 14,
}


def _plot_options(overrides: dict | None = None) -> dict:
    out = dict(DEFAULT_REAL_PLOT_OPTIONS)
    if overrides:
        out.update(overrides)
    return out


def _domain_colors(domains: Sequence[str]) -> dict[str, tuple]:
    unique = list(dict.fromkeys(domains))
    cmap = plt.get_cmap("tab10")
    return {d: cmap(i % 10) for i, d in enumerate(unique)}


def _adjacent(D: np.ndarray, order: Sequence[int]) -> np.ndarray:
    return np.asarray([D[int(order[i]), int(order[i+1])] for i in range(len(order)-1)], dtype=float) if len(order) > 1 else np.empty(0)


def _path_cost(D: np.ndarray, order: Sequence[int]) -> float:
    return float(_adjacent(D, order).sum())


def _exact_neighborhood_order(D: np.ndarray) -> np.ndarray:
    """Exact shortest open Hamiltonian path via Held-Karp (O(n^2 2^n))."""
    D = validate_distance_matrix(D)
    n = len(D)
    if n <= 1:
        return np.arange(n)
    if n > 12:
        raise ValueError("exact neighborhood order is restricted to n<=12")
    dp: dict[tuple[int,int], tuple[float,int]] = {}
    for j in range(n):
        dp[(1<<j,j)] = (0.0,-1)
    for mask in range(1,1<<n):
        if mask & (mask-1) == 0:
            continue
        for j in range(n):
            if not (mask & (1<<j)):
                continue
            prev_mask = mask ^ (1<<j)
            best=(float("inf"),-1)
            for i in range(n):
                if not (prev_mask & (1<<i)):
                    continue
                prev=dp.get((prev_mask,i))
                if prev is None:
                    continue
                cost=prev[0]+float(D[i,j])
                if cost < best[0]-1e-15 or (abs(cost-best[0])<=1e-15 and (best[1]<0 or i<best[1])):
                    best=(cost,i)
            dp[(mask,j)] = best
    full=(1<<n)-1
    end=min(range(n),key=lambda j:(dp[(full,j)][0],j))
    path=[]; mask=full; j=end
    while j>=0:
        path.append(j); _,prev=dp[(mask,j)]; mask ^= 1<<j; j=prev
    path.reverse(); a=tuple(path); b=tuple(reversed(a))
    return np.asarray(min(a,b),dtype=int)

def _greedy_2opt(D: np.ndarray) -> np.ndarray:
    n = len(D)
    best = None; best_cost = float("inf")
    for start in range(n):
        remaining = set(range(n)); remaining.remove(start); order=[start]
        while remaining:
            u=order[-1]; v=min(remaining, key=lambda j:(D[u,j],j)); order.append(v); remaining.remove(v)
        improved=True
        while improved:
            improved=False
            for i in range(n-1):
                for j in range(i+1,n):
                    cand=order[:i]+list(reversed(order[i:j+1]))+order[j+1:]
                    if _path_cost(D,cand) < _path_cost(D,order)-1e-15:
                        order=cand; improved=True
        tup=tuple(order); rev=tuple(reversed(order)); tup=min(tup,rev)
        c=_path_cost(D,tup)
        if c < best_cost-1e-15 or (abs(c-best_cost)<=1e-15 and (best is None or tup<best)):
            best,best_cost=tup,c
    return np.asarray(best,dtype=int)


def neighborhood_order(D: np.ndarray) -> SeriationResult:
    D = validate_distance_matrix(D)
    order = _exact_neighborhood_order(D) if len(D) <= 10 else _greedy_2opt(D)
    a = _adjacent(D, order)
    return SeriationResult(order, a, float(a.sum()), "exact_shortest_path" if len(D)<=10 else "multistart_nn_2opt")


def domain_block_order(D: np.ndarray, domains: Sequence[str]) -> SeriationResult:
    D = validate_distance_matrix(D)
    order=[]
    for domain in dict.fromkeys(domains):
        idx=[i for i,d in enumerate(domains) if d==domain]
        local=neighborhood_order(D[np.ix_(idx,idx)]).order
        order.extend(idx[int(k)] for k in local)
    order=np.asarray(order,dtype=int); a=_adjacent(D,order)
    return SeriationResult(order,a,float(a.sum()),"true_domain_blocks_plus_within_block_neighborhood")


def _write_order(path: Path, order: Sequence[int], labels: Sequence[str], domains: Sequence[str] | None, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for pos,i in enumerate(order):
        rows.append({"position":pos+1,"original_index":int(i),"label":labels[int(i)],"domain":"" if domains is None else domains[int(i)],"ordering_method":method})
    pd.DataFrame(rows).to_csv(path,index=False,encoding="utf-8-sig")


def _heatmap(
    ax, D, labels, order, title, *,
    y_fontsize: int | None = None,
    x_fontsize: int | None = None,
    x_rotation: float = 90,
):
    D = validate_distance_matrix(D)
    order = np.asarray(order, dtype=int)
    shown = D[np.ix_(order, order)]
    im = ax.imshow(shown, cmap=DMEASURE_CMAP, aspect="equal", vmin=0, interpolation="nearest")
    labs = [labels[int(i)] for i in order]
    n = len(labs)
    if y_fontsize is None:
        y_fontsize = 15 if n <= 40 else (13 if n <= 64 else 11)
    if x_fontsize is None:
        x_fontsize = 14 if n <= 40 else (12 if n <= 64 else 10)
    # One tick per matrix cell.  This is intentionally not thinned: every data
    # set name corresponds to exactly one row/column in the 32-network figure.
    centers = np.arange(n, dtype=float)
    ax.set_xticks(centers, labs, rotation=x_rotation, ha="center", va="top", fontsize=x_fontsize)
    ax.set_yticks(centers, labs, fontsize=y_fontsize)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="x", which="major", pad=7, length=0)
    ax.tick_params(axis="y", which="major", pad=7, length=0)
    ax.set_title(title, fontweight="bold")
    return im


def plot_cluster_heatmap(
    D, labels, clustering: ClusterResult, base: Path, title: str,
    *, plot_options: dict | None = None,
) -> None:
    """Unsupervised average-linkage + optimal-leaf-order dendrogram heatmap.

    Geometry + display contract:
      * dendrogram and heatmap occupy the *same GridSpec column*;
      * scipy dendrogram leaf centres are at 5,15,...,10n-5;
      * heatmap cell centres are at 0,1,...,n-1;
      * both are mapped to identical fractional x positions (i+0.5)/n;
      * in addition, an explicit ConnectionPatch is drawn from every terminal
        dendrogram leaf to the centre-x of its corresponding heatmap column.

    The connector is a visible mapping cue, not a layout workaround: changing
    subplot spacing cannot change which leaf is connected to which matrix cell.
    """
    D = validate_distance_matrix(D)
    order = np.asarray(clustering.leaf_order, dtype=int)
    n = len(order)
    opt = _plot_options(plot_options)

    fig = plt.figure(figsize=(float(opt["cluster_figure_width"]), float(opt["cluster_figure_height"])))
    gs = GridSpec(
        2, 2,
        width_ratios=[1.0, 0.035],
        height_ratios=[float(opt["cluster_dendrogram_height_ratio"]), float(opt["cluster_heatmap_height_ratio"])],
        hspace=0.055, wspace=0.06,
        left=0.18, right=0.94, bottom=0.20, top=0.91,
        figure=fig,
    )
    da = fig.add_subplot(gs[0, 0])
    ha = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    blank = fig.add_subplot(gs[0, 1]); blank.axis("off")

    if n > 1:
        ddata = dendrogram(
            clustering.linkage_matrix,
            ax=da,
            no_labels=True,
            color_threshold=None,
            above_threshold_color="0.15",
            link_color_func=lambda _: "0.15",
        )
        dendro_order = np.asarray(ddata["leaves"], dtype=int)
        if not np.array_equal(dendro_order, order):
            raise RuntimeError(
                "Dendrogram leaf order disagrees with the optimal leaf order used by the heatmap."
            )
        for coll in da.collections:
            coll.set_linewidth(float(opt["cluster_dendrogram_linewidth"]))
        # Explicit leaf geometry.  scipy's terminal x positions are 5+10*i.
        da.set_xlim(0.0, 10.0 * n)
        da.set_xticks(5.0 + 10.0 * np.arange(n))
        da.set_xticklabels([])
        da.tick_params(axis="x", length=5, width=1.2, direction="out")
    else:
        da.set_xlim(0.0, 10.0)
        da.set_xticks([5.0]); da.set_xticklabels([])

    da.set_ylabel("Linkage distance", fontsize=LABEL_SIZE)
    da.tick_params(axis="y", labelsize=int(opt["cluster_linkage_tick_fontsize"]))
    da.spines["top"].set_visible(False)
    da.spines["right"].set_visible(False)
    da.spines["bottom"].set_visible(False)

    im = _heatmap(
        ha, D, labels, order, "",
        y_fontsize=int(opt["heatmap_label_fontsize"]),
        x_fontsize=int(opt["heatmap_x_label_fontsize"]),
        x_rotation=float(opt["heatmap_rotation"]),
    )

    # Make the dendrogram-to-matrix correspondence explicit rather than merely
    # relying on shared subplot geometry.  For leaf position ``pos`` scipy uses
    # x = 5 + 10*pos, while imshow uses column centre x = pos.  The heatmap
    # target y=-0.5 is the top matrix boundary, exactly above that column centre.
    # ConnectionPatch transforms both points through their own data transforms,
    # so the visible line remains exact even if figure size, margins, title
    # spacing, font size, or colorbar layout changes later.
    if bool(opt.get("cluster_leaf_connector", True)):
        connector_lw = float(opt.get("cluster_leaf_connector_linewidth", 1.25))
        connector_alpha = float(opt.get("cluster_leaf_connector_alpha", 0.72))
        connector_color = opt.get("cluster_leaf_connector_color", "0.42")
        for pos in range(n):
            leaf_x = 5.0 + 10.0 * pos
            cell_x = float(pos)
            con = ConnectionPatch(
                xyA=(leaf_x, 0.0), coordsA=da.transData,
                xyB=(cell_x, -0.5), coordsB=ha.transData,
                axesA=da, axesB=ha,
                arrowstyle="-", shrinkA=0.0, shrinkB=0.0,
                linewidth=connector_lw, color=connector_color,
                alpha=connector_alpha, zorder=8, clip_on=False,
            )
            fig.add_artist(con)

    # Keep the connector corridor visually clean.  The heatmap descriptor is
    # placed above the dendrogram rather than in the leaf-to-cell bridge gap.
    fig.text(
        0.56, 0.925, "Raw pairwise dissimilarity — optimal leaf order",
        ha="center", va="bottom", fontsize=TITLE_SIZE, fontweight="bold",
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Raw dissimilarity", fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    fig.suptitle(
        title + " — average-linkage hierarchical clustering",
        fontsize=SUPTITLE_SIZE, fontweight="bold", y=.972,
    )
    _write_order(
        base.parent / (base.name + "_order.csv"), order, labels, None,
        "average_linkage_optimal_leaf_order",
    )
    _save(fig, base)


def plot_domain_block_heatmap(
    D, labels, domains, base: Path, title: str, *, plot_options: dict | None = None
) -> None:
    """Supervised descriptive display: fixed domain blocks + local ordering."""
    D = validate_distance_matrix(D)
    result = domain_block_order(D, domains)
    opt = _plot_options(plot_options)
    display = [f"{domains[i][:4]} | {labels[i]}" for i in range(len(labels))]
    fig, ax = plt.subplots(
        figsize=(float(opt["domain_block_figure_width"]), float(opt["domain_block_figure_height"]))
    )
    im = _heatmap(
        ax, D, display, result.order, title + " — true-domain blocks (descriptive only)",
        y_fontsize=int(opt["heatmap_label_fontsize"]),
        x_fontsize=int(opt["heatmap_x_label_fontsize"]),
        x_rotation=float(opt["heatmap_rotation"]),
    )
    ordered = [domains[int(i)] for i in result.order]
    for i in range(1, len(ordered)):
        if ordered[i] != ordered[i - 1]:
            ax.axhline(i - .5, color="white", linewidth=3.2)
            ax.axvline(i - .5, color="white", linewidth=3.2)
    cb = fig.colorbar(im, ax=ax, fraction=.032, pad=.025)
    cb.set_label("Raw dissimilarity", fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    _write_order(base.parent / (base.name + "_order.csv"), result.order, labels, domains, result.method)
    _save(fig, base)


def plot_mst(D, labels, domains, base: Path, title: str, seed: int=20260826) -> None:
    D=validate_distance_matrix(D); T=minimum_spanning_tree(D); colors=_domain_colors(domains)
    fig,ax=plt.subplots(figsize=(17,14)); pos=nx.spring_layout(T,seed=int(seed),weight=None)
    nx.draw_networkx_nodes(T,pos,node_size=MST_NODE_SIZE_GLOBAL,node_color=[colors[domains[i]] for i in range(len(labels))],edgecolors="0.15",ax=ax)
    nx.draw_networkx_edges(T,pos,width=MST_EDGE_WIDTH,edge_color="0.25",ax=ax)
    nx.draw_networkx_labels(T,pos,labels={i:labels[i] for i in range(len(labels))},font_size=MST_LABEL_SIZE,ax=ax)
    handles=[plt.Line2D([0],[0],marker="o",linestyle="",markerfacecolor=colors[d],markeredgecolor="0.15",label=d) for d in colors]
    ax.legend(handles=handles,frameon=False,loc="best"); ax.set_title(title+" — minimum spanning tree",fontweight="bold"); ax.set_axis_off()
    pd.DataFrame([{"u":labels[u],"v":labels[v],"domain_u":domains[u],"domain_v":domains[v],"distance":data["weight"]} for u,v,data in T.edges(data=True)]).to_csv(base.parent/(base.name+"_edges.csv"),index=False,encoding="utf-8-sig")
    _save(fig,base)


def plot_within_domain_panels(
    D, labels, domains, base: Path, title: str, seed: int=20260826, *, plot_options: dict | None = None
) -> None:
    """Within-domain descriptive heatmap + MST panels for any domain count."""
    opt = _plot_options(plot_options)
    unique=list(dict.fromkeys(domains))
    if not unique:
        return
    n_domains=len(unique)
    fig_width=max(18.0, 6.5*n_domains)
    fig=plt.figure(figsize=(fig_width,15)); outer=GridSpec(2,n_domains,height_ratios=[1.15,.9],wspace=.28,hspace=.25,figure=fig)
    last_im=None
    for col,domain in enumerate(unique):
        idx=[i for i,d in enumerate(domains) if d==domain]; local=D[np.ix_(idx,idx)]; local_labels=[labels[i] for i in idx]
        result=neighborhood_order(local)
        ax=fig.add_subplot(outer[0,col]); last_im=_heatmap(
            ax, local, local_labels, result.order,
            f"{chr(65+col)}  {domain}\nwithin-domain neighborhood order",
            y_fontsize=int(opt["within_domain_label_fontsize"]),
            x_fontsize=max(10, int(opt["within_domain_label_fontsize"]) - 1), x_rotation=60,
        )
        tree_ax=fig.add_subplot(outer[1,col]); T=minimum_spanning_tree(local); pos=nx.spring_layout(T,seed=seed+col,weight=None)
        nx.draw_networkx_nodes(T,pos,node_size=MST_NODE_SIZE,node_color="0.86",edgecolors="0.15",ax=tree_ax)
        nx.draw_networkx_edges(T,pos,width=MST_EDGE_WIDTH,edge_color="0.25",ax=tree_ax)
        nx.draw_networkx_labels(
            T, pos, labels={i:local_labels[i] for i in range(len(local_labels))},
            font_size=max(10, int(opt["within_domain_label_fontsize"]) - 2), ax=tree_ax
        )
        tree_ax.set_title("Minimum spanning tree",fontweight="bold"); tree_ax.set_axis_off()
        _write_order(base.parent/f"{base.name}_{domain}_order.csv",result.order,local_labels,[domain]*len(local_labels),result.method)
    if last_im is not None:
        cax=fig.add_axes([.93,.55,.010,.3]); cb=fig.colorbar(last_im,cax=cax); cb.set_label("Raw dissimilarity")
    # On the five-domain 50-network canvas, Matplotlib's PDF backend can spend
    # minutes recomputing a tight bounding box around all heatmap/MST labels.
    # The full fixed canvas already contains every artist, so omitting the tight
    # bbox here is rendering-equivalent and avoids that pathological export cost.
    fig.subplots_adjust(left=.035,right=.91,bottom=.07,top=.93)
    fig.suptitle(title,fontsize=SUPTITLE_SIZE,fontweight="bold",y=.985)
    _save(fig,base,tight_bbox=False)


def plot_mds(D, labels, domains, base: Path, title: str) -> None:
    coords,_=classical_mds(D,2); colors=_domain_colors(domains)
    fig,ax=plt.subplots(figsize=(13,10))
    for d in dict.fromkeys(domains):
        idx=[i for i,x in enumerate(domains) if x==d]
        ax.scatter(coords[idx,0],coords[idx,1],s=130,label=d,color=colors[d],edgecolors="0.15")
    for i,label in enumerate(labels):
        ax.annotate(label,(coords[i,0],coords[i,1]),xytext=(5,4),textcoords="offset points",fontsize=10)
    ax.set_xlabel("Classical MDS dimension 1"); ax.set_ylabel("Classical MDS dimension 2")
    ax.set_title(title+" — 2D distance map",fontweight="bold"); ax.legend(frameon=False)
    _save(fig,base)


def plot_retrieval_summary(metrics: pd.DataFrame, base: Path) -> None:
    cols=["auroc_same_domain","nn1_accuracy","nn1_macro_f1","precision_at_1","mAP","ndcg_at_7"]
    vals=[float(metrics.iloc[0][c]) for c in cols]; labs=["AUROC","1-NN Acc.","Macro-F1","P@1","mAP","NDCG@7"]
    fig,ax=plt.subplots(figsize=(12,8)); bars=ax.bar(labs,vals); ax.set_ylim(0,1); ax.set_ylabel("Score (higher is better)")
    ax.set_title("Real-network domain/retrieval evaluation",fontweight="bold")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.02,f"{v:.3f}",ha="center",fontsize=14)
    _save(fig,base)


def plot_core_metrics_comparison(metrics: pd.DataFrame, base: Path, *, plot_options: dict | None = None) -> None:
    """All formal methods on the same bounded [0,1] core metrics."""
    opt=_plot_options(plot_options)
    columns=["auroc_same_domain","nn1_accuracy","nn1_macro_f1","precision_at_1","mAP","ndcg_at_7"]
    labels=["AUROC","1-NN Acc.","Macro-F1","P@1","mAP","NDCG@7"]
    methods=metrics["method"].astype(str).tolist()
    x=np.arange(len(columns),dtype=float); width=0.8/max(1,len(methods))
    fig,ax=plt.subplots(figsize=(float(opt.get("core_metrics_figure_width",18.0)),float(opt.get("core_metrics_figure_height",10.5))))
    for mi,method in enumerate(methods):
        row=metrics.iloc[mi]
        values=[float(row[c]) for c in columns]
        offset=(mi-(len(methods)-1)/2)*width
        bars=ax.bar(x+offset,values,width=width*0.92,label=method,color=METHOD_COLORS.get(method))
        for b,v in zip(bars,values):
            ax.text(b.get_x()+b.get_width()/2,v+0.012,f"{v:.2f}",ha="center",va="bottom",fontsize=10,rotation=90)
    ax.set_xticks(x,labels); ax.set_ylim(0,1.08); ax.set_ylabel("Score (higher is better)")
    ax.set_title("Expanded real-network benchmark — core domain/retrieval metrics",fontweight="bold")
    ax.legend(frameon=False,ncol=min(len(methods),5),loc="upper center")
    fig.tight_layout(); _save(fig,base)


def plot_domain_metrics_comparison(domain_metrics: pd.DataFrame, base: Path, *, plot_options: dict | None = None) -> None:
    """Per-domain core retrieval metrics, retaining the same method palette."""
    opt=_plot_options(plot_options)
    metrics=[("nn1_accuracy","1-NN accuracy"),("mAP","mAP"),("ndcg_at_7","NDCG@7")]
    domains=list(dict.fromkeys(domain_metrics["domain"].astype(str).tolist()))
    methods=list(dict.fromkeys(domain_metrics["method"].astype(str).tolist()))
    fig,axes=plt.subplots(len(metrics),1,figsize=(float(opt.get("domain_metrics_figure_width",24.0)),float(opt.get("domain_metrics_figure_height",18.0))),sharex=True)
    if len(metrics)==1: axes=[axes]
    x=np.arange(len(domains),dtype=float); width=0.82/max(1,len(methods))
    for ax,(col,title) in zip(axes,metrics):
        for mi,method in enumerate(methods):
            sub=domain_metrics[domain_metrics.method==method].set_index("domain")
            vals=[float(sub.loc[d,col]) if d in sub.index else np.nan for d in domains]
            offset=(mi-(len(methods)-1)/2)*width
            ax.bar(x+offset,vals,width=width*0.92,label=method,color=METHOD_COLORS.get(method))
        ax.set_ylim(0,1.02); ax.set_ylabel(title); ax.set_title(title,fontweight="bold")
    axes[-1].set_xticks(x,domains,rotation=22,ha="right")
    axes[0].legend(frameon=False,ncol=min(len(methods),5),loc="upper center")
    fig.suptitle("Expanded real-network benchmark — per-domain retrieval",fontsize=SUPTITLE_SIZE,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.96]); _save(fig,base)


def plot_runtime_comparison(runtime: pd.DataFrame, base: Path) -> None:
    """Median uncached representation runtime per method.

    ``--force`` intentionally reuses legal computation caches.  A rerun can
    therefore contain only cache hits, whose ``elapsed_sec`` is zero by design.
    Those zeros are not uncached computational costs and must not be placed on a
    logarithmic axis.  Such methods are labeled ``cache hits only`` instead.
    """
    rows=[]
    for method,grp in runtime.groupby("method",sort=False):
        misses=grp[~grp["cache_hit"].astype(bool)]
        if len(misses):
            median=float(misses["elapsed_sec"].median())
            total=float(misses["elapsed_sec"].sum())
            n_uncached=int(len(misses))
        else:
            median=float("nan")
            total=float("nan")
            n_uncached=0
        rows.append((str(method),median,total,n_uncached))

    methods=[r[0] for r in rows]
    med=np.asarray([r[1] for r in rows],dtype=float)
    x=np.arange(len(methods),dtype=float)
    valid=np.isfinite(med) & (med>0)
    fig,ax=plt.subplots(figsize=(12,8))
    if np.any(valid):
        for i,(method,value) in enumerate(zip(methods,med)):
            if np.isfinite(value) and value>0:
                bar=ax.bar(i,value,color=METHOD_COLORS.get(method),width=.8)[0]
                ax.text(bar.get_x()+bar.get_width()/2,value*1.08,f"{value:.3g}",ha="center",fontsize=12)
        ax.set_yscale("log")
        ymin=float(np.min(med[valid]))
        for i,is_valid in enumerate(valid):
            if not is_valid:
                ax.text(i,ymin,"cache\nhits only",ha="center",va="bottom",fontsize=11)
        ax.set_ylabel("Median uncached representation time per graph (s, log scale)")
    else:
        ax.set_ylim(0,1)
        ax.text(.5,.52,"No uncached representation timings in this run",transform=ax.transAxes,ha="center",va="center",fontsize=16)
        ax.set_ylabel("Uncached representation time")
    ax.set_xticks(x,methods)
    ax.set_title("Real-network computational cost",fontweight="bold")
    fig.tight_layout(); _save(fig,base)

