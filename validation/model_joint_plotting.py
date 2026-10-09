from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
from .style import DMEASURE_CMAP, FIG_DPI, METHOD_COLORS, SUPTITLE_SIZE, TICK_SIZE, TITLE_SIZE, apply_style


def _save(fig, base:Path):
    base.parent.mkdir(parents=True,exist_ok=True); fig.savefig(base.with_suffix('.png'),dpi=FIG_DPI,bbox_inches='tight',facecolor='white'); fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',facecolor='white'); plt.close(fig)


def _cluster_heatmap(D,labels,title,base):
    W=np.asarray(D,float).copy(); np.fill_diagonal(W,0.0); cond=squareform(W,checks=False); Z=linkage(cond,method='average'); Z=optimal_leaf_ordering(Z,cond) if len(W)>2 else Z; order=leaves_list(Z).astype(int)
    fig=plt.figure(figsize=(17,19)); gs=fig.add_gridspec(2,2,height_ratios=[3.0,12],width_ratios=[12,.5],hspace=.02,wspace=.1); axd=fig.add_subplot(gs[0,0]); ax=fig.add_subplot(gs[1,0]); cax=fig.add_subplot(gs[1,1])
    dd=dendrogram(Z,ax=axd,no_labels=True,color_threshold=None,above_threshold_color='0.15',link_color_func=lambda _:'0.15'); axd.set_xticks([]); axd.set_yticks([])
    for s in axd.spines.values(): s.set_visible(False)
    if not np.array_equal(np.asarray(dd['leaves']),order): raise RuntimeError('leaf order mismatch')
    M=np.asarray(D)[np.ix_(order,order)]; im=ax.imshow(M,cmap=DMEASURE_CMAP,aspect='equal',vmin=0); ol=[labels[i] for i in order]; fs=max(5,TICK_SIZE-10); ax.set_xticks(np.arange(len(order)),ol,rotation=90,fontsize=fs); ax.set_yticks(np.arange(len(order)),ol,fontsize=fs); fig.colorbar(im,cax=cax,label='Raw dissimilarity'); fig.suptitle(title,fontweight='bold',fontsize=TITLE_SIZE); _save(fig,base)
    return order


def _cmds(D):
    D=np.asarray(D,float).copy(); np.fill_diagonal(D,0); n=len(D); J=np.eye(n)-np.ones((n,n))/n; B=-.5*J@(D**2)@J; vals,vecs=np.linalg.eigh(B); idx=np.argsort(vals)[::-1]; vals=vals[idx]; vecs=vecs[:,idx]; pos=np.clip(vals[:2],0,None); X=vecs[:,:2]*np.sqrt(pos)[None,:]
    for k in range(2):
        p=int(np.argmax(np.abs(X[:,k])));
        if X[p,k]<0: X[:,k]*=-1
    return X


def plot_joint_discrimination(run_dir:str|Path):
    apply_style(); run=Path(run_dir); plots=run/'plots'; plots.mkdir(exist_ok=True); metrics=pd.read_csv(run/'tables'/'metrics_all_methods.csv'); cells=pd.read_csv(run/'tables'/'parameter_cells.csv').sort_values('cell_index')
    methods=[m for m in ['Hodge','DMeasure','NetLSD','Portrait','Count'] if m in set(metrics.method)]; protocols=['matched_N_k','cross_N_only','cross_k_only','cross_N_k','global']; labels=['Matched N & k','Cross N only','Cross k only','Cross N & k','Global']; core=['balanced_accuracy','macro_f1','same_class_auroc','mean_average_precision','ndcg_at_5']; corelab=['Bal. Acc.','Macro-F1','AUROC','mAP','NDCG@5']
    for protocol,ptitle in zip(protocols,labels):
        sub=metrics[metrics.protocol==protocol]; fig,axes=plt.subplots(len(methods),1,figsize=(15.5,max(8,3.6*len(methods))),sharex=True,squeeze=False); x=np.arange(len(core))
        for ax,m in zip(axes[:,0],methods):
            r=sub[sub.method==m];
            if r.empty: ax.axis('off'); continue
            vals=[float(r.iloc[0][c]) for c in core]; bars=ax.bar(x,vals,color=METHOD_COLORS.get(m)); ax.set_ylim(0,1.08); ax.set_ylabel(m)
            for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.02,f'{v:.3f}',ha='center',va='bottom',fontsize=max(10,TICK_SIZE-5))
        axes[-1,0].set_xticks(x,corelab,rotation=20,ha='right'); fig.suptitle(f'ER–WS–RGG joint N × mean-degree experiment: {ptitle}',fontweight='bold',fontsize=SUPTITLE_SIZE); fig.tight_layout(rect=(0,0,1,.98)); _save(fig,plots/f'core_metrics_{protocol}')
    clabel=cells.cell_label.astype(str).tolist()
    for m in methods:
        p=run/'arrays'/f'cell_distance_{m}.npy'
        if not p.exists(): continue
        D=np.load(p); order=_cluster_heatmap(D,clabel,f'{m}: joint mechanism × N × mean-degree raw dissimilarity',plots/f'joint_cell_heatmap_{m}'); pd.DataFrame({'order':np.arange(len(order)),'cell_index':order,'cell_label':[clabel[i] for i in order]}).to_csv(run/'tables'/f'joint_heatmap_order_{m}.csv',index=False)
        X=_cmds(D); out=cells[['cell_index','cell_label','mechanism_state','network_size','target_mean_degree']].copy(); out['mds_x']=X[:,0]; out['mds_y']=X[:,1]; out.to_csv(run/'tables'/f'mds_cell_{m}.csv',index=False)
        fig,ax=plt.subplots(figsize=(15.5,10.5)); states=out.mechanism_state.drop_duplicates().tolist(); cmap=plt.get_cmap('tab10'); colors={s:cmap(i%10) for i,s in enumerate(states)}; sizes=sorted(out.network_size.unique()); markers=['o','s','^','D','P']; mmap={n:markers[i%len(markers)] for i,n in enumerate(sizes)}
        kmin,kmax=float(out.target_mean_degree.min()),float(out.target_mean_degree.max())
        for r in out.itertuples(index=False):
            ms=90+110*((float(r.target_mean_degree)-kmin)/(max(kmax-kmin,1e-12))); ax.scatter(r.mds_x,r.mds_y,s=ms,marker=mmap[int(r.network_size)],c=[colors[str(r.mechanism_state)]],edgecolors='white',linewidths=.8,alpha=.85)
        ax.set_xlabel('Classical MDS dimension 1'); ax.set_ylabel('Classical MDS dimension 2'); ax.set_title(f'{m}: cell-level 2-D MDS\ncolor = mechanism, marker = N, size = mean degree',fontweight='bold'); ax.set_aspect('equal',adjustable='datalim'); fig.tight_layout(); _save(fig,plots/f'joint_mds_cell_{m}')
    audit=pd.read_csv(run/'tables'/'structural_audit.csv'); fig,ax=plt.subplots(figsize=(14,9))
    for (state,n),sub in audit.groupby(['mechanism_state','network_size'],sort=False):
        sub=sub.sort_values('target_mean_degree'); ax.plot(sub.target_mean_degree,sub.realized_mean_degree_mean,marker='o',linewidth=1.8,alpha=.7,label=f'{state}, N={n}')
    lo=min(audit.target_mean_degree); hi=max(audit.target_mean_degree); ax.plot([lo,hi],[lo,hi],'--',color='0.25',linewidth=2,label='target = realized'); ax.set_xlabel('Target mean degree'); ax.set_ylabel('Realized mean degree'); ax.set_title('Structural audit across the full N × mean-degree grid',fontweight='bold'); ax.legend(frameon=False,ncol=3,fontsize=max(8,TICK_SIZE-7)); fig.tight_layout(); _save(fig,plots/'structural_audit_realized_degree')
