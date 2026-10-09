from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import pickle

import networkx as nx
import numpy as np

from .cache import ArtifactStore
from .cliques import enumerate_maximal_clique_layers, retained_layers
from .config import MethodConfig
from .graph_io import GraphRecord, graph_hash
from .hodge import get_hodge_laplacian
from .simplices import layer_simplices_at_order, required_max_hodge_order, required_max_simplex_order
from .spectral import heat_trace, time_grid

Channel = tuple[int, int]  # (q, k) everywhere in VNext.


@dataclass
class GraphProfile:
    graph_hash: str
    num_nodes: int
    num_edges: int
    times: np.ndarray
    channels: dict[Channel, np.ndarray]
    retained_q_layers: list[int]
    q_max: int | None
    clique_counts: dict[int, int]
    simplex_counts: dict[Channel, int]
    hodge_dimensions: dict[Channel, int]
    hodge_nonzeros: dict[Channel, int]
    solver: str
    max_clique_size: int | None
    max_hodge_order: int | None
    normalization: str


def _profile_cache_key(cfg: MethodConfig) -> dict:
    # Alpha only changes pair aggregation, never the graph profile or its cache key.
    return {
        "max_clique_size": cfg.max_clique_size,
        "max_hodge_order": cfg.max_hodge_order,
        "min_layer_q": cfg.min_layer_q,
        "time_min": cfg.time_min,
        "time_max": cfg.time_max,
        "time_points": cfg.time_points,
        "normalization": cfg.normalization,
        "solver": cfg.solver,
        "exact": cfg.exact.__dict__,
        "slq": cfg.slq.__dict__,
    }


def _normalize(raw: np.ndarray, cfg: MethodConfig, n_nodes: int, n_simplices: int) -> np.ndarray:
    if cfg.normalization == "none":
        return raw
    if cfg.normalization == "per_node":
        return raw / max(n_nodes, 1)
    if cfg.normalization == "per_simplex":
        # Divide by dim(L_k^(q)); do not rescale L or its eigenvalues.
        return raw / max(n_simplices, 1)
    raise ValueError(cfg.normalization)


def compute_graph_profile(
    record: GraphRecord,
    cfg: MethodConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> GraphProfile:
    cfg.validate()
    G = record.graph
    gid = record.graph_hash
    if gid != graph_hash(G):
        raise RuntimeError("GraphRecord hash does not match graph content.")
    store = ArtifactStore(record.cache_root, cfg.cache)

    profile_file = store.profile_file(_profile_cache_key(cfg))
    cached = store.load_pickle(profile_file)
    if cached is not None:
        if progress:
            progress(f"[PROFILE] graph={gid}: cache hit")
        return cached

    all_layers = enumerate_maximal_clique_layers(G, store, progress)
    layers = retained_layers(all_layers, cfg.max_clique_size, cfg.min_layer_q)
    times = time_grid(cfg)

    channels: dict[Channel, np.ndarray] = {}
    simplex_counts: dict[Channel, int] = {}
    hodge_dimensions: dict[Channel, int] = {}
    hodge_nonzeros: dict[Channel, int] = {}

    for q, cliques in sorted(layers.items()):
        k_max = required_max_hodge_order(q, cfg.max_hodge_order)
        # Materialize/cache only what is requested. Raising Q later adds missing K_r/B_r/L_k
        # in the same q directory and reuses all existing lower-order artifacts.
        r_max = required_max_simplex_order(q, cfg.max_hodge_order)
        for r in range(0, r_max + 1):
            layer_simplices_at_order(q, cliques, r, store, progress)

        for k in range(0, k_max + 1):
            simplices_k = layer_simplices_at_order(q, cliques, k, store, progress)
            n_k = len(simplices_k)
            L = get_hodge_laplacian(q, k, cliques, store, progress)
            raw = heat_trace(L, gid, q, k, cfg, store, progress)
            trace = _normalize(raw, cfg, G.number_of_nodes(), n_k)
            channels[(q, k)] = trace
            simplex_counts[(q, k)] = n_k
            hodge_dimensions[(q, k)] = int(L.shape[0])
            hodge_nonzeros[(q, k)] = int(L.nnz)

    retained_q = sorted(layers)
    profile = GraphProfile(
        graph_hash=gid,
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        times=times,
        channels=channels,
        retained_q_layers=retained_q,
        q_max=max(retained_q) if retained_q else None,
        clique_counts={q: len(v) for q, v in layers.items()},
        simplex_counts=simplex_counts,
        hodge_dimensions=hodge_dimensions,
        hodge_nonzeros=hodge_nonzeros,
        solver=cfg.solver,
        max_clique_size=cfg.max_clique_size,
        max_hodge_order=cfg.max_hodge_order,
        normalization=cfg.normalization,
    )
    store.save_pickle(profile_file, profile, enabled=store.config.persist_profiles)
    if progress:
        progress(f"[PROFILE] graph={gid}: completed q_layers={retained_q}")
    return profile
