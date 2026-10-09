from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional
import networkx as nx

from .cache import ArtifactStore

CliqueLayers = dict[int, list[tuple[int, ...]]]


def enumerate_maximal_clique_layers(
    G: nx.Graph,
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> CliqueLayers:
    """Enumerate all globally maximal cliques once and cache by simplex dimension q.

    The experimental clique cutoff S is applied later when selecting layers. This
    keeps the base algorithm definition and cache independent of S and makes S
    increases/decreases pure downstream selections.
    """
    path = store.cliques_file()
    cached = store.load_pickle(path)
    if cached is not None:
        if progress:
            progress("[CLIQUES] cache hit: complete maximal-clique enumeration")
        return cached

    if progress:
        progress("[CLIQUES] enumerating all maximal cliques")
    layers: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for clique in nx.find_cliques(G):
        c = tuple(sorted(int(v) for v in clique))
        q = len(c) - 1
        layers[q].append(c)
    out = {q: sorted(items) for q, items in sorted(layers.items())}
    store.save_pickle(path, out, enabled=True)
    store.write_manifest_once(
        store.cliques_manifest(),
        {
            "complete": True,
            "layer_definition": "exclusive maximal q-simplex, q=clique_size-1",
            "counts": {str(q): len(items) for q, items in out.items()},
        },
    )
    if progress:
        progress("[CLIQUES] completed: " + ", ".join(f"q={q}:{len(v)}" for q, v in out.items()))
    return out


def retained_layers(layers: CliqueLayers, max_clique_size: int | None, min_layer_q: int) -> CliqueLayers:
    max_q = None if max_clique_size is None else max_clique_size - 1
    return {
        q: cliques
        for q, cliques in layers.items()
        if q >= min_layer_q and (max_q is None or q <= max_q)
    }
