from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional

from .cache import ArtifactStore


def required_max_simplex_order(q: int, max_hodge_order: int | None) -> int:
    if max_hodge_order is None:
        return q
    return min(q, max_hodge_order + 1)


def required_max_hodge_order(q: int, max_hodge_order: int | None) -> int:
    return q if max_hodge_order is None else min(q, max_hodge_order)


def layer_simplices_at_order(
    q: int,
    cliques: list[tuple[int, ...]],
    r: int,
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> list[tuple[int, ...]]:
    if r < 0 or r > q:
        raise ValueError(f"Invalid simplex order r={r} for q-layer {q}.")
    path = store.simplex_file(q, r)
    cached = store.load_pickle(path)
    if cached is not None:
        if progress:
            progress(f"[SIMPLICES] q={q}, r={r}: cache hit ({len(cached)})")
        return cached

    simplex_set: set[tuple[int, ...]] = set()
    size = r + 1
    for clique in cliques:
        if len(clique) != q + 1:
            raise ValueError("Clique assigned to wrong exclusive layer.")
        simplex_set.update(combinations(clique, size))
    result = sorted(simplex_set)
    store.save_pickle(path, result, enabled=store.config.persist_simplices)
    if progress:
        progress(f"[SIMPLICES] q={q}, r={r}: built ({len(result)})")
    return result
