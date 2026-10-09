from __future__ import annotations

from pathlib import Path
import time

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from hodge_vnext.config import CacheConfig, ExactConfig, MethodConfig, SLQConfig
from hodge_vnext.distance import profile_distance
from hodge_vnext.graph_io import GraphRecord, graph_hash, save_graph_npz
from hodge_vnext.models import synthetic_graph_record
from hodge_vnext.profile import compute_graph_profile
from hodge_vnext.spectral import lanczos_tridiagonal_full_reorth
from scipy.linalg import eigh_tridiagonal


def record_for(G: nx.Graph, root: Path) -> GraphRecord:
    G = nx.convert_node_labels_to_integers(G)
    gid = graph_hash(G)
    graph_root = root / f"graph_{gid}"
    graph_root.mkdir(parents=True, exist_ok=True)
    save_graph_npz(graph_root / "graph.npz", G)
    return GraphRecord(G, gid, graph_root, {"kind": "test"})


def cfg(tmp_path: Path, **kwargs) -> MethodConfig:
    payload = dict(
        max_clique_size=None,
        max_hodge_order=None,
        min_layer_q=1,
        time_min=1e-2,
        time_max=1e2,
        time_points=8,
        normalization="none",
        alpha=1.0,
        solver="exact",
        exact=ExactConfig(dimension_guard=100),
        slq=SLQConfig(num_probes=8, lanczos_steps=8, seed=1234),
        cache=CacheConfig(root=str(tmp_path / "cache")),
    )
    payload.update(kwargs)
    out = MethodConfig(**payload)
    out.validate()
    return out


def test_triangle_profile_and_self_distance(tmp_path: Path) -> None:
    G = nx.complete_graph(3)
    c = cfg(tmp_path)
    p = compute_graph_profile(record_for(G, tmp_path / "r1"), c)
    assert set(p.channels) == {(2, 0), (2, 1), (2, 2)}
    assert p.hodge_dimensions[(2, 0)] == 3
    assert p.hodge_dimensions[(2, 1)] == 3
    assert p.hodge_dimensions[(2, 2)] == 1
    assert profile_distance(p, p, c).total_distance == pytest.approx(0.0, abs=1e-14)


def test_pair_adaptive_schema_includes_common_missing_intermediate_q(tmp_path: Path) -> None:
    c = cfg(tmp_path)
    p4 = compute_graph_profile(record_for(nx.complete_graph(4), tmp_path / "g4"), c)  # q=3
    p6 = compute_graph_profile(record_for(nx.complete_graph(6), tmp_path / "g6"), c)  # q=5
    d = profile_distance(p4, p6, c)
    assert d.q_star == 5
    assert set(d.layer_weights) == {1, 2, 3, 4, 5}
    # Neither K4 nor K6 has an exclusive q=4 layer; common absence is similarity.
    assert d.layer_distances[4] == pytest.approx(0.0, abs=1e-14)
    # For alpha=1 all individual Hodge matrices in the pair schema have equal weight.
    weights = list(d.matrix_weights.values())
    assert max(weights) - min(weights) < 1e-14


def test_q_cutoff_adds_artifacts_incrementally(tmp_path: Path) -> None:
    rec = record_for(nx.complete_graph(6), tmp_path / "g")  # q=5
    c2 = cfg(tmp_path, max_hodge_order=2)
    p2 = compute_graph_profile(rec, c2)
    assert set(k for (q, k) in p2.channels if q == 5) == {0, 1, 2}
    stage = rec.cache_root / "stages" / "hodge-vnext-stage-cache-v2"
    assert (stage / "Q2_simplices" / "q0005" / "K_r0003.pkl").exists()  # needed for B3 and L2
    assert not (stage / "Q4_hodge" / "q0005" / "L_k0003.npz").exists()
    lower = stage / "Q4_hodge" / "q0005" / "L_k0002.npz"
    before = lower.stat().st_mtime_ns

    c4 = cfg(tmp_path, max_hodge_order=4)
    p4 = compute_graph_profile(rec, c4)
    assert set(k for (q, k) in p4.channels if q == 5) == {0, 1, 2, 3, 4}
    assert (stage / "Q4_hodge" / "q0005" / "L_k0004.npz").exists()
    assert lower.stat().st_mtime_ns == before  # lower artifact was reused, not overwritten


def test_independent_parameter_seeds_and_repeat_cache(tmp_path: Path) -> None:
    a = synthetic_graph_record(tmp_path / "cache", "ER", 40, {"p": 0.1}, 22, 1)
    b = synthetic_graph_record(tmp_path / "cache", "ER", 40, {"p": 0.2}, 22, 1)
    assert a.source["repeat_seed"] == b.source["repeat_seed"] == 22
    assert a.source["graph_seed"] != b.source["graph_seed"]
    a2 = synthetic_graph_record(tmp_path / "cache", "ER", 40, {"p": 0.1}, 22, 1)
    assert a2.graph_hash == a.graph_hash
    assert a2.source["graph_seed"] == a.source["graph_seed"]


def test_tridiagonal_solver_matches_dense_projection_spectrum() -> None:
    rng = np.random.default_rng(7)
    A = rng.normal(size=(8, 8))
    A = A.T @ A
    L = sparse.csr_matrix(A)
    z = rng.choice(np.array([-1.0, 1.0]), size=8)
    alpha, beta, _ = lanczos_tridiagonal_full_reorth(L, z, 6, 1e-14)
    T = np.diag(alpha)
    if beta.size:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    dense_vals = np.linalg.eigvalsh(T)
    tri_vals = eigh_tridiagonal(alpha, beta, eigvals_only=True, check_finite=False)
    assert np.allclose(dense_vals, tri_vals, rtol=1e-12, atol=1e-12)


def test_auto_solver_is_forbidden(tmp_path: Path) -> None:
    c = cfg(tmp_path, solver="exact")
    data = c.to_dict()
    data["solver"] = "auto"
    with pytest.raises(ValueError):
        MethodConfig.from_dict(data)
