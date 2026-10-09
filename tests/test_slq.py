from __future__ import annotations

from pathlib import Path
import networkx as nx
import numpy as np

from hodge_vnext.config import CacheConfig, MethodConfig, SLQConfig
from hodge_vnext.distance import profile_distance
from hodge_vnext.graph_io import GraphRecord, graph_hash, save_graph_npz
from hodge_vnext.profile import compute_graph_profile


def test_slq_self_distance_is_zero_and_probe_cache_is_incremental(tmp_path: Path) -> None:
    G = nx.complete_graph(5)
    gid = graph_hash(G)
    root = tmp_path / f"graph_{gid}"
    root.mkdir(parents=True)
    save_graph_npz(root / "graph.npz", G)
    rec = GraphRecord(G, gid, root, {"kind": "test"})

    base = dict(
        min_layer_q=1,
        time_points=10,
        normalization="per_node",
        solver="slq",
        cache=CacheConfig(root=str(tmp_path / "cache")),
    )
    c2 = MethodConfig(**base, slq=SLQConfig(num_probes=2, lanczos_steps=5, seed=99))
    p2 = compute_graph_profile(rec, c2)
    assert profile_distance(p2, p2, c2).total_distance == 0.0

    slq_root = root / "stages" / "hodge-vnext-stage-cache-v2" / "Q5_spectral" / "slq" / "q0004"
    probe_files_before = list(slq_root.glob("**/probe_*.npz"))
    assert probe_files_before

    c4 = MethodConfig(**base, slq=SLQConfig(num_probes=4, lanczos_steps=5, seed=99))
    p4 = compute_graph_profile(rec, c4)
    assert profile_distance(p4, p4, c4).total_distance == 0.0
    # Each matrix cache branch should now have probes 1..4; at least one branch must show 4.
    branches = list(slq_root.glob("k*_*/"))
    assert any(len(list(branch.glob("probe_*.npz"))) == 4 for branch in branches)
