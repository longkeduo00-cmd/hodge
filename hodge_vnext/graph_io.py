from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np

from .utils import stable_hash, write_json

GRAPH_SCHEMA = "hodge-vnext-graph-v2"
GraphInput = Union[str, Path, nx.Graph]


@dataclass(frozen=True)
class GraphRecord:
    graph: nx.Graph
    graph_hash: str
    cache_root: Path
    source: dict


def canonicalize_graph(graph: GraphInput, nodes_path: Optional[str | Path] = None) -> nx.Graph:
    if isinstance(graph, nx.Graph):
        G = graph.copy()
    else:
        path = Path(graph)
        suffix = path.suffix.lower()
        if suffix == ".graphml":
            G = nx.read_graphml(path)
        elif suffix == ".gml":
            G = nx.read_gml(path)
        elif suffix in {".gpickle", ".pickle", ".pkl"}:
            # networkx 3 removed top-level read_gpickle on some installs.
            import pickle
            with path.open("rb") as f:
                G = pickle.load(f)
        else:
            G = nx.read_edgelist(path, nodetype=str, data=False)
        if nodes_path is not None and str(nodes_path):
            with Path(nodes_path).open("r", encoding="utf-8") as handle:
                G.add_nodes_from(line.strip() for line in handle if line.strip())

    if G.is_directed() or G.is_multigraph():
        raise ValueError("Input must be an undirected simple graph.")
    if nx.number_of_selfloops(G):
        raise ValueError("Self-loops are not allowed.")
    if G.number_of_nodes() == 0:
        raise ValueError("Input graph must be non-empty.")

    ordered = sorted(G.nodes(), key=repr)
    mapping = {node: i for i, node in enumerate(ordered)}
    return nx.relabel_nodes(G, mapping, copy=True)


def graph_hash(G: nx.Graph) -> str:
    payload = {
        "schema": GRAPH_SCHEMA,
        "n": int(G.number_of_nodes()),
        "edges": sorted((int(min(u, v)), int(max(u, v))) for u, v in G.edges()),
    }
    return stable_hash(payload, length=20)


def save_graph_npz(path: Path, G: nx.Graph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(sorted((min(u, v), max(u, v)) for u, v in G.edges()), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    np.savez_compressed(path, n=np.int64(G.number_of_nodes()), edges=edges)


def load_graph_npz(path: Path) -> nx.Graph:
    data = np.load(path, allow_pickle=False)
    n = int(data["n"])
    edges = np.asarray(data["edges"], dtype=np.int64).reshape((-1, 2))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for u, v in edges)
    return G


def _graph_stage(root: Path) -> Path:
    p = root / "C0_graph"
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_real_record(
    graph: GraphInput,
    cache_root: str | Path,
    label: str,
    nodes_path: Optional[str | Path] = None,
) -> GraphRecord:
    G = canonicalize_graph(graph, nodes_path)
    gid = graph_hash(G)
    root = Path(cache_root) / "real" / str(label) / f"graph_{gid}"
    stage = _graph_stage(root)
    graph_file = stage / "graph.npz"
    if not graph_file.exists():
        save_graph_npz(graph_file, G)
    source = {
        "graph_schema": GRAPH_SCHEMA,
        "kind": "real",
        "label": str(label),
        "graph_hash": gid,
        "nodes_path_used": bool(nodes_path),
    }
    write_json(stage / "source.json", source)
    return GraphRecord(G, gid, root, source)
