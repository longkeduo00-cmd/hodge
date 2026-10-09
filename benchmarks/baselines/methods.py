from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.metadata
import math
import pickle
import time

import networkx as nx
import numpy as np

from hodge_vnext.cache import ArtifactStore
from hodge_vnext.cliques import enumerate_maximal_clique_layers, retained_layers
from hodge_vnext.config import MethodConfig
from hodge_vnext.graph_io import GraphRecord
from hodge_vnext.utils import stable_hash

from .netlsd_local import (
    heat_signature as netlsd_heat_signature,
    compare as netlsd_compare,
    AUTO_FULL_THRESHOLD as NETLSD_AUTO_FULL_THRESHOLD,
    AUTO_EIGENVALUES_PER_SIDE as NETLSD_AUTO_EIGENVALUES_PER_SIDE,
)
from .netrd_local import (
    degree_histogram_official,
    degree_js_distance,
    netsimile_signature,
    netsimile_distance,
)
from .portrait_local import portrait_distribution_sparse, portrait_divergence_sparse
from .dmeasure_local import dmeasure_representation, dmeasure_distance

SUPPORTED_METHODS = (
    "Count",
    "DegreeJS",
    "NetLSD",
    "NetSimile",
    "Portrait",
    "DMeasure",
    "DMeasureNoComplement",
)

UPSTREAM_PROVENANCE = {
    "NetLSD": {
        "repository": "https://github.com/xgfs/NetLSD",
        "commit": "86c73cc8c6533b4547cbc12a2b81e7cc23f9817a",
        "reference_files": ["netlsd/kernels.py", "netlsd/util.py"],
        "local_status": "faithful local reproduction; no external package required",
    },
    "DegreeJS": {
        "repository": "https://github.com/netsiphd/netrd",
        "commit": "c1c85aa1517de49d335fb3c0e62cc0edbc5c161b",
        "reference_file": "netrd/distance/degree_divergence.py",
        "local_status": "faithful local reproduction",
    },
    "NetSimile": {
        "repository": "https://github.com/netsiphd/netrd",
        "commit": "c1c85aa1517de49d335fb3c0e62cc0edbc5c161b",
        "reference_file": "netrd/distance/netsimile.py",
        "local_status": "same 7 features/35 moments; ego construction algebraically optimized",
    },
    "Portrait": {
        "repository": "https://github.com/bagrow/network-portrait-divergence",
        "commit": "72993c368114c2e834142787579466d673232fa4",
        "reference_file": "portrait_divergence.py",
        "local_status": "exact sparse representation of occupied portrait mass",
    },
    "DMeasure": {
        "repository": "https://github.com/tischieber/Quantifying-Network-Structural-Dissimilarities",
        "commit": "27e1361884da3f7f28b38bfced4279422231ad38",
        "reference_file": "d_distance.r",
        "reference_blob": "620ac811c6706531eff91cf66a3349477d4b50df",
        "local_status": "full uppercase D; chunked APSP + exact sparse complement solve",
    },
    "DMeasureNoComplement": {
        "repository": "https://github.com/tischieber/Quantifying-Network-Structural-Dissimilarities",
        "commit": "27e1361884da3f7f28b38bfced4279422231ad38",
        "reference_file": "d_distance.r",
        "reference_blob": "620ac811c6706531eff91cf66a3349477d4b50df",
        "local_status": "official lowercase d (complement term omitted)",
    },
    "Count": {
        "repository": "local Hodge VNext clique cache",
        "implementation": "exclusive globally maximal q-layer counts only",
    },
}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def dependency_versions() -> dict[str, str | None]:
    # No extra baseline package is required.
    return {
        "networkx": _package_version("networkx"),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "scikit-learn": _package_version("scikit-learn"),
        "matplotlib": _package_version("matplotlib"),
    }


def _cache_path(record: GraphRecord, method: str, settings: dict[str, Any]) -> Path:
    store = ArtifactStore(record.cache_root, record_cache_config(record))
    return store.baseline_file(method, settings)


def record_cache_config(record: GraphRecord):
    # Baseline cache must follow the same graph-level cache switch as Hodge.
    # BaselineContext passes the authoritative MethodConfig; this helper exists
    # only for the legacy standalone cache-path function and is replaced by the
    # context-owned implementation below when called normally.
    from hodge_vnext.config import CacheConfig
    return CacheConfig(root=str(record.cache_root.parents[0]))


def _load_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_cache(path: Path, obj: Any) -> None:
    if path.exists():
        return
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _count_representation(
    record: GraphRecord,
    cfg: MethodConfig,
    max_clique_size: int | None,
    min_layer_q: int,
) -> dict[int, float]:
    store = ArtifactStore(record.cache_root, cfg.cache)
    layers = enumerate_maximal_clique_layers(record.graph, store, progress=None)
    selected = retained_layers(layers, max_clique_size, min_layer_q)
    return {int(q): float(len(items)) for q, items in selected.items()}


@dataclass
class BaselineContext:
    method_cfg: MethodConfig
    count_normalization: str = "distribution"
    count_max_clique_size: int | None = None
    count_min_layer_q: int = 1
    netlsd_t_min: float = 1e-2
    netlsd_t_max: float = 1e2
    netlsd_t_points: int = 250
    netlsd_eigenvalues: str = "auto"
    netlsd_normalization: str = "empty"
    netlsd_normalized_laplacian: bool = True
    dmeasure_w1: float = 0.45
    dmeasure_w2: float = 0.45
    dmeasure_w3: float = 0.10
    dmeasure_shortest_path_chunk_size: int = 128

    def validate(self) -> None:
        if self.count_normalization not in {"raw", "per_node", "distribution"}:
            raise ValueError("count_normalization must be raw, per_node, or distribution")
        if self.count_max_clique_size is not None and self.count_max_clique_size < 1:
            raise ValueError("count_max_clique_size must be None or >=1")
        if self.count_min_layer_q < 0:
            raise ValueError("count_min_layer_q must be >=0")
        if self.netlsd_t_min <= 0 or self.netlsd_t_max <= self.netlsd_t_min:
            raise ValueError("invalid NetLSD time range")
        if self.netlsd_t_points < 2:
            raise ValueError("NetLSD time points must be >=2")
        if self.netlsd_eigenvalues not in {"auto", "full"}:
            raise ValueError("netlsd_eigenvalues must be auto or full")
        if self.netlsd_normalization not in {"empty", "complete", "none"}:
            raise ValueError("netlsd_normalization must be empty, complete, or none")
        if not isinstance(self.netlsd_normalized_laplacian, bool):
            raise ValueError("netlsd_normalized_laplacian must be boolean")
        if self.dmeasure_shortest_path_chunk_size < 1:
            raise ValueError("DMeasure shortest-path chunk size must be >=1")

    def method_settings(self, method: str) -> dict[str, Any]:
        if method == "Count":
            return {
                "normalization": self.count_normalization,
                "max_clique_size": self.count_max_clique_size,
                "min_layer_q": self.count_min_layer_q,
            }
        if method == "DegreeJS":
            return {"backend": "local-netrd-official-logic", "base": 2}
        if method == "NetLSD":
            return {
                "backend": "local-xgfs-official-logic",
                "t_min": self.netlsd_t_min,
                "t_max": self.netlsd_t_max,
                "t_points": self.netlsd_t_points,
                "eigenvalues": self.netlsd_eigenvalues,
                "normalization": self.netlsd_normalization,
                "normalized_laplacian": self.netlsd_normalized_laplacian,
                "auto_policy": {
                    "full_eigendecomposition_if_n_le": NETLSD_AUTO_FULL_THRESHOLD,
                    "large_graph_eigenvalues_per_side": NETLSD_AUTO_EIGENVALUES_PER_SIDE,
                    "large_graph_middle_spectrum": "linear_interpolation",
                    "uses_slq": False,
                },
            }
        if method == "NetSimile":
            return {
                "backend": "local-netrd-official-logic",
                "constant_moment_policy": "strict-upstream-NaN-propagation",
            }
        if method == "Portrait":
            return {
                "backend": "local-bagrow-official-logic",
                "storage": "sparse-occupied-cells-exact",
            }
        if method in {"DMeasure", "DMeasureNoComplement"}:
            return {
                "backend": "local-Schieber-official-logic",
                "w1": self.dmeasure_w1,
                "w2": self.dmeasure_w2,
                "w3": self.dmeasure_w3,
                "include_complement": method == "DMeasure",
                "shortest_path_chunk_size": self.dmeasure_shortest_path_chunk_size,
            }
        raise KeyError(method)

    def representation(self, method: str, record: GraphRecord) -> tuple[Any, float, bool]:
        self.validate()
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method {method}; choose from {SUPPORTED_METHODS}")
        settings = self.method_settings(method)
        path = ArtifactStore(record.cache_root, self.method_cfg.cache).baseline_file(method, settings)
        cached = _load_cache(path)
        if cached is not None:
            return cached, 0.0, True

        t0 = time.perf_counter()
        graph = record.graph
        if method == "Count":
            rep = _count_representation(
                record,
                self.method_cfg,
                self.count_max_clique_size,
                self.count_min_layer_q,
            )
        elif method == "DegreeJS":
            rep = degree_histogram_official(graph)
        elif method == "NetLSD":
            times = np.logspace(
                math.log10(self.netlsd_t_min),
                math.log10(self.netlsd_t_max),
                self.netlsd_t_points,
            )
            rep = netlsd_heat_signature(
                graph,
                timescales=times,
                eigenvalues=self.netlsd_eigenvalues,
                normalization=self.netlsd_normalization,
                normalized_laplacian=self.netlsd_normalized_laplacian,
            )
        elif method == "NetSimile":
            rep = netsimile_signature(graph)
        elif method == "Portrait":
            rep = portrait_distribution_sparse(graph)
        elif method == "DMeasure":
            rep = dmeasure_representation(
                graph,
                include_alpha=self.dmeasure_w3 > 0,
                include_complement=True,
                shortest_path_chunk_size=self.dmeasure_shortest_path_chunk_size,
            )
        elif method == "DMeasureNoComplement":
            rep = dmeasure_representation(
                graph,
                include_alpha=self.dmeasure_w3 > 0,
                include_complement=False,
                shortest_path_chunk_size=self.dmeasure_shortest_path_chunk_size,
            )
        else:  # pragma: no cover
            raise AssertionError(method)

        elapsed = time.perf_counter() - t0
        _save_cache(path, rep)
        return rep, elapsed, False

    def distance(self, method: str, rep_a: Any, rep_b: Any, n_a: int, n_b: int) -> float:
        if method == "Count":
            qs = sorted(set(rep_a) | set(rep_b))
            if not qs:
                return 0.0
            a = np.asarray([rep_a.get(q, 0.0) for q in qs], dtype=float)
            b = np.asarray([rep_b.get(q, 0.0) for q in qs], dtype=float)
            if self.count_normalization == "per_node":
                a = a / max(int(n_a), 1)
                b = b / max(int(n_b), 1)
            elif self.count_normalization == "distribution":
                a = a / a.sum() if a.sum() > 0 else a
                b = b / b.sum() if b.sum() > 0 else b
            return float(np.linalg.norm(a - b))
        if method == "DegreeJS":
            return degree_js_distance(rep_a, rep_b)
        if method == "NetLSD":
            return netlsd_compare(rep_a, rep_b)
        if method == "NetSimile":
            return netsimile_distance(rep_a, rep_b)
        if method == "Portrait":
            return portrait_divergence_sparse(rep_a, rep_b)
        if method == "DMeasure":
            return dmeasure_distance(
                rep_a,
                rep_b,
                w1=self.dmeasure_w1,
                w2=self.dmeasure_w2,
                w3=self.dmeasure_w3,
                include_complement=True,
            )
        if method == "DMeasureNoComplement":
            return dmeasure_distance(
                rep_a,
                rep_b,
                w1=self.dmeasure_w1,
                w2=self.dmeasure_w2,
                w3=self.dmeasure_w3,
                include_complement=False,
            )
        raise KeyError(method)
