from __future__ import annotations

"""Unified method interface for synthetic/real validation.

The Hodge branch calls the unchanged VNext core.  Classical baselines are kept
outside the method core under ``benchmarks.baselines``.
"""

from dataclasses import dataclass
from typing import Any
import time

from hodge_vnext.config import MethodConfig
from hodge_vnext.distance import profile_distance
from hodge_vnext.graph_io import GraphRecord
from hodge_vnext.profile import GraphProfile, compute_graph_profile
from benchmarks.baselines.methods import BaselineContext, SUPPORTED_METHODS

PRIMARY_METHOD = "Hodge"
DEFAULT_METHODS = ("Hodge", "DMeasure", "NetLSD", "Portrait", "Count")


@dataclass
class MethodManager:
    cfg: MethodConfig
    baseline: BaselineContext

    @classmethod
    def from_config(cls, cfg: MethodConfig, baseline_options: dict | None = None) -> "MethodManager":
        opts = dict(baseline_options or {})
        baseline = BaselineContext(
            method_cfg=cfg,
            count_normalization=str(opts.get("count_normalization", "distribution")),
            count_max_clique_size=opts.get("count_max_clique_size", None),
            # The historical Hodge Count baseline used q>=1. Keep that default
            # unless the experiment explicitly requests q=0.
            count_min_layer_q=int(opts.get("count_min_layer_q", cfg.min_layer_q)),
            netlsd_t_min=float(opts.get("netlsd_t_min", cfg.time_min)),
            netlsd_t_max=float(opts.get("netlsd_t_max", cfg.time_max)),
            netlsd_t_points=int(opts.get("netlsd_t_points", cfg.time_points)),
            netlsd_eigenvalues=str(opts.get("netlsd_eigenvalues", "auto")),
            netlsd_normalization=str(opts.get("netlsd_normalization", "empty")),
            netlsd_normalized_laplacian=bool(opts.get("netlsd_normalized_laplacian", True)),
            dmeasure_shortest_path_chunk_size=int(opts.get("dmeasure_chunk_size", 128)),
        )
        return cls(cfg, baseline)

    @staticmethod
    def validate_methods(methods: list[str] | tuple[str, ...]) -> list[str]:
        allowed = {PRIMARY_METHOD, *SUPPORTED_METHODS}
        bad = [m for m in methods if m not in allowed]
        if bad:
            raise ValueError(f"Unsupported methods {bad}; allowed={sorted(allowed)}")
        return list(methods)

    def representation(self, method: str, record: GraphRecord) -> tuple[Any, float, bool]:
        if method == PRIMARY_METHOD:
            t0 = time.perf_counter()
            # compute_graph_profile reports cache hits internally only when progress
            # is supplied. We conservatively report elapsed and leave hit=False.
            rep = compute_graph_profile(record, self.cfg, progress=None)
            return rep, time.perf_counter() - t0, False
        return self.baseline.representation(method, record)

    def distance(self, method: str, a: Any, b: Any, na: int, nb: int) -> float:
        if method == PRIMARY_METHOD:
            if not isinstance(a, GraphProfile) or not isinstance(b, GraphProfile):
                raise TypeError("Hodge representations must be GraphProfile objects")
            return float(profile_distance(a, b, self.cfg).total_distance)
        return float(self.baseline.distance(method, a, b, na, nb))
