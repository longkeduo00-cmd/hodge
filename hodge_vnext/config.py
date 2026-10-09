from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import json


@dataclass(frozen=True)
class CacheConfig:
    root: str = "cache_vnext"
    enabled: bool = True
    persist_simplices: bool = True
    persist_boundaries: bool = True
    persist_hodge: bool = True
    persist_spectral: bool = True
    persist_heat: bool = True
    persist_profiles: bool = True


@dataclass(frozen=True)
class ExactConfig:
    dimension_guard: Optional[int] = None
    eigen_zero_tolerance: float = 1e-10
    eigen_negative_tolerance: float = 1e-8


@dataclass(frozen=True)
class SLQConfig:
    num_probes: int = 24
    lanczos_steps: int = 40
    seed: int = 20260814
    breakdown_tolerance: float = 1e-12


@dataclass(frozen=True)
class MethodConfig:
    # Structural cutoffs.
    # max_clique_size S: retained maximal-clique size, None means all.
    # max_hodge_order Q: highest L_k^(q) computed in each retained q-layer, None means all.
    max_clique_size: Optional[int] = None
    max_hodge_order: Optional[int] = None
    min_layer_q: int = 1

    # Heat-signature parameters. Sampling mode is intentionally fixed to logspace.
    time_min: float = 1e-2
    time_max: float = 1e2
    time_points: int = 250
    normalization: str = "per_node"  # none | per_node | per_simplex

    # Pair-adaptive layer/matrix weighting.
    alpha: float = 1.0

    # A run is globally exact or globally SLQ. No auto/mixed mode is allowed.
    solver: str = "exact"  # exact | slq
    exact: ExactConfig = field(default_factory=ExactConfig)
    slq: SLQConfig = field(default_factory=SLQConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def validate(self) -> None:
        if self.max_clique_size is not None and self.max_clique_size < 1:
            raise ValueError("max_clique_size must be None or >= 1.")
        if self.max_hodge_order is not None and self.max_hodge_order < 0:
            raise ValueError("max_hodge_order must be None or >= 0.")
        if self.min_layer_q < 0:
            raise ValueError("min_layer_q must be >= 0.")
        if not (0 < self.time_min < self.time_max):
            raise ValueError("Require 0 < time_min < time_max.")
        if self.time_points < 2:
            raise ValueError("time_points must be >= 2.")
        if self.normalization not in {"none", "per_node", "per_simplex"}:
            raise ValueError("normalization must be none, per_node, or per_simplex.")
        if self.solver not in {"exact", "slq"}:
            raise ValueError("solver must be exactly 'exact' or 'slq'; auto/mixed mode is forbidden.")
        if self.exact.dimension_guard is not None and self.exact.dimension_guard < 1:
            raise ValueError("exact.dimension_guard must be None or >= 1.")
        if min(self.exact.eigen_zero_tolerance, self.exact.eigen_negative_tolerance) <= 0:
            raise ValueError("Exact eigenvalue tolerances must be positive.")
        if self.slq.num_probes < 1 or self.slq.lanczos_steps < 1:
            raise ValueError("SLQ num_probes and lanczos_steps must be positive.")
        if self.slq.breakdown_tolerance <= 0:
            raise ValueError("SLQ breakdown_tolerance must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Optional[Path] = None) -> "MethodConfig":
        exact = ExactConfig(**data.get("exact", {}))
        slq = SLQConfig(**data.get("slq", {}))
        cache_payload = dict(data.get("cache", {}))
        if root is not None and "root" in cache_payload:
            p = Path(cache_payload["root"])
            if not p.is_absolute():
                cache_payload["root"] = str((root / p).resolve())
        cache = CacheConfig(**cache_payload)
        payload = {k: v for k, v in data.items() if k not in {"exact", "slq", "cache"}}
        cfg = cls(**payload, exact=exact, slq=slq, cache=cache)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_method_config(path: str | Path) -> MethodConfig:
    p = Path(path).resolve()
    payload = load_json(p)
    data = payload.get("method", payload)
    return MethodConfig.from_dict(data, root=p.parent)
