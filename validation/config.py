from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json

from hodge_vnext.config import MethodConfig


def project_root(config_path: str | Path) -> Path:
    p = Path(config_path).resolve()
    for candidate in (p.parent, *p.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def load_payload(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path


def resolve_path(value: str | Path, config_path: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root(config_path) / p).resolve()


def method_from_payload(payload: dict[str, Any], config_path: str | Path) -> MethodConfig:
    return MethodConfig.from_dict(payload["method"], root=project_root(config_path))


def override_method(
    cfg: MethodConfig,
    *,
    solver: str | None = None,
    max_clique_size: int | None | str = "inherit",
    max_hodge_order: int | None | str = "inherit",
    normalization: str | None = None,
    alpha: float | None = None,
    time_min: float | None = None,
    time_max: float | None = None,
    time_points: int | None = None,
    slq_probes: int | None = None,
    slq_steps: int | None = None,
) -> MethodConfig:
    S = cfg.max_clique_size if max_clique_size == "inherit" else max_clique_size
    Q = cfg.max_hodge_order if max_hodge_order == "inherit" else max_hodge_order
    slq = cfg.slq
    if slq_probes is not None or slq_steps is not None:
        slq = replace(
            slq,
            num_probes=slq.num_probes if slq_probes is None else int(slq_probes),
            lanczos_steps=slq.lanczos_steps if slq_steps is None else int(slq_steps),
        )
    out = replace(
        cfg,
        solver=cfg.solver if solver is None else solver,
        max_clique_size=S,
        max_hodge_order=Q,
        normalization=cfg.normalization if normalization is None else normalization,
        alpha=cfg.alpha if alpha is None else float(alpha),
        time_min=cfg.time_min if time_min is None else float(time_min),
        time_max=cfg.time_max if time_max is None else float(time_max),
        time_points=cfg.time_points if time_points is None else int(time_points),
        slq=slq,
    )
    out.validate()
    return out
