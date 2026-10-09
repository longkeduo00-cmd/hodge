from __future__ import annotations

"""Finite-size density-matched ER vs RGG comparison.

This module does not alter either model or the Hodge method.  It constructs two
ordinary synthetic-validation sub-runs on a shared control coordinate and then
computes cross-model distances at matched coordinates.

For the standard 2-D Euclidean RGG in the unit square, the default matching
uses the exact pair-connection probability for radius r<=1,

    P(||X-Y||_2 <= r) = pi r^2 - 8/3 r^3 + 1/2 r^4,

so that ER p=c/N and the RGG connection probability are both c/N.  Therefore
both models have the same finite-N expected mean degree (N-1)c/N.  The RGG
core itself remains parameterized by lambda=N*pi*r^2; only the comparison
runner converts the common c coordinate into the lambda value required by the
existing RGG generator.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
import json
import math
import shutil
import time

import numpy as np
import pandas as pd

from hodge_vnext.config import MethodConfig
from hodge_vnext.graph_io import GraphRecord, load_graph_npz
from hodge_vnext.models import rgg_resolved_parameters
from hodge_vnext.utils import safe_token, stable_hash, stable_seed, write_json

from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .runner import run_synthetic_validation


def _progress(msg: str) -> None:
    print(msg, flush=True)


def square_euclidean_connection_probability(radius: float) -> float:
    """Exact P(distance<=r) for two U([0,1]^2) points, valid on 0<=r<=1."""
    r = float(radius)
    if not 0.0 <= r <= 1.0:
        raise ValueError("finite-square matching currently requires 0 <= radius <= 1")
    return float(math.pi * r * r - (8.0 / 3.0) * r**3 + 0.5 * r**4)


def square_euclidean_radius_from_probability(probability: float) -> float:
    """Invert the exact unit-square connection probability by bisection."""
    p = float(probability)
    p1 = square_euclidean_connection_probability(1.0)
    if not 0.0 <= p <= p1:
        raise ValueError(f"connection probability must lie in [0,{p1:.12g}] for r<=1; got {p}")
    if p == 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if square_euclidean_connection_probability(mid) < p:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def matched_rgg_lambda_from_er_c(n: int, c: float, *, dim: int = 2, p_norm: float = 2.0) -> tuple[float, float]:
    """Return (lambda,radius) whose expected degree matches ER p=c/N.

    Exact finite-square matching is deliberately limited to the formal RGG
    setting used by this project: d=2, Euclidean norm, unit square.
    """
    n = int(n)
    c = float(c)
    if n < 2:
        raise ValueError("n must be >=2")
    if not (int(dim) == 2 and math.isclose(float(p_norm), 2.0, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("finite-square ER/RGG matching currently supports only dim=2, p_norm=2")
    if not math.isfinite(c) or c < 0.0 or c > n:
        raise ValueError("c must satisfy 0 <= c <= N because ER uses p=c/N")
    edge_probability = c / float(n)
    radius = square_euclidean_radius_from_probability(edge_probability)
    lam = float(n) * math.pi * radius * radius
    return lam, radius


def _run_stem(exp: dict[str, Any], cfg: MethodConfig, controls: list[float], matching: str) -> str:
    S = "all" if cfg.max_clique_size is None else str(cfg.max_clique_size)
    Q = "all" if cfg.max_hodge_order is None else str(cfg.max_hodge_order)
    identity = {
        "n": int(exp["n"]),
        "repeats": int(exp["repeats"]),
        "base_seed": int(exp["base_seed"]),
        "controls": [float(x) for x in controls],
        "matching": matching,
        "rgg_dim": int(exp.get("rgg_dim", 2)),
        "rgg_p_norm": float(exp.get("rgg_p_norm", 2.0)),
        "methods": list(exp.get("methods", [])),
        "method": cfg.to_dict(),
    }
    parts = [
        f"N{int(exp['n'])}", f"rep{int(exp['repeats'])}", f"seed{int(exp['base_seed'])}",
        f"match-{safe_token(matching)}", f"c{safe_token(controls[0])}-{safe_token(controls[-1])}-P{len(controls)}",
        f"solver-{cfg.solver}", f"S{S}", f"Q{Q}", f"qmin{cfg.min_layer_q}",
        f"alpha{safe_token(cfg.alpha)}", f"norm-{safe_token(cfg.normalization)}",
        f"cmpid-{stable_hash(identity, length=10)}",
    ]
    if cfg.solver == "slq":
        parts += [f"R{cfg.slq.num_probes}", f"M{cfg.slq.lanczos_steps}"]
    return "_".join(parts)


def _clear_outputs(run_dir: Path) -> None:
    for name in ("tables", "plots", "subruns"):
        p = run_dir / name
        if p.exists():
            shutil.rmtree(p)
    for name in ("effective_config.json", "run_manifest.json", "RUN_COMPLETE.json"):
        p = run_dir / name
        if p.exists():
            p.unlink()


def _record_from_row(row: pd.Series) -> GraphRecord:
    root = Path(str(row.cache_graph_root))
    G = load_graph_npz(root / "C0_graph" / "graph.npz")
    source = json.loads((root / "C0_graph" / "source.json").read_text(encoding="utf-8"))
    return GraphRecord(G, str(row.graph_hash), root, source)


def _cross_stats(distance_matrix: np.ndarray, *, bootstrap_samples: int, seed: int) -> dict[str, float | int]:
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.size == 0:
        raise ValueError("cross-model distance matrix must be non-empty 2-D")
    flat = D.ravel()
    mean = float(flat.mean())
    std = float(flat.std(ddof=1)) if flat.size > 1 else 0.0
    if bootstrap_samples <= 0:
        lo = hi = mean
    else:
        rng = np.random.default_rng(int(seed))
        vals = np.empty(int(bootstrap_samples), dtype=float)
        # Realization-level bootstrap: resample ER and RGG graphs, then rebuild
        # the full cross product. This avoids pretending the R^2 pair distances
        # are independent observations.
        for b in range(int(bootstrap_samples)):
            ii = rng.integers(0, D.shape[0], size=D.shape[0])
            jj = rng.integers(0, D.shape[1], size=D.shape[1])
            vals[b] = float(D[np.ix_(ii, jj)].mean())
        lo, hi = np.quantile(vals, [0.025, 0.975]).tolist()
    return {
        "mean": mean, "std": std, "ci95_low": float(lo), "ci95_high": float(hi),
        "pair_count": int(flat.size),
    }


def _noise_lookup(run_dir: Path) -> dict[tuple[str, int], float]:
    df = pd.read_csv(run_dir / "tables" / "same_parameter_noise.csv")
    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        out[(str(row.method), int(row.parameter_index))] = float(row["mean"])
    return out


def _build_subpayloads(payload: dict[str, Any], controls: list[float], comparison_root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    comp = dict(payload["comparison"])
    n = int(comp["n"])
    repeats = int(comp["repeats"])
    base_seed = int(comp.get("base_seed", 22))
    methods = list(comp.get("methods", ["Hodge", "DMeasure", "NetLSD", "Portrait", "Count"]))
    dim = int(comp.get("rgg_dim", 2))
    p_norm = float(comp.get("rgg_p_norm", 2.0))
    matching = str(comp.get("matching", "finite_square_expected_degree"))
    if matching != "finite_square_expected_degree":
        raise ValueError("Only matching='finite_square_expected_degree' is currently implemented")

    rgg_lambdas: list[float] = []
    rows = []
    for i, c in enumerate(controls):
        lam, radius = matched_rgg_lambda_from_er_c(n, c, dim=dim, p_norm=p_norm)
        p_er = float(c) / float(n)
        expected_degree = (n - 1) * p_er
        resolved = rgg_resolved_parameters(n, {"lambda": lam, "dim": dim, "p_norm": p_norm})
        if not math.isclose(float(resolved["radius"]), radius, rel_tol=0.0, abs_tol=5e-13):
            raise RuntimeError("RGG lambda/radius inversion mismatch")
        rgg_lambdas.append(lam)
        rows.append({
            "parameter_index": i,
            "control_c": float(c),
            "er_p": p_er,
            "er_expected_mean_degree": expected_degree,
            "rgg_lambda_generator": lam,
            "rgg_radius": radius,
            "rgg_edge_probability_exact": square_euclidean_connection_probability(radius),
            "rgg_expected_mean_degree": (n - 1) * square_euclidean_connection_probability(radius),
            "rgg_dim": dim,
            "rgg_p_norm": p_norm,
        })

    common_validation = {
        "n": n, "repeats": repeats, "base_seed": base_seed, "methods": methods,
        "bootstrap_samples": int(comp.get("bootstrap_samples", 1000)),
        "display_max_q": comp.get("display_max_q", 8),
        "save_pair_samples": bool(comp.get("save_pair_samples", False)),
        "path_length_mode": str(comp.get("path_length_mode", "none")),
        "path_sample_sources": int(comp.get("path_sample_sources", 64)),
        "plotting": deepcopy(comp.get("plotting", {})),
        "results_root": str(comparison_root / "subruns"),
    }

    er = {"method": deepcopy(payload["method"]), "baselines": deepcopy(payload.get("baselines", {}))}
    er["validation"] = {
        **common_validation,
        "model": "ER", "parameter_name": "p", "coordinate_label": "matched c = Np",
        "grid": {"mode": "explicit_np", "values": [float(x) for x in controls]},
        "fixed_params": {},
    }

    rgg = {"method": deepcopy(payload["method"]), "baselines": deepcopy(payload.get("baselines", {}))}
    rgg["validation"] = {
        **common_validation,
        "model": "RGG", "parameter_name": "lambda", "coordinate_label": "matched c = Np",
        "grid": {"mode": "explicit_pairs"},
        "parameter_values": [float(x) for x in rgg_lambdas],
        "control_values": [float(x) for x in controls],
        "fixed_params": {"dim": dim, "p_norm": p_norm},
    }
    # The RGG sub-run is displayed on the shared matched-c coordinate rather
    # than its native lambda coordinate, so map its theory guides as well.
    def _map_lambda_to_c(lam: float) -> float:
        rr = math.sqrt(float(lam) / (float(n) * math.pi))
        return float(n) * square_euclidean_connection_probability(rr)
    rgg["validation"]["theory_references_override"] = {
        "2D percolation guide (mapped to matched c)": _map_lambda_to_c(4.512),
        "connectivity guide ln N (mapped to matched c)": _map_lambda_to_c(math.log(n)),
    }
    return er, rgg, pd.DataFrame(rows)


def run_er_rgg_comparison(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or _progress
    if "comparison" not in payload:
        raise ValueError("comparison config requires a top-level 'comparison' object")
    cfg = method_from_payload(payload, config_path)
    comp = dict(payload["comparison"])
    n = int(comp.get("n", 1000))
    repeats = int(comp.get("repeats", 10))
    base_seed = int(comp.get("base_seed", 22))
    controls = [float(x) for x in comp.get("control_values", [])]
    if repeats < 2:
        raise ValueError("repeats must be >=2")
    if not controls or any((not math.isfinite(x) or x <= 0.0) for x in controls):
        raise ValueError("comparison.control_values must be a non-empty positive finite list")
    if any(b <= a for a, b in zip(controls[:-1], controls[1:])):
        raise ValueError("comparison.control_values must be strictly increasing")
    if controls[-1] > n:
        raise ValueError("control c cannot exceed N because ER uses p=c/N")
    matching = str(comp.get("matching", "finite_square_expected_degree"))

    results_root = resolve_path(comp.get("results_root", "results/validation/er_rgg_matched"), config_path)
    run_dir = results_root / _run_stem({**comp, "n": n, "repeats": repeats, "base_seed": base_seed}, cfg, controls, matching)
    complete = run_dir / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed ER/RGG comparison exists: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_outputs(run_dir)
    for d in ("tables", "plots", "subruns"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    er_payload, rgg_payload, mapping = _build_subpayloads(payload, controls, run_dir)
    mapping.to_csv(run_dir / "tables" / "matched_parameter_mapping.csv", index=False, encoding="utf-8-sig")

    progress("[SUBRUN] ER")
    er_run = run_synthetic_validation(er_payload, config_path, force=force, progress=progress)
    progress("[SUBRUN] RGG (finite-square expected-degree matched)")
    rgg_run = run_synthetic_validation(rgg_payload, config_path, force=force, progress=progress)

    er_graphs = pd.read_csv(er_run / "tables" / "graph_index.csv")
    rgg_graphs = pd.read_csv(rgg_run / "tables" / "graph_index.csv")
    methods = list(comp.get("methods", ["Hodge", "DMeasure", "NetLSD", "Portrait", "Count"]))
    manager = MethodManager.from_config(cfg, payload.get("baselines", {}))
    bootstrap_samples = int(comp.get("bootstrap_samples", 1000))
    er_noise = _noise_lookup(er_run)
    rgg_noise = _noise_lookup(rgg_run)

    summary_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    # Realized matching audit.
    realized_rows = []
    for i, c in enumerate(controls):
        ea = er_graphs[er_graphs.parameter_index == i]
        ra = rgg_graphs[rgg_graphs.parameter_index == i]
        realized_rows.append({
            "parameter_index": i, "control_c": c,
            "er_realized_mean_degree_mean": float(ea.realized_mean_degree.mean()),
            "er_realized_mean_degree_std": float(ea.realized_mean_degree.std(ddof=1)) if len(ea) > 1 else 0.0,
            "rgg_realized_mean_degree_mean": float(ra.realized_mean_degree.mean()),
            "rgg_realized_mean_degree_std": float(ra.realized_mean_degree.std(ddof=1)) if len(ra) > 1 else 0.0,
            "realized_degree_mean_difference_rgg_minus_er": float(ra.realized_mean_degree.mean() - ea.realized_mean_degree.mean()),
            "er_average_clustering_mean": float(ea.average_clustering.mean()),
            "rgg_average_clustering_mean": float(ra.average_clustering.mean()),
            "er_lcc_fraction_mean": float(ea.largest_component_fraction.mean()),
            "rgg_lcc_fraction_mean": float(ra.largest_component_fraction.mean()),
        })
    realized = mapping.merge(pd.DataFrame(realized_rows), on=["parameter_index", "control_c"], how="left")
    realized.to_csv(run_dir / "tables" / "matched_structural_audit.csv", index=False, encoding="utf-8-sig")

    for method in methods:
        progress(f"[COMPARE METHOD] {method}")
        for i, c in enumerate(controls):
            egrp = er_graphs[er_graphs.parameter_index == i].sort_values("repeat")
            rgrp = rgg_graphs[rgg_graphs.parameter_index == i].sort_values("repeat")
            ereps: list[tuple[GraphRecord, Any]] = []
            rreps: list[tuple[GraphRecord, Any]] = []
            for model_name, grp, target in (("ER", egrp, ereps), ("RGG", rgrp, rreps)):
                for _, row in grp.iterrows():
                    rec = _record_from_row(row)
                    t0 = time.perf_counter()
                    rep, elapsed, cache_hit = manager.representation(method, rec)
                    target.append((rec, rep))
                    runtime_rows.append({
                        "method": method, "model": model_name, "parameter_index": i,
                        "control_c": c, "repeat": int(row["repeat"]), "graph_hash": rec.graph_hash,
                        "elapsed_sec": float(elapsed), "cache_hit": bool(cache_hit),
                        "wall_sec": float(time.perf_counter() - t0),
                    })
            D = np.empty((len(ereps), len(rreps)), dtype=float)
            for a, (er_rec, er_rep) in enumerate(ereps):
                for b, (rg_rec, rg_rep) in enumerate(rreps):
                    D[a, b] = manager.distance(
                        method, er_rep, rg_rep,
                        er_rec.graph.number_of_nodes(), rg_rec.graph.number_of_nodes(),
                    )
            stats = _cross_stats(
                D, bootstrap_samples=bootstrap_samples,
                seed=stable_seed("er-rgg-cross-bootstrap", method, n, c, base_seed),
            )
            enoise = er_noise.get((method, i), float("nan"))
            rnoise = rgg_noise.get((method, i), float("nan"))
            background = float(np.nanmean([enoise, rnoise]))
            summary_rows.append({
                "method": method, "parameter_index": i, "control_c": c,
                "distance_mean": stats["mean"], "distance_std": stats["std"],
                "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
                "cross_pair_count": stats["pair_count"],
                "er_same_parameter_noise": enoise,
                "rgg_same_parameter_noise": rnoise,
                "background_noise_mean": background,
                "model_separation_snr": float(stats["mean"] / background) if background > 0 else np.inf,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "tables" / "matched_model_distance_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(run_dir / "tables" / "runtime_comparison_cache_audit.csv", index=False, encoding="utf-8-sig")

    effective = deepcopy(payload)
    effective["comparison"]["n"] = n
    effective["comparison"]["repeats"] = repeats
    effective["comparison"]["base_seed"] = base_seed
    effective["comparison"]["control_values"] = controls
    effective["resolved_matching"] = mapping.to_dict("records")
    effective["subruns"] = {"ER": str(er_run), "RGG": str(rgg_run)}
    write_json(run_dir / "effective_config.json", effective)

    from .plotting import plot_er_rgg_matched_comparison
    plot_er_rgg_matched_comparison(run_dir)

    write_json(run_dir / "run_manifest.json", {
        "experiment": "finite-square expected-degree-matched ER vs RGG",
        "method_core_changed": False,
        "n": n, "repeats": repeats, "base_seed": base_seed,
        "control_coordinate": "c=Np",
        "matching": "ER p=c/N; RGG radius solves exact U([0,1]^2) Euclidean connection probability=c/N",
        "sampling": "independent across model, control point, and repeat; no trajectory coupling",
        "distance_aggregation": "all ER x RGG realization pairs at each matched control coordinate",
        "cross_model_ci": "realization-level two-sample bootstrap over ER and RGG graphs",
        "separation_snr": "cross-model mean distance divided by mean of ER and RGG same-parameter noise means",
        "methods": methods,
        "subruns": {"ER": str(er_run), "RGG": str(rgg_run)},
        "method": cfg.to_dict(),
    })
    write_json(complete, {"status": "complete", "run_dir": str(run_dir), "method_core_changed": False})
    progress(f"[COMPLETE] {run_dir}")
    return run_dir
