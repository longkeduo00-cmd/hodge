from __future__ import annotations

"""MCID-style synthetic validation runner for the unchanged Hodge VNext core."""

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
import json
import math
import shutil
import time

import networkx as nx
import numpy as np
import pandas as pd

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers
from hodge_vnext.config import MethodConfig
from hodge_vnext.distance import active_hodge_count, profile_distance
from hodge_vnext.graph_io import GraphRecord, load_graph_npz
from hodge_vnext.grid import ParameterGrid, build_parameter_grid
from hodge_vnext.models import rgg_resolved_parameters, synthetic_graph_record
from hodge_vnext.profile import GraphProfile, compute_graph_profile
from hodge_vnext.utils import safe_token, stable_hash, stable_seed, write_json

from .config import method_from_payload, project_root, resolve_path
from .methods import DEFAULT_METHODS, MethodManager, PRIMARY_METHOD
from .theory import model_references


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _run_stem(exp: dict[str, Any], cfg: MethodConfig, grid: ParameterGrid) -> str:
    S = "all" if cfg.max_clique_size is None else str(cfg.max_clique_size)
    Q = "all" if cfg.max_hodge_order is None else str(cfg.max_hodge_order)
    parts = [
        f"N{int(exp['n'])}", f"rep{int(exp['repeats'])}", f"seed{int(exp['base_seed'])}",
        f"solver-{cfg.solver}", f"S{S}", f"Q{Q}", f"qmin{cfg.min_layer_q}",
        f"alpha{safe_token(cfg.alpha)}", f"norm-{safe_token(cfg.normalization)}",
        f"t{safe_token(cfg.time_min)}-{safe_token(cfg.time_max)}-T{cfg.time_points}",
        f"grid{safe_token(float(grid.coordinates[0]))}-{safe_token(float(grid.coordinates[-1]))}-P{len(grid.values)}",
    ]
    # Preserve result paths for ER/WS/SBM. RGG includes its generator parameters.
    # identity additionally protects against dim/p-norm or internal-grid
    # collisions that share the same endpoints and point count.
    if str(exp.get("model", "")).upper() == "RGG":
        fixed = dict(exp.get("fixed_params", {}))
        dim = int(fixed.get("dim", 2))
        p_norm = float(fixed.get("p_norm", 2.0))
        identity = {
            "parameter_name": grid.parameter_name,
            "fixed_params": fixed,
            "generator_values": [float(x) for x in grid.values],
            "control_coordinates": [float(x) for x in grid.coordinates],
        }
        parts += [
            f"dim{dim}", f"p{safe_token(p_norm)}",
            f"rggid-{stable_hash(identity, length=10)}",
        ]
    if cfg.solver == "slq":
        parts += [f"R{cfg.slq.num_probes}", f"M{cfg.slq.lanczos_steps}"]
    return "_".join(parts)


def _clear_run_outputs(run_dir: Path) -> None:
    for name in ("tables", "arrays", "plots"):
        p = run_dir / name
        if p.exists():
            shutil.rmtree(p)
    for p in (run_dir / "RUN_COMPLETE.json", run_dir / "run_manifest.json"):
        if p.exists():
            p.unlink()


def _graph_observables(G: nx.Graph, model: str, path_mode: str, sample_sources: int, seed: int) -> dict[str, float | int]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    components = sorted((set(c) for c in nx.connected_components(G)), key=len, reverse=True)
    largest = components[0]
    second = components[1] if len(components) > 1 else set()
    lcc = G.subgraph(largest)
    if len(largest) <= 1 or path_mode == "none":
        apl = 0.0 if len(largest) <= 1 else float("nan")
    elif path_mode == "exact":
        apl = float(nx.average_shortest_path_length(lcc))
    elif path_mode == "sampled":
        nodes = np.asarray(sorted(lcc.nodes()), dtype=int)
        r = min(int(sample_sources), len(nodes))
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(nodes, size=r, replace=False)
        vals = []
        for s in chosen:
            d = nx.single_source_shortest_path_length(lcc, int(s))
            vals.append(sum(d.values()) / max(len(d) - 1, 1))
        apl = float(np.mean(vals))
    else:
        raise ValueError("path_length_mode must be none, exact, or sampled")

    finite = [len(c) for c in components[1:]]
    denom = sum(finite)
    susceptibility = float(sum(x * x for x in finite) / denom) if denom else 0.0
    isolates = int(nx.number_of_isolates(G))
    return {
        "num_nodes": int(n),
        "num_edges": int(m),
        "realized_mean_degree": float(2 * m / max(n, 1)),
        "num_components": int(len(components)),
        "largest_component_fraction": float(len(largest) / n),
        "second_component_fraction": float(len(second) / n),
        "finite_cluster_susceptibility": susceptibility,
        "num_isolates": isolates,
        "isolated_fraction": float(isolates / n),
        "average_clustering": float(nx.average_clustering(G)),
        "lcc_average_path_length": apl,
    }


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1 or samples <= 0:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=float)
    for b in range(int(samples)):
        means[b] = float(np.mean(rng.choice(x, size=x.size, replace=True)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _stats(values: list[float] | np.ndarray, bootstrap_samples: int, seed: int) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "count": 0}
    lo, hi = _bootstrap_ci(x, bootstrap_samples, seed)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "count": int(x.size),
    }


def _midpoint(model: str, left: float, right: float) -> float:
    if model in {"ER", "WS", "RGG"}:
        if left <= 0:
            return right / 2.0
        return math.sqrt(left * right)
    return 0.5 * (left + right)


def _record_from_row(row: pd.Series, cfg: MethodConfig) -> GraphRecord:
    root = Path(str(row.cache_graph_root))
    G = load_graph_npz(root / "C0_graph" / "graph.npz")
    source = json.loads((root / "C0_graph" / "source.json").read_text(encoding="utf-8"))
    return GraphRecord(G, str(row.graph_hash), root, source)


def _distance_for_pair(manager: MethodManager, method: str, ra: GraphRecord, rb: GraphRecord) -> tuple[float, Any, Any]:
    a, _, _ = manager.representation(method, ra)
    b, _, _ = manager.representation(method, rb)
    return manager.distance(method, a, b, ra.graph.number_of_nodes(), rb.graph.number_of_nodes()), a, b


def _weighted_layer_contributions(result, cfg: MethodConfig) -> dict[int, float]:
    out: dict[int, float] = {}
    for q, local_d in result.layer_distances.items():
        count = active_hodge_count(q, cfg.max_hodge_order)
        w = result.layer_weights[q] / count
        out[q] = float(math.sqrt(w) * local_d)
    return out


def run_synthetic_validation(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or _progress
    cfg = method_from_payload(payload, config_path)
    exp = dict(payload["validation"])
    model = str(exp["model"]).upper()
    n = int(exp["n"])
    repeats = int(exp["repeats"])
    base_seed = int(exp.get("base_seed", 22))
    if repeats < 2:
        raise ValueError("repeats must be >=2 for same-parameter noise and ensemble distances")
    grid_exp = {
        "n": n,
        "parameter_name": exp["parameter_name"],
        "coordinate_label": exp.get("coordinate_label", exp["parameter_name"]),
        "grid": exp.get("grid", {"mode": "explicit"}),
        "parameter_values": exp.get("parameter_values", []),
        "control_values": exp.get("control_values", []),
    }
    grid = build_parameter_grid(grid_exp)
    fixed_params = dict(exp.get("fixed_params", {}))
    if model == "RGG":
        if str(exp["parameter_name"]) != "lambda":
            raise ValueError("RGG validation requires parameter_name='lambda'.")
        # Validate dim/p-norm once even before graph generation.
        probe_lambda = float(grid.values[0])
        rgg_resolved_parameters(n, {**fixed_params, "lambda": probe_lambda})
        if np.any(~np.isfinite(grid.values)) or np.any(np.asarray(grid.values) <= 0.0):
            raise ValueError("RGG lambda grid must contain finite positive values for the standard log-scale plots.")
        if np.any(np.diff(np.asarray(grid.coordinates, dtype=float)) <= 0.0):
            raise ValueError("RGG lambda grid must be strictly increasing for adjacent-distance analysis.")
    methods = MethodManager.validate_methods(list(exp.get("methods", [PRIMARY_METHOD])))
    manager = MethodManager.from_config(cfg, payload.get("baselines", {}))
    bootstrap_samples = int(exp.get("bootstrap_samples", 1000))
    display_max_q = exp.get("display_max_q", 8)
    display_max_q = None if display_max_q is None else int(display_max_q)
    save_pair_samples = bool(exp.get("save_pair_samples", False))
    path_mode = str(exp.get("path_length_mode", "none"))
    path_sources = int(exp.get("path_sample_sources", 64))

    results_root = resolve_path(exp.get("results_root", "results/validation/synthetic"), config_path)
    run_dir = results_root / model / _run_stem(exp, cfg, grid)
    complete = run_dir / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed run exists: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_run_outputs(run_dir)
    for d in ("tables", "arrays", "plots"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    effective = json.loads(json.dumps(payload))
    effective["resolved_grid"] = {
        "parameter_name": grid.parameter_name,
        "coordinate_label": grid.coordinate_label,
        "generator_values": grid.values.tolist(),
        "control_coordinates": grid.coordinates.tolist(),
    }
    if model == "RGG":
        resolved = [rgg_resolved_parameters(n, {**fixed_params, "lambda": float(v)}) for v in grid.values]
        effective["resolved_grid"]["derived_radius"] = [float(x["radius"]) for x in resolved]
        effective["resolved_grid"]["rgg_dim"] = int(resolved[0]["dim"])
        effective["resolved_grid"]["rgg_p_norm"] = float(resolved[0]["p_norm"])
        effective["resolved_grid"]["rgg_unit_ball_volume"] = float(resolved[0]["unit_ball_volume"])
    effective["cache_schema"] = CACHE_SCHEMA
    write_json(run_dir / "effective_config.json", effective)

    # ------------------------------------------------------------------
    # 1) Independent graph ensemble + graph observables + clique abundance.
    # ------------------------------------------------------------------
    graph_rows: list[dict[str, Any]] = []
    clique_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    if model == "RGG":
        mapping_rows = []
        for pi, value in enumerate(grid.values):
            rr = rgg_resolved_parameters(n, {**fixed_params, "lambda": float(value)})
            mapping_rows.append({
                "parameter_index": pi,
                "N": n,
                "dim": int(rr["dim"]),
                "p_norm": float(rr["p_norm"]),
                "lambda": float(rr["lambda"]),
                "unit_ball_volume": float(rr["unit_ball_volume"]),
                "radius": float(rr["radius"]),
            })
        pd.DataFrame(mapping_rows).to_csv(
            run_dir / "tables" / "rgg_parameter_mapping.csv", index=False, encoding="utf-8-sig"
        )
    for pi, value in enumerate(grid.values):
        params = dict(fixed_params)
        params[str(exp["parameter_name"])] = float(value)
        progress(f"[PARAM] {pi+1}/{len(grid.values)} {grid.coordinate_label}={grid.coordinates[pi]:.8g}")
        for repeat in range(1, repeats + 1):
            rec = synthetic_graph_record(cfg.cache.root, model, n, params, base_seed, repeat)
            obs = _graph_observables(
                rec.graph, model, path_mode, path_sources,
                stable_seed("observable", rec.graph_hash, path_sources),
            )
            store = ArtifactStore(rec.cache_root, cfg.cache)
            layers = enumerate_maximal_clique_layers(rec.graph, store, progress=None)
            total_cliques = sum(len(v) for v in layers.values())
            for q, items in sorted(layers.items()):
                clique_rows.append({
                    "parameter_index": pi, "parameter_value": float(value),
                    "coordinate": float(grid.coordinates[pi]), "repeat": repeat,
                    "graph_hash": rec.graph_hash, "q": int(q),
                    "clique_size": int(q + 1), "maximal_clique_count": int(len(items)),
                    "maximal_clique_share": float(len(items) / total_cliques) if total_cliques else 0.0,
                })
            row = {
                "model": model, "N": n, "parameter_index": pi,
                "parameter_value": float(value), "coordinate": float(grid.coordinates[pi]),
                "repeat": repeat, "repeat_seed": base_seed + repeat - 1,
                "graph_seed": int(rec.source["graph_seed"]), "graph_hash": rec.graph_hash,
                "cache_graph_root": str(rec.cache_root), "num_maximal_cliques": int(total_cliques),
                "maximal_q": int(max(layers)) if layers else -1, **obs,
            }
            if model == "RGG":
                rr = rec.source["resolved_params"]
                row.update({
                    "rgg_lambda": float(rr["lambda"]),
                    "rgg_radius": float(rr["radius"]),
                    "rgg_dim": int(rr["dim"]),
                    "rgg_p_norm": float(rr["p_norm"]),
                    "rgg_unit_ball_volume": float(rr["unit_ball_volume"]),
                })
            graph_rows.append(row)

    graph_df = pd.DataFrame(graph_rows)
    clique_df = pd.DataFrame(clique_rows)
    # Ensemble abundance statistics require structural zeros.  If a q-layer is
    # absent in one realization it contributes count/share=0 rather than being
    # silently dropped from the group mean.  This fixes the previous
    # E[C_q | C_q>0] bias at q-layer appearance/disappearance boundaries.
    if not graph_df.empty and int(graph_df.maximal_q.max()) >= 0:
        q_values = list(range(int(graph_df.maximal_q.max()) + 1))
        dense_rows: list[dict[str, Any]] = []
        lookup = {
            (int(r.parameter_index), int(r.repeat), int(r.q)): r
            for r in clique_df.itertuples(index=False)
        }
        for g in graph_df.itertuples(index=False):
            total = int(g.num_maximal_cliques)
            for q in q_values:
                key = (int(g.parameter_index), int(g.repeat), int(q))
                hit = lookup.get(key)
                count = int(hit.maximal_clique_count) if hit is not None else 0
                dense_rows.append({
                    "parameter_index": int(g.parameter_index),
                    "parameter_value": float(g.parameter_value),
                    "coordinate": float(g.coordinate),
                    "repeat": int(g.repeat),
                    "graph_hash": str(g.graph_hash),
                    "q": int(q),
                    "clique_size": int(q + 1),
                    "maximal_clique_count": count,
                    "maximal_clique_share": float(count / total) if total else 0.0,
                })
        clique_df = pd.DataFrame(dense_rows)
    graph_df.to_csv(run_dir / "tables" / "graph_index.csv", index=False, encoding="utf-8-sig")
    graph_df.to_csv(run_dir / "tables" / "graph_observables.csv", index=False, encoding="utf-8-sig")
    clique_df.to_csv(run_dir / "tables" / "clique_abundance_per_graph.csv", index=False, encoding="utf-8-sig")
    clique_summary = clique_df.groupby(["parameter_index", "coordinate", "q"], as_index=False).agg(
        mean_count=("maximal_clique_count", "mean"), std_count=("maximal_clique_count", "std"),
        mean_share=("maximal_clique_share", "mean"), std_share=("maximal_clique_share", "std"),
    )
    clique_summary.to_csv(run_dir / "tables" / "clique_abundance_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 2) Method-by-method ensemble distances. Each pair uses raw distance.
    # ------------------------------------------------------------------
    adjacent_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    snr_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    all_pair_samples: dict[str, list[np.ndarray]] = defaultdict(list)

    grouped = {i: graph_df[graph_df.parameter_index == i].reset_index(drop=True) for i in range(len(grid.values))}

    for method in methods:
        progress(f"[METHOD] {method}")
        m = len(grid.values)
        mean_mat = np.zeros((m, m), dtype=float)
        std_mat = np.zeros((m, m), dtype=float)
        count_mat = np.zeros((m, m), dtype=int)
        cell_rows = []

        # Preload representations parameter by parameter for this method only.
        reps: dict[int, list[tuple[GraphRecord, Any]]] = {}
        for i in range(m):
            bucket = []
            for _, row in grouped[i].iterrows():
                rec = _record_from_row(row, cfg)
                t0 = time.perf_counter()
                rep, elapsed, cache_hit = manager.representation(method, rec)
                bucket.append((rec, rep))
                runtime_rows.append({
                    "method": method, "graph_hash": rec.graph_hash, "parameter_index": i,
                    "repeat": int(row["repeat"]), "elapsed_sec": float(elapsed), "cache_hit": bool(cache_hit),
                    "wall_sec": float(time.perf_counter() - t0),
                })
            reps[i] = bucket

        for i in range(m):
            for j in range(i, m):
                vals: list[float] = []
                q_contrib_samples: dict[int, list[float]] = defaultdict(list)
                left, right = reps[i], reps[j]
                if i == j:
                    pairs = [(a, b) for a in range(len(left)) for b in range(a + 1, len(right))]
                else:
                    pairs = [(a, b) for a in range(len(left)) for b in range(len(right))]
                for a, b in pairs:
                    ra, repa = left[a]
                    rb, repb = right[b]
                    if method == PRIMARY_METHOD:
                        result = profile_distance(repa, repb, cfg)
                        d = float(result.total_distance)
                        if j == i + 1:
                            for q, v in _weighted_layer_contributions(result, cfg).items():
                                q_contrib_samples[int(q)].append(v)
                    else:
                        d = manager.distance(method, repa, repb, ra.graph.number_of_nodes(), rb.graph.number_of_nodes())
                    vals.append(d)
                stats = _stats(vals, bootstrap_samples, stable_seed("cell-bootstrap", model, method, i, j, base_seed))
                mean_mat[i, j] = mean_mat[j, i] = float(stats["mean"])
                std_mat[i, j] = std_mat[j, i] = float(stats["std"])
                count_mat[i, j] = count_mat[j, i] = int(stats["count"])
                cell_rows.append({
                    "method": method, "i": i, "j": j,
                    "coordinate_i": float(grid.coordinates[i]), "coordinate_j": float(grid.coordinates[j]),
                    "distance_mean": stats["mean"], "distance_std": stats["std"],
                    "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
                    "pair_count": stats["count"],
                })
                if i == j:
                    noise_rows.append({"method": method, "parameter_index": i, "coordinate": float(grid.coordinates[i]), **stats})
                if j == i + 1:
                    xmid = _midpoint(model, float(grid.coordinates[i]), float(grid.coordinates[j]))
                    adjacent_rows.append({
                        "method": method, "left_index": i, "right_index": j,
                        "coordinate_left": float(grid.coordinates[i]), "coordinate_right": float(grid.coordinates[j]),
                        "coordinate_mid": xmid, "distance_mean": stats["mean"], "distance_std": stats["std"],
                        "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"], "pair_count": stats["count"],
                    })
                    if method == PRIMARY_METHOD:
                        for q, samples in sorted(q_contrib_samples.items()):
                            qstats = _stats(samples, bootstrap_samples, stable_seed("q-bootstrap", model, q, i, base_seed))
                            order_rows.append({
                                "left_index": i, "right_index": j, "coordinate_mid": xmid, "q": q,
                                "weighted_q_distance_mean": qstats["mean"],
                                "weighted_q_distance_std": qstats["std"],
                                "ci95_low": qstats["ci95_low"], "ci95_high": qstats["ci95_high"],
                                "pair_count": qstats["count"],
                            })
                    if save_pair_samples:
                        all_pair_samples[method].append(np.asarray(vals, dtype=float))

        np.save(run_dir / "arrays" / f"distance_mean_{method}.npy", mean_mat)
        np.save(run_dir / "arrays" / f"distance_std_{method}.npy", std_mat)
        np.save(run_dir / "arrays" / f"distance_count_{method}.npy", count_mat)
        pd.DataFrame(cell_rows).to_csv(run_dir / "tables" / f"distance_cells_{method}.csv", index=False, encoding="utf-8-sig")
        if save_pair_samples:
            np.savez_compressed(run_dir / "arrays" / f"pair_samples_{method}.npz", **{
                f"adjacent_{i:03d}": a for i, a in enumerate(all_pair_samples[method])
            })

    adjacent_df = pd.DataFrame(adjacent_rows)
    noise_df = pd.DataFrame(noise_rows)
    order_df = pd.DataFrame(order_rows)
    adjacent_df.to_csv(run_dir / "tables" / "adjacent_distance_summary.csv", index=False, encoding="utf-8-sig")
    noise_df.to_csv(run_dir / "tables" / "same_parameter_noise.csv", index=False, encoding="utf-8-sig")
    order_df.to_csv(run_dir / "tables" / "hodge_order_adjacent.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(run_dir / "tables" / "runtime_by_graph.csv", index=False, encoding="utf-8-sig")

    # SNR uses mean adjacent signal / average noise at the two endpoints.
    for _, row in adjacent_df.iterrows():
        method = row.method
        i, j = int(row.left_index), int(row.right_index)
        ni = noise_df[(noise_df.method == method) & (noise_df.parameter_index == i)]["mean"]
        nj = noise_df[(noise_df.method == method) & (noise_df.parameter_index == j)]["mean"]
        background = float(np.nanmean([float(ni.iloc[0]) if len(ni) else np.nan, float(nj.iloc[0]) if len(nj) else np.nan]))
        snr_rows.append({
            "method": method, "left_index": i, "right_index": j,
            "coordinate_mid": float(row.coordinate_mid), "signal": float(row.distance_mean),
            "background_noise": background,
            "signal_noise_ratio": float(row.distance_mean / background) if background > 0 else np.inf,
        })
    pd.DataFrame(snr_rows).to_csv(run_dir / "tables" / "signal_noise_ratio.csv", index=False, encoding="utf-8-sig")

    # Transition detection: peak distance per log interval for ER/WS/RGG, per linear interval for SBM/others.
    transition_rows = []
    refs = dict(exp.get("theory_references_override") or model_references(model, n, fixed_params))
    for method, sub in adjacent_df.groupby("method", sort=False):
        sub = sub.sort_values("left_index")
        sens = []
        for _, r in sub.iterrows():
            l, rr = float(r.coordinate_left), float(r.coordinate_right)
            if model in {"ER", "WS", "RGG"} and l > 0:
                width = abs(math.log10(rr) - math.log10(l))
            else:
                width = abs(rr - l)
            sens.append(float(r.distance_mean) / width if width > 0 else np.nan)
        sens = np.asarray(sens, dtype=float)
        if np.isfinite(sens).any():
            idx = int(np.nanargmax(sens))
            peak = float(sub.iloc[idx].coordinate_mid)
            row = {"method": method, "detected_peak": peak, "peak_sensitivity": float(sens[idx])}
            for name, value in refs.items():
                row[f"error_to_{safe_token(name)}"] = abs(peak - value)
                if peak > 0 and value > 0:
                    row[f"log10_error_to_{safe_token(name)}"] = abs(math.log10(peak) - math.log10(value))
            transition_rows.append(row)
    pd.DataFrame(transition_rows).to_csv(run_dir / "tables" / "transition_detection_metrics.csv", index=False, encoding="utf-8-sig")

    # Hodge self-distance contract check.
    first = _record_from_row(graph_df.iloc[0], cfg)
    p0 = compute_graph_profile(first, cfg)
    self_d = profile_distance(p0, p0, cfg).total_distance
    if abs(self_d) > 1e-12:
        raise RuntimeError(f"Hodge self-distance contract failed: {self_d}")

    write_json(run_dir / "run_manifest.json", {
        "architecture": "MCID-style separation: method core / validation / benchmarks",
        "method_core_changed": False,
        "cache_schema": CACHE_SCHEMA,
        "model": model, "N": n, "repeats": repeats, "methods": methods,
        "fixed_params": fixed_params,
        "resolved_grid": effective["resolved_grid"],
        "sampling": "independent",
        "distance_aggregation": "all cross-realization pairs; same-parameter unordered pairs",
        "bootstrap_samples": bootstrap_samples,
        "theory_references": refs,
        "method": cfg.to_dict(),
        "baseline_settings": {
            m: manager.baseline.method_settings(m)
            for m in methods if m != PRIMARY_METHOD
        },
    })

    from .plotting import plot_synthetic_validation
    plot_synthetic_validation(run_dir, model=model, display_max_q=display_max_q)
    write_json(complete, {"status": "complete", "run_dir": str(run_dir), "method_core_changed": False})
    progress(f"[COMPLETE] {run_dir}")
    return run_dir
