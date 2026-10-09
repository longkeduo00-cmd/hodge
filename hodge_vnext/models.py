from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import networkx as nx

from .graph_io import GraphRecord, GRAPH_SCHEMA, graph_hash, load_graph_npz, save_graph_npz
from .utils import params_slug, read_json, stable_seed, write_json

GENERATOR_VERSION = "vnext-independent-2"
RGG_GENERATOR_VERSION = "vnext-rgg-independent-1"


def lp_unit_ball_volume(dim: int, p_norm: float) -> float:
    """Volume of the unit L^p ball in ``dim`` dimensions.

    This is used only to parameterize RGG radius via
    lambda = N * V_{d,p} * r^d.  It does not alter the hard-threshold
    geometric graph definition used by NetworkX.
    """
    dim = int(dim)
    p_norm = float(p_norm)
    if dim < 1:
        raise ValueError("RGG dim must be >= 1.")
    if math.isnan(p_norm) or p_norm < 1.0:
        raise ValueError("RGG p_norm must satisfy 1 <= p_norm <= infinity.")
    if math.isinf(p_norm):
        return float(2.0 ** dim)
    return float((2.0 * math.gamma(1.0 + 1.0 / p_norm)) ** dim / math.gamma(1.0 + dim / p_norm))


def rgg_radius_from_lambda(n: int, lambda_value: float, dim: int = 2, p_norm: float = 2.0) -> float:
    """Map the RGG control coordinate lambda to the hard connection radius."""
    n = int(n)
    lam = float(lambda_value)
    dim = int(dim)
    if n < 1:
        raise ValueError("RGG n must be >= 1.")
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("RGG lambda must be finite and nonnegative.")
    volume = lp_unit_ball_volume(dim, float(p_norm))
    return float((lam / (float(n) * volume)) ** (1.0 / dim))


def rgg_resolved_parameters(n: int, params: dict[str, Any]) -> dict[str, Any]:
    """Return validated RGG parameters including deterministic derived radius."""
    if "lambda" not in params:
        raise ValueError("RGG requires parameter 'lambda'.")
    dim = int(params.get("dim", 2))
    p_norm = float(params.get("p_norm", 2.0))
    lam = float(params["lambda"] )
    radius = rgg_radius_from_lambda(n, lam, dim=dim, p_norm=p_norm)
    return {
        "lambda": lam,
        "dim": dim,
        "p_norm": p_norm,
        "unit_ball_volume": lp_unit_ball_volume(dim, p_norm),
        "radius": radius,
    }


def _generate_er(n: int, params: dict[str, Any], seed: int) -> nx.Graph:
    p = float(params["p"])
    if not 0.0 <= p <= 1.0:
        raise ValueError("ER p must be in [0,1].")
    return nx.fast_gnp_random_graph(n, p, seed=seed, directed=False)


def _generate_ws(n: int, params: dict[str, Any], seed: int) -> nx.Graph:
    degree = int(params["degree"])
    p = float(params["p"])
    if degree <= 0 or degree >= n or degree % 2:
        raise ValueError("WS degree must be positive, even, and < n.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("WS p must be in [0,1].")
    return nx.watts_strogatz_graph(n, degree, p, seed=seed)


def _generate_sbm(n: int, params: dict[str, Any], seed: int) -> nx.Graph:
    blocks = int(params["blocks"])
    mean_degree = float(params["mean_degree"])
    mu = float(params["mu"])
    if blocks < 2 or blocks > n:
        raise ValueError("SBM blocks must satisfy 2 <= blocks <= n.")
    if not 0.0 <= mu <= 1.0:
        raise ValueError("SBM mu must be in [0,1].")
    if mean_degree < 0:
        raise ValueError("SBM mean_degree must be nonnegative.")
    base, rem = divmod(n, blocks)
    sizes = [base + (1 if i < rem else 0) for i in range(blocks)]
    sbar = n / blocks
    p_in = (1.0 - mu) * mean_degree / max(sbar - 1.0, 1.0)
    p_out = mu * mean_degree / max(n - sbar, 1.0)
    if p_in > 1.0 + 1e-12 or p_out > 1.0 + 1e-12:
        raise ValueError("SBM probabilities exceed 1; reduce mean_degree or adjust blocks/mu.")
    p_in = min(max(p_in, 0.0), 1.0)
    p_out = min(max(p_out, 0.0), 1.0)
    probs = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
    return nx.stochastic_block_model(sizes, probs, seed=seed, directed=False, selfloops=False)


def _generate_rgg(n: int, params: dict[str, Any], seed: int) -> nx.Graph:
    resolved = rgg_resolved_parameters(n, params)
    return nx.random_geometric_graph(
        n,
        resolved["radius"],
        dim=resolved["dim"],
        p=resolved["p_norm"],
        seed=seed,
    )


_GENERATORS = {"ER": _generate_er, "WS": _generate_ws, "SBM": _generate_sbm, "RGG": _generate_rgg}


def generate_independent_graph(model: str, n: int, params: dict[str, Any], graph_seed: int) -> nx.Graph:
    model = model.upper()
    if model not in _GENERATORS:
        raise ValueError(f"Unsupported synthetic model {model}; supported={sorted(_GENERATORS)}")
    G = _GENERATORS[model](n, params, graph_seed)
    if G.number_of_nodes() != n:
        raise RuntimeError("Generator returned unexpected node count.")
    return G


def synthetic_graph_record(
    cache_root: str | Path,
    model: str,
    n: int,
    params: dict[str, Any],
    base_seed: int,
    repeat_index: int,
) -> GraphRecord:
    if repeat_index < 1:
        raise ValueError("repeat_index is 1-based and must be >= 1.")
    model = model.upper()
    repeat_seed = int(base_seed) + int(repeat_index) - 1
    generator_version = RGG_GENERATOR_VERSION if model == "RGG" else GENERATOR_VERSION
    source_spec = {
        "graph_schema": GRAPH_SCHEMA,
        "generator_version": generator_version,
        "model": model,
        "n": int(n),
        "params": {str(k): params[k] for k in sorted(params)},
        "base_seed": int(base_seed),
        "repeat_index": int(repeat_index),
        "repeat_seed": repeat_seed,
    }
    if model == "RGG":
        source_spec["resolved_params"] = rgg_resolved_parameters(n, params)
    graph_seed = stable_seed(
        generator_version, model, n,
        json.dumps(source_spec["params"], sort_keys=True, separators=(",", ":")),
        repeat_seed,
    )
    source_spec["graph_seed"] = int(graph_seed)

    root = (
        Path(cache_root) / "synthetic" / model / f"N{n}" / params_slug(params)
        / f"base_seed{base_seed}" / f"repeat_{repeat_index:03d}_seed{repeat_seed}"
    )
    root.mkdir(parents=True, exist_ok=True)

    for candidate in sorted(root.glob("graph_*/C0_graph/source.json")):
        try:
            existing = read_json(candidate)
        except Exception:
            continue
        if existing == source_spec:
            graph_root = candidate.parents[1]
            G = load_graph_npz(graph_root / "C0_graph" / "graph.npz")
            gid = graph_root.name.removeprefix("graph_").split("_")[0]
            return GraphRecord(G, gid, graph_root, source_spec)

    G = generate_independent_graph(model, n, params, int(graph_seed))
    gid = graph_hash(G)
    graph_root = root / f"graph_{gid}"
    if graph_root.exists():
        stage = graph_root / "C0_graph"
        if (stage / "source.json").exists() and read_json(stage / "source.json") == source_spec:
            return GraphRecord(load_graph_npz(stage / "graph.npz"), gid, graph_root, source_spec)
        idx = 2
        while (root / f"graph_{gid}_{idx:03d}").exists():
            idx += 1
        graph_root = root / f"graph_{gid}_{idx:03d}"
    stage = graph_root / "C0_graph"
    stage.mkdir(parents=True, exist_ok=False)
    save_graph_npz(stage / "graph.npz", G)
    write_json(stage / "source.json", source_spec)
    return GraphRecord(G, gid, graph_root, source_spec)
