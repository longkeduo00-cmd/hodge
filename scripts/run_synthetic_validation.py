from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.runner import run_synthetic_validation


def _cutoff(text: str):
    t = str(text).strip().lower()
    return None if t in {"all", "none", "null"} else int(t)


def _float_or_inf(text: str) -> float:
    t = str(text).strip().lower()
    if t in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    value = float(t)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be finite or +inf")
    return value


def _float_list(text: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numeric values") from exc
    if not values or any(not math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError("expected non-empty finite comma-separated numeric values")
    return values


def main():
    ap = argparse.ArgumentParser(description="MCID-style synthetic validation for unchanged Hodge VNext core")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--base-seed", type=int)
    ap.add_argument("--methods")
    ap.add_argument("--solver", choices=["exact", "slq"])
    ap.add_argument("--max-clique-size")
    ap.add_argument("--max-hodge-order")
    ap.add_argument("--normalization", choices=["none", "per_node", "per_simplex"])
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--time-min", type=float)
    ap.add_argument("--time-max", type=float)
    ap.add_argument("--num-times", type=int)
    ap.add_argument("--slq-probes", type=int)
    ap.add_argument("--slq-steps", type=int)
    ap.add_argument("--bootstrap-samples", type=int)
    ap.add_argument("--display-max-q", type=int)
    ap.add_argument(
        "--parameter-values",
        type=_float_list,
        help="Override the explicit generator/control parameter grid; for RGG these are lambda values.",
    )
    ap.add_argument(
        "--rgg-lambda-values",
        type=_float_list,
        help="RGG-only alias for --parameter-values.",
    )
    ap.add_argument("--rgg-dim", type=int, help="RGG embedding dimension (>=1).")
    ap.add_argument(
        "--rgg-p-norm",
        type=_float_or_inf,
        help="RGG Minkowski p-norm (>=1, or 'inf').",
    )
    ap.add_argument("--output-root")
    ap.add_argument("--cache-root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp = payload["validation"]
    method = payload["method"]
    model = str(exp["model"]).upper()

    for key, val in (
        ("n", args.n),
        ("repeats", args.repeats),
        ("base_seed", args.base_seed),
        ("bootstrap_samples", args.bootstrap_samples),
        ("display_max_q", args.display_max_q),
    ):
        if val is not None:
            exp[key] = val

    if args.methods:
        exp["methods"] = [x.strip() for x in args.methods.split(",") if x.strip()]
    if args.output_root:
        exp["results_root"] = args.output_root

    if args.parameter_values is not None and args.rgg_lambda_values is not None:
        ap.error("use only one of --parameter-values and --rgg-lambda-values")
    grid_override = args.rgg_lambda_values if args.rgg_lambda_values is not None else args.parameter_values
    if args.rgg_lambda_values is not None and model != "RGG":
        ap.error("--rgg-lambda-values is only valid for an RGG config")
    if grid_override is not None:
        exp["parameter_values"] = [float(x) for x in grid_override]
        exp["grid"] = {"mode": "explicit"}

    if args.rgg_dim is not None or args.rgg_p_norm is not None:
        if model != "RGG":
            ap.error("--rgg-dim/--rgg-p-norm are only valid for an RGG config")
        fixed = exp.setdefault("fixed_params", {})
        if args.rgg_dim is not None:
            if args.rgg_dim < 1:
                ap.error("--rgg-dim must be >= 1")
            fixed["dim"] = int(args.rgg_dim)
        if args.rgg_p_norm is not None:
            if args.rgg_p_norm < 1.0:
                ap.error("--rgg-p-norm must be >= 1")
            fixed["p_norm"] = float(args.rgg_p_norm)

    if args.solver:
        method["solver"] = args.solver
    if args.max_clique_size is not None:
        method["max_clique_size"] = _cutoff(args.max_clique_size)
    if args.max_hodge_order is not None:
        method["max_hodge_order"] = _cutoff(args.max_hodge_order)
    if args.normalization:
        method["normalization"] = args.normalization
    if args.alpha is not None:
        method["alpha"] = args.alpha
    if args.time_min is not None:
        method["time_min"] = args.time_min
    if args.time_max is not None:
        method["time_max"] = args.time_max
    if args.num_times is not None:
        method["time_points"] = args.num_times
    if args.slq_probes is not None:
        method.setdefault("slq", {})["num_probes"] = args.slq_probes
    if args.slq_steps is not None:
        method.setdefault("slq", {})["lanczos_steps"] = args.slq_steps
    if args.cache_root:
        method.setdefault("cache", {})["root"] = args.cache_root

    run_synthetic_validation(payload, path, force=args.force)


if __name__ == "__main__":
    main()
