from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.er_rgg_comparison import run_er_rgg_comparison


def _cutoff(text: str):
    t = str(text).strip().lower()
    return None if t in {"all", "none", "null"} else int(t)


def _float_list(text: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numeric values") from exc
    if not values or any(not math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError("expected non-empty finite comma-separated numeric values")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run finite-square expected-degree-matched ER vs 2-D Euclidean RGG comparison"
    )
    ap.add_argument("--config", default="experiments/synthetic/ER_RGG/er_rgg_matched.json")
    ap.add_argument("--n", type=int)
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--base-seed", type=int)
    ap.add_argument("--control-values", type=_float_list, help="Shared c=Np grid for ER and RGG matching")
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
    ap.add_argument("--output-root")
    ap.add_argument("--cache-root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    comp = payload["comparison"]
    method = payload["method"]

    for key, value in (
        ("n", args.n), ("repeats", args.repeats), ("base_seed", args.base_seed),
        ("bootstrap_samples", args.bootstrap_samples),
    ):
        if value is not None:
            comp[key] = value
    if args.control_values is not None:
        comp["control_values"] = [float(x) for x in args.control_values]
    if args.methods:
        comp["methods"] = [x.strip() for x in args.methods.split(",") if x.strip()]
    if args.output_root:
        comp["results_root"] = args.output_root

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

    run_er_rgg_comparison(payload, path, force=args.force)


if __name__ == "__main__":
    main()
