from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.real_alpha_sweep import run_real_alpha_sweep


def _cutoff(text: str):
    t = str(text).strip().lower()
    return None if t in {"all", "none", "null"} else int(t)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-network Hodge low-order alpha sweep with full historical real plots")
    ap.add_argument("--config", required=True)
    ap.add_argument("--alphas", help="Comma-separated values, e.g. 1,0.75,0.5,0.25,0,-0.25,-0.5,-0.75,-1")
    ap.add_argument("--reference-alpha", type=float)
    ap.add_argument("--baseline-policy", choices=["reference_only", "all_methods_each_alpha"])
    ap.add_argument("--labels", help="Comma-separated real-network labels")
    ap.add_argument("--max-per-domain", type=int)
    ap.add_argument("--methods", help="Comma-separated methods; default comes from the real config")
    ap.add_argument("--max-clique-size")
    ap.add_argument("--max-hodge-order")
    ap.add_argument("--normalization", choices=["none", "per_node", "per_simplex"])
    ap.add_argument("--output-root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    p = Path(args.config).resolve()
    payload = json.loads(p.read_text(encoding="utf-8"))
    sweep = payload.setdefault("real_alpha_sweep", {})
    if args.alphas:
        sweep["alphas"] = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if args.reference_alpha is not None: sweep["reference_alpha"] = args.reference_alpha
    if args.baseline_policy: sweep["baseline_policy"] = args.baseline_policy
    if args.output_root: sweep["results_root"] = args.output_root
    exp = payload.setdefault("real_validation", {})
    if args.labels: exp["labels"] = [x.strip() for x in args.labels.split(",") if x.strip()]
    if args.max_per_domain is not None: exp["max_per_domain"] = args.max_per_domain
    if args.methods: exp["methods"] = [x.strip() for x in args.methods.split(",") if x.strip()]
    if args.max_clique_size is not None: payload["method"]["max_clique_size"] = _cutoff(args.max_clique_size)
    if args.max_hodge_order is not None: payload["method"]["max_hodge_order"] = _cutoff(args.max_hodge_order)
    if args.normalization: payload["method"]["normalization"] = args.normalization
    run_real_alpha_sweep(payload, p, force=args.force)


if __name__ == "__main__":
    main()
