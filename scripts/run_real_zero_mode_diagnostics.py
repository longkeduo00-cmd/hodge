from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.real_zero_mode_diagnostics import run_real_zero_mode_diagnostics


def _cutoff(text: str):
    t = str(text).strip().lower()
    return None if t in {"all", "none", "null"} else int(t)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-network Hodge zero-mode / eigenvalue diagnostics")
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", help="Comma-separated real-network labels")
    ap.add_argument("--max-q", type=int)
    ap.add_argument("--max-k", type=int)
    ap.add_argument("--max-clique-size")
    ap.add_argument("--max-hodge-order")
    ap.add_argument("--dense-dimension-guard", type=int)
    ap.add_argument("--sparse-max-k", type=int)
    ap.add_argument("--no-store-eigenvalues", action="store_true")
    ap.add_argument("--output-root")
    ap.add_argument("--force", action="store_true", help="Rebuild run-level tables/plots while reusing legal D0 diagnostics.")
    ap.add_argument("--force-diagnostic-cache", action="store_true", help="Recompute D0 zero-mode and spectrum diagnostics.")
    args = ap.parse_args()

    p = Path(args.config).resolve()
    payload = json.loads(p.read_text(encoding="utf-8"))
    diag = payload.setdefault("real_zero_mode_diagnostics", {})
    if args.labels:
        payload.setdefault("real_validation", {})["labels"] = [x.strip() for x in args.labels.split(",") if x.strip()]
    if args.max_q is not None: diag["max_q"] = args.max_q
    if args.max_k is not None: diag["max_k"] = args.max_k
    if args.dense_dimension_guard is not None: diag["dense_dimension_guard"] = args.dense_dimension_guard
    if args.sparse_max_k is not None: diag["sparse_max_k"] = args.sparse_max_k
    if args.no_store_eigenvalues: diag["store_eigenvalues"] = False
    if args.output_root: diag["results_root"] = args.output_root
    if args.force_diagnostic_cache: diag["force_diagnostic_cache"] = True
    if args.max_clique_size is not None: payload["method"]["max_clique_size"] = _cutoff(args.max_clique_size)
    if args.max_hodge_order is not None: payload["method"]["max_hodge_order"] = _cutoff(args.max_hodge_order)
    run_real_zero_mode_diagnostics(payload, p, force=args.force)


if __name__ == "__main__":
    main()
