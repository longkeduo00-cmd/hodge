from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.hodge_alpha_sweep import run_hodge_alpha_sweep


def _csv(text: str, cast):
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Hodge low-order alpha sweep using shared cached q,k heat profiles")
    ap.add_argument("--config", required=True)
    ap.add_argument("--alphas", help="Comma-separated alpha values, e.g. 1,0.75,0.5,0.25,0,-0.5,-1")
    ap.add_argument("--network-sizes")
    ap.add_argument("--target-degrees")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--persist-graph-distance-matrices", action="store_true")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    p = Path(args.config).resolve()
    payload = json.loads(p.read_text(encoding="utf-8"))
    sweep = payload.setdefault("alpha_sweep", {})
    if args.alphas:
        sweep["alphas"] = _csv(args.alphas, float)
    if args.network_sizes:
        sweep["network_sizes"] = _csv(args.network_sizes, int)
    if args.target_degrees:
        sweep["target_degrees"] = _csv(args.target_degrees, float)
    if args.repeats is not None:
        sweep["repeats"] = args.repeats
    if args.block_size is not None:
        sweep["block_size"] = args.block_size
    if args.persist_graph_distance_matrices:
        sweep["persist_graph_distance_matrices"] = True
    if args.keep_scratch:
        sweep["keep_scratch"] = True
    run_hodge_alpha_sweep(payload, p, force=args.force)


if __name__ == "__main__":
    main()
