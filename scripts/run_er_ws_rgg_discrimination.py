from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.model_discrimination import run_model_discrimination


def _csv_floats(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty float list")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Matched-density ER/WS/RGG mechanism discrimination")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--base-seed", type=int)
    ap.add_argument("--target-degrees", type=_csv_floats)
    ap.add_argument("--methods")
    ap.add_argument("--output-root")
    ap.add_argument("--cache-root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp = payload["validation"]
    method = payload["method"]
    if args.n is not None:
        exp["n"] = args.n
    if args.repeats is not None:
        exp["repeats"] = args.repeats
    if args.base_seed is not None:
        exp["base_seed"] = args.base_seed
    if args.target_degrees is not None:
        exp["target_degrees"] = args.target_degrees
    if args.methods:
        exp["methods"] = [x.strip() for x in args.methods.split(",") if x.strip()]
    if args.output_root:
        exp["results_root"] = args.output_root
    if args.cache_root:
        method.setdefault("cache", {})["root"] = args.cache_root

    run_model_discrimination(payload, path, force=args.force)


if __name__ == "__main__":
    main()
