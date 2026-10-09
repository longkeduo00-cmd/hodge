from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Repeated stratified 5-fold diagnosis: train-only q scaling × Uniform/Single-alpha/Free-q"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--output-root")
    ap.add_argument("--cache-root")
    ap.add_argument("--precomputed-layer-tensor")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--folds", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--profile-workers", type=int)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute-hodge", action="store_true")
    ap.add_argument("--recompute-baselines", action="store_true")
    args = ap.parse_args()

    if args.blas_threads < 1:
        raise ValueError("--blas-threads must be >=1")
    threads = str(int(args.blas_threads))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads

    from validation.real_q_weight_scale_cv import run_real_q_weight_scale_cv

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp = payload["real_validation"]
    method = payload["method"]
    learning = payload.setdefault("learning", {})
    outer = learning.setdefault("outer_cv", {})
    opt = learning.setdefault("free_q_optimizer", {})
    execution = payload.setdefault("execution", {})

    if args.manifest:
        exp["manifest"] = args.manifest
    if args.output_root:
        learning["results_root"] = args.output_root
    if args.cache_root:
        method.setdefault("cache", {})["root"] = args.cache_root
    if args.precomputed_layer_tensor:
        learning["precomputed_layer_tensor"] = args.precomputed_layer_tensor
    if args.repeats is not None:
        if args.repeats < 1:
            raise ValueError("--repeats must be >=1")
        outer["repeats"] = args.repeats
    if args.folds is not None:
        if args.folds < 2:
            raise ValueError("--folds must be >=2")
        outer["folds"] = args.folds
    if args.seed is not None:
        outer["seed"] = args.seed
    if args.max_epochs is not None:
        opt["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        opt["learning_rate"] = args.learning_rate
    if args.profile_workers is not None:
        if args.profile_workers < 1:
            raise ValueError("--profile-workers must be >=1")
        execution["profile_workers"] = args.profile_workers
    if args.no_plots:
        learning["generate_plots"] = False

    run_real_q_weight_scale_cv(
        payload,
        path,
        force=args.force,
        recompute_hodge=args.recompute_hodge,
        recompute_baselines=args.recompute_baselines,
    )


if __name__ == "__main__":
    main()
