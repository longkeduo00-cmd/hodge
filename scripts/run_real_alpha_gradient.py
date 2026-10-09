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
    ap = argparse.ArgumentParser(description="Continuous gradient learning of Hodge alpha on real domains")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--output-root")
    ap.add_argument("--cache-root")
    ap.add_argument("--outer-strategy", choices=["repeated_stratified_kfold", "repeated_stratified_holdout", "kfold", "holdout"])
    ap.add_argument("--outer-repeats", type=int)
    ap.add_argument("--outer-folds", type=int)
    ap.add_argument("--outer-seed", type=int)
    ap.add_argument("--test-fraction", type=float)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--initial-alpha", type=float)
    ap.add_argument("--inner-diagnostic-folds", type=int)
    ap.add_argument("--profile-workers", type=int, help="Parallel CPU worker processes for independent network Hodge profiles")
    ap.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads per worker; keep at 1 when profile-workers>1")
    ap.add_argument("--solver", choices=["exact", "slq"])
    ap.add_argument("--slq-probes", type=int)
    ap.add_argument("--slq-steps", type=int)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute-hodge", action="store_true")
    ap.add_argument("--recompute-baselines", action="store_true")
    args = ap.parse_args()

    if args.blas_threads < 1:
        raise ValueError("--blas-threads must be >=1")
    # Must be set before importing NumPy/SciPy through the validation package.
    threads = str(int(args.blas_threads))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads

    from validation.real_alpha_gradient import run_real_alpha_gradient

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp = payload["real_validation"]
    method = payload["method"]
    learning = payload.setdefault("learning", {})
    execution = payload.setdefault("execution", {})
    outer = learning.setdefault("outer_cv", {})
    opt = learning.setdefault("gradient_optimizer", {})

    if args.manifest: exp["manifest"] = args.manifest
    if args.output_root: learning["results_root"] = args.output_root
    if args.cache_root: method.setdefault("cache", {})["root"] = args.cache_root
    if args.outer_strategy: outer["strategy"] = args.outer_strategy
    if args.outer_repeats is not None: outer["repeats"] = args.outer_repeats
    if args.outer_folds is not None: outer["folds"] = args.outer_folds
    if args.outer_seed is not None: outer["seed"] = args.outer_seed
    if args.test_fraction is not None: outer["test_fraction"] = args.test_fraction
    if args.learning_rate is not None: opt["learning_rate"] = args.learning_rate
    if args.max_epochs is not None: opt["max_epochs"] = args.max_epochs
    if args.initial_alpha is not None: opt["initial_alpha"] = args.initial_alpha
    if args.inner_diagnostic_folds is not None: learning["inner_diagnostic_folds"] = args.inner_diagnostic_folds
    if args.profile_workers is not None:
        if args.profile_workers < 1:
            raise ValueError("--profile-workers must be >=1")
        execution["profile_workers"] = args.profile_workers
    if args.solver: method["solver"] = args.solver
    if args.slq_probes is not None: method.setdefault("slq", {})["num_probes"] = args.slq_probes
    if args.slq_steps is not None: method.setdefault("slq", {})["lanczos_steps"] = args.slq_steps
    if args.no_plots: learning["generate_plots"] = False

    run_real_alpha_gradient(
        payload, path,
        force=args.force,
        recompute_hodge=args.recompute_hodge,
        recompute_baselines=args.recompute_baselines,
    )


if __name__ == "__main__":
    main()
