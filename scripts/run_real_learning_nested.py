from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.real_learning_nested import run_real_learning_nested


def main() -> None:
    ap = argparse.ArgumentParser(description="Repeated nested CV / repeated outer hold-out for real48 Hodge learning")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--output-root")
    ap.add_argument("--cache-root")
    ap.add_argument("--outer-strategy", choices=["repeated_stratified_kfold", "repeated_stratified_holdout", "kfold", "holdout"])
    ap.add_argument("--outer-repeats", type=int)
    ap.add_argument("--outer-folds", type=int)
    ap.add_argument("--outer-seed", type=int)
    ap.add_argument("--test-fraction", type=float)
    ap.add_argument("--inner-folds", type=int)
    ap.add_argument("--inner-seed", type=int)
    ap.add_argument("--primary-metric", choices=["nn1_macro_f1", "nn1_accuracy", "mAP", "auroc_same_domain", "ndcg_at_7", "precision_at_1"])
    ap.add_argument("--mst-permutations", type=int)
    ap.add_argument("--solver", choices=["exact", "slq"])
    ap.add_argument("--slq-probes", type=int)
    ap.add_argument("--slq-steps", type=int)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--force", action="store_true", help="Regenerate nested run-level reports while reusing legal staged/candidate/full-distance caches")
    ap.add_argument("--recompute-candidates", action="store_true")
    ap.add_argument("--recompute-baselines", action="store_true")
    args = ap.parse_args()

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp = payload["real_validation"]
    method = payload["method"]
    learning = payload.setdefault("learning", {})
    nested = learning.setdefault("nested_cv", {})

    if args.manifest:
        exp["manifest"] = args.manifest
    if args.output_root:
        learning["results_root"] = args.output_root
    if args.cache_root:
        method.setdefault("cache", {})["root"] = args.cache_root
    if args.outer_strategy:
        nested["outer_strategy"] = args.outer_strategy
    if args.outer_repeats is not None:
        nested["outer_repeats"] = args.outer_repeats
    if args.outer_folds is not None:
        nested["outer_folds"] = args.outer_folds
    if args.outer_seed is not None:
        nested["outer_seed"] = args.outer_seed
    if args.test_fraction is not None:
        nested["test_fraction"] = args.test_fraction
    if args.inner_folds is not None:
        nested["inner_folds"] = args.inner_folds
    if args.inner_seed is not None:
        nested["inner_seed"] = args.inner_seed
    if args.primary_metric:
        learning["primary_metric"] = args.primary_metric
    if args.mst_permutations is not None:
        nested["mst_permutations"] = args.mst_permutations
    if args.solver:
        method["solver"] = args.solver
    if args.slq_probes is not None:
        method.setdefault("slq", {})["num_probes"] = args.slq_probes
    if args.slq_steps is not None:
        method.setdefault("slq", {})["lanczos_steps"] = args.slq_steps
    if args.no_plots:
        learning["generate_plots"] = False

    run_real_learning_nested(
        payload,
        path,
        force=args.force,
        recompute_candidates=args.recompute_candidates,
        recompute_baselines=args.recompute_baselines,
    )


if __name__ == "__main__":
    main()
