from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def _int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description='Synthetic-network independent-q Hodge zero-eigenvalue diagnostic')
    ap.add_argument('--config', required=True)
    ap.add_argument('--cache-root')
    ap.add_argument('--output-root')
    ap.add_argument('--n', type=int)
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--target-degrees')
    ap.add_argument('--max-q', type=int)
    ap.add_argument('--max-k', type=int)
    ap.add_argument('--dense-guard', type=int)
    ap.add_argument('--sparse-max-k', type=int)
    ap.add_argument('--report-k-orders', help='comma-separated Hodge orders for real-style spectrum reports; default from config, normally 0')
    ap.add_argument('--report-q-min', type=int)
    ap.add_argument('--report-q-max', type=int)
    ap.add_argument('--report-columns', type=int)
    ap.add_argument('--no-spectrum-reports', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--force-diagnostics', action='store_true')
    ap.add_argument('--blas-threads', type=int, default=1)
    args = ap.parse_args()

    if args.blas_threads < 1:
        raise ValueError('--blas-threads must be >=1')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(int(args.blas_threads))

    from validation.synthetic_zero_modes import run_synthetic_zero_modes

    path = Path(args.config).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    exp = payload.setdefault('synthetic_zero_modes', {})
    method = payload['method']
    diag = exp.setdefault('diagnostics', {})
    if args.cache_root:
        method.setdefault('cache', {})['root'] = args.cache_root
    if args.output_root:
        exp['results_root'] = args.output_root
    if args.n is not None:
        exp['n'] = int(args.n)
    if args.repeats is not None:
        exp['repeats'] = int(args.repeats)
    if args.target_degrees:
        exp['target_mean_degrees'] = _float_list(args.target_degrees)
    if args.max_q is not None:
        exp['max_q'] = int(args.max_q)
    if args.max_k is not None:
        exp['max_k'] = int(args.max_k)
    if args.dense_guard is not None:
        diag['dense_dimension_guard'] = int(args.dense_guard)
    if args.sparse_max_k is not None:
        diag['sparse_max_k'] = int(args.sparse_max_k)
    report = exp.setdefault('spectrum_reports', {})
    if args.report_k_orders:
        report['k_orders'] = _int_list(args.report_k_orders)
    if args.report_q_min is not None:
        report['q_min'] = int(args.report_q_min)
    if args.report_q_max is not None:
        report['q_max'] = int(args.report_q_max)
    if args.report_columns is not None:
        report['columns'] = int(args.report_columns)
    if args.no_spectrum_reports:
        report['enabled'] = False
    if args.no_plots:
        exp['generate_plots'] = False

    run_synthetic_zero_modes(payload, path, force=args.force, force_diagnostics=args.force_diagnostics)


if __name__ == '__main__':
    main()
