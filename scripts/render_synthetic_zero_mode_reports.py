from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.synthetic_zero_modes_plotting import (
    render_per_graph_spectrum_reports,
    zero_count_statistics,
    plot_k_order_zero_count_boxplots,
    plot_k_order_zero_fraction_boxplots,
)


def _ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Render real-style synthetic zero-mode spectral reports from an existing completed run; no Hodge recomputation.'
    )
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--k-orders', default='0')
    ap.add_argument('--q-min', type=int, default=0)
    ap.add_argument('--q-max', type=int)
    ap.add_argument('--columns', type=int, default=5)
    ap.add_argument('--zero-tolerance', type=float, default=1e-10)
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    table = run / 'tables' / 'synthetic_zero_modes_per_graph_channel.csv'
    if not table.exists():
        raise FileNotFoundError(f'missing channel table: {table}')
    channels = pd.read_csv(table)
    q_max = int(channels['q'].max()) if args.q_max is None else int(args.q_max)

    stats = zero_count_statistics(channels)
    stats.to_csv(run / 'tables' / 'synthetic_matrix_zero_statistics_by_model_degree_qk.csv', index=False, encoding='utf-8-sig')

    report_index = render_per_graph_spectrum_reports(
        channels, run, run / 'plots',
        k_orders=_ints(args.k_orders),
        q_min=int(args.q_min), q_max=q_max, columns=int(args.columns),
        zero_tolerance=float(args.zero_tolerance),
    )
    report_index.to_csv(run / 'tables' / 'synthetic_zero_mode_report_index.csv', index=False, encoding='utf-8-sig')
    for k in _ints(args.k_orders):
        plot_k_order_zero_count_boxplots(channels, run / 'plots', k=k)
        plot_k_order_zero_fraction_boxplots(channels, run / 'plots', k=k)
    print(f'[DONE] reports rendered under {run / "plots" / "zero_mode_report"}', flush=True)


if __name__ == '__main__':
    main()
