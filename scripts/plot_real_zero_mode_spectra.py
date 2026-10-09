from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.real_zero_mode_reporting import generate_real_zero_mode_report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate exact/resolved zero-mode tables and per-network spectral dashboards from an existing real zero-mode run"
    )
    ap.add_argument("--run-dir", required=True, help="Path to real_zero_modes_<hash> result directory")
    ap.add_argument("--labels", help="Optional comma-separated subset of network labels")
    ap.add_argument("--max-q", type=int, help="Optional maximum exclusive q-layer to plot/report")
    ap.add_argument("--no-spectral-dashboards", action="store_true", help="Skip per-network per-q positive-spectrum dashboards")
    ap.add_argument("--no-network-heatmaps", action="store_true", help="Skip per-network q-by-k zero-count/fraction heatmaps")
    ap.add_argument("--pdf", action="store_true", help="Also save PDF copies; PNG is always saved")
    args = ap.parse_args()

    labels = None
    if args.labels:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    result = generate_real_zero_mode_report(
        args.run_dir,
        labels=labels,
        max_q=args.max_q,
        spectral_dashboards=not args.no_spectral_dashboards,
        network_heatmaps=not args.no_network_heatmaps,
        save_pdf=args.pdf,
    )
    print(f"[DONE] networks={result.networks} channels={result.channels} resolved={result.resolved_channels}")
    print(f"[TABLES] {result.tables_dir}")
    print(f"[PLOTS]  {result.plots_dir}")


if __name__ == "__main__":
    main()
