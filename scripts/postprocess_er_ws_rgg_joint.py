from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.model_joint_postprocess import postprocess_joint_run


def main() -> None:
    ap = argparse.ArgumentParser(description='Pure post-processing for existing ER/WS/RGG joint N×mean-degree results')
    ap.add_argument('--run-dir', required=True, help='Existing completed run directory containing tables/ and arrays/.')
    args = ap.parse_args()
    postprocess_joint_run(Path(args.run_dir).resolve())


if __name__ == '__main__':
    main()
