from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from validation.model_size_discrimination import run_model_size_discrimination

def main():
    ap=argparse.ArgumentParser(description='ER/WS/RGG network-size discrimination at fixed mean degree')
    ap.add_argument('--config',required=True); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    p=Path(args.config).resolve(); payload=json.loads(p.read_text(encoding='utf-8')); run_model_size_discrimination(payload,p,force=args.force)
if __name__=='__main__': main()
