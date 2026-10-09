from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from validation.model_joint_discrimination import run_joint_discrimination

def _csv_num(text,cast): return [cast(x.strip()) for x in str(text).split(',') if x.strip()]
def main():
    ap=argparse.ArgumentParser(description='Joint ER/WS/RGG discrimination over network size and mean degree')
    ap.add_argument('--config',required=True); ap.add_argument('--network-sizes'); ap.add_argument('--target-degrees'); ap.add_argument('--repeats',type=int); ap.add_argument('--methods'); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    p=Path(args.config).resolve(); payload=json.loads(p.read_text(encoding='utf-8')); exp=payload['validation']
    if args.network_sizes: exp['network_sizes']=_csv_num(args.network_sizes,int)
    if args.target_degrees: exp['target_degrees']=_csv_num(args.target_degrees,float)
    if args.repeats is not None: exp['repeats']=args.repeats
    if args.methods: exp['methods']=[x.strip() for x in args.methods.split(',') if x.strip()]
    run_joint_discrimination(payload,p,force=args.force)
if __name__=='__main__': main()
