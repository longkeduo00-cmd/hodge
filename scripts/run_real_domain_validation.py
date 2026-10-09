from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from validation.real_domains import run_real_domain_validation


def _cutoff(text: str):
    t=str(text).strip().lower(); return None if t in {"all","none","null"} else int(t)


def main():
    ap=argparse.ArgumentParser(description="MCID-style real-domain validation for Hodge VNext")
    ap.add_argument("--config",required=True); ap.add_argument("--manifest"); ap.add_argument("--methods")
    ap.add_argument("--max-nodes",type=int); ap.add_argument("--max-edges",type=int); ap.add_argument("--max-per-domain",type=int)
    ap.add_argument("--solver",choices=["exact","slq"]); ap.add_argument("--max-clique-size"); ap.add_argument("--max-hodge-order")
    ap.add_argument("--normalization",choices=["none","per_node","per_simplex"]); ap.add_argument("--alpha",type=float)
    ap.add_argument("--time-min",type=float); ap.add_argument("--time-max",type=float); ap.add_argument("--num-times",type=int)
    ap.add_argument("--slq-probes",type=int); ap.add_argument("--slq-steps",type=int)
    ap.add_argument("--output-root"); ap.add_argument("--cache-root"); ap.add_argument("--force",action="store_true")
    args=ap.parse_args(); path=Path(args.config).resolve(); payload=json.loads(path.read_text(encoding="utf-8"))
    exp=payload["real_validation"]; method=payload["method"]
    if args.manifest: exp["manifest"]=args.manifest
    if args.methods: exp["methods"]=[x.strip() for x in args.methods.split(",") if x.strip()]
    for key,val in (("max_nodes",args.max_nodes),("max_edges",args.max_edges),("max_per_domain",args.max_per_domain)):
        if val is not None: exp[key]=val
    if args.output_root: exp["results_root"]=args.output_root
    if args.solver: method["solver"]=args.solver
    if args.max_clique_size is not None: method["max_clique_size"]=_cutoff(args.max_clique_size)
    if args.max_hodge_order is not None: method["max_hodge_order"]=_cutoff(args.max_hodge_order)
    if args.normalization: method["normalization"]=args.normalization
    if args.alpha is not None: method["alpha"]=args.alpha
    if args.time_min is not None: method["time_min"]=args.time_min
    if args.time_max is not None: method["time_max"]=args.time_max
    if args.num_times is not None: method["time_points"]=args.num_times
    if args.slq_probes is not None: method.setdefault("slq",{})["num_probes"]=args.slq_probes
    if args.slq_steps is not None: method.setdefault("slq",{})["lanczos_steps"]=args.slq_steps
    if args.cache_root: method.setdefault("cache",{})["root"]=args.cache_root
    run_real_domain_validation(payload,path,force=args.force)

if __name__=="__main__": main()
