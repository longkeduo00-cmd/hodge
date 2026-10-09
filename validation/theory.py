from __future__ import annotations

import math


def er_references(n: int) -> dict[str, float]:
    return {"c=1": 1.0, "c=ln N": math.log(float(n))}


def ws_references() -> dict[str, float]:
    return {"beta=1e-3": 1e-3, "beta=0.1": 1e-1}


def rgg_references(n: int, fixed_params: dict) -> dict[str, float]:
    dim = int(fixed_params.get("dim", fixed_params.get("rgg_dim", 2)))
    p_norm = float(fixed_params.get("p_norm", fixed_params.get("rgg_p_norm", 2.0)))
    if dim == 2 and abs(p_norm - 2.0) <= 1e-12:
        return {"2D percolation guide": 4.512, "connectivity guide ln N": math.log(float(n))}
    return {}


def sbm_references(blocks: int, mean_degree: float) -> dict[str, float]:
    q = float(blocks)
    mu_er = (q - 1.0) / q
    mu_ks = mu_er * (1.0 - 1.0 / math.sqrt(float(mean_degree)))
    return {"KS": mu_ks, "ER-like": mu_er}


def model_references(model: str, n: int, fixed_params: dict) -> dict[str, float]:
    model = model.upper()
    if model == "ER":
        return er_references(n)
    if model == "WS":
        return ws_references()
    if model == "RGG":
        return rgg_references(n, fixed_params)
    if model == "SBM":
        return sbm_references(int(fixed_params.get("blocks", 4)), float(fixed_params.get("mean_degree", 10)))
    return {}
