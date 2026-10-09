from __future__ import annotations

"""Stage-addressed cache for Hodge VNext.

The mathematical method is unchanged.  This module changes only artifact
organization and cache identity.  It follows the same design principle used by
MCID: graph identity is separated from method stages, and each numerical stage
is cached under the parameters that can actually change that stage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import numpy as np
from scipy import sparse

from .config import CacheConfig
from .utils import read_json, stable_hash, write_json

CACHE_SCHEMA = "hodge-vnext-stage-cache-v2"


@dataclass
class ArtifactStore:
    graph_root: Path
    config: CacheConfig

    @property
    def root(self) -> Path:
        p = Path(self.graph_root) / "stages" / CACHE_SCHEMA
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stage_root(self, stage: str) -> Path:
        p = self.root / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def layer_root(self, q: int) -> Path:
        # Backward-compatible helper for callers that expect one q directory.
        p = self.root / "Q_layers" / f"q{int(q):04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # C1: complete globally maximal-clique enumeration.
    def cliques_file(self) -> Path:
        p = self.stage_root("C1_cliques")
        return p / "maximal_cliques.pkl"

    def cliques_manifest(self) -> Path:
        return self.stage_root("C1_cliques") / "manifest.json"

    # Q2: downward closure of each independent exclusive q-layer.
    def simplex_file(self, q: int, r: int) -> Path:
        p = self.stage_root("Q2_simplices") / f"q{int(q):04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"K_r{int(r):04d}.pkl"

    # Q3: oriented boundary matrices per q-layer.
    def boundary_file(self, q: int, r: int) -> Path:
        p = self.stage_root("Q3_boundary") / f"q{int(q):04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"B_r{int(r):04d}.npz"

    # Q4: Hodge Laplacians per (q,k).
    def hodge_file(self, q: int, k: int) -> Path:
        p = self.stage_root("Q4_hodge") / f"q{int(q):04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"L_k{int(k):04d}.npz"

    # Q5: exact/SLQ spectral objects.
    def exact_dir(self, q: int, k: int, numerical_key: dict[str, Any]) -> Path:
        digest = stable_hash({"schema": CACHE_SCHEMA, **numerical_key}, 16)
        p = self.stage_root("Q5_spectral") / "exact" / f"q{q:04d}" / f"k{k:04d}_{digest}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def slq_dir(self, q: int, k: int, numerical_key: dict[str, Any]) -> Path:
        digest = stable_hash({"schema": CACHE_SCHEMA, **numerical_key}, 16)
        p = self.stage_root("Q5_spectral") / "slq" / f"q{q:04d}" / f"k{k:04d}_{digest}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Q6: raw heat traces. Normalization is deliberately not cached here.
    def heat_file(self, q: int, k: int, solver_key: dict[str, Any], time_key: dict[str, Any]) -> Path:
        solver_digest = stable_hash({"schema": CACHE_SCHEMA, **solver_key}, 16)
        time_digest = stable_hash(time_key, 16)
        p = self.stage_root("Q6_heat") / f"q{q:04d}" / f"k{k:04d}" / solver_digest
        p.mkdir(parents=True, exist_ok=True)
        return p / f"raw_{time_digest}.npy"

    # Q7: complete per-graph profile under structural/numerical settings.
    def profile_file(self, profile_key: dict[str, Any]) -> Path:
        p = self.stage_root("Q7_profiles")
        return p / f"profile_{stable_hash({'schema': CACHE_SCHEMA, **profile_key}, 20)}.pkl"

    # B0: classical baseline representations, outside the Hodge core stages.
    def baseline_file(self, method: str, settings: dict[str, Any]) -> Path:
        p = self.stage_root("B0_baselines")
        digest = stable_hash({"schema": CACHE_SCHEMA, "method": method, **settings}, 20)
        return p / f"{method.lower()}_{digest}.pkl"

    def diagnostics_dir(self) -> Path:
        return self.stage_root("D0_diagnostics")

    def load_pickle(self, path: Path) -> Any | None:
        if not self.config.enabled or not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save_pickle(self, path: Path, obj: Any, enabled: bool = True) -> None:
        if not self.config.enabled or not enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        with path.open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_sparse(self, path: Path) -> sparse.csr_matrix | None:
        if not self.config.enabled or not path.exists():
            return None
        return sparse.load_npz(path).tocsr()

    def save_sparse(self, path: Path, matrix: sparse.spmatrix, enabled: bool = True) -> None:
        if not self.config.enabled or not enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        sparse.save_npz(path, matrix.tocsr())

    def load_array(self, path: Path) -> np.ndarray | None:
        if not self.config.enabled or not path.exists():
            return None
        return np.load(path, allow_pickle=False)

    def save_array(self, path: Path, arr: np.ndarray, enabled: bool = True) -> None:
        if not self.config.enabled or not enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        np.save(path, arr)

    def write_manifest_once(self, path: Path, payload: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        payload = {"cache_schema": CACHE_SCHEMA, **payload}
        if path.exists():
            existing = read_json(path)
            if existing != payload:
                idx = 2
                while path.with_name(f"{path.stem}_{idx:03d}{path.suffix}").exists():
                    idx += 1
                write_json(path.with_name(f"{path.stem}_{idx:03d}{path.suffix}"), payload)
            return
        write_json(path, payload)
