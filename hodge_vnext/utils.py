from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def float_token(value: float) -> str:
    if math.isfinite(value):
        text = format(float(value), ".12g")
    else:
        text = str(value)
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def safe_token(value: Any) -> str:
    if isinstance(value, float):
        return float_token(value)
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    return text.strip("-") or "x"


def params_slug(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    return "__".join(f"{safe_token(k)}={safe_token(v)}" for k, v in sorted(params.items()))


def collision_safe_dir(parent: Path, stem: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / stem
    if not candidate.exists():
        candidate.mkdir()
        return candidate
    idx = 2
    while True:
        candidate = parent / f"{stem}_{idx:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        idx += 1


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted(set(int(x) for x in values))
