from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ParameterGrid:
    values: np.ndarray
    coordinates: np.ndarray
    parameter_name: str
    coordinate_label: str


def _explicit(exp: dict[str, Any]) -> ParameterGrid:
    values = np.asarray(exp.get("parameter_values", []), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("explicit grid requires a non-empty experiment.parameter_values list.")
    name = str(exp["parameter_name"])
    return ParameterGrid(values, values.copy(), name, str(exp.get("coordinate_label", name)))




def _explicit_pairs(exp: dict[str, Any]) -> ParameterGrid:
    """Explicit generator values with distinct display/control coordinates.

    This is used by controlled cross-model experiments where the generator
    parameter must be transformed to preserve a common scientific coordinate
    (for example finite-square degree-matched ER vs RGG).
    """
    values = np.asarray(exp.get("parameter_values", []), dtype=float)
    coordinates = np.asarray(exp.get("control_values", []), dtype=float)
    if values.ndim != 1 or coordinates.ndim != 1 or values.size == 0:
        raise ValueError("explicit_pairs requires non-empty 1-D parameter_values and control_values")
    if values.size != coordinates.size:
        raise ValueError("explicit_pairs requires parameter_values and control_values of equal length")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coordinates)):
        raise ValueError("explicit_pairs values and coordinates must be finite")
    name = str(exp["parameter_name"])
    return ParameterGrid(values, coordinates, name, str(exp.get("coordinate_label", name)))

def _explicit_np(exp: dict[str, Any], n: int, grid: dict[str, Any]) -> ParameterGrid:
    coordinates = np.asarray(grid.get("values", exp.get("control_values", [])), dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("explicit_np grid requires grid.values or experiment.control_values")
    values = coordinates / float(n)
    return ParameterGrid(values, coordinates, str(exp["parameter_name"]), str(exp.get("coordinate_label", "c=Np")))


def _piecewise_linear_np(exp: dict[str, Any], n: int, grid: dict[str, Any]) -> ParameterGrid:
    anchors = np.asarray(grid["anchors"], dtype=float)
    points_per_interval = int(grid["points_per_interval"])
    if anchors.ndim != 1 or anchors.size < 2 or np.any(np.diff(anchors) <= 0):
        raise ValueError("ER piecewise_linear_np anchors must be a strictly increasing 1-D list.")
    if anchors[0] < 0 or anchors[-1] > n:
        raise ValueError("ER Np anchors must satisfy 0 <= Np <= N.")
    if points_per_interval < 2:
        raise ValueError("points_per_interval must be >= 2.")
    pieces = []
    for idx, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        segment = np.linspace(float(left), float(right), points_per_interval)
        if idx:
            segment = segment[1:]
        pieces.append(segment)
    coordinates = np.concatenate(pieces)
    values = coordinates / float(n)
    return ParameterGrid(values, coordinates, str(exp["parameter_name"]), "Np")


def _decade(exp: dict[str, Any], grid: dict[str, Any]) -> ParameterGrid:
    values: list[float] = []
    if bool(grid.get("include_zero", True)):
        values.append(float(grid.get("zero_value", 0.0)))
    base = float(grid.get("base", 10.0))
    for exponent in grid["exponents"]:
        for multiplier in grid["multipliers"]:
            values.append(float(multiplier) * base ** float(exponent))
    if bool(grid.get("include_endpoint", True)):
        values.append(float(grid.get("endpoint_value", 1.0)))
    arr = np.asarray(sorted(set(values)), dtype=float)
    return ParameterGrid(arr, arr.copy(), str(exp["parameter_name"]), str(exp.get("coordinate_label", exp["parameter_name"])))


def _linear(exp: dict[str, Any], grid: dict[str, Any]) -> ParameterGrid:
    arr = np.linspace(float(grid["minimum"]), float(grid["maximum"]), int(grid["points"]))
    return ParameterGrid(arr, arr.copy(), str(exp["parameter_name"]), str(exp.get("coordinate_label", exp["parameter_name"])))


def build_parameter_grid(exp: dict[str, Any]) -> ParameterGrid:
    n = int(exp["n"])
    grid = dict(exp.get("grid", {}))
    mode = str(grid.get("mode", "explicit"))
    if mode == "explicit":
        return _explicit(exp)
    if mode == "explicit_np":
        return _explicit_np(exp, n, grid)
    if mode == "explicit_pairs":
        return _explicit_pairs(exp)
    if mode == "piecewise_linear_np":
        return _piecewise_linear_np(exp, n, grid)
    if mode == "decade":
        return _decade(exp, grid)
    if mode == "linear":
        return _linear(exp, grid)
    raise ValueError(f"Unsupported parameter grid mode: {mode}")
