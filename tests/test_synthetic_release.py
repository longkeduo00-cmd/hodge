import json
from pathlib import Path

import numpy as np
import pytest

from hodge_vnext.models import _GENERATORS, generate_independent_graph
from validation.synthetic_zero_modes import _params_for_target

ROOT = Path(__file__).resolve().parents[1]


def test_release_generators_exclude_ba():
    assert set(_GENERATORS) == {"ER", "WS", "SBM", "RGG"}
    with pytest.raises(ValueError, match="Unsupported synthetic model"):
        generate_independent_graph("BA", 20, {"m": 2}, 42)
    with pytest.raises(ValueError, match="unsupported model"):
        _params_for_target("BA", 20, 4, {})


def test_no_ba_experiment_or_zero_mode_preset():
    for path in (ROOT / "experiments/synthetic").rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "BA" not in path.parts
        assert payload.get("validation", {}).get("model") != "BA"
        assert "BA" not in payload.get("synthetic_zero_modes", {}).get("models", {})


def test_rgg_zero_preset_uses_existing_radius_parameterization():
    payload = json.loads((ROOT / "experiments/synthetic/ZERO_MODES/synthetic_zero_modes_rgg.json").read_text())
    exp = payload["synthetic_zero_modes"]
    assert set(exp["models"]) == {"RGG"}
    for coordinate in exp["target_mean_degrees"]:
        params = _params_for_target("RGG", exp["n"], coordinate, exp["models"]["RGG"])
        assert np.isclose(params["lambda"], coordinate)
        assert params["dim"] == 2 and params["p_norm"] == 2
