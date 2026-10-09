from dataclasses import replace
from pathlib import Path

import pytest

from hodge_vnext.config import MethodConfig
from validation.real_domains import _resolve_run_dir


def test_override_compacts_only_run_output_path(tmp_path):
    cfg = MethodConfig(max_clique_size=10, min_layer_q=0, normalization="per_simplex", solver="slq")
    default = _resolve_run_dir(tmp_path, 48, cfg, "real48-v1-curated")
    compact = _resolve_run_dir(tmp_path, 48, cfg, "real48-v1-curated", "a0p5")
    assert "S10" in default.name and "qmin0" in default.name and "norm-per_simplex" in default.name
    assert len(default.name) < 90
    assert compact == tmp_path / "a0p5"
    assert default != _resolve_run_dir(tmp_path, 48, replace(cfg, alpha=2), "real48-v1-curated")


@pytest.mark.parametrize("name", ["../other", "..", "C:/other", "a/b", "a\\b"])
def test_run_name_cannot_escape_output_root(tmp_path, name):
    with pytest.raises(ValueError):
        _resolve_run_dir(tmp_path, 48, MethodConfig(), "real48", name)
