from __future__ import annotations

import math
import numpy as np

from validation.model_discrimination import rgg_square_pair_probability
from validation.model_size_discrimination import build_size_cells, _eligible_mask


def test_size_cells_hold_target_mean_degree_contract():
    cells = build_size_cells([100, 200, 500], 10.0, [0.001, 0.01, 0.1, 1.0])
    assert len(cells) == 18
    er = cells[cells.mechanism_state == "ER"]
    for row in er.itertuples(index=False):
        assert math.isclose(float(row.er_p) * (int(row.network_size) - 1), 10.0, rel_tol=0, abs_tol=1e-12)
    rgg = cells[cells.mechanism_state == "RGG"]
    for row in rgg.itertuples(index=False):
        expected = (int(row.network_size) - 1) * rgg_square_pair_probability(float(row.rgg_radius))
        assert math.isclose(expected, 10.0, rel_tol=0, abs_tol=1e-9)
    ws = cells[cells.mechanism_state.str.startswith("WS-")]
    assert set(ws.ws_degree.astype(int)) == {10}


def test_size_protocol_candidate_masks():
    sizes = np.array([100, 100, 200, 200, 500])
    m = _eligible_mask(sizes, 0, "matched_size")
    assert m.tolist() == [False, True, False, False, False]
    c = _eligible_mask(sizes, 0, "cross_size")
    assert c.tolist() == [False, False, True, True, True]
    g = _eligible_mask(sizes, 0, "global")
    assert g.tolist() == [False, True, True, True, True]
