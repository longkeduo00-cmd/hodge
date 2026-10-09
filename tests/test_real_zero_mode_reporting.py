from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.real_zero_mode_reporting import generate_real_zero_mode_report


def test_real_zero_mode_report_writes_exact_tables_and_figures(tmp_path: Path) -> None:
    run = tmp_path / "real_zero_modes_test"
    (run / "tables").mkdir(parents=True)
    (run / "arrays" / "eigenvalues").mkdir(parents=True)

    channels = pd.DataFrame([
        {"graph_index":0,"label":"g1","domain":"D","N":4,"E":3,"q":1,"k":0,"dimension":4,"zero_count":1,"count_resolved":True,"zero_fraction":0.25,"smallest_positive_eigenvalue":1.0,"method":"k0_connected_components","status":"resolved_exact_k0","spectrum_scope":"full","spectrum_method":"dense_eigvalsh"},
        {"graph_index":0,"label":"g1","domain":"D","N":4,"E":3,"q":1,"k":1,"dimension":3,"zero_count":0,"count_resolved":True,"zero_fraction":0.0,"smallest_positive_eigenvalue":2.0,"method":"dense_eigvalsh","status":"resolved_under_tolerance","spectrum_scope":"full","spectrum_method":"dense_eigvalsh"},
    ])
    channels.to_csv(run / "tables" / "real_zero_modes_per_network_channel.csv", index=False)
    pd.DataFrame([{"graph_index":0,"label":"g1","domain":"D","N":4,"E":3,"retained_q_max":1,"retained_q_count":1}]).to_csv(run / "tables" / "real_graph_index.csv", index=False)
    np.save(run / "arrays" / "eigenvalues" / "g1_q001_k000.npy", np.array([0.0,1.0,2.0,3.0]))
    np.save(run / "arrays" / "eigenvalues" / "g1_q001_k001.npy", np.array([2.0,2.0,2.0]))
    pd.DataFrame([
        {"graph_index":0,"label":"g1","domain":"D","q":1,"k":0,"dimension":4,"spectrum_scope":"full","spectrum_method":"dense_eigvalsh","eigenvalues_saved":4,"spectrum_file":"arrays/eigenvalues/g1_q001_k000.npy","observed_eigenvalue_min":0.0,"observed_eigenvalue_max":3.0,"full_spectrum_available":True},
        {"graph_index":0,"label":"g1","domain":"D","q":1,"k":1,"dimension":3,"spectrum_scope":"full","spectrum_method":"dense_eigvalsh","eigenvalues_saved":3,"spectrum_file":"arrays/eigenvalues/g1_q001_k001.npy","observed_eigenvalue_min":2.0,"observed_eigenvalue_max":2.0,"full_spectrum_available":True},
    ]).to_csv(run / "tables" / "real_eigenvalue_index.csv", index=False)
    (run / "effective_config.json").write_text(json.dumps({"resolved_zero_mode_settings":{"zero_tolerance":1e-10}}))

    result = generate_real_zero_mode_report(run, save_pdf=False)
    assert result.networks == 1
    assert result.channels == 2
    assert result.resolved_channels == 2
    qtab = pd.read_csv(result.tables_dir / "real_zero_count_exact_by_network_q.csv")
    assert int(qtab.loc[0, "k0"]) == 1
    assert int(qtab.loc[0, "k1"]) == 0
    assert (result.plots_dir / "g1" / "g1_q01_spectral_dashboard.png").exists()
    assert (result.plots_dir / "g1" / "g1_zero_count_qk.png").exists()
