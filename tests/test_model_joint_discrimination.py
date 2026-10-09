from validation.model_joint_discrimination import build_joint_cells, _protocol_pair_ok

def test_joint_grid_shape_and_order():
    df=build_joint_cells([100,250],[4,10],[.001,.01,.1,1.0])
    assert len(df)==6*2*2
    assert set(df.network_size)=={100,250}
    assert set(df.target_mean_degree)=={4.0,10.0}

def test_joint_protocols():
    assert _protocol_pair_ok(100,4,100,4,'matched_N_k')
    assert _protocol_pair_ok(100,4,250,4,'cross_N_only')
    assert _protocol_pair_ok(100,4,100,10,'cross_k_only')
    assert _protocol_pair_ok(100,4,250,10,'cross_N_k')
