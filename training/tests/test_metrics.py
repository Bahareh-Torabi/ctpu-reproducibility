import numpy as np
from ctpu.metrics import compute_metrics, label_group_indices


def test_metrics_finite():
    y = np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1]], dtype=np.uint8)
    logits = np.array([[2,-1,-2],[-1,2,-2],[1,1,-2],[-2,-2,2]], dtype=np.float32)
    m = compute_metrics(y, logits, train_support=[2,2,1])
    assert 0 <= m["brier"] <= 1
    assert m["macro_ap"] > 0.5
    assert np.isfinite(m["soft_cardinality_signed_bias"])


def test_groups_disjoint():
    g = label_group_indices([0, 1, 2, 3, 4, 5, 6])
    seen = set(g["tail"]) | set(g["middle"]) | set(g["head"])
    assert seen == set(range(1,7))
    assert set(g["zero_train_support"]) == {0}


def test_xmc_metrics_present_and_bounded():
    y = np.array([[1,0,1,0,0],[0,1,0,0,1]], dtype=np.uint8)
    logits = np.array([[3,-2,2,-1,-3],[-2,3,-1,-3,2]], dtype=np.float32)
    m = compute_metrics(y, logits, train_support=[1,1,1,0,1])
    for k in (1,3,5):
        assert 0 <= m[f"precision_at_{k}"] <= 1
        assert 0 <= m[f"ndcg_at_{k}"] <= 1
    assert "topk_flat_ece_5_diagnostic" in m
    assert "ece_at_5" not in m


def test_hard_cardinality_signed_bias_handles_underprediction_without_uint_underflow():
    y = np.array([[1,1,0],[1,0,0]], dtype=np.uint8)
    # All predictions are negative at 0.5, so hard cardinality is 0 vs true mean 1.5.
    logits = np.full((2,3), -5.0, dtype=np.float32)
    m = compute_metrics(y, logits, train_support=[2,1,0], tune_threshold=False)
    assert m["hard_cardinality_mean_at_0_5"] == 0.0
    assert m["hard_cardinality_signed_bias_at_0_5"] == -1.5
    assert abs(m["hard_cardinality_signed_bias_at_0_5"]) < 10
