import numpy as np
from util import bootstrap_diff_ci, roi_from_pairs

def test_bootstrap_separates_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 5, 200)   # 明らかに大きい群
    b = rng.normal(80, 5, 200)    # 明らかに小さい群
    lo, hi, mean = bootstrap_diff_ci(a, b, iters=2000, seed=1)
    assert lo > 0, (lo, hi)        # 差(a-b)のCIが0を跨がない
    assert 15 < mean < 25, mean

def test_bootstrap_overlapping_includes_zero():
    rng = np.random.default_rng(2)
    a = rng.normal(100, 30, 60); b = rng.normal(100, 30, 60)
    lo, hi, _ = bootstrap_diff_ci(a, b, iters=2000, seed=1)
    assert lo < 0 < hi

def test_roi_from_pairs():
    # (amount, return) ペア。投資300, 払戻450 -> ROI 150%
    assert abs(roi_from_pairs([(100, 0), (100, 0), (100, 450)]) - 150.0) < 1e-6

if __name__ == "__main__":
    test_bootstrap_separates_clear_difference()
    test_bootstrap_overlapping_includes_zero()
    test_roi_from_pairs()
    print("ALL PASS")
