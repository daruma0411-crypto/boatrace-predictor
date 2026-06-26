import numpy as np


def bootstrap_diff_ci(a, b, iters=10000, seed=42, lo_pct=2.5, hi_pct=97.5):
    """群aの平均 - 群bの平均 の差を、各群リサンプルでブートストラップし95%区間を返す。"""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(iters, len(a)))
    ib = rng.integers(0, len(b), size=(iters, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(diffs, lo_pct)), float(np.percentile(diffs, hi_pct)), float(diffs.mean())


def roi_from_pairs(pairs):
    """[(amount, return_amount), ...] -> ROI%（払戻合計/投資合計*100）。"""
    inv = sum(float(a) for a, _ in pairs)
    ret = sum(float(r) for _, r in pairs)
    return (ret / inv * 100.0) if inv else 0.0
