"""P改用: レース軟らかさ採点（凍結パラメータ使用）。analysis/soft_regime/features.py の計算を boats_data 用に移植。"""
import numpy as np
from src.softness_params import FEATURE_NAMES, MU, SD, W


def _f(v):
    return float(v) if v is not None else float("nan")


def score_race(boats_data):
    """boats_data(本番形式のdictリスト) -> 軟らかさ点数。計算不能なら None。高いほど緩い。"""
    try:
        rows = [( b.get("boat_number"), _f(b.get("win_rate")),
                  _f(b.get("motor_win_rate_2")), _f(b.get("avg_st")) ) for b in boats_data]
        wrs = np.array([x[1] for x in rows], dtype=float)
        valid = wrs[~np.isnan(wrs)]
        if len(rows) < 4 or len(valid) < 4:
            return None
        motors = np.array([x[2] for x in rows], dtype=float)
        sts = np.array([x[3] for x in rows], dtype=float)
        wr_sorted = np.sort(valid)[::-1]
        top_gap = float(wr_sorted[0] - wr_sorted[1]) if len(wr_sorted) >= 2 else 0.0
        inside = [x[1] for x in rows if x[0] == 1]
        inside_wr = float(inside[0]) if inside and not np.isnan(inside[0]) else float(np.nanmean(wrs))
        feat = {
            "wr_std": float(np.nanstd(wrs)),
            "wr_top_gap": top_gap,
            "inside_wr": inside_wr,
            "motor_std": float(np.nanstd(motors)) if not np.isnan(motors).all() else 0.0,
            "st_std": float(np.nanstd(sts)) if not np.isnan(sts).all() else 0.0,
        }
        x = np.array([feat[n] for n in FEATURE_NAMES], dtype=float)
        z = (x - np.array(MU)) / np.array(SD)
        return float(np.dot(np.append(z, 1.0), np.array(W)))
    except Exception:
        return None
