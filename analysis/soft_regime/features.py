import numpy as np
from db import fetch, gw_flag
import datetime

# 事前登録した少数特徴（5個）。後から増やさない。
FEATURE_NAMES = ["wr_std", "wr_top_gap", "inside_wr", "motor_std", "st_std"]


def race_features(conn, date_from, date_to):
    """race_id -> {特徴名: 値, gw_flag, dow, venue_id, race_date} の辞書。"""
    sql = """
        select b.race_id, r.race_date, r.venue_id,
               b.boat_number, b.win_rate, b.motor_win_rate_2, b.avg_st
        from boats b join races r on b.race_id = r.id
        where r.race_date between %s and %s
        order by b.race_id, b.boat_number
    """
    _, rows = fetch(conn, sql, (date_from, date_to))
    by_race = {}
    for race_id, rdate, venue, bno, wr, motor, st in rows:
        by_race.setdefault(race_id, {"race_date": rdate, "venue_id": venue, "rows": []})
        by_race[race_id]["rows"].append(
            (bno, _f(wr), _f(motor), _f(st))
        )

    out = {}
    for race_id, d in by_race.items():
        rws = d["rows"]
        wrs = np.array([x[1] for x in rws], dtype=float)
        motors = np.array([x[2] for x in rws], dtype=float)
        sts = np.array([x[3] for x in rws], dtype=float)
        if np.isnan(wrs).all() or len(rws) < 4:
            continue  # データ不足レースは除外
        wr_sorted = np.sort(wrs[~np.isnan(wrs)])[::-1]
        top_gap = float(wr_sorted[0] - wr_sorted[1]) if len(wr_sorted) >= 2 else 0.0
        inside = [x[1] for x in rws if x[0] == 1]
        inside_wr = float(inside[0]) if inside and not np.isnan(inside[0]) else float(np.nanmean(wrs))
        rdate = d["race_date"]
        dow = rdate.weekday() if isinstance(rdate, (datetime.date, datetime.datetime)) else 0
        out[race_id] = {
            "wr_std": float(np.nanstd(wrs)),
            "wr_top_gap": top_gap,
            "inside_wr": inside_wr,
            "motor_std": float(np.nanstd(motors)) if not np.isnan(motors).all() else 0.0,
            "st_std": float(np.nanstd(sts)) if not np.isnan(sts).all() else 0.0,
            "gw_flag": 1 if gw_flag(rdate) else 0,
            "dow": dow,
            "venue_id": d["venue_id"],
            "race_date": rdate,
        }
    return out


def _f(v):
    return float(v) if v is not None else float("nan")
