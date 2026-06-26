import numpy as np
from db import connect, fetch, SPLIT_DATE, ARE_TRAIN_START, STRATS
from features import race_features, FEATURE_NAMES
from util import bootstrap_diff_ci, roi_from_pairs


def _payout_map(conn, date_from, date_to):
    sql = """select id, payout_sanrentan from races
             where race_date between %s and %s and payout_sanrentan is not null
                   and result_1st is not null"""
    _, rows = fetch(conn, sql, (date_from, date_to))
    return {rid: float(p) for rid, p in rows}


def _strategy_pairs_by_race(conn, date_from, date_to):
    sql = """select b.race_id, b.amount, coalesce(b.return_amount,b.payout,0)
             from bets b join races r on b.race_id=r.id
             where b.strategy_type = any(%s) and r.race_date between %s and %s
                   and r.result_1st is not null"""
    _, rows = fetch(conn, sql, (list(STRATS), date_from, date_to))
    m = {}
    for rid, a, r in rows:
        m.setdefault(rid, []).append((float(a), float(r)))
    return m


def _matrix(feats, payout, race_ids):
    X = np.array([[feats[r][n] for n in FEATURE_NAMES] for r in race_ids], dtype=float)
    y = np.log(np.array([payout[r] for r in race_ids], dtype=float))
    return X, y


def run_stage_a(conn):
    feats = race_features(conn, ARE_TRAIN_START, "2026-12-31")
    payout = _payout_map(conn, ARE_TRAIN_START, "2026-12-31")
    ids = [r for r in feats if r in payout]
    train_ids = [r for r in ids if str(feats[r]["race_date"]) < SPLIT_DATE]
    test_ids = [r for r in ids if str(feats[r]["race_date"]) >= SPLIT_DATE]

    Xtr, ytr = _matrix(feats, payout, train_ids)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Ztr = (Xtr - mu) / sd
    Ztr1 = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    w, *_ = np.linalg.lstsq(Ztr1, ytr, rcond=None)  # 線形: log(payout) ~ 標準化特徴

    Xte, yte = _matrix(feats, payout, test_ids)
    Zte = (Xte - mu) / sd
    score = np.hstack([Zte, np.ones((len(Zte), 1))]) @ w  # 高いほど波乱(=緩い)予測

    q1, q2 = np.percentile(score, [33.333, 66.667])
    tier = np.where(score <= q1, "low", np.where(score <= q2, "mid", "high"))
    pay_te = np.exp(yte)

    payout_by_tercile = {t: round(float(pay_te[tier == t].mean()), 0) for t in ("low", "mid", "high")}
    monotonic = payout_by_tercile["low"] < payout_by_tercile["mid"] < payout_by_tercile["high"]
    ci = bootstrap_diff_ci(pay_te[tier == "high"], pay_te[tier == "low"])

    # 確認: P/P2 ROI を tercile 別に
    pairs_by_race = _strategy_pairs_by_race(conn, SPLIT_DATE, "2026-12-31")
    score_of = {test_ids[i]: tier[i] for i in range(len(test_ids))}
    roi_by_tercile = {}
    for t in ("low", "mid", "high"):
        pp = [p for rid, ps in pairs_by_race.items() if score_of.get(rid) == t for p in ps]
        roi_by_tercile[t] = round(roi_from_pairs(pp), 1) if pp else None

    # ベースライン比較: カレンダー(gw_flag)単独 vs softスコア の |相関| with log(payout) on test
    gw_te = np.array([feats[r]["gw_flag"] for r in test_ids], dtype=float)
    corr_score = abs(float(np.corrcoef(score, yte)[0, 1]))
    corr_gw = abs(float(np.corrcoef(gw_te, yte)[0, 1])) if gw_te.std() > 0 else 0.0

    return {
        "train_n": len(train_ids), "test_n": len(test_ids),
        "train_max_date": str(max(feats[r]["race_date"] for r in train_ids)),
        "test_min_date": str(min(feats[r]["race_date"] for r in test_ids)),
        "weights": dict(zip(FEATURE_NAMES + ["bias"], [round(float(x), 4) for x in w])),
        "payout_by_tercile": payout_by_tercile,
        "payout_monotonic": bool(monotonic),
        "softscore_top_bottom_payout_ci": [round(x, 1) for x in ci],
        "roi_by_tercile": roi_by_tercile,
        "corr_softscore": round(corr_score, 4),
        "corr_calendar_baseline": round(corr_gw, 4),
        "beats_calendar_baseline": bool(corr_score > corr_gw),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_stage_a(connect()), default=str, ensure_ascii=False, indent=2))
