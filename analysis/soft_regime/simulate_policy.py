import numpy as np
from db import connect, fetch, SPLIT_DATE, STRATS
from stage_a_softscore import run_stage_a, race_features  # noqa: F401


def _tier_of_test_races(conn):
    """test期間の race_id -> tercile を Stage A から得る（再計算）。"""
    from features import race_features as rf
    from stage_a_softscore import _payout_map, _matrix
    from db import ARE_TRAIN_START
    from features import FEATURE_NAMES
    feats = rf(conn, ARE_TRAIN_START, "2026-12-31")
    payout = _payout_map(conn, ARE_TRAIN_START, "2026-12-31")
    ids = [r for r in feats if r in payout]
    train_ids = [r for r in ids if str(feats[r]["race_date"]) < SPLIT_DATE]
    test_ids = [r for r in ids if str(feats[r]["race_date"]) >= SPLIT_DATE]
    Xtr, ytr = _matrix(feats, payout, train_ids)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0); sd[sd == 0] = 1.0
    w, *_ = np.linalg.lstsq(np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))]), ytr, rcond=None)
    Xte, _ = _matrix(feats, payout, test_ids)
    score = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))]) @ w
    q1, q2 = np.percentile(score, [33.333, 66.667])
    tier = np.where(score <= q1, "low", np.where(score <= q2, "mid", "high"))
    return {test_ids[i]: tier[i] for i in range(len(test_ids))}


def _test_bets(conn):
    sql = """select b.race_id, r.race_date, b.amount, coalesce(b.return_amount,b.payout,0)
             from bets b join races r on b.race_id=r.id
             where b.strategy_type = any(%s) and r.race_date >= %s and r.result_1st is not null
             order by r.race_date, b.id"""
    _, rows = fetch(conn, sql, (list(STRATS), SPLIT_DATE))
    return rows


def simulate(conn, multipliers):
    """multipliers: {'low':x,'mid':y,'high':z} か None(=均等1.0)。"""
    tier_of = {} if multipliers is None else _tier_of_test_races(conn)
    bets = _test_bets(conn)
    inv = ret = 0.0
    equity = 0.0; peak = 0.0; max_dd = 0.0; n = 0
    for rid, _d, amount, payout in bets:
        m = 1.0 if multipliers is None else float(multipliers.get(tier_of.get(rid, "mid"), 1.0))
        a = float(amount) * m; r = float(payout) * m
        inv += a; ret += r; equity += (r - a); n += 1
        peak = max(peak, equity); max_dd = min(max_dd, equity - peak)
    return {
        "n_bets": n,
        "invested": round(inv, 0),
        "roi": round(ret / inv * 100.0, 1) if inv else 0.0,
        "profit": round(ret - inv, 0),
        "max_drawdown": round(max_dd, 0),
    }


if __name__ == "__main__":
    import json
    c = connect()
    print("FLAT :", json.dumps(simulate(c, None), ensure_ascii=False))
    print("POLICY:", json.dumps(simulate(c, {"low": 0.5, "mid": 1.0, "high": 1.5}), ensure_ascii=False))
