"""P改 forward トラッカー — 定義凍結版 (2026-07-30)

P改 = P (mc_venue_focus) のベット部分集合＋賭け金スケール:
  1. 堅レース見送り: レース軟らかさ点数が train(2026-02-01〜05-15) 分布の下位1/3 (堅tier) なら bet しない
  2. 低温半額(守り専用): 「市場温度」= 前日までの直近7日の日次平均軟らかさ。
     train 期間の温度中央値未満の日は賭け金×0.5（増し張りはしない）

規律:
- 重み・カットオフ・温度閾値はすべて train(〜5/15) から算出（リーク無し・凍結）。
- この定義は 2026-07-30 に凍結。以後変更しない（変更する場合は別名の新定義として登録）。
- P改 は P の部分集合なのでデプロイ不要。P の bet 記録から本スクリプトでいつでも forward 比較できる。

実行: venv/Scripts/python.exe analysis/soft_regime/p_kai_track.py
出力: P そのまま vs P改 の n/的中/PnL/ROI/最大DD と月別（5/16以降の全期間）
"""
import os
import sys

import numpy as np
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import connect, SPLIT_DATE, ARE_TRAIN_START  # noqa: E402
from features import race_features  # noqa: E402
from stage_a_softscore import _payout_map, _matrix  # noqa: E402

STRATEGY = "mc_venue_focus"          # P
COLD_SCALE = 0.5                     # 低温日の賭け金倍率（凍結: 守りのみ、増し張り禁止）
TEMP_WINDOW = 7                      # 温度 = 直近7日の日次平均軟らかさ（前日まで）


def build_model(conn):
    """train(〜5/15) で重み・堅カットオフ・温度閾値を確定（凍結ロジック）。"""
    feats = race_features(conn, ARE_TRAIN_START, "2026-12-31")
    payout = _payout_map(conn, ARE_TRAIN_START, "2026-12-31")
    ids = [r for r in feats if r in payout]
    train_ids = [r for r in ids if str(feats[r]["race_date"]) < SPLIT_DATE]
    Xtr, ytr = _matrix(feats, payout, train_ids)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    w, *_ = np.linalg.lstsq(
        np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))]), ytr, rcond=None)
    tr_sc = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))]) @ w
    hard_cut = float(np.percentile(tr_sc, 33.333))       # 堅tier境界（train分布）

    # 全レースの点数 → 日次平均 → 温度（前日までの7日平均）
    allids = ids
    Xall, _ = _matrix(feats, payout, allids)
    sca = np.hstack([(Xall - mu) / sd, np.ones((len(Xall), 1))]) @ w
    from collections import defaultdict
    day_scores = defaultdict(list)
    score_of = {}
    for i, rid in enumerate(allids):
        d = str(feats[rid]["race_date"])
        day_scores[d].append(float(sca[i]))
        score_of[rid] = float(sca[i])
    day_mean = {d: float(np.mean(v)) for d, v in day_scores.items()}
    days_sorted = sorted(day_mean)
    temp = {}
    for i, d in enumerate(days_sorted):
        prev = [day_mean[x] for x in days_sorted[max(0, i - TEMP_WINDOW):i]]
        if prev:
            temp[d] = float(np.mean(prev))
    temp_thr = float(np.median(
        [temp[d] for d in days_sorted if d < SPLIT_DATE and d in temp]))
    return score_of, hard_cut, temp, temp_thr


def track(date_from=SPLIT_DATE, date_to="2026-12-31"):
    conn = connect()
    score_of, hard_cut, temp, temp_thr = build_model(conn)
    print(f"凍結パラメータ: 堅カットオフ={hard_cut:.3f}  温度閾値={temp_thr:.3f}  (train〜{SPLIT_DATE})")

    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""SELECT b.race_id, r.race_date::text d, b.amount::float a,
                          coalesce(b.return_amount,b.payout,0)::float ret, b.is_hit
                   FROM bets b JOIN races r ON r.id=b.race_id
                   WHERE b.strategy_type=%s AND r.race_date BETWEEN %s AND %s
                         AND r.result_1st IS NOT NULL ORDER BY r.race_date, b.id""",
                (STRATEGY, date_from, date_to))
    rows = cur.fetchall()

    def transform(x):
        """P改ルール適用: 堅は捨てる / 低温日は×0.5。返り値 (amount, ret) or None。"""
        sc = score_of.get(x["race_id"])
        if sc is not None and sc <= hard_cut:
            return None                                   # 堅レース見送り
        scale = COLD_SCALE if temp.get(x["d"], temp_thr) < temp_thr else 1.0
        return (x["a"] * scale, x["ret"] * scale)

    def summarize(name, pairs, dates, hits):
        inv = sum(a for a, _ in pairs)
        ret = sum(r for _, r in pairs)
        bank = peak = dd = 0.0
        from collections import OrderedDict
        byday = OrderedDict()
        for (a, r), d in zip(pairs, dates):
            byday.setdefault(d, []).append((a, r))
        for d, xs in byday.items():
            bank += sum(r - a for a, r in xs)
            peak = max(peak, bank)
            dd = min(dd, bank - peak)
        roi = ret / inv * 100 if inv else 0.0
        print(f"  {name}: n={len(pairs)} 的中={hits} PnL={int(ret-inv):+,} "
              f"ROI={roi:.1f}% 最大DD={int(dd):,}")

    print(f"\n=== P vs P改 ({date_from}〜, 対象bet={len(rows)}) ===")
    summarize("P そのまま", [(x["a"], x["ret"]) for x in rows],
              [x["d"] for x in rows], sum(1 for x in rows if x["is_hit"]))
    kai = [(transform(x), x) for x in rows]
    kai = [(t, x) for t, x in kai if t is not None]
    summarize("P改       ", [t for t, _ in kai], [x["d"] for _, x in kai],
              sum(1 for _, x in kai if x["is_hit"]))

    from collections import OrderedDict
    mon = OrderedDict()
    for t, x in kai:
        m = x["d"][:7]
        mon.setdefault(m, [0.0, 0.0, 0])
        mon[m][0] += t[0]; mon[m][1] += t[1]; mon[m][2] += (1 if x["is_hit"] else 0)
    print("  P改 月別: " + " / ".join(
        f"{m} {v[1]/v[0]*100:.0f}%({int(v[1]-v[0]):+,},h{v[2]})" for m, v in mon.items() if v[0]))


if __name__ == "__main__":
    track()
