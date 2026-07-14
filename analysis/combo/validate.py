"""忠実度ゲート: 実 bets を sim で決済(当たり目=確定配当)し、DB実績PnLを再現する。

sim の決済メカニクスが本番の実払戻を忠実に再現できることを証明する。
再現できれば、新手法(Task 8)の sim 数値も信頼できる根拠になる。READ-ONLY。
"""
from db import connect
from sim import run_sim

_SQL = """
SELECT b.race_id, b.combination, b.amount, b.odds,
       coalesce(b.return_amount, b.payout, 0) AS ret, b.is_hit,
       r.result_1st, r.result_2nd, r.result_3rd, r.payout_sanrentan
FROM bets b JOIN races r ON r.id = b.race_id
WHERE b.strategy_type = %s AND r.race_date BETWEEN %s AND %s
      AND r.result_1st IS NOT NULL
ORDER BY b.id
"""


def replay_strategy_from_db(strat, date_from, date_to):
    """実 bets を sim で決済(当たり目=確定配当)し、DB実績PnLと突合。(db_summary, sim_summary)。"""
    conn = connect(); cur = conn.cursor()
    cur.execute(_SQL, (strat, date_from, date_to))
    rows = cur.fetchall()
    db_inv = db_ret = 0.0
    by_race = {}
    for r in rows:
        db_inv += float(r["amount"]); db_ret += float(r["ret"])
        rid = r["race_id"]
        if rid not in by_race:
            by_race[rid] = {
                "win_combo": f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}",
                "payout_odds": (float(r["payout_sanrentan"]) / 100.0 if r["payout_sanrentan"] else None),
                "bets": [],
            }
        by_race[rid]["bets"].append((r["combination"], float(r["amount"]), float(r["odds"])))
    per_race = [{"race_id": rid, **v} for rid, v in by_race.items()]
    sim = run_sim(per_race, settle_only=True)
    db = {"n": len(rows), "invested": round(db_inv), "returned": round(db_ret),
          "pnl": round(db_ret - db_inv)}
    return db, sim
