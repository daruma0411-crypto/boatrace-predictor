"""P改ロールアウト以降の forward 成績サマリ (read-only)。8/31チェックポイント等で使用。
実行: python scripts/forward_status.py  (要 psycopg2-binary)
出力: P/P改/P全場/P2 の同一窓成績・P的中のP改採否・温度日数・V11停止確認"""
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable",
)
DEPLOY_AT = "2026-07-31 06:00"   # P改ロールアウトのデプロイ時刻
TEMP_THR = 7.972                  # 凍結温度閾値 (src/softness_params.py と同値)


def main():
    conn = psycopg2.connect(DSN, cursor_factory=RealDictCursor)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()

    print(f"=== forward成績 ({DEPLOY_AT} 以降, 同一窓, 全て紙) ===")
    for s, label in [("mc_venue_focus", "P     "), ("p_kai", "P改   "),
                     ("mc_venue_all", "P全場 "), ("mc2_venue_focus", "P2    ")]:
        cur.execute("""SELECT count(*) n, sum(CASE WHEN b.is_hit THEN 1 ELSE 0 END) h,
            sum(b.amount) inv, sum(coalesce(b.return_amount,b.payout,0)) ret,
            count(distinct b.race_id) races
            FROM bets b JOIN races r ON r.id=b.race_id
            WHERE b.strategy_type=%s AND b.created_at>%s AND r.result_1st IS NOT NULL""",
            (s, DEPLOY_AT))
        r = cur.fetchone()
        inv = float(r["inv"] or 0); ret = float(r["ret"] or 0)
        roi = ret / inv * 100 if inv else 0.0
        print(f"  {label} bet={r['n']:4} 的中={r['h'] or 0:3} レース={r['races']:4} "
              f"PnL={int(ret-inv):+9,} ROI={roi:6.1f}%")

    print("\n=== P の的中を P改 が取れたか (堅見送りの実弾検証) ===")
    cur.execute("""SELECT b.race_id, b.combination, b.amount, coalesce(b.return_amount,0) ret
        FROM bets b WHERE b.strategy_type='mc_venue_focus' AND b.is_hit=true
              AND b.created_at>%s ORDER BY b.created_at""", (DEPLOY_AT,))
    pwins = cur.fetchall()
    missed = 0
    for w in pwins:
        cur.execute("""SELECT amount, coalesce(return_amount,0) ret FROM bets
            WHERE strategy_type='p_kai' AND race_id=%s AND combination=%s""",
            (w["race_id"], w["combination"]))
        k = cur.fetchone()
        if k:
            print(f"  {w['combination']} P:+{int(w['ret']-w['amount']):,} → P改:取った(+{int(k['ret']-k['amount']):,})")
        else:
            missed += 1
            print(f"  {w['combination']} P:+{int(w['ret']-w['amount']):,} → P改:堅判定で見送り(逸失)")
    print(f"  → P的中 {len(pwins)}件中、P改が見送った勝ち {missed}件")

    print("\n=== 市場温度 (低温=P改半額運転の日) ===")
    cur.execute("""SELECT race_date, avg(score) m FROM race_softness
        WHERE race_date >= %s::date GROUP BY race_date ORDER BY race_date""", (DEPLOY_AT,))
    rows = cur.fetchall()
    cold = sum(1 for r in rows if float(r["m"]) < TEMP_THR)
    print(f"  {len(rows)}日中 低温 {cold}日 (閾値 {TEMP_THR})")

    print("\n=== P全場: 会場別の芽 (仮説どまり・事後選択注意) ===")
    cur.execute("""SELECT r.venue_id v, count(*) n,
        sum(CASE WHEN b.is_hit THEN 1 ELSE 0 END) h,
        sum(coalesce(b.return_amount,b.payout,0)) - sum(b.amount) pnl,
        CASE WHEN sum(b.amount)>0 THEN
          (sum(coalesce(b.return_amount,b.payout,0))/sum(b.amount)*100)::int ELSE 0 END roi
        FROM bets b JOIN races r ON r.id=b.race_id
        WHERE b.strategy_type='mc_venue_all' AND b.created_at>%s AND r.result_1st IS NOT NULL
        GROUP BY r.venue_id ORDER BY pnl DESC""", (DEPLOY_AT,))
    for r in cur.fetchall():
        print(f"  場{r['v']:2}: n={r['n']:3} 的中={r['h']:2} PnL={int(r['pnl']):+8,} ROI={r['roi']}%")

    print("\n=== V11 停止確認 ===")
    cur.execute("SELECT count(*) n FROM bets WHERE strategy_type='v11_var13' AND created_at>%s",
                (DEPLOY_AT,))
    print(f"  v11_var13 bets = {cur.fetchone()['n']} (0であること)")


if __name__ == "__main__":
    main()
