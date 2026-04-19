"""V3ベットデータ詳細分析"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.database import get_db_connection

def main():
    with get_db_connection() as conn:
        cur = conn.cursor()

        print("=== 1. V3 戦略別成績 ===")
        cur.execute("""
            SELECT strategy_type, COUNT(*) as bets, SUM(amount) as wagered,
                   SUM(payout) as payout,
                   ROUND(SUM(payout)::numeric / NULLIF(SUM(amount),0) * 100, 1) as roi,
                   SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins
            FROM bets WHERE result IS NOT NULL AND created_at < '2026-04-01'
            GROUP BY strategy_type ORDER BY roi DESC
        """)
        for r in cur.fetchall():
            print(f"  {r['strategy_type']:25s} bets={r['bets']:5d} "
                  f"wagered={r['wagered']:>10,} payout={r['payout']:>10,} "
                  f"ROI={r['roi']}% wins={r['wins']}")

        print("\n=== 2. 月次推移 ===")
        cur.execute("""
            SELECT to_char(DATE_TRUNC('month', created_at), 'YYYY-MM') as month,
                   COUNT(*) as bets, SUM(amount) as wagered, SUM(payout) as payout,
                   ROUND(SUM(payout)::numeric / NULLIF(SUM(amount),0) * 100, 1) as roi
            FROM bets WHERE result IS NOT NULL AND created_at < '2026-04-01'
            GROUP BY month ORDER BY month
        """)
        for r in cur.fetchall():
            print(f"  {r['month']} bets={r['bets']:5d} wagered={r['wagered']:>10,} "
                  f"payout={r['payout']:>10,} ROI={r['roi']}%")

        print("\n=== 3. 会場別成績 (10件以上) ===")
        cur.execute("""
            SELECT r.venue_id, COUNT(*) as bets, SUM(b.amount) as wagered,
                   SUM(b.payout) as payout,
                   ROUND(SUM(b.payout)::numeric / NULLIF(SUM(b.amount),0) * 100, 1) as roi
            FROM bets b JOIN races r ON b.race_id = r.id
            WHERE b.result IS NOT NULL AND b.created_at < '2026-04-01'
            GROUP BY r.venue_id HAVING COUNT(*) >= 10
            ORDER BY roi DESC
        """)
        for r in cur.fetchall():
            print(f"  場{r['venue_id']:2d}  bets={r['bets']:5d} wagered={r['wagered']:>10,} "
                  f"payout={r['payout']:>10,} ROI={r['roi']}%")

        print("\n=== 4. オッズ帯別成績 ===")
        cur.execute("""
            SELECT CASE
              WHEN odds < 10 THEN '01:<10x'
              WHEN odds < 25 THEN '02:10-25x'
              WHEN odds < 50 THEN '03:25-50x'
              WHEN odds < 100 THEN '04:50-100x'
              ELSE '05:100x+'
            END as band,
            COUNT(*) as bets, SUM(amount) as wagered, SUM(payout) as payout,
            ROUND(SUM(payout)::numeric / NULLIF(SUM(amount),0) * 100, 1) as roi,
            SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins
            FROM bets WHERE result IS NOT NULL AND created_at < '2026-04-01'
              AND odds IS NOT NULL
            GROUP BY band ORDER BY band
        """)
        for r in cur.fetchall():
            print(f"  {r['band']:10s} bets={r['bets']:5d} wagered={r['wagered']:>10,} "
                  f"payout={r['payout']:>10,} ROI={r['roi']}% wins={r['wins']}")

        print("\n=== 5. レース番号別成績 ===")
        cur.execute("""
            SELECT r.race_number, COUNT(*) as bets, SUM(b.amount) as wagered,
                   SUM(b.payout) as payout,
                   ROUND(SUM(b.payout)::numeric / NULLIF(SUM(b.amount),0) * 100, 1) as roi
            FROM bets b JOIN races r ON b.race_id = r.id
            WHERE b.result IS NOT NULL AND b.created_at < '2026-04-01'
            GROUP BY r.race_number
            ORDER BY r.race_number
        """)
        for r in cur.fetchall():
            print(f"  R{r['race_number']:2d}  bets={r['bets']:5d} wagered={r['wagered']:>10,} "
                  f"payout={r['payout']:>10,} ROI={r['roi']}%")

        print("\n=== 6. 1号艇1着予測 vs 2-6号艇1着予測 ===")
        cur.execute("""
            SELECT
              CASE WHEN LEFT(combination, 1) = '1' THEN '1号艇軸' ELSE '2-6号艇軸' END as axis,
              COUNT(*) as bets, SUM(amount) as wagered, SUM(payout) as payout,
              ROUND(SUM(payout)::numeric / NULLIF(SUM(amount),0) * 100, 1) as roi,
              SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins
            FROM bets WHERE result IS NOT NULL AND created_at < '2026-04-01'
              AND bet_type = 'sanrentan'
            GROUP BY axis ORDER BY axis
        """)
        for r in cur.fetchall():
            print(f"  {r['axis']:10s} bets={r['bets']:5d} wagered={r['wagered']:>10,} "
                  f"payout={r['payout']:>10,} ROI={r['roi']}% wins={r['wins']}")

        print("\n=== 7. model_version分布 (3月以降) ===")
        cur.execute("""
            SELECT model_version, COUNT(*) as cnt
            FROM predictions WHERE created_at >= '2026-03-01'
            GROUP BY model_version ORDER BY cnt DESC
        """)
        for r in cur.fetchall():
            ver = r['model_version'] or 'NULL'
            print(f"  {ver:30s} count={r['cnt']}")

if __name__ == '__main__':
    main()
