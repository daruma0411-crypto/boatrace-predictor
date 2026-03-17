"""過去レース結果のバックフィル: result_1st/2nd/3rd, payout_sanrentan を埋める"""
import os
import sys
import psycopg2
import psycopg2.extras

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable')

def main():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # 1) バックフィル: actual_result_trifecta があるが result_1st がNULL
    cur.execute("""
        UPDATE races
        SET result_1st = CAST(split_part(actual_result_trifecta, '-', 1) AS INTEGER),
            result_2nd = CAST(split_part(actual_result_trifecta, '-', 2) AS INTEGER),
            result_3rd = CAST(split_part(actual_result_trifecta, '-', 3) AS INTEGER),
            payout_sanrentan = COALESCE(payout_sanrentan, payout_amount)
        WHERE actual_result_trifecta IS NOT NULL
          AND actual_result_trifecta LIKE '%-%-%'
          AND result_1st IS NULL
    """)
    print(f"バックフィル完了: {cur.rowcount}件更新")

    # 2) 確認
    cur.execute("""
        SELECT COUNT(*) as total,
               COUNT(result_1st) as has_1st,
               COUNT(payout_sanrentan) as has_payout
        FROM races
        WHERE actual_result_trifecta IS NOT NULL
    """)
    r = cur.fetchone()
    print(f"確認: 結果あり={r['total']}件, result_1st={r['has_1st']}件, payout_sanrentan={r['has_payout']}件")

    # 3) テストモード確認: 本日のbet amountの分布
    cur.execute("""
        SELECT amount, COUNT(*) as cnt
        FROM bets
        WHERE created_at::date = CURRENT_DATE
        GROUP BY amount
        ORDER BY amount
    """)
    rows = cur.fetchall()
    print("\n本日のベット金額分布:")
    for row in rows:
        print(f"  {row['amount']}円: {row['cnt']}件")

    conn.close()

if __name__ == '__main__':
    main()
