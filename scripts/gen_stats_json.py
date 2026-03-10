"""起動時にDB統計をJSONとして出力するスクリプト"""
import os
import sys
import json
from datetime import date, timedelta

def main():
    db_url = os.environ.get('DATABASE_URL', '')
    if not db_url:
        print("DATABASE_URL not set, skipping stats generation")
        return

    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        print("psycopg2 not available")
        return

    try:
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return

    cur = conn.cursor()
    today = date.today()
    result = {"generated_at": today.isoformat(), "strategies": {}, "daily": [], "totals": {}}

    # 全期間の戦略別サマリー
    cur.execute("""
        SELECT
            b.strategy_type,
            COUNT(*) as total_bets,
            SUM(b.amount) as total_amount,
            SUM(b.payout) as total_payout,
            CASE WHEN SUM(b.amount) > 0
                 THEN SUM(b.payout)::float / SUM(b.amount) * 100
                 ELSE 0 END as roi,
            COUNT(CASE WHEN b.payout > 0 THEN 1 END) as wins,
            COUNT(DISTINCT b.race_id) as total_races
        FROM bets b
        JOIN races r ON b.race_id = r.id
        WHERE b.result IS NOT NULL
        GROUP BY b.strategy_type
        ORDER BY b.strategy_type
    """)
    for row in cur.fetchall():
        st = row['strategy_type']
        result["strategies"][st] = {
            "total_bets": row['total_bets'],
            "total_amount": int(row['total_amount'] or 0),
            "total_payout": int(row['total_payout'] or 0),
            "roi": round(row['roi'] or 0, 2),
            "wins": row['wins'],
            "total_races": row['total_races'],
            "profit": int((row['total_payout'] or 0) - (row['total_amount'] or 0)),
        }

    # 全体合計
    cur.execute("""
        SELECT
            COUNT(*) as total_bets,
            SUM(amount) as total_amount,
            SUM(payout) as total_payout,
            CASE WHEN SUM(amount) > 0
                 THEN SUM(payout)::float / SUM(amount) * 100
                 ELSE 0 END as roi,
            COUNT(CASE WHEN payout > 0 THEN 1 END) as wins,
            COUNT(DISTINCT race_id) as total_races
        FROM bets
        WHERE result IS NOT NULL
    """)
    row = cur.fetchone()
    result["totals"] = {
        "total_bets": row['total_bets'],
        "total_amount": int(row['total_amount'] or 0),
        "total_payout": int(row['total_payout'] or 0),
        "roi": round(row['roi'] or 0, 2),
        "wins": row['wins'],
        "total_races": row['total_races'],
        "profit": int((row['total_payout'] or 0) - (row['total_amount'] or 0)),
    }

    # 日別推移
    cur.execute("""
        SELECT
            r.race_date,
            b.strategy_type,
            COUNT(*) as total_bets,
            SUM(b.amount) as total_amount,
            SUM(b.payout) as total_payout,
            COUNT(CASE WHEN b.payout > 0 THEN 1 END) as wins
        FROM bets b
        JOIN races r ON b.race_id = r.id
        WHERE b.result IS NOT NULL
        GROUP BY r.race_date, b.strategy_type
        ORDER BY r.race_date
    """)
    for row in cur.fetchall():
        result["daily"].append({
            "date": row['race_date'].isoformat(),
            "strategy": row['strategy_type'],
            "bets": row['total_bets'],
            "amount": int(row['total_amount'] or 0),
            "payout": int(row['total_payout'] or 0),
            "wins": row['wins'],
        })

    # 未結果のベット数
    cur.execute("SELECT COUNT(*) as cnt FROM bets WHERE result IS NULL")
    result["pending_bets"] = cur.fetchone()['cnt']

    conn.close()

    # 出力先
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'streamlit_app', 'static')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Stats written to {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
