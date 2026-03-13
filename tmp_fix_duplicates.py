"""重複ベット修正 + race_processing テーブル作成 + UNIQUE制約追加"""
import psycopg2

DB_URL = 'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
conn = psycopg2.connect(DB_URL, connect_timeout=15)
cur = conn.cursor()

# 1. race_processing テーブル作成
print("1. race_processing テーブル作成...")
cur.execute('''
    CREATE TABLE IF NOT EXISTS race_processing (
        race_id INTEGER PRIMARY KEY,
        locked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )
''')
conn.commit()
print("   OK")

# 2. 重複ベットを削除（最古のIDだけ残す）
print("2. 重複ベット削除...")
cur.execute('''
    DELETE FROM bets
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM bets
        GROUP BY race_id, strategy_type, combination
    )
''')
deleted = cur.rowcount
conn.commit()
print(f"   {deleted}件削除")

# 3. UNIQUE制約を追加
print("3. UNIQUE制約追加...")
try:
    cur.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS uq_bets_race_strategy_combo
        ON bets (race_id, strategy_type, combination)
    ''')
    conn.commit()
    print("   OK")
except Exception as e:
    conn.rollback()
    print(f"   失敗: {e}")

# 4. 残りのベット数確認
cur.execute('SELECT COUNT(*) FROM bets')
remaining = cur.fetchone()[0]
print(f"4. 残ベット数: {remaining}")

# 5. 日別確認
cur.execute('''
    SELECT r.race_date, COUNT(*) as cnt, SUM(b.amount) as total
    FROM bets b JOIN races r ON b.race_id = r.id
    GROUP BY r.race_date ORDER BY r.race_date
''')
print("5. 日別:")
for row in cur.fetchall():
    print(f"   {row[0]}: {row[1]}件 / Y{row[2]:,}")

conn.close()
print("完了")
