"""過去レースの軟らかさを race_softness に一括投入（デプロイ時に1回実行）。
書込み先は新テーブル race_softness のみ。既存テーブルには一切書かない。"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import psycopg2
from psycopg2.extras import RealDictCursor
from src.softness import score_race
from src.softness_store import CREATE_SQL, INSERT_SQL

DSN = os.environ.get("DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable")
SINCE = "2026-06-15"   # 温度計算に十分な過去分

conn = psycopg2.connect(DSN, cursor_factory=RealDictCursor)
cur = conn.cursor()
cur.execute(CREATE_SQL)
cur.execute("SELECT id, race_date FROM races WHERE race_date >= %s ORDER BY id", (SINCE,))
races = cur.fetchall()
n_ok = 0
for r in races:
    cur.execute("SELECT * FROM boats WHERE race_id=%s ORDER BY boat_number", (r["id"],))
    boats = [dict(b) for b in cur.fetchall()]
    s = score_race(boats)
    if s is not None:
        cur.execute(INSERT_SQL, (r["id"], str(r["race_date"]), float(s)))
        n_ok += 1
conn.commit()
print(f"backfilled {n_ok}/{len(races)} races since {SINCE}")
