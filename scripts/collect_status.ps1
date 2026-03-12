# データ収集進捗確認
# $env:DATABASE_URL を事前に設定してから実行
$env:PYTHONIOENCODING = "utf-8"
cd "$env:USERPROFILE\.openclaw\workspace\boatrace-predictor"

python -c @"
import os, psycopg2, psycopg2.extras
from datetime import date, timedelta

conn = psycopg2.connect(
    os.environ['DATABASE_URL'],
    cursor_factory=psycopg2.extras.RealDictCursor
)
cur = conn.cursor()

cur.execute('''
    SELECT MIN(race_date) as first, MAX(race_date) as last,
           COUNT(DISTINCT race_date) as days, COUNT(*) as races
    FROM races WHERE status='finished'
''')
r = cur.fetchone()
print(f'Collected: {r["days"]} days, {r["races"]} races ({r["first"]} ~ {r["last"]})')

# 月別
cur.execute('''
    SELECT to_char(race_date, 'YYYY-MM') as month,
           COUNT(DISTINCT race_date) as days,
           COUNT(*) as races
    FROM races WHERE status='finished'
    GROUP BY 1 ORDER BY 1
''')
print()
print('Month       | Days | Races')
print('------------|------|------')
for row in cur.fetchall():
    print(f'{row["month"]}      | {row["days"]:4d} | {row["races"]:5d}')

# boats テーブル
cur.execute('SELECT COUNT(*) as cnt FROM boats')
boats = cur.fetchone()['cnt']
cur.execute('SELECT COUNT(DISTINCT race_id) as cnt FROM boats')
boat_races = cur.fetchone()['cnt']
print(f'\nBoats: {boats} (for {boat_races} races)')

# 学習可能データ
cur.execute('''
    SELECT COUNT(*) FROM (
        SELECT b.race_id FROM boats b
        JOIN races r ON b.race_id = r.id
        WHERE r.status='finished' AND r.result_1st IS NOT NULL
        GROUP BY b.race_id HAVING COUNT(*) = 6
    ) t
''')
trainable = cur.fetchone()['count']
print(f'Trainable (6 boats + result): {trainable}')
print(f'\nTarget: 30,000+ races for full retrain')
print(f'Progress: {trainable/30000*100:.1f}%')

conn.close()
"@

pause
