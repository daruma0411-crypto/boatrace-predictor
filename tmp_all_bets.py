"""全ベット一覧出力"""
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = 'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
cur = conn.cursor()

cur.execute('''
    SELECT r.race_date, r.venue_id, r.race_number,
           b.strategy_type, b.combination, b.amount, b.odds,
           b.expected_value, b.is_hit,
           COALESCE(b.return_amount, b.payout, 0) as return_amount,
           r.actual_result_trifecta, r.is_finished,
           b.created_at
    FROM bets b
    JOIN races r ON b.race_id = r.id
    ORDER BY r.race_date, r.venue_id, r.race_number, b.strategy_type, b.expected_value DESC
''')
rows = cur.fetchall()

VENUES = {
    1:'桐生',2:'戸田',3:'江戸川',4:'平和島',5:'多摩川',
    6:'浜名湖',7:'蒲郡',8:'常滑',9:'津',10:'三国',
    11:'びわこ',12:'住之江',13:'尼崎',14:'鳴門',15:'丸亀',
    16:'児島',17:'宮島',18:'徳山',19:'下関',20:'若松',
    21:'芦屋',22:'福岡',23:'唐津',24:'大村',
}

STRAT = {
    'conservative':'A保守','standard':'B普通','divergence':'C乖離',
    'high_confidence':'D確信','ensemble':'E合議','div_confidence':'F乖離確信',
    'bt_none':'G-BT基本','bt_entropy':'H-BT確信','bt_ensemble':'I-BT合議',
}

print(f'全ベット数: {len(rows)}')
print()

current_race = None
for r in rows:
    race_key = (r['race_date'], r['venue_id'], r['race_number'])
    if race_key != current_race:
        current_race = race_key
        venue = VENUES.get(r['venue_id'], f"場{r['venue_id']}")
        finished = r['is_finished']
        result = r['actual_result_trifecta'] or ''
        status = f'結果:{result}' if finished else '未確定'
        print(f'=== {r["race_date"]} {venue} {r["race_number"]}R [{status}] ===')

    strat = STRAT.get(r['strategy_type'], r['strategy_type'])
    hit_mark = '-'
    if r['is_hit'] is True:
        hit_mark = 'O'
    elif r['is_hit'] is False:
        hit_mark = 'X'
    ret_amt = r['return_amount'] or 0
    ret_str = f'Y{int(ret_amt):,}' if ret_amt > 0 else '-'
    print(f'  {strat:10s} | {r["combination"]:8s} | Y{r["amount"]:,} | odds={r["odds"]:6.1f} | EV={r["expected_value"]:.2f} | {hit_mark} | {ret_str}')

print()

# サマリー
cur.execute('''
    SELECT b.strategy_type,
           COUNT(*) as cnt,
           SUM(b.amount) as invested,
           SUM(COALESCE(b.return_amount, b.payout, 0)) as returned,
           COUNT(CASE WHEN b.is_hit = TRUE THEN 1 END) as hits
    FROM bets b
    JOIN races r ON b.race_id = r.id
    GROUP BY b.strategy_type
    ORDER BY b.strategy_type
''')
summary = cur.fetchall()
print('=== 戦略別サマリー ===')
print(f'{"戦略":12s} | {"件数":>4s} | {"投資額":>10s} | {"払戻額":>10s} | {"損益":>10s} | {"的中":>3s} | {"回収率":>6s}')
print('-' * 80)
total_inv = 0
total_ret = 0
total_hits = 0
total_cnt = 0
for s in summary:
    strat = STRAT.get(s['strategy_type'], s['strategy_type'])
    inv = s['invested'] or 0
    ret = s['returned'] or 0
    profit = ret - inv
    hits = s['hits'] or 0
    roi = (ret / inv * 100) if inv > 0 else 0
    total_inv += inv
    total_ret += ret
    total_hits += hits
    total_cnt += s['cnt']
    sign = '+' if profit >= 0 else ''
    print(f'{strat:12s} | {s["cnt"]:4d} | Y{inv:>9,} | Y{ret:>9,} | {sign}Y{int(profit):>8,} | {hits:3d} | {roi:5.1f}%')
print('-' * 80)
total_profit = total_ret - total_inv
total_roi = (total_ret / total_inv * 100) if total_inv > 0 else 0
sign = '+' if total_profit >= 0 else ''
print(f'{"合計":12s} | {total_cnt:4d} | Y{total_inv:>9,} | Y{total_ret:>9,} | {sign}Y{int(total_profit):>8,} | {total_hits:3d} | {total_roi:5.1f}%')

# 日別集計
print()
cur.execute('''
    SELECT r.race_date,
           COUNT(*) as cnt,
           SUM(b.amount) as invested,
           SUM(COALESCE(b.return_amount, b.payout, 0)) as returned,
           COUNT(CASE WHEN b.is_hit = TRUE THEN 1 END) as hits,
           COUNT(CASE WHEN b.is_hit IS NULL AND b.result IS NULL THEN 1 END) as pending
    FROM bets b
    JOIN races r ON b.race_id = r.id
    GROUP BY r.race_date
    ORDER BY r.race_date
''')
daily = cur.fetchall()
print('=== 日別集計 ===')
print(f'{"日付":12s} | {"件数":>4s} | {"投資額":>10s} | {"払戻額":>10s} | {"損益":>10s} | {"的中":>3s} | {"未確定":>4s}')
print('-' * 75)
for d in daily:
    inv = d['invested'] or 0
    ret = d['returned'] or 0
    profit = ret - inv
    hits = d['hits'] or 0
    pending = d['pending'] or 0
    sign = '+' if profit >= 0 else ''
    print(f'{d["race_date"]} | {d["cnt"]:4d} | Y{inv:>9,} | Y{ret:>9,} | {sign}Y{int(profit):>8,} | {hits:3d} | {pending:4d}')

conn.close()
