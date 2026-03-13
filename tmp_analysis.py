"""ベット分析 - 修正後の全データ"""
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = 'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
conn = psycopg2.connect(DB_URL, connect_timeout=15, cursor_factory=RealDictCursor)
cur = conn.cursor()

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

# ========== 1. 全ベット一覧 ==========
cur.execute('''
    SELECT r.race_date, r.venue_id, r.race_number,
           b.strategy_type, b.combination, b.amount, b.odds,
           b.expected_value, b.is_hit,
           COALESCE(b.return_amount, 0) as return_amount,
           r.actual_result_trifecta, r.is_finished
    FROM bets b
    JOIN races r ON b.race_id = r.id
    ORDER BY r.race_date, r.venue_id, r.race_number, b.strategy_type, b.expected_value DESC
''')
rows = cur.fetchall()

print(f'=== 全ベット一覧 ({len(rows)}件) ===')
print()

current_race = None
for r in rows:
    race_key = (r['race_date'], r['venue_id'], r['race_number'])
    if race_key != current_race:
        current_race = race_key
        venue = VENUES.get(r['venue_id'], f"場{r['venue_id']}")
        result = r['actual_result_trifecta'] or ''
        if r['is_finished']:
            status = f'結果:{result}'
        else:
            status = '未確定'
        print(f'--- {r["race_date"]} {venue} {r["race_number"]}R [{status}] ---')

    strat = STRAT.get(r['strategy_type'], r['strategy_type'])
    if r['is_hit'] is True:
        hit_mark = 'HIT!'
    elif r['is_hit'] is False:
        hit_mark = 'X'
    else:
        hit_mark = '?'
    ret = f'Y{int(r["return_amount"]):,}' if r['return_amount'] > 0 else '-'
    print(f'  {strat:10s} | {r["combination"]:8s} | Y{r["amount"]:>5,} | odds={r["odds"]:6.1f} | EV={r["expected_value"]:.2f} | {hit_mark:4s} | {ret}')

# ========== 2. 的中ベット詳細 ==========
print()
print('=' * 60)
cur.execute('''
    SELECT r.race_date, r.venue_id, r.race_number,
           b.strategy_type, b.combination, b.amount, b.odds,
           b.expected_value, b.return_amount,
           r.actual_result_trifecta
    FROM bets b
    JOIN races r ON b.race_id = r.id
    WHERE b.is_hit = TRUE
    ORDER BY r.race_date, r.venue_id, r.race_number
''')
hits = cur.fetchall()
print(f'=== 的中ベット詳細 ({len(hits)}件) ===')
if hits:
    total_win_invest = 0
    total_win_return = 0
    for h in hits:
        venue = VENUES.get(h['venue_id'], f"場{h['venue_id']}")
        strat = STRAT.get(h['strategy_type'], h['strategy_type'])
        ret = h['return_amount'] or 0
        profit = ret - h['amount']
        total_win_invest += h['amount']
        total_win_return += ret
        print(f'  {h["race_date"]} {venue} {h["race_number"]}R | {strat:10s} | {h["combination"]} | Y{h["amount"]:,} -> Y{int(ret):,} (profit: Y{int(profit):+,})')
    print(f'  的中合計: 投資Y{total_win_invest:,} -> 払戻Y{int(total_win_return):,} (profit: Y{int(total_win_return - total_win_invest):+,})')
else:
    print('  (的中ベットなし)')

# ========== 3. 戦略別サマリー ==========
print()
print('=' * 60)
cur.execute('''
    SELECT b.strategy_type,
           COUNT(*) as total,
           COUNT(CASE WHEN b.is_hit IS NOT NULL THEN 1 END) as settled,
           COUNT(CASE WHEN b.is_hit = TRUE THEN 1 END) as hits,
           SUM(b.amount) as invested,
           SUM(CASE WHEN b.is_hit IS NOT NULL THEN b.amount ELSE 0 END) as settled_invested,
           SUM(COALESCE(b.return_amount, 0)) as returned
    FROM bets b
    GROUP BY b.strategy_type
    ORDER BY b.strategy_type
''')
summary = cur.fetchall()
print('=== 戦略別サマリー ===')
print(f'{"戦略":12s} | {"全件":>4s} | {"確定":>4s} | {"的中":>3s} | {"的中率":>6s} | {"投資(確定)":>10s} | {"払戻":>10s} | {"損益":>10s} | {"回収率":>6s}')
print('-' * 95)
t_total = t_settled = t_hits = t_inv = t_ret = 0
for s in summary:
    strat = STRAT.get(s['strategy_type'], s['strategy_type'])
    settled = s['settled'] or 0
    hits = s['hits'] or 0
    inv = s['settled_invested'] or 0
    ret = s['returned'] or 0
    profit = ret - inv
    hit_rate = (hits / settled * 100) if settled > 0 else 0
    roi = (ret / inv * 100) if inv > 0 else 0
    t_total += s['total']
    t_settled += settled
    t_hits += hits
    t_inv += inv
    t_ret += ret
    sign = '+' if profit >= 0 else ''
    print(f'{strat:12s} | {s["total"]:4d} | {settled:4d} | {hits:3d} | {hit_rate:5.1f}% | Y{inv:>9,} | Y{int(ret):>9,} | {sign}Y{int(profit):>8,} | {roi:5.1f}%')
print('-' * 95)
t_profit = t_ret - t_inv
t_hr = (t_hits / t_settled * 100) if t_settled > 0 else 0
t_roi = (t_ret / t_inv * 100) if t_inv > 0 else 0
sign = '+' if t_profit >= 0 else ''
print(f'{"合計":12s} | {t_total:4d} | {t_settled:4d} | {t_hits:3d} | {t_hr:5.1f}% | Y{t_inv:>9,} | Y{int(t_ret):>9,} | {sign}Y{int(t_profit):>8,} | {t_roi:5.1f}%')

# ========== 4. 日別集計 ==========
print()
print('=' * 60)
cur.execute('''
    SELECT r.race_date,
           COUNT(*) as total,
           COUNT(CASE WHEN b.is_hit IS NOT NULL THEN 1 END) as settled,
           COUNT(CASE WHEN b.is_hit = TRUE THEN 1 END) as hits,
           SUM(b.amount) as invested,
           SUM(CASE WHEN b.is_hit IS NOT NULL THEN b.amount ELSE 0 END) as settled_inv,
           SUM(COALESCE(b.return_amount, 0)) as returned,
           COUNT(DISTINCT r.id) as races
    FROM bets b
    JOIN races r ON b.race_id = r.id
    GROUP BY r.race_date
    ORDER BY r.race_date
''')
daily = cur.fetchall()
print('=== 日別集計 ===')
print(f'{"日付":12s} | {"R数":>3s} | {"件数":>4s} | {"確定":>4s} | {"的中":>3s} | {"投資":>10s} | {"払戻":>10s} | {"損益":>10s} | {"回収率":>6s}')
print('-' * 85)
for d in daily:
    settled = d['settled'] or 0
    hits = d['hits'] or 0
    inv = d['settled_inv'] or 0
    ret = d['returned'] or 0
    profit = ret - inv
    roi = (ret / inv * 100) if inv > 0 else 0
    sign = '+' if profit >= 0 else ''
    print(f'{d["race_date"]} | {d["races"]:3d} | {d["total"]:4d} | {settled:4d} | {hits:3d} | Y{inv:>9,} | Y{int(ret):>9,} | {sign}Y{int(profit):>8,} | {roi:5.1f}%')

# ========== 5. オッズ帯別分析 ==========
print()
print('=' * 60)
print('=== オッズ帯別分析（確定ベットのみ） ===')
cur.execute('''
    SELECT
        CASE
            WHEN b.odds < 20 THEN '01: ~20倍'
            WHEN b.odds < 40 THEN '02: 20~40倍'
            WHEN b.odds < 60 THEN '03: 40~60倍'
            ELSE '04: 60倍~'
        END as odds_band,
        COUNT(*) as total,
        COUNT(CASE WHEN b.is_hit = TRUE THEN 1 END) as hits,
        SUM(b.amount) as invested,
        SUM(COALESCE(b.return_amount, 0)) as returned,
        AVG(b.odds) as avg_odds,
        AVG(b.expected_value) as avg_ev
    FROM bets b
    WHERE b.is_hit IS NOT NULL
    GROUP BY odds_band
    ORDER BY odds_band
''')
odds_analysis = cur.fetchall()
print(f'{"オッズ帯":12s} | {"件数":>4s} | {"的中":>3s} | {"的中率":>6s} | {"投資":>10s} | {"払戻":>10s} | {"回収率":>6s} | {"平均odds":>8s} | {"平均EV":>6s}')
print('-' * 90)
for o in odds_analysis:
    hits = o['hits'] or 0
    inv = o['invested'] or 0
    ret = o['returned'] or 0
    hr = (hits / o['total'] * 100) if o['total'] > 0 else 0
    roi = (ret / inv * 100) if inv > 0 else 0
    print(f'{o["odds_band"]:12s} | {o["total"]:4d} | {hits:3d} | {hr:5.1f}% | Y{inv:>9,} | Y{int(ret):>9,} | {roi:5.1f}% | {o["avg_odds"]:7.1f} | {o["avg_ev"]:.2f}')

# ========== 6. 結果照合状況 ==========
print()
print('=' * 60)
print('=== 結果照合状況 ===')
cur.execute('''
    SELECT
        r.race_date,
        COUNT(*) as total_races,
        COUNT(CASE WHEN r.is_finished = TRUE THEN 1 END) as finished,
        COUNT(CASE WHEN r.is_finished = FALSE OR r.is_finished IS NULL THEN 1 END) as pending
    FROM races r
    WHERE r.race_date >= '2026-03-11'
    GROUP BY r.race_date
    ORDER BY r.race_date
''')
reconcile = cur.fetchall()
for rc in reconcile:
    print(f'  {rc["race_date"]}: 全{rc["total_races"]}R / 確定{rc["finished"]} / 未確定{rc["pending"]}')

# ベットがあるレースの照合状況
cur.execute('''
    SELECT
        COUNT(DISTINCT r.id) as bet_races,
        COUNT(DISTINCT CASE WHEN r.is_finished = TRUE THEN r.id END) as finished,
        COUNT(DISTINCT CASE WHEN r.is_finished = FALSE OR r.is_finished IS NULL THEN r.id END) as pending
    FROM bets b
    JOIN races r ON b.race_id = r.id
''')
br = cur.fetchone()
print(f'  ベット対象レース: {br["bet_races"]}R / 確定{br["finished"]} / 未確定{br["pending"]}')

conn.close()
