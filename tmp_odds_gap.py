"""購入オッズ vs 確定オッズの乖離分析"""
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

# ========== 1. 的中ベットでの購入オッズ vs 確定オッズ比較 ==========
print('=== 1. 的中ベット: 購入オッズ vs 確定オッズ ===')
print('  (確定オッズ = payout_amount / 100 = 100円あたりの払戻)')
print()
cur.execute('''
    SELECT r.race_date, r.venue_id, r.race_number,
           b.strategy_type, b.combination, b.amount, b.odds as buy_odds,
           b.return_amount,
           r.payout_amount,
           r.actual_result_trifecta
    FROM bets b
    JOIN races r ON b.race_id = r.id
    WHERE b.is_hit = TRUE
    ORDER BY r.race_date, r.venue_id, r.race_number
''')
hits = cur.fetchall()

if hits:
    print(f'{"レース":16s} | {"組合せ":8s} | {"購入odds":>8s} | {"確定odds":>8s} | {"差":>8s} | {"乖離率":>6s}')
    print('-' * 70)
    for h in hits:
        venue = VENUES.get(h['venue_id'], f"場{h['venue_id']}")
        race_label = f'{h["race_date"]} {venue} {h["race_number"]}R'
        buy_odds = float(h['buy_odds'])
        # 確定オッズ = payout_amount / 100 (100円あたりの払戻金)
        confirmed_odds = float(h['payout_amount']) / 100.0 if h['payout_amount'] else None
        # もしくは return_amount / (amount / 100) で逆算
        actual_odds = float(h['return_amount']) / (float(h['amount']) / 100.0) / 100.0 if h['return_amount'] and h['amount'] else None

        if confirmed_odds:
            diff = confirmed_odds - buy_odds
            ratio = (diff / buy_odds) * 100
            print(f'{race_label:16s} | {h["combination"]:8s} | {buy_odds:8.1f} | {confirmed_odds:8.1f} | {diff:+8.1f} | {ratio:+5.1f}%')
        else:
            print(f'{race_label:16s} | {h["combination"]:8s} | {buy_odds:8.1f} | {"N/A":>8s} | {"":>8s} | {"":>6s}')
else:
    print('  (的中ベットなし)')

# ========== 2. 全レースの確定3連単払戻を確認 ==========
print()
print('=== 2. ベット対象レースの確定3連単オッズ一覧 ===')
print('  (payout_amount = 3連単100円あたり払戻)')
print()
cur.execute('''
    SELECT DISTINCT r.race_date, r.venue_id, r.race_number,
           r.actual_result_trifecta, r.payout_amount,
           r.is_finished
    FROM bets b
    JOIN races r ON b.race_id = r.id
    WHERE r.is_finished = TRUE
    ORDER BY r.race_date, r.venue_id, r.race_number
''')
races = cur.fetchall()

low_payout = 0
mid_payout = 0
high_payout = 0
for rc in races:
    venue = VENUES.get(rc['venue_id'], f"場{rc['venue_id']}")
    payout = rc['payout_amount'] or 0
    confirmed_odds = payout / 100.0 if payout else 0
    print(f'  {rc["race_date"]} {venue:4s} {rc["race_number"]:2d}R | 結果: {rc["actual_result_trifecta"] or "?":8s} | 確定払戻: Y{payout:>7,} (={confirmed_odds:.1f}倍)')
    if confirmed_odds < 30:
        low_payout += 1
    elif confirmed_odds < 60:
        mid_payout += 1
    else:
        high_payout += 1

print(f'\n  確定オッズ分布: <30倍:{low_payout}件, 30-60倍:{mid_payout}件, 60倍超:{high_payout}件')

# ========== 3. 購入オッズと確定結果オッズの比較（購入した組み合わせが的中していた場合の理論比較） ==========
print()
print('=== 3. 惜しかったレース分析（結果の着順と予測の近さ） ===')
print()

# 各レースで、実際の結果に最も近い買い目を探す
cur.execute('''
    SELECT r.race_date, r.venue_id, r.race_number,
           r.actual_result_trifecta,
           r.payout_amount,
           b.combination, b.odds, b.amount, b.strategy_type,
           b.expected_value
    FROM bets b
    JOIN races r ON b.race_id = r.id
    WHERE r.is_finished = TRUE
      AND r.actual_result_trifecta IS NOT NULL
    ORDER BY r.race_date, r.venue_id, r.race_number, b.strategy_type
''')
all_bets = cur.fetchall()

# レースごとに分析
from collections import defaultdict
race_bets = defaultdict(list)
for b in all_bets:
    key = (b['race_date'], b['venue_id'], b['race_number'])
    race_bets[key].append(b)

near_miss_count = 0
for key, bets in race_bets.items():
    result = bets[0]['actual_result_trifecta']
    if not result:
        continue
    result_parts = result.split('-')
    if len(result_parts) != 3:
        continue
    r1, r2, r3 = result_parts

    # 各ベットの「近さ」を判定
    for b in bets:
        combo_parts = b['combination'].split('-')
        if len(combo_parts) != 3:
            continue
        c1, c2, c3 = combo_parts

        # 1着一致で2,3着が逆（裏目）
        if c1 == r1 and c2 == r3 and c3 == r2:
            venue = VENUES.get(b['venue_id'], f"場{b['venue_id']}")
            near_miss_count += 1
            if near_miss_count <= 15:  # 表示数制限
                print(f'  裏目: {b["race_date"]} {venue} {b["race_number"]}R | 買:{b["combination"]} vs 結果:{result} | Y{b["amount"]:,} | odds={b["odds"]:.1f}')

        # 1着一致（2,3着は違う）
        elif c1 == r1 and (c2 != r2 or c3 != r3):
            pass  # 多すぎるので省略

# ========== 4. 購入時オッズの分布 vs 的中率 ==========
print()
print('=== 4. 購入時オッズ帯別の詳細分析 ===')
cur.execute('''
    SELECT
        CASE
            WHEN b.odds < 15 THEN '01: ~15'
            WHEN b.odds < 25 THEN '02: 15~25'
            WHEN b.odds < 35 THEN '03: 25~35'
            WHEN b.odds < 45 THEN '04: 35~45'
            WHEN b.odds < 55 THEN '05: 45~55'
            WHEN b.odds < 65 THEN '06: 55~65'
            ELSE '07: 65~'
        END as odds_band,
        COUNT(*) as total,
        COUNT(CASE WHEN b.is_hit = TRUE THEN 1 END) as hits,
        SUM(b.amount) as invested,
        SUM(COALESCE(b.return_amount, 0)) as returned,
        AVG(b.odds) as avg_odds,
        AVG(b.expected_value) as avg_ev,
        SUM(b.amount) / COUNT(*) as avg_bet
    FROM bets b
    WHERE b.is_hit IS NOT NULL
    GROUP BY odds_band
    ORDER BY odds_band
''')
bands = cur.fetchall()
print(f'{"帯":8s} | {"件数":>4s} | {"的中":>3s} | {"率":>5s} | {"投資":>10s} | {"払戻":>10s} | {"回収":>5s} | {"平均bet":>7s} | {"平均EV":>6s}')
print('-' * 85)
for b in bands:
    hits = b['hits'] or 0
    inv = b['invested'] or 0
    ret = b['returned'] or 0
    hr = (hits / b['total'] * 100) if b['total'] > 0 else 0
    roi = (ret / inv * 100) if inv > 0 else 0
    print(f'{b["odds_band"]:8s} | {b["total"]:4d} | {hits:3d} | {hr:4.1f}% | Y{inv:>9,} | Y{int(ret):>9,} | {roi:4.0f}% | Y{int(b["avg_bet"]):>6,} | {b["avg_ev"]:.2f}')

# ========== 5. 締切何分前にオッズ取得したか ==========
print()
print('=== 5. オッズ取得タイミング（締切からの分数） ===')
cur.execute('''
    SELECT r.race_date, r.venue_id, r.race_number,
           r.deadline_time, b.created_at,
           EXTRACT(EPOCH FROM (r.deadline_time - b.created_at)) / 60.0 as minutes_before
    FROM bets b
    JOIN races r ON b.race_id = r.id
    WHERE r.deadline_time IS NOT NULL
      AND b.created_at IS NOT NULL
    ORDER BY r.race_date, r.venue_id, r.race_number
''')
timing = cur.fetchall()

# 統計集計
if timing:
    minutes_list = [t['minutes_before'] for t in timing if t['minutes_before'] is not None]
    if minutes_list:
        avg_min = sum(minutes_list) / len(minutes_list)
        min_min = min(minutes_list)
        max_min = max(minutes_list)
        print(f'  ベット数: {len(minutes_list)}')
        print(f'  締切前平均: {avg_min:.1f}分')
        print(f'  締切前最小: {min_min:.1f}分')
        print(f'  締切前最大: {max_min:.1f}分')

        # 分布
        bins = {'~1分': 0, '1~2分': 0, '2~3分': 0, '3~4分': 0, '4~5分': 0, '5分~': 0}
        for m in minutes_list:
            if m < 1:
                bins['~1分'] += 1
            elif m < 2:
                bins['1~2分'] += 1
            elif m < 3:
                bins['2~3分'] += 1
            elif m < 4:
                bins['3~4分'] += 1
            elif m < 5:
                bins['4~5分'] += 1
            else:
                bins['5分~'] += 1
        print(f'  分布:')
        for label, count in bins.items():
            bar = '#' * (count // 2)
            print(f'    {label:6s}: {count:3d} {bar}')

conn.close()
