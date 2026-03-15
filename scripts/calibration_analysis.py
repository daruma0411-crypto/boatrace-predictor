"""キャリブレーション分析スクリプト

betsテーブルの実績データから、モデル確率の過大推定率を定量化し、
確率帯別のキャリブレーション係数を算出する。

出力: config/calibration.json
"""
import json
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

# DB接続
DB_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
)


def get_dynamic_discount(raw_odds):
    """betting.pyと同じ動的ディスカウント"""
    if raw_odds < 25.0:
        return 0.85
    elif raw_odds < 40.0:
        return 0.88
    else:
        return 0.95


def recover_probability(expected_value, raw_odds):
    """EV = prob * discounted_odds から prob を逆算"""
    if raw_odds <= 0:
        return 0.0
    discount = get_dynamic_discount(raw_odds)
    discounted_odds = raw_odds * discount
    if discounted_odds <= 0:
        return 0.0
    return expected_value / discounted_odds


def main():
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # 3/11以降の確定済みベットを取得
    cur.execute("""
        SELECT b.id, b.odds, b.expected_value, b.result, b.payout,
               b.amount, b.is_hit, b.strategy_type, b.combination,
               b.created_at
        FROM bets b
        WHERE b.result IS NOT NULL
          AND b.created_at >= '2026-03-11'
        ORDER BY b.created_at
    """)
    rows = cur.fetchall()
    print(f"\n=== キャリブレーション分析 ===")
    print(f"確定済みベット: {len(rows)}件")

    if not rows:
        print("データなし。終了。")
        conn.close()
        return

    # 確率帯別集計
    bands = [
        (0.00, 0.01, "0-1%"),
        (0.01, 0.02, "1-2%"),
        (0.02, 0.03, "2-3%"),
        (0.03, 0.05, "3-5%"),
        (0.05, 0.10, "5-10%"),
        (0.10, 1.00, "10%+"),
    ]

    band_stats = {label: {'count': 0, 'hits': 0, 'total_prob': 0.0,
                           'total_wagered': 0, 'total_payout': 0}
                  for _, _, label in bands}

    all_probs = []
    all_hits = []

    for row in rows:
        odds = float(row['odds']) if row['odds'] else 0
        ev = float(row['expected_value']) if row['expected_value'] else 0
        is_hit = bool(row['is_hit']) if row['is_hit'] is not None else False
        amount = int(row['amount']) if row['amount'] else 0
        payout = int(row['payout']) if row['payout'] else 0

        prob = recover_probability(ev, odds)
        if prob <= 0:
            continue

        all_probs.append(prob)
        all_hits.append(1 if is_hit else 0)

        for lo, hi, label in bands:
            if lo <= prob < hi:
                band_stats[label]['count'] += 1
                band_stats[label]['hits'] += (1 if is_hit else 0)
                band_stats[label]['total_prob'] += prob
                band_stats[label]['total_wagered'] += amount
                band_stats[label]['total_payout'] += payout
                break

    # 結果表示
    print(f"\n{'帯':>8} | {'件数':>6} | {'的中':>4} | {'的中率':>8} | {'平均モデルP':>10} | {'係数':>6} | {'投資':>10} | {'回収':>10} | {'ROI':>6}")
    print("-" * 100)

    calibration_data = []
    for lo, hi, label in bands:
        s = band_stats[label]
        if s['count'] == 0:
            continue

        actual_hit_rate = s['hits'] / s['count']
        avg_model_prob = s['total_prob'] / s['count']

        # キャリブレーション係数 = 実際の的中率 / モデルの平均確率
        if avg_model_prob > 0:
            cal_factor = actual_hit_rate / avg_model_prob
        else:
            cal_factor = 0.0

        roi = (s['total_payout'] / s['total_wagered'] * 100) if s['total_wagered'] > 0 else 0

        print(f"{label:>8} | {s['count']:>6} | {s['hits']:>4} | {actual_hit_rate:>7.1%} | {avg_model_prob:>9.3%} | {cal_factor:>5.2f}x | Y{s['total_wagered']:>8,} | Y{s['total_payout']:>8,} | {roi:>5.1f}%")

        calibration_data.append({
            'min': lo,
            'max': hi,
            'label': label,
            'count': s['count'],
            'hits': s['hits'],
            'actual_hit_rate': round(actual_hit_rate, 6),
            'avg_model_prob': round(avg_model_prob, 6),
            'calibration_factor': round(cal_factor, 4),
        })

    # 全体サマリ
    total_count = len(all_probs)
    total_hits = sum(all_hits)
    total_wagered = sum(s['total_wagered'] for s in band_stats.values())
    total_payout = sum(s['total_payout'] for s in band_stats.values())
    overall_hit_rate = total_hits / total_count if total_count > 0 else 0
    avg_prob = sum(all_probs) / len(all_probs) if all_probs else 0
    overall_roi = (total_payout / total_wagered * 100) if total_wagered > 0 else 0

    print("-" * 100)
    print(f"{'ALL':>8} | {total_count:>6} | {total_hits:>4} | {overall_hit_rate:>7.1%} | {avg_prob:>9.3%} | {'':>6} | Y{total_wagered:>8,} | Y{total_payout:>8,} | {overall_roi:>5.1f}%")

    # 戦略別サマリ
    print(f"\n=== 戦略別サマリ ===")
    strategy_stats = {}
    for row in rows:
        st = row['strategy_type']
        if st not in strategy_stats:
            strategy_stats[st] = {'count': 0, 'hits': 0, 'wagered': 0, 'payout': 0}
        strategy_stats[st]['count'] += 1
        strategy_stats[st]['hits'] += (1 if row['is_hit'] else 0)
        strategy_stats[st]['wagered'] += int(row['amount']) if row['amount'] else 0
        strategy_stats[st]['payout'] += int(row['payout']) if row['payout'] else 0

    print(f"{'戦略':>20} | {'件数':>6} | {'的中':>4} | {'的中率':>8} | {'投資':>10} | {'回収':>10} | {'ROI':>6}")
    print("-" * 80)
    for st, s in sorted(strategy_stats.items()):
        hit_rate = s['hits'] / s['count'] if s['count'] > 0 else 0
        roi = (s['payout'] / s['wagered'] * 100) if s['wagered'] > 0 else 0
        print(f"{st:>20} | {s['count']:>6} | {s['hits']:>4} | {hit_rate:>7.1%} | Y{s['wagered']:>8,} | Y{s['payout']:>8,} | {roi:>5.1f}%")

    # EV帯別の実績分析
    print(f"\n=== EV帯別実績 ===")
    ev_bands = [
        (0.0, 1.0, "EV<1.0"),
        (1.0, 1.5, "1.0-1.5"),
        (1.5, 2.0, "1.5-2.0"),
        (2.0, 3.0, "2.0-3.0"),
        (3.0, 100.0, "3.0+"),
    ]
    ev_stats = {label: {'count': 0, 'hits': 0, 'wagered': 0, 'payout': 0}
                for _, _, label in ev_bands}

    for row in rows:
        ev = float(row['expected_value']) if row['expected_value'] else 0
        is_hit = bool(row['is_hit']) if row['is_hit'] is not None else False
        for lo, hi, label in ev_bands:
            if lo <= ev < hi:
                ev_stats[label]['count'] += 1
                ev_stats[label]['hits'] += (1 if is_hit else 0)
                ev_stats[label]['wagered'] += int(row['amount']) if row['amount'] else 0
                ev_stats[label]['payout'] += int(row['payout']) if row['payout'] else 0
                break

    print(f"{'EV帯':>10} | {'件数':>6} | {'的中':>4} | {'的中率':>8} | {'投資':>10} | {'回収':>10} | {'ROI':>6}")
    print("-" * 80)
    for _, _, label in ev_bands:
        s = ev_stats[label]
        if s['count'] == 0:
            continue
        hit_rate = s['hits'] / s['count'] if s['count'] > 0 else 0
        roi = (s['payout'] / s['wagered'] * 100) if s['wagered'] > 0 else 0
        print(f"{label:>10} | {s['count']:>6} | {s['hits']:>4} | {hit_rate:>7.1%} | Y{s['wagered']:>8,} | Y{s['payout']:>8,} | {roi:>5.1f}%")

    # config/calibration.json に出力
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    os.makedirs(config_dir, exist_ok=True)
    output_path = os.path.join(config_dir, 'calibration.json')

    output = {
        'analysis_date': '2026-03-15',
        'data_period': '2026-03-11 to 2026-03-14',
        'total_bets': total_count,
        'total_hits': total_hits,
        'overall_hit_rate': round(overall_hit_rate, 6),
        'overall_roi': round(overall_roi, 2),
        'prob_bands': calibration_data,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n出力: {output_path}")
    conn.close()


if __name__ == '__main__':
    main()
