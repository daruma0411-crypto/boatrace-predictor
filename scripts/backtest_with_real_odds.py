"""バックテスト: 実オッズデータで最適パラメータ探索

betsテーブルに蓄積された実オッズ・モデル確率・結果データを使い、
異なるキャリブレーション係数・EV閾値・max_betsの組み合わせで
シミュレーションを実行し、ROI最大化パラメータを発見する。
"""
import json
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
)


def get_dynamic_discount(raw_odds):
    if raw_odds < 25.0:
        return 0.85
    elif raw_odds < 40.0:
        return 0.88
    else:
        return 0.95


def recover_probability(expected_value, raw_odds):
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

    # レース単位でグルーピングするためrace_idも取得
    cur.execute("""
        SELECT b.id, b.race_id, b.odds, b.expected_value, b.result,
               b.payout, b.amount, b.is_hit, b.strategy_type,
               b.combination, b.created_at
        FROM bets b
        WHERE b.result IS NOT NULL
          AND b.created_at >= '2026-03-11'
        ORDER BY b.race_id, b.expected_value DESC
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"=== backtest with real odds ===")
    print(f"total bets: {len(rows)}")

    if not rows:
        print("no data.")
        return

    # 各ベットに確率を復元
    bets = []
    for row in rows:
        odds = float(row['odds']) if row['odds'] else 0
        ev = float(row['expected_value']) if row['expected_value'] else 0
        prob = recover_probability(ev, odds)
        is_hit = bool(row['is_hit']) if row['is_hit'] is not None else False
        payout_per_100 = float(row['payout']) / (float(row['amount']) / 100) if row['is_hit'] and float(row['amount']) > 0 else 0

        bets.append({
            'id': row['id'],
            'race_id': row['race_id'],
            'odds': odds,
            'ev': ev,
            'prob': prob,
            'is_hit': is_hit,
            'amount': int(row['amount']) if row['amount'] else 0,
            'payout': int(row['payout']) if row['payout'] else 0,
            'strategy': row['strategy_type'],
            'combo': row['combination'],
        })

    # --- Grid Search ---
    calibration_factors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_evs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    max_bets_options = [1, 2, 3, 5]

    best_roi = -999
    best_params = {}
    results = []

    for cal_f in calibration_factors:
        for min_ev in min_evs:
            for max_b in max_bets_options:
                # シミュレーション: 各ベットにキャリブレーション適用
                total_wagered = 0
                total_payout = 0
                bet_count = 0
                hit_count = 0

                # レース単位でmax_betsを適用
                race_bets = {}
                for b in bets:
                    rid = b['race_id']
                    if rid not in race_bets:
                        race_bets[rid] = []
                    race_bets[rid].append(b)

                for rid, race_bet_list in race_bets.items():
                    # キャリブレーション適用 & EV再計算
                    qualified = []
                    for b in race_bet_list:
                        cal_prob = b['prob'] * cal_f
                        discount = get_dynamic_discount(b['odds'])
                        cal_ev = cal_prob * b['odds'] * discount

                        if cal_ev >= min_ev:
                            qualified.append((cal_ev, b))

                    # EV上位max_b件に絞る
                    qualified.sort(key=lambda x: x[0], reverse=True)
                    selected = qualified[:max_b]

                    for _, b in selected:
                        total_wagered += b['amount']
                        total_payout += b['payout']
                        bet_count += 1
                        if b['is_hit']:
                            hit_count += 1

                roi = (total_payout / total_wagered * 100) if total_wagered > 0 else 0
                hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0

                results.append({
                    'cal_factor': cal_f,
                    'min_ev': min_ev,
                    'max_bets': max_b,
                    'bet_count': bet_count,
                    'hit_count': hit_count,
                    'hit_rate': round(hit_rate, 2),
                    'wagered': total_wagered,
                    'payout': total_payout,
                    'roi': round(roi, 2),
                    'profit': total_payout - total_wagered,
                })

                if roi > best_roi and bet_count >= 5:  # 最低5件
                    best_roi = roi
                    best_params = {
                        'cal_factor': cal_f,
                        'min_ev': min_ev,
                        'max_bets': max_b,
                        'bet_count': bet_count,
                        'hit_count': hit_count,
                        'roi': round(roi, 2),
                    }

    # ROI上位20件を表示
    results.sort(key=lambda x: x['roi'], reverse=True)
    print(f"\n=== ROI Top 20 (min 5 bets) ===")
    print(f"{'cal_f':>6} | {'min_ev':>6} | {'max_b':>5} | {'bets':>5} | {'hits':>4} | {'hit%':>6} | {'wagered':>10} | {'payout':>10} | {'ROI':>7} | {'profit':>8}")
    print("-" * 95)
    shown = 0
    for r in results:
        if r['bet_count'] < 5:
            continue
        print(f"{r['cal_factor']:>6.1f} | {r['min_ev']:>6.1f} | {r['max_bets']:>5} | {r['bet_count']:>5} | {r['hit_count']:>4} | {r['hit_rate']:>5.1f}% | {r['wagered']:>10,} | {r['payout']:>10,} | {r['roi']:>6.1f}% | {r['profit']:>+8,}")
        shown += 1
        if shown >= 20:
            break

    # ROI > 100%の件数
    profitable = [r for r in results if r['roi'] > 100 and r['bet_count'] >= 5]
    print(f"\nROI > 100%: {len(profitable)} parameter sets")

    print(f"\n=== Best Parameters ===")
    print(json.dumps(best_params, indent=2))

    # ベット数10件以上でベスト
    results_10 = [r for r in results if r['bet_count'] >= 10]
    results_10.sort(key=lambda x: x['roi'], reverse=True)
    if results_10:
        print(f"\n=== Best (min 10 bets) ===")
        b = results_10[0]
        print(f"cal_factor={b['cal_factor']}, min_ev={b['min_ev']}, max_bets={b['max_bets']}")
        print(f"bets={b['bet_count']}, hits={b['hit_count']}, ROI={b['roi']}%, profit={b['profit']:+,}")

    # ベット数30件以上でベスト（信頼度が高い）
    results_30 = [r for r in results if r['bet_count'] >= 30]
    results_30.sort(key=lambda x: x['roi'], reverse=True)
    if results_30:
        print(f"\n=== Best (min 30 bets) ===")
        b = results_30[0]
        print(f"cal_factor={b['cal_factor']}, min_ev={b['min_ev']}, max_bets={b['max_bets']}")
        print(f"bets={b['bet_count']}, hits={b['hit_count']}, ROI={b['roi']}%, profit={b['profit']:+,}")

    # ベット数50件以上でベスト
    results_50 = [r for r in results if r['bet_count'] >= 50]
    results_50.sort(key=lambda x: x['roi'], reverse=True)
    if results_50:
        print(f"\n=== Best (min 50 bets) ===")
        b = results_50[0]
        print(f"cal_factor={b['cal_factor']}, min_ev={b['min_ev']}, max_bets={b['max_bets']}")
        print(f"bets={b['bet_count']}, hits={b['hit_count']}, ROI={b['roi']}%, profit={b['profit']:+,}")


if __name__ == '__main__':
    main()
