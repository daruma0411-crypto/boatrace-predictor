"""43次元モデル バックテスト（実配当ベース）

モデル予測 → 3連単上位N候補 → 的中時は実配当で回収
市場オッズの代わりに実配当(payout_sanrentan/100)を使用

Usage:
    DATABASE_URL=... python scripts/simulate_with_payout.py
"""
import sys
import os
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load_model
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_sim_data(days=30):
    """直近N日分のレースデータを取得"""
    fe = FeatureEngineer()
    cutoff = datetime.now() - timedelta(days=days)

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_number, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.payout_sanrentan,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.race_date >= %s AND r.status = 'finished'
              AND r.result_1st IS NOT NULL
              AND r.payout_sanrentan IS NOT NULL
              AND r.payout_sanrentan > 0
              AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date, r.venue_id, r.race_number
        """, (cutoff.date(),))
        races = cur.fetchall()
        logger.info(f"Races: {len(races):,} ({cutoff.date()} ~)")

        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()

    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    data = []
    for race in races:
        boats = boats_by_race.get(race['id'], [])
        if len(boats) != 6:
            continue
        race_data = {
            'venue_id': race['venue_id'],
            'month': race['race_date'].month,
            'distance': 1800,
            'wind_speed': race.get('wind_speed') or 0,
            'wind_direction': race.get('wind_direction') or 'calm',
            'temperature': race.get('temperature') or 20,
            'wave_height': race.get('wave_height') or 0,
            'water_temperature': race.get('water_temperature') or 20,
        }
        try:
            features = fe.transform(race_data, boats)
            data.append({
                'features': features,
                'result_combo': f"{race['result_1st']}-{race['result_2nd']}-{race['result_3rd']}",
                'payout': race['payout_sanrentan'],  # 100円あたり配当
                'date': race['race_date'],
                'venue_id': race['venue_id'],
                'race_number': race['race_number'],
            })
        except Exception:
            continue

    logger.info(f"Sim data: {len(data):,}")
    return data


def compute_entropy(probs):
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log2(p))


def get_top_combos(probs_1st, probs_2nd, probs_3rd, top_n=20):
    """モデル確率から3連単上位N候補を取得"""
    combos = {}
    # 1着上位4 × 2着上位5 × 3着上位5 で最大100候補
    top_1st = np.argsort(probs_1st)[::-1][:4]
    top_2nd = np.argsort(probs_2nd)[::-1][:5]
    top_3rd = np.argsort(probs_3rd)[::-1][:5]

    for i in top_1st:
        for j in top_2nd:
            if j == i:
                continue
            for k in top_3rd:
                if k == i or k == j:
                    continue
                prob = probs_1st[i] * probs_2nd[j] * probs_3rd[k]
                combos[f"{i+1}-{j+1}-{k+1}"] = prob

    # 上位N個をソート
    sorted_combos = sorted(combos.items(), key=lambda x: -x[1])[:top_n]
    return sorted_combos


def simulate(model, data, configs):
    """バックテスト実行"""
    device = torch.device('cpu')
    model.eval()

    results = {}
    for name in configs:
        results[name] = {
            'bets': 0, 'wins': 0, 'wagered': 0, 'payout': 0,
            'daily': defaultdict(lambda: {'w': 0, 'p': 0, 'bets': 0, 'wins': 0}),
        }

    for d in data:
        X = torch.FloatTensor(d['features']).unsqueeze(0).to(device)
        with torch.no_grad():
            out1, out2, out3 = model(X)
            p1 = torch.softmax(out1, dim=1).cpu().numpy()[0]
            p2 = torch.softmax(out2, dim=1).cpu().numpy()[0]
            p3 = torch.softmax(out3, dim=1).cpu().numpy()[0]

        entropy = compute_entropy(p1)
        top_boat = np.argmax(p1)

        # 5-6号艇1着予測スキップ
        if top_boat >= 4:
            continue

        # 3連単候補
        combos = get_top_combos(p1, p2, p3, top_n=30)

        actual = d['result_combo']
        actual_payout = d['payout']  # 100円あたり配当
        actual_odds = actual_payout / 100.0

        for strat_name, cfg in configs.items():
            max_ent = cfg.get('max_entropy', 999)
            if entropy >= max_ent:
                continue

            min_odds = cfg.get('min_odds', 1.5)
            max_odds = cfg.get('max_odds', 50)
            min_ev = cfg.get('min_ev', 0.0)
            discount = cfg.get('odds_discount', 0.92)
            max_bets = cfg.get('max_bets', 4)
            min_prob = cfg.get('min_prob', 0.02)

            bet_count = 0
            r = results[strat_name]

            for combo, prob in combos:
                if prob < min_prob:
                    continue

                # オッズ推定: 配当データがないので、モデル確率の逆数を市場オッズの近似とする
                # ただし実際の的中判定と配当は実データを使う
                # EV判定には「推定市場オッズ」を使う
                # ここでは万人オッズの近似として: market_prob ≈ prob * 0.7 (モデルが市場より正確な前提)
                # → estimated_odds ≈ 1 / (prob * 0.7)
                est_market_odds = 1.0 / (prob * 0.75) if prob > 0.001 else 1000

                if est_market_odds < min_odds or est_market_odds > max_odds:
                    continue

                ev = prob * est_market_odds * discount
                if ev < min_ev:
                    continue

                # ベット実行
                amount = 100
                r['bets'] += 1
                r['wagered'] += amount
                r['daily'][d['date']]['w'] += amount
                r['daily'][d['date']]['bets'] += 1
                bet_count += 1

                if combo == actual:
                    payout = actual_payout * amount // 100
                    r['wins'] += 1
                    r['payout'] += payout
                    r['daily'][d['date']]['p'] += payout
                    r['daily'][d['date']]['wins'] += 1

                if bet_count >= max_bets:
                    break

    return results


def print_results(data, results, configs):
    print("\n" + "=" * 85)
    print("43次元モデル バックテスト（実配当ベース）")
    print(f"期間: {data[0]['date']} ~ {data[-1]['date']} ({len(data):,}レース)")
    print("=" * 85)
    print(f"{'Strategy':<20} {'Bets':>6} {'Wins':>5} {'Hit%':>6} {'Wagered':>10} {'Payout':>10} {'P/L':>10} {'ROI':>7}")
    print("-" * 85)

    for name in configs:
        r = results[name]
        hit = r['wins'] / r['bets'] * 100 if r['bets'] > 0 else 0
        pl = r['payout'] - r['wagered']
        roi = r['payout'] / r['wagered'] * 100 if r['wagered'] > 0 else 0
        mark = " ***" if roi >= 100 else ""
        print(f"{name:<20} {r['bets']:>6} {r['wins']:>5} {hit:>5.1f}% {r['wagered']:>10,} {r['payout']:>10,} {pl:>+10,} {roi:>6.1f}%{mark}")

    # high_confidence 日別
    for target in ['high_confidence', 'hc_strict', 'hc_ev1']:
        if target not in results or results[target]['bets'] == 0:
            continue
        r = results[target]
        print(f"\n--- {target} 日別P/L ---")
        cumulative = 0
        peak = 0
        max_dd = 0
        for date in sorted(r['daily'].keys()):
            day = r['daily'][date]
            dpl = day['p'] - day['w']
            cumulative += dpl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
            print(f"  {date} | B:{day['bets']:>3} W:{day['wins']:>2} | "
                  f"W:{day['w']:>6,} P:{day['p']:>7,} | "
                  f"daily:{dpl:>+7,} | cum:{cumulative:>+8,}")
        print(f"  MaxDD: {max_dd:,} | Final P/L: {cumulative:+,}")


def main():
    logger.info("=== 43次元モデル バックテスト（実配当ベース）===")

    device = torch.device('cpu')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'models', 'boatrace_model.pth')
    model = load_model(model_path, device)
    data = load_sim_data(days=30)
    if not data:
        logger.error("データなし")
        return

    configs = {
        'high_confidence': {
            'max_entropy': 2.3, 'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.92, 'max_bets': 4, 'min_prob': 0.02,
        },
        'hc_strict': {
            'max_entropy': 2.0, 'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.92, 'max_bets': 4, 'min_prob': 0.02,
        },
        'hc_ev1': {
            'max_entropy': 2.3, 'min_odds': 5.0, 'max_odds': 80,
            'min_ev': 1.05, 'odds_discount': 0.92, 'max_bets': 4, 'min_prob': 0.02,
        },
        'conservative': {
            'max_entropy': 999, 'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.90, 'max_bets': 3, 'min_prob': 0.02,
        },
        'standard': {
            'max_entropy': 999, 'min_odds': 1.5, 'max_odds': 150,
            'min_ev': 0.0, 'odds_discount': 0.95, 'max_bets': 5, 'min_prob': 0.02,
        },
        'top1_only': {
            'max_entropy': 2.3, 'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.92, 'max_bets': 1, 'min_prob': 0.05,
        },
        'top2_only': {
            'max_entropy': 2.3, 'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.92, 'max_bets': 2, 'min_prob': 0.03,
        },
    }

    results = simulate(model, data, configs)
    print_results(data, results, configs)


if __name__ == '__main__':
    main()
