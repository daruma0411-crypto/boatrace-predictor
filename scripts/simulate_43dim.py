"""43次元モデル バックテストシミュレーション

直近データでモデルの予測確率 → 3連単オッズ照合 → Kelly基準ベット → ROI計測
high_confidence戦略（エントロピー<2.3フィルタ）での実運用シミュレーション

Usage:
    DATABASE_URL=... python scripts/simulate_43dim.py
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
    """直近N日分のレースデータ（オッズ・結果付き）を取得"""
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
        logger.info(f"レース取得: {len(races):,}件 ({cutoff.date()} ~)")

        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats
            WHERE race_id = ANY(%s)
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
            result_combo = f"{race['result_1st']}-{race['result_2nd']}-{race['result_3rd']}"
            data.append({
                'features': features,
                'result_1st': race['result_1st'] - 1,
                'result_2nd': race['result_2nd'] - 1,
                'result_3rd': race['result_3rd'] - 1,
                'result_combo': result_combo,
                'payout': race['payout_sanrentan'],
                'date': race['race_date'],
                'venue_id': race['venue_id'],
                'race_number': race['race_number'],
            })
        except Exception:
            continue

    logger.info(f"シミュレーション用データ: {len(data):,}件")
    return data


def compute_entropy(probs):
    """Shannon entropy"""
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log2(p))


def simulate(model, data, strategy_configs):
    """各戦略でシミュレーション実行"""
    device = torch.device('cpu')
    model.eval()

    results = {name: {'bets': 0, 'wins': 0, 'wagered': 0, 'payout': 0,
                       'daily': defaultdict(lambda: {'w': 0, 'p': 0})}
               for name in strategy_configs}

    for d in data:
        X = torch.FloatTensor(d['features']).unsqueeze(0).to(device)

        with torch.no_grad():
            out1, out2, out3 = model(X)
            probs_1st = torch.softmax(out1, dim=1).cpu().numpy()[0]
            probs_2nd = torch.softmax(out2, dim=1).cpu().numpy()[0]
            probs_3rd = torch.softmax(out3, dim=1).cpu().numpy()[0]

        entropy = compute_entropy(probs_1st)
        top_boat = np.argmax(probs_1st)

        # 5-6号艇軸スキップ
        if top_boat >= 4:
            continue

        # 3連単確率計算 (上位N個)
        top3_1st = np.argsort(probs_1st)[::-1][:3]
        top3_2nd = np.argsort(probs_2nd)[::-1][:4]
        top3_3rd = np.argsort(probs_3rd)[::-1][:4]

        combos = {}
        for i in top3_1st:
            for j in top3_2nd:
                if j == i:
                    continue
                for k in top3_3rd:
                    if k == i or k == j:
                        continue
                    prob = probs_1st[i] * probs_2nd[j] * probs_3rd[k]
                    combo = f"{i+1}-{j+1}-{k+1}"
                    combos[combo] = prob

        actual_combo = d['result_combo']
        actual_payout = d['payout']

        for strat_name, config in strategy_configs.items():
            # エントロピーフィルタ
            max_entropy = config.get('max_entropy', 999)
            if entropy >= max_entropy:
                continue

            min_odds = config.get('min_odds', 1.5)
            max_odds = config.get('max_odds', 50)
            min_ev = config.get('min_ev', 0.0)
            odds_discount = config.get('odds_discount', 0.92)
            max_bets = config.get('max_bets', 4)

            # 各コンボのEV計算
            bet_candidates = []
            for combo, prob in sorted(combos.items(), key=lambda x: -x[1]):
                # 市場オッズ推定: 1/prob（実際はオッズデータを使うが、ここでは推定）
                estimated_odds = 1.0 / prob if prob > 0.001 else 1000
                if estimated_odds < min_odds or estimated_odds > max_odds:
                    continue
                ev = prob * estimated_odds * odds_discount
                if ev > min_ev:
                    bet_candidates.append((combo, prob, ev))

                if len(bet_candidates) >= max_bets:
                    break

            for combo, prob, ev in bet_candidates:
                amount = 100  # フラット100円ベット
                r = results[strat_name]
                r['bets'] += 1
                r['wagered'] += amount
                r['daily'][d['date']]['w'] += amount

                if combo == actual_combo:
                    payout = actual_payout * amount // 100
                    r['wins'] += 1
                    r['payout'] += payout
                    r['daily'][d['date']]['p'] += payout

    return results


def main():
    logger.info("=== 43次元モデル バックテストシミュレーション ===")

    device = torch.device('cpu')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'models', 'boatrace_model.pth')
    model = load_model(model_path, device)

    data = load_sim_data(days=30)
    if not data:
        logger.error("シミュレーションデータなし")
        return

    # 戦略設定
    strategies = {
        'high_confidence': {
            'max_entropy': 2.3,
            'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.92,
            'max_bets': 4,
        },
        'conservative': {
            'max_entropy': 999,  # フィルタなし
            'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.90,
            'max_bets': 3,
        },
        'standard': {
            'max_entropy': 999,
            'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.95,
            'max_bets': 5,
        },
        'strict_entropy': {
            'max_entropy': 2.0,  # より厳しい
            'min_odds': 1.5, 'max_odds': 50,
            'min_ev': 0.0, 'odds_discount': 0.92,
            'max_bets': 4,
        },
        'wide_odds': {
            'max_entropy': 2.3,
            'min_odds': 5.0, 'max_odds': 150,
            'min_ev': 1.0, 'odds_discount': 0.92,
            'max_bets': 4,
        },
    }

    results = simulate(model, data, strategies)

    # 結果表示
    print("\n" + "=" * 80)
    print("43次元モデル バックテスト結果")
    print(f"データ期間: {data[0]['date']} ~ {data[-1]['date']} ({len(data):,}レース)")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Bets':>6} {'Wins':>5} {'HitRate':>8} {'Wagered':>10} {'Payout':>10} {'P/L':>10} {'ROI':>7}")
    print("-" * 80)

    for name in strategies:
        r = results[name]
        hit = r['wins'] / r['bets'] * 100 if r['bets'] > 0 else 0
        pl = r['payout'] - r['wagered']
        roi = r['payout'] / r['wagered'] * 100 if r['wagered'] > 0 else 0
        print(f"{name:<20} {r['bets']:>6} {r['wins']:>5} {hit:>7.1f}% {r['wagered']:>10,} {r['payout']:>10,} {pl:>+10,} {roi:>6.1f}%")

    # 日別P/L for high_confidence
    print(f"\n--- high_confidence 日別P/L ---")
    hc = results['high_confidence']
    cumulative = 0
    max_dd = 0
    peak = 0
    for date in sorted(hc['daily'].keys()):
        day = hc['daily'][date]
        daily_pl = day['p'] - day['w']
        cumulative += daily_pl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
        print(f"  {date} | W:{day['w']:>6,} P:{day['p']:>7,} | daily:{daily_pl:>+7,} | cum:{cumulative:>+8,} | DD:{dd:>6,}")

    print(f"\n  最大ドローダウン: {max_dd:,}")
    print(f"  最終累積P/L: {cumulative:+,}")


if __name__ == '__main__':
    main()
