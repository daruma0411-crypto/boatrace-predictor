"""EVスキャン戦略バックテスト（改訂版）

市場オッズが不明な状況での最も正直なバックテスト：
  - Top-N分散投資（確率上位N通りに分散ベット）
  - 信頼度フィルター（1号艇確率が低い＝荒れ予想のレースに集中）
  - 確率比例配分（ケリー的に高確率の組み合わせに多く配分）

精算は全て実際の払戻金で行う。

使い方:
    python scripts/backtest_ev_strategy.py
    python scripts/backtest_ev_strategy.py --bankroll 100000
"""
import sys
import os
import logging
import numpy as np
import torch
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load_model
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_all_data():
    """全データ一括取得"""
    logger.info("=== データ読み込み ===")
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, venue_id, race_date,
                   result_1st, result_2nd, result_3rd,
                   payout_sanrentan
            FROM races
            WHERE result_1st IS NOT NULL AND status = 'finished'
              AND payout_sanrentan IS NOT NULL AND payout_sanrentan > 0
            ORDER BY race_date
        """)
        races = cur.fetchall()
        logger.info(f"レース: {len(races):,}件")

        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()
        logger.info(f"ボート: {len(all_boats):,}件")

    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))
    return races, boats_by_race


def batch_predict(races, boats_by_race, model, device):
    """全レースの特徴量生成 + バッチ推論"""
    feature_engineer = FeatureEngineer()

    logger.info("特徴量生成中...")
    features_list = []
    valid_races = []

    for race in races:
        boats = boats_by_race.get(race['id'], [])
        if len(boats) != 6:
            continue
        race_data = {
            'venue_id': race['venue_id'],
            'month': race['race_date'].month,
            'distance': 1800, 'wind_speed': 0,
            'wind_direction': 'calm', 'temperature': 20,
        }
        try:
            features = feature_engineer.transform(race_data, boats)
            features_list.append(features)
            valid_races.append(race)
        except Exception:
            continue

    logger.info(f"有効レース: {len(valid_races):,}件")
    logger.info("バッチ推論中...")
    X = torch.FloatTensor(np.array(features_list)).to(device)
    BS = 4096
    all_p1, all_p2, all_p3 = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), BS):
            o1, o2, o3 = model(X[i:i+BS])
            all_p1.append(torch.softmax(o1, dim=1).numpy())
            all_p2.append(torch.softmax(o2, dim=1).numpy())
            all_p3.append(torch.softmax(o3, dim=1).numpy())

    logger.info("推論完了\n")
    return (valid_races,
            np.concatenate(all_p1),
            np.concatenate(all_p2),
            np.concatenate(all_p3))


def calc_sanrentan_probs(p1, p2, p3):
    """条件付き確率で3連単全120通りの確率を計算"""
    result = {}
    for combo in permutations(range(6), 3):
        i, j, k = combo
        pi = p1[i]
        if pi <= 0:
            continue
        rem2 = sum(p2[x] for x in range(6) if x != i)
        if rem2 <= 0:
            continue
        pj = p2[j] / rem2
        rem3 = sum(p3[x] for x in range(6) if x != i and x != j)
        if rem3 <= 0:
            continue
        pk = p3[k] / rem3
        prob = pi * pj * pk
        if prob > 0:
            result[f"{i+1}-{j+1}-{k+1}"] = prob
    return result


# ====================================================================
#  戦略関数群
# ====================================================================

def strategy_old_argmax(race, p1, p2, p3, **kw):
    """旧戦略: argmax 1点買い 100円"""
    pred = f"{np.argmax(p1)+1}-{np.argmax(p2)+1}-{np.argmax(p3)+1}"
    return [{'combo': pred, 'amount': 100}]


def strategy_topN_equal(race, p1, p2, p3, N=3, bet_per_point=100, **kw):
    """Top-N均等分散: 確率上位N通りに均等額ベット"""
    sp = calc_sanrentan_probs(p1, p2, p3)
    ranked = sorted(sp.items(), key=lambda x: x[1], reverse=True)[:N]
    return [{'combo': c, 'amount': bet_per_point} for c, _ in ranked]


def strategy_topN_weighted(race, p1, p2, p3, N=5, total_bet=500, **kw):
    """Top-N確率比例配分: 確率に比例して配分"""
    sp = calc_sanrentan_probs(p1, p2, p3)
    ranked = sorted(sp.items(), key=lambda x: x[1], reverse=True)[:N]
    if not ranked:
        return []
    total_p = sum(p for _, p in ranked)
    if total_p <= 0:
        return []
    bets = []
    for combo, prob in ranked:
        amount = int(round(total_bet * prob / total_p / 100) * 100)
        if amount >= 100:
            bets.append({'combo': combo, 'amount': amount})
    return bets


def strategy_upset_hunter(race, p1, p2, p3,
                           N=5, total_bet=500,
                           max_boat1_prob=0.45, **kw):
    """荒れレース狙い: 1号艇確率が閾値以下のレースだけ投資

    1号艇が弱いレース = 穴展開 = 高配当チャンス
    """
    if p1[0] > max_boat1_prob:
        return []  # 1号艇が強すぎるレースは見送り

    sp = calc_sanrentan_probs(p1, p2, p3)
    ranked = sorted(sp.items(), key=lambda x: x[1], reverse=True)[:N]
    if not ranked:
        return []
    total_p = sum(p for _, p in ranked)
    if total_p <= 0:
        return []
    bets = []
    for combo, prob in ranked:
        amount = int(round(total_bet * prob / total_p / 100) * 100)
        if amount >= 100:
            bets.append({'combo': combo, 'amount': amount})
    return bets


def strategy_non_favorite_focus(race, p1, p2, p3,
                                 N=5, total_bet=500, **kw):
    """穴目フォーカス: 1号艇頭の組み合わせを除外し、残りのTop-Nに投資"""
    sp = calc_sanrentan_probs(p1, p2, p3)
    # 1号艇頭を除外
    filtered = {c: p for c, p in sp.items() if not c.startswith('1-')}
    ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:N]
    if not ranked:
        return []
    total_p = sum(p for _, p in ranked)
    if total_p <= 0:
        return []
    bets = []
    for combo, prob in ranked:
        amount = int(round(total_bet * prob / total_p / 100) * 100)
        if amount >= 100:
            bets.append({'combo': combo, 'amount': amount})
    return bets


def strategy_entropy_filter(race, p1, p2, p3,
                             N=5, total_bet=500,
                             min_entropy=1.2, **kw):
    """エントロピーフィルター: 1着確率の不確実性が高いレースに投資

    エントロピーが高い = 混戦 = 穴が出やすい
    """
    # 1着確率のエントロピー
    entropy = -sum(p * np.log(p + 1e-10) for p in p1)
    if entropy < min_entropy:
        return []  # 確実なレースはスキップ

    sp = calc_sanrentan_probs(p1, p2, p3)
    ranked = sorted(sp.items(), key=lambda x: x[1], reverse=True)[:N]
    if not ranked:
        return []
    total_p = sum(p for _, p in ranked)
    if total_p <= 0:
        return []
    bets = []
    for combo, prob in ranked:
        amount = int(round(total_bet * prob / total_p / 100) * 100)
        if amount >= 100:
            bets.append({'combo': combo, 'amount': amount})
    return bets


# ====================================================================
#  バックテスト実行エンジン
# ====================================================================

def run_backtest(valid_races, p1_all, p2_all, p3_all, strategy_fn,
                 strategy_name='', **strategy_kw):
    """汎用バックテスト実行"""
    total_bet = 0
    total_payout = 0
    hits = 0
    bets_count = 0
    races_with_bets = 0

    for idx, race in enumerate(valid_races):
        actual = f"{race['result_1st']}-{race['result_2nd']}-{race['result_3rd']}"
        actual_payout = race['payout_sanrentan']

        bets = strategy_fn(
            race, p1_all[idx], p2_all[idx], p3_all[idx],
            **strategy_kw
        )

        if not bets:
            continue

        races_with_bets += 1

        for b in bets:
            total_bet += b['amount']
            bets_count += 1
            if b['combo'] == actual:
                payout = int(b['amount'] / 100 * actual_payout)
                total_payout += payout
                hits += 1

    roi = total_payout / total_bet if total_bet > 0 else 0
    hit_rate = hits / bets_count * 100 if bets_count > 0 else 0
    skip_rate = 1 - races_with_bets / len(valid_races)

    return {
        'name': strategy_name,
        'total_races': len(valid_races),
        'races_bet': races_with_bets,
        'bets': bets_count,
        'hits': hits,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_payout': total_payout,
        'roi': roi,
        'profit': total_payout - total_bet,
        'skip_rate': skip_rate,
    }


def print_results(results_list):
    """全戦略の比較テーブル"""
    logger.info("=" * 100)
    logger.info("  バックテスト全戦略比較レポート")
    logger.info("=" * 100)

    header = (f"  {'戦略':<28} {'ベットR':>7} {'点数':>7} {'的中':>5} "
              f"{'的中率':>7} {'投資':>12} {'回収':>12} {'ROI':>7} {'損益':>14} {'見送':>6}")
    logger.info(header)
    logger.info("-" * 100)

    for r in results_list:
        line = (
            f"  {r['name']:<28} {r['races_bet']:>7,} {r['bets']:>7,} "
            f"{r['hits']:>5,} {r['hit_rate']:>6.2f}% "
            f"{r['total_bet']:>12,} {r['total_payout']:>12,} "
            f"{r['roi']*100:>6.1f}% {r['profit']:>+13,} "
            f"{r['skip_rate']*100:>5.1f}%"
        )
        logger.info(line)

    logger.info("=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='全戦略バックテスト')
    parser.add_argument('--bankroll', type=int, default=100000)
    args = parser.parse_args()

    model = load_model('models/boatrace_model.pth')
    model.eval()
    device = torch.device('cpu')

    races, boats_by_race = load_all_data()
    valid_races, p1, p2, p3 = batch_predict(races, boats_by_race, model, device)

    results = []

    # === 戦略群 ===

    # 1. 旧戦略: argmax 1点買い
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_old_argmax, '旧: argmax 1点 100円')
    results.append(r)

    # 2. Top-3 均等
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_topN_equal, 'Top-3 均等 100円×3',
                     N=3, bet_per_point=100)
    results.append(r)

    # 3. Top-5 均等
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_topN_equal, 'Top-5 均等 100円×5',
                     N=5, bet_per_point=100)
    results.append(r)

    # 4. Top-10 均等
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_topN_equal, 'Top-10 均等 100円×10',
                     N=10, bet_per_point=100)
    results.append(r)

    # 5. Top-5 確率比例 500円
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_topN_weighted, 'Top-5 比例配分 500円',
                     N=5, total_bet=500)
    results.append(r)

    # 6. 荒れレース狙い (1号艇<45%) Top-5
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_upset_hunter, '荒れ狙い(B1<45%) Top-5',
                     N=5, total_bet=500, max_boat1_prob=0.45)
    results.append(r)

    # 7. 荒れレース狙い (1号艇<35%) Top-5
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_upset_hunter, '荒れ狙い(B1<35%) Top-5',
                     N=5, total_bet=500, max_boat1_prob=0.35)
    results.append(r)

    # 8. 穴目フォーカス（1号艇頭除外）Top-5
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_non_favorite_focus, '穴目(1号頭除外) Top-5',
                     N=5, total_bet=500)
    results.append(r)

    # 9. エントロピーフィルター (>1.2) Top-5
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_entropy_filter, 'エントロピー(>1.2) Top-5',
                     N=5, total_bet=500, min_entropy=1.2)
    results.append(r)

    # 10. エントロピーフィルター (>1.5) Top-5
    r = run_backtest(valid_races, p1, p2, p3,
                     strategy_entropy_filter, 'エントロピー(>1.5) Top-5',
                     N=5, total_bet=500, min_entropy=1.5)
    results.append(r)

    # === 結果出力 ===
    print_results(results)

    # === パラメータ感度分析: 荒れレース閾値 ===
    logger.info("\n=== 感度分析: 1号艇確率フィルター ===")
    logger.info(f"  {'B1閾値':>8} {'対象R':>7} {'点数':>7} {'的中':>5} "
                f"{'投資':>12} {'回収':>12} {'ROI':>7} {'損益':>14}")
    logger.info("-" * 80)
    for thresh in [1.0, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
        r = run_backtest(valid_races, p1, p2, p3,
                         strategy_upset_hunter, '',
                         N=5, total_bet=500, max_boat1_prob=thresh)
        logger.info(
            f"  {thresh:>7.2f} {r['races_bet']:>7,} {r['bets']:>7,} "
            f"{r['hits']:>5,} {r['total_bet']:>12,} {r['total_payout']:>12,} "
            f"{r['roi']*100:>6.1f}% {r['profit']:>+13,}"
        )

    # === パラメータ感度分析: Top-N ===
    logger.info("\n=== 感度分析: Top-N (全レース) ===")
    logger.info(f"  {'N':>4} {'点数':>8} {'的中':>5} "
                f"{'投資':>12} {'回収':>12} {'ROI':>7} {'損益':>14}")
    logger.info("-" * 65)
    for n in [1, 2, 3, 5, 10, 15, 20, 30]:
        r = run_backtest(valid_races, p1, p2, p3,
                         strategy_topN_equal, '',
                         N=n, bet_per_point=100)
        logger.info(
            f"  {n:>4} {r['bets']:>8,} {r['hits']:>5,} "
            f"{r['total_bet']:>12,} {r['total_payout']:>12,} "
            f"{r['roi']*100:>6.1f}% {r['profit']:>+13,}"
        )


if __name__ == '__main__':
    main()
