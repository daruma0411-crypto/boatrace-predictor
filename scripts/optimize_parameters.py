"""パラメータ最適化スクリプト: グリッドサーチによる最適ベッティングパラメータ探索

過去の精算済みベットデータを使い、異なるフィルタ閾値での仮想ROIを総当たり計算。
テストモード (min_ev=0.0) で蓄積された幅広いEV帯のデータを活用する。

データリーク防止: 保存済みの予測確率・オッズのみ使用（未来情報を参照しない）。

使い方:
    DATABASE_URL=xxx python scripts/optimize_parameters.py
    DATABASE_URL=xxx python scripts/optimize_parameters.py --min-bets 20
    DATABASE_URL=xxx python scripts/optimize_parameters.py --strategy conservative
"""
import sys
import os
import argparse
import logging
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# === グリッドサーチ変数 ===
# 実データ分布に基づく探索範囲 (odds: 7.5~77, ev: 0.3~3.3, prob: 0.01~0.07)
GRID = {
    'min_ev':          [0.8, 1.0, 1.05, 1.10, 1.15, 1.20, 1.30],
    'min_odds':        [10.0, 15.0, 20.0, 25.0, 30.0],
    'max_odds':        [30.0, 40.0, 50.0, 60.0, 77.0],
    'min_probability': [0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
}


def load_settled_bets(strategy=None):
    """精算済みベットをDB一括取得

    Returns:
        list of dict: [{odds, expected_value, amount, payout, result, strategy_type}, ...]
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        query = """
            SELECT odds, expected_value, amount, payout, result, strategy_type
            FROM bets
            WHERE result IS NOT NULL
              AND odds > 0
              AND expected_value IS NOT NULL
        """
        params = []
        if strategy:
            query += " AND strategy_type = %s"
            params.append(strategy)

        cur.execute(query, params or None)
        rows = cur.fetchall()

    bets = []
    for r in rows:
        odds = float(r['odds'])
        ev = float(r['expected_value'])
        prob = ev / odds if odds > 0 else 0
        bets.append({
            'odds': odds,
            'ev': ev,
            'prob': prob,
            'amount': int(r['amount']),
            'payout': int(r['payout'] or 0),
            'hit': r['result'] == 'win',
            'strategy': r['strategy_type'],
        })
    return bets


def simulate(bets, min_ev, min_odds, max_odds, min_prob):
    """指定パラメータでフィルタし、仮想ベット成績を計算"""
    total_amount = 0
    total_payout = 0
    hits = 0
    count = 0

    for b in bets:
        if b['ev'] < min_ev:
            continue
        if b['odds'] < min_odds:
            continue
        if b['odds'] > max_odds:
            continue
        if b['prob'] < min_prob:
            continue

        count += 1
        total_amount += b['amount']
        total_payout += b['payout']
        if b['hit']:
            hits += 1

    if count == 0:
        return None

    roi = (total_payout - total_amount) / total_amount * 100 if total_amount > 0 else 0
    hit_rate = hits / count * 100

    return {
        'count': count,
        'total_amount': total_amount,
        'total_payout': total_payout,
        'roi': roi,
        'hit_rate': hit_rate,
        'hits': hits,
    }


def grid_search(bets, min_bets=10):
    """全パラメータ組み合わせを探索"""
    keys = list(GRID.keys())
    values = list(GRID.values())
    total_combos = 1
    for v in values:
        total_combos *= len(v)

    logger.info(f"グリッドサーチ: {total_combos}通り探索")

    results = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        result = simulate(
            bets,
            min_ev=params['min_ev'],
            min_odds=params['min_odds'],
            max_odds=params['max_odds'],
            min_prob=params['min_probability'],
        )
        if result and result['count'] >= min_bets:
            results.append({**params, **result})

    results.sort(key=lambda x: x['roi'], reverse=True)
    return results


def print_results(results, top_n=10):
    """トップN結果を表示"""
    print("\n" + "=" * 100)
    print(f"{'Rank':>4} | {'min_ev':>6} {'min_odds':>8} {'max_odds':>8} {'min_prob':>8} | "
          f"{'Bets':>5} {'Hits':>4} {'HitRate':>7} | "
          f"{'Wagered':>10} {'Payout':>10} {'ROI':>8}")
    print("-" * 100)

    for i, r in enumerate(results[:top_n], 1):
        print(f"{i:>4} | {r['min_ev']:>6.2f} {r['min_odds']:>8.1f} {r['max_odds']:>8.1f} "
              f"{r['min_probability']:>8.3f} | "
              f"{r['count']:>5} {r['hits']:>4} {r['hit_rate']:>6.1f}% | "
              f"¥{r['total_amount']:>9,} ¥{r['total_payout']:>9,} {r['roi']:>+7.1f}%")

    print("=" * 100)

    if results:
        best = results[0]
        print(f"\n最適パラメータ:")
        print(f"  min_ev         = {best['min_ev']:.2f}")
        print(f"  min_odds       = {best['min_odds']:.1f}")
        print(f"  max_odds       = {best['max_odds']:.1f}")
        print(f"  min_probability = {best['min_probability']:.3f}")
        print(f"  → ROI: {best['roi']:+.1f}% ({best['count']}件, 的中率{best['hit_rate']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='ベッティングパラメータ最適化 (グリッドサーチ)')
    parser.add_argument('--strategy', help='特定戦略のみ (例: conservative, standard)')
    parser.add_argument('--min-bets', type=int, default=10, help='最低ベット件数 (まぐれ排除)')
    parser.add_argument('--top', type=int, default=10, help='表示するトップN')
    args = parser.parse_args()

    logger.info("=== パラメータ最適化 (Grid Search) ===")

    bets = load_settled_bets(strategy=args.strategy)
    logger.info(f"精算済みベット: {len(bets):,}件"
                + (f" (戦略: {args.strategy})" if args.strategy else " (全戦略)"))

    if not bets:
        logger.error("精算済みベットがありません。")
        return

    # 現状サマリ
    total_a = sum(b['amount'] for b in bets)
    total_p = sum(b['payout'] for b in bets)
    total_h = sum(1 for b in bets if b['hit'])
    current_roi = (total_p - total_a) / total_a * 100 if total_a else 0
    logger.info(f"現状: {len(bets)}件, 投資¥{total_a:,}, 回収¥{total_p:,}, "
                f"ROI={current_roi:+.1f}%, 的中率={total_h/len(bets)*100:.1f}%")

    # 戦略別サマリ
    strategies = {}
    for b in bets:
        s = b['strategy']
        if s not in strategies:
            strategies[s] = {'count': 0, 'amount': 0, 'payout': 0, 'hits': 0}
        strategies[s]['count'] += 1
        strategies[s]['amount'] += b['amount']
        strategies[s]['payout'] += b['payout']
        if b['hit']:
            strategies[s]['hits'] += 1
    print("\n戦略別現状:")
    for s, d in sorted(strategies.items()):
        roi = (d['payout'] - d['amount']) / d['amount'] * 100 if d['amount'] else 0
        hr = d['hits'] / d['count'] * 100 if d['count'] else 0
        print(f"  {s:20s}: {d['count']:>5}件 ROI={roi:>+7.1f}% 的中率={hr:.1f}%")

    # グリッドサーチ
    results = grid_search(bets, min_bets=args.min_bets)
    logger.info(f"有効な組み合わせ: {len(results)}通り (ベット件数>={args.min_bets})")

    print_results(results, top_n=args.top)


if __name__ == '__main__':
    main()
