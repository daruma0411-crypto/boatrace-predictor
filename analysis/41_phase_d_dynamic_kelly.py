"""Phase D D2: 動的 kelly_prob_gain backtest

Task 1 のベスト (kelly_fraction, min_expected_value) を固定し、
kelly_prob_gain を entropy 別に変動させて改善幅を確認する。
"""
import os
import sys
import copy
import json
import pickle
import logging
import math
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import pandas as pd

from src.betting import KellyBettingStrategy

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
m40 = import_module('40_phase_d_backtest')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'phase_d_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'phase_d_dynamic.md'
GRID_JSON = ROOT / 'analysis' / 'reports' / 'phase_d_grid.json'

INITIAL_BANKROLL = 200000


def entropy_of_probs(probs):
    """3 連単確率分布のエントロピー (高いほど低確信)"""
    total = sum(probs.values())
    if total <= 0:
        return float('inf')
    e = 0.0
    for p in probs.values():
        if p > 0:
            r = p / total
            e -= r * math.log(r)
    return e


def gain_for_entropy(entropy):
    """entropy → kelly_prob_gain map"""
    if entropy < 1.5:
        return 1.5
    if entropy < 2.0:
        return 1.2
    return 1.0


def evaluate_dynamic(cache, base_config, override, kb):
    base_config_local = copy.deepcopy(base_config)
    base_config_local['kelly_fraction'] = override['kelly_fraction']
    base_config_local['min_expected_value'] = override['min_expected_value']
    bankroll = INITIAL_BANKROLL
    bets_log = []
    gain_counts = {1.0: 0, 1.2: 0, 1.5: 0}
    for c in cache:
        ok, _ = m40.race_level_filter(c['probs_1st'], c['venue_id'], c['race_number'], base_config_local)
        if not ok:
            continue
        # 動的 gain (毎回 deepcopy して config 変更)
        config = copy.deepcopy(base_config_local)
        e = entropy_of_probs(c['mc_probs'])
        config['kelly_prob_gain'] = gain_for_entropy(e)
        gain_counts[config['kelly_prob_gain']] = gain_counts.get(config['kelly_prob_gain'], 0) + 1
        try:
            bets = kb._strategy_kelly(
                config=config, sanrentan_probs=c['mc_probs'],
                odds_data=c['odds_data'], bankroll=bankroll,
                strategy_name='mc3_venue_focus_r4',
                venue_id=c['venue_id'], race_number=c['race_number'],
            )
        except Exception:
            continue
        if not bets:
            continue
        actual = c['actual_trifecta']
        payout = c['payout_sanrentan']
        for bet in bets:
            stake = bet.get('amount', 100)
            combo = bet.get('combination')
            is_hit = (combo == actual)
            actual_return = (stake / 100 * payout) if is_hit else 0
            pnl = actual_return - stake
            bets_log.append({
                'race_id': c['race_id'], 'race_date': c['race_date'],
                'combo': combo, 'stake': stake, 'is_hit': is_hit,
                'actual_return': actual_return, 'pnl': pnl,
            })
            bankroll += pnl
    return m40.summarize(bets_log), gain_counts


def main():
    if not CACHE_PATH.exists():
        raise SystemExit(f"cache 不在: {CACHE_PATH}。先に 40_phase_d_backtest.py を実行してください")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f'cache 読込: {len(cache)} レース')

    if not GRID_JSON.exists():
        raise SystemExit(f"Task 1 結果 JSON 不在: {GRID_JSON}")
    with open(GRID_JSON, 'r', encoding='utf-8') as f:
        grid_results = json.load(f)
    valid = [r for r in grid_results if r['metrics'].get('n_bets', 0) > 0]
    best = max(valid, key=lambda r: r['metrics']['roi'])
    logger.info(f"Task 1 ベスト: {best['label']} ROI {best['metrics']['roi']:.1f}%")

    kb = KellyBettingStrategy(initial_bankroll=INITIAL_BANKROLL)
    base_config = kb.config['strategies']['mc3_venue_focus_r4']

    static = best['metrics']
    logger.info("動的 gain backtest 実行")
    dynamic, gain_counts = evaluate_dynamic(cache, base_config, best['override'], kb)
    logger.info(f"動的 gain 分布: {gain_counts}")
    logger.info(f"動的結果: bets={dynamic.get('n_bets', 0)} ROI={dynamic.get('roi', 0):.1f}%")

    lines = []
    lines.append("# Phase D D2 動的 kelly_prob_gain 結果\n\n")
    lines.append(f"Task 1 ベース: {best['label']}\n")
    lines.append(f"gain rule: entropy<1.5→1.5, 1.5≤<2.0→1.2, ≥2.0→1.0\n\n")
    lines.append(f"## entropy 分布 (cache レース数別)\n\n")
    total = sum(gain_counts.values())
    lines.append("| 条件 | gain | レース数 | % |\n|---|---|---|---|\n")
    lines.append(f"| entropy<1.5 (高確信) | 1.5 | {gain_counts.get(1.5,0)} | {100*gain_counts.get(1.5,0)/total:.1f}% |\n")
    lines.append(f"| 1.5≤entropy<2.0 | 1.2 | {gain_counts.get(1.2,0)} | {100*gain_counts.get(1.2,0)/total:.1f}% |\n")
    lines.append(f"| entropy≥2.0 (通常) | 1.0 | {gain_counts.get(1.0,0)} | {100*gain_counts.get(1.0,0)/total:.1f}% |\n")
    lines.append("\n## 比較\n\n")
    lines.append("| 指標 | 静的 gain=1.0 (Task 1 ベスト) | 動的 gain | 差 |\n|---|---|---|---|\n")
    lines.append(f"| n_bets | {static['n_bets']} | {dynamic.get('n_bets',0)} | {dynamic.get('n_bets',0)-static['n_bets']:+d} |\n")
    if dynamic.get('n_bets', 0) > 0:
        lines.append(f"| hit_rate | {static['hit_rate']:.1f}% | {dynamic['hit_rate']:.1f}% | {dynamic['hit_rate']-static['hit_rate']:+.1f}pt |\n")
        lines.append(f"| ROI | {static['roi']:.1f}% | {dynamic['roi']:.1f}% | {dynamic['roi']-static['roi']:+.1f}pt |\n")
        lines.append(f"| PnL | ¥{static['total_pnl']:+,} | ¥{dynamic['total_pnl']:+,} | ¥{dynamic['total_pnl']-static['total_pnl']:+,} |\n")
        lines.append(f"| Sharpe | {static['sharpe_daily']:.3f} | {dynamic['sharpe_daily']:.3f} | {dynamic['sharpe_daily']-static['sharpe_daily']:+.3f} |\n")
        lines.append(f"| MDD | ¥{static['mdd']:,} | ¥{dynamic['mdd']:,} | ¥{dynamic['mdd']-static['mdd']:+,} |\n")
        diff = dynamic['roi'] - static['roi']
    else:
        diff = -999

    lines.append("\n## 最終提言\n\n")
    if diff > 5:
        verdict = f"✅ 動的 gain 採用、本番反映候補 (差 {diff:+.1f}pt)"
    elif diff > 0:
        verdict = f"🟡 微改善 ({diff:+.1f}pt)。静的 gain で十分、動的化は見送り"
    else:
        verdict = f"❌ 動的 gain は逆効果 ({diff:+.1f}pt)、静的 gain 採用"
    lines.append(f"{verdict}\n\n")
    lines.append(f"### 本番反映候補のパラメータ (mc3_venue_focus_r4)\n\n")
    lines.append(f"- `kelly_fraction`: {best['override']['kelly_fraction']}\n")
    lines.append(f"- `min_expected_value`: {best['override']['min_expected_value']}\n")
    lines.append(f"- `kelly_prob_gain`: {'動的 (entropy 別)' if diff > 5 else '1.0 (静的、現状維持)'}\n\n")
    lines.append("**本番反映は別セッションで実施** (shadow 1-2 週並走後)。\n\n")
    lines.append("## Phase D 総合判定\n\n")
    # Task 1 ベスト = P6 default なら、Phase D 全体としては「現状維持推奨」
    if 'default' in best['label']:
        lines.append("Task 1 で baseline (P6 default) を超える組合せが見つからず、本 Task 2 でも動的化の効果が小さい場合、"
                     "**Phase D 全体としては現状の P6 設定を維持** することを推奨。サードパーティ推奨の Kelly 増額・EV フィルタは "
                     "P6 既存フィルタ (`min_probability=0.005`, `max_odds=80`) との重複により当該データセットでは効果なしと判明。\n")
    else:
        lines.append(f"**ベスト設定**: {best['label']} を本番候補とする。\n")
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
