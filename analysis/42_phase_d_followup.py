"""Phase D' 後続研究 (5 分析、cache 流用)

1. R 番号別 ROI 推移 (R4 不調仮説)
2. 会場別 ROI (大村追加・除外会場)
3. max_recommended_bets 変動 (1, 2, 3, 5)
4. 外し時 actual payout 分布 (万舟券パターン)
5. 戦略間日次 PnL 相関 + アンサンブル + 後知恵 oracle

入力: analysis/phase_d_cache.pkl (Task 1 で生成済み)
出力: analysis/reports/phase_d_followup.md
"""
import os
import sys
import copy
import json
import pickle
import logging
from datetime import date
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import pandas as pd

from src.betting import KellyBettingStrategy

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
m40 = import_module('40_phase_d_backtest')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'phase_d_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'phase_d_followup.md'
INITIAL_BANKROLL = 200000

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川', 6: '浜名湖',
    7: '蒲郡', 8: '常滑', 9: '津', 10: '三国', 11: 'びわこ', 12: '住之江',
    13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島', 17: '宮島', 18: '徳山',
    19: '下関', 20: '若松', 21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}


def run_strategy(cache, strategy_name, kb, config_override=None):
    """戦略 config を kb から取得し、override 適用 + cache を backtest"""
    base = copy.deepcopy(kb.config['strategies'][strategy_name])
    if config_override:
        base.update(config_override)
    bankroll = INITIAL_BANKROLL
    bets_log = []
    for c in cache:
        ok, _ = m40.race_level_filter(c['probs_1st'], c['venue_id'], c['race_number'], base)
        if not ok:
            continue
        try:
            bets = kb._strategy_kelly(
                config=base, sanrentan_probs=c['mc_probs'],
                odds_data=c['odds_data'], bankroll=bankroll,
                strategy_name=strategy_name,
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
                'race_id': c['race_id'],
                'race_date': c['race_date'],
                'venue_id': c['venue_id'],
                'race_number': c['race_number'],
                'combo': combo,
                'stake': stake,
                'is_hit': is_hit,
                'actual_return': actual_return,
                'pnl': pnl,
                'actual_payout': payout,
            })
            bankroll += pnl
    return bets_log


def metrics(bets_log):
    if not bets_log:
        return {'n_bets': 0}
    df = pd.DataFrame(bets_log)
    n_hit = int(df['is_hit'].sum())
    total_stake = int(df['stake'].sum())
    total_payout = int(df['actual_return'].sum())
    total_pnl = int(df['pnl'].sum())
    roi = total_payout / total_stake * 100 if total_stake else 0
    daily = df.groupby('race_date')['pnl'].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    cum = df['pnl'].cumsum()
    mdd = int((cum.cummax() - cum).max())
    return {
        'n_bets': len(df), 'n_hit': n_hit, 'hit_rate': n_hit/len(df)*100,
        'total_stake': total_stake, 'total_payout': total_payout,
        'total_pnl': total_pnl, 'roi': float(roi),
        'sharpe_daily': float(sharpe), 'mdd': mdd,
    }


def analysis1_race_number(p6_bets):
    """R 番号別 ROI + 月内推移"""
    df = pd.DataFrame(p6_bets)
    by_r = df.groupby('race_number').apply(lambda g: pd.Series({
        'bets': len(g),
        'hit_rate': g['is_hit'].mean() * 100,
        'roi': g['actual_return'].sum() / g['stake'].sum() * 100 if g['stake'].sum() else 0,
        'pnl': g['pnl'].sum(),
    })).reset_index()
    # 上下旬比較 (4 月)
    df['half'] = df['race_date'].apply(lambda d: 'first' if d.day <= 15 else 'second')
    by_r_half = df.groupby(['race_number', 'half']).apply(lambda g: pd.Series({
        'bets': len(g), 'roi': g['actual_return'].sum() / g['stake'].sum() * 100 if g['stake'].sum() else 0,
    })).reset_index()
    return by_r, by_r_half


def analysis2_venue(cache, kb):
    """会場別 ROI: include_venues=None (全 24 会場) で P6 を回す"""
    bets = run_strategy(cache, 'mc3_venue_focus_r4', kb, config_override={'include_venues': []})
    if not bets:
        return None, None
    df = pd.DataFrame(bets)
    by_v = df.groupby('venue_id').apply(lambda g: pd.Series({
        'bets': len(g),
        'hit_rate': g['is_hit'].mean() * 100,
        'roi': g['actual_return'].sum() / g['stake'].sum() * 100 if g['stake'].sum() else 0,
        'pnl': g['pnl'].sum(),
    })).reset_index()
    by_v['venue_name'] = by_v['venue_id'].map(VENUE_NAMES)
    by_v = by_v.sort_values('roi', ascending=False)
    current_include = [2, 4, 5, 6, 9, 10, 12, 13, 17, 23]
    by_v['in_p6_include'] = by_v['venue_id'].isin(current_include)
    return by_v, df


def analysis3_max_bets(cache, kb):
    """max_recommended_bets を 1, 2, 3, 5 で比較"""
    results = []
    for max_bets in [1, 2, 3, 5]:
        bets = run_strategy(cache, 'mc3_venue_focus_r4', kb,
                            config_override={'max_recommended_bets': max_bets})
        m = metrics(bets)
        results.append({'max_bets': max_bets, **m})
    return results


def analysis4_miss_payout(p6_bets):
    """外し時の actual payout 分布"""
    df = pd.DataFrame(p6_bets)
    # bets 単位ではなく race 単位 (同じ race の bets を集約)
    by_race = df.groupby('race_id').agg({
        'is_hit': 'max',  # 1bets でも当たれば hit
        'actual_payout': 'first',
        'stake': 'sum',
        'pnl': 'sum',
    }).reset_index()
    missed = by_race[by_race['is_hit'] == False]
    hit = by_race[by_race['is_hit'] == True]
    return {
        'n_races': len(by_race),
        'n_hit_races': int((by_race['is_hit'] == True).sum()),
        'n_missed_races': len(missed),
        'missed_payout_mean': float(missed['actual_payout'].mean()) if len(missed) else 0,
        'missed_payout_median': float(missed['actual_payout'].median()) if len(missed) else 0,
        'missed_manshu': int((missed['actual_payout'] >= 10000).sum()),  # 万舟 = 1万円以上
        'missed_manshu_rate': float((missed['actual_payout'] >= 10000).mean() * 100) if len(missed) else 0,
        'all_manshu_rate': float((by_race['actual_payout'] >= 10000).mean() * 100),
        'hit_payout_mean': float(hit['actual_payout'].mean()) if len(hit) else 0,
        'hit_payout_median': float(hit['actual_payout'].median()) if len(hit) else 0,
    }


def analysis5_strategy_correlation(cache, kb, strategies):
    """戦略間日次 PnL 相関 + アンサンブル + oracle"""
    all_bets = {}
    for s in strategies:
        if s not in kb.config['strategies']:
            logger.warning(f'戦略 {s} 不在、スキップ')
            continue
        bets = run_strategy(cache, s, kb)
        all_bets[s] = bets
        logger.info(f'  戦略 {s}: bets={len(bets)} ROI={metrics(bets).get("roi", 0):.1f}%')
    # 日次 PnL 行列
    daily_pnl = {}
    all_dates = set()
    for s, bets in all_bets.items():
        if not bets:
            daily_pnl[s] = {}
            continue
        df = pd.DataFrame(bets)
        daily = df.groupby('race_date')['pnl'].sum()
        daily_pnl[s] = daily.to_dict()
        all_dates.update(daily.index)
    all_dates = sorted(all_dates)
    # date × strategy DataFrame
    matrix = pd.DataFrame(index=all_dates, columns=list(all_bets.keys()))
    for s, dmap in daily_pnl.items():
        for d in all_dates:
            matrix.loc[d, s] = dmap.get(d, 0)
    matrix = matrix.fillna(0).astype(float)
    # 相関
    correlation = matrix.corr()
    # 均等配分アンサンブル
    matrix['ensemble_equal'] = matrix.mean(axis=1)
    # 後知恵 oracle (毎日のベスト戦略)
    matrix['oracle_best'] = matrix[list(all_bets.keys())].max(axis=1)
    # 簡易ルール: 前日ベスト戦略を選ぶ
    prev_best = None
    rule_pnl = []
    for d in all_dates:
        if prev_best is None or prev_best not in all_bets:
            # 初日 or NA: 均等配分
            rule_pnl.append(matrix.loc[d, list(all_bets.keys())].mean())
        else:
            rule_pnl.append(matrix.loc[d, prev_best])
        # 当日のベスト戦略を記録
        today_pnl = matrix.loc[d, list(all_bets.keys())]
        prev_best = today_pnl.idxmax() if today_pnl.max() > 0 else prev_best
    matrix['rule_prev_day'] = rule_pnl

    # サマリ
    summary = {}
    for col in matrix.columns:
        total = matrix[col].sum()
        days_pos = int((matrix[col] > 0).sum())
        days_neg = int((matrix[col] < 0).sum())
        summary[col] = {
            'total_pnl': int(total),
            'days_pos': days_pos,
            'days_neg': days_neg,
            'mean_daily': float(matrix[col].mean()),
            'std_daily': float(matrix[col].std()),
            'sharpe': float(matrix[col].mean() / matrix[col].std()) if matrix[col].std() > 0 else 0,
        }
    return {
        'all_bets': all_bets,
        'correlation': correlation,
        'matrix': matrix,
        'summary': summary,
    }


def write_report(p6_bets, p6_metrics, ana1, ana2, ana3, ana4, ana5):
    lines = []
    lines.append("# Phase D' 後続研究レポート (5 分析)\n\n")
    lines.append(f"期間: 2026-04-01 〜 2026-04-30 (4320 races cache)\n")
    lines.append(f"P6 baseline: bets={p6_metrics['n_bets']} ROI={p6_metrics['roi']:.1f}% PnL=¥{p6_metrics['total_pnl']:+,}\n\n")

    # 1. R 番号別
    lines.append("## 1. R 番号別 ROI 推移\n\n")
    by_r, by_r_half = ana1
    lines.append("### 月通算\n\n| R | bets | hit_rate | ROI | PnL |\n|---|---|---|---|---|\n")
    for _, row in by_r.iterrows():
        lines.append(f"| R{int(row['race_number'])} | {int(row['bets'])} | {row['hit_rate']:.1f}% | {row['roi']:.1f}% | ¥{int(row['pnl']):+,} |\n")
    lines.append("\n### 上下旬比較 (4 月)\n\n| R | half | bets | ROI |\n|---|---|---|---|\n")
    for _, row in by_r_half.iterrows():
        lines.append(f"| R{int(row['race_number'])} | {row['half']} | {int(row['bets'])} | {row['roi']:.1f}% |\n")

    # 2. 会場別
    lines.append("\n## 2. 会場別 ROI (P6 + include_venues 制限解除)\n\n")
    by_v, _ = ana2
    if by_v is not None:
        lines.append("| 会場 | bets | hit | ROI | PnL | P6現含 |\n|---|---|---|---|---|---|\n")
        for _, row in by_v.iterrows():
            mark = '✓' if row['in_p6_include'] else ''
            lines.append(f"| V{int(row['venue_id'])} {row['venue_name']} | {int(row['bets'])} | {row['hit_rate']:.1f}% | "
                         f"{row['roi']:.1f}% | ¥{int(row['pnl']):+,} | {mark} |\n")
        lines.append("\n**追加候補 (現非含 × ROI≥150% × bets≥5):**\n\n")
        cand = by_v[(~by_v['in_p6_include']) & (by_v['roi'] >= 150) & (by_v['bets'] >= 5)]
        if len(cand) == 0:
            lines.append("該当なし\n")
        else:
            for _, row in cand.iterrows():
                lines.append(f"- V{int(row['venue_id'])} {row['venue_name']}: ROI {row['roi']:.1f}% (bets {int(row['bets'])})\n")
        lines.append("\n**削除候補 (現含 × ROI<100% × bets≥5):**\n\n")
        rem = by_v[(by_v['in_p6_include']) & (by_v['roi'] < 100) & (by_v['bets'] >= 5)]
        if len(rem) == 0:
            lines.append("該当なし\n")
        else:
            for _, row in rem.iterrows():
                lines.append(f"- V{int(row['venue_id'])} {row['venue_name']}: ROI {row['roi']:.1f}% (bets {int(row['bets'])})\n")

    # 3. max_bets
    lines.append("\n## 3. max_recommended_bets 変動\n\n")
    lines.append("| max_bets | bets | hit_rate | ROI | PnL | Sharpe | MDD |\n|---|---|---|---|---|---|---|\n")
    for r in ana3:
        if r['n_bets'] == 0:
            lines.append(f"| {r['max_bets']} | 0 | - | - | - | - | - |\n")
        else:
            lines.append(f"| {r['max_bets']} | {r['n_bets']} | {r['hit_rate']:.1f}% | "
                         f"{r['roi']:.1f}% | ¥{r['total_pnl']:+,} | {r['sharpe_daily']:.3f} | ¥{r['mdd']:,} |\n")

    # 4. 外し時 payout
    lines.append("\n## 4. 外し時 actual payout 分布\n\n")
    a = ana4
    lines.append(f"- 全 race: {a['n_races']} (hit {a['n_hit_races']}, missed {a['n_missed_races']})\n")
    lines.append(f"- 全 race 万舟率 (payout≥¥10,000): {a['all_manshu_rate']:.1f}%\n")
    lines.append(f"- **missed race の 万舟率**: **{a['missed_manshu_rate']:.1f}%** ({a['missed_manshu']} / {a['n_missed_races']})\n")
    lines.append(f"- missed race の actual payout 平均: ¥{a['missed_payout_mean']:,.0f}、中央値: ¥{a['missed_payout_median']:,.0f}\n")
    lines.append(f"- hit race の actual payout 平均: ¥{a['hit_payout_mean']:,.0f}、中央値: ¥{a['hit_payout_median']:,.0f}\n\n")
    if a['missed_manshu_rate'] > a['all_manshu_rate'] + 5:
        lines.append("→ **missed race の方が万舟率が顕著に高い**。本命狙い戦略が万舟券パターンで崩れる傾向あり。\n")
    elif a['missed_manshu_rate'] > a['all_manshu_rate']:
        lines.append("→ missed の万舟率がやや高い (差 +α pt) が顕著ではない。\n")
    else:
        lines.append("→ missed の万舟率は全体と同等以下。仮説 (本命崩れ時に万舟) は反証。\n")

    # 5. 戦略間相関
    lines.append("\n## 5. 戦略間日次 PnL 相関 + アンサンブル + Oracle\n\n")
    lines.append("### 戦略別パフォーマンス\n\n| 戦略 | total_pnl | days+ | days- | sharpe |\n|---|---|---|---|---|\n")
    for s, m in ana5['summary'].items():
        lines.append(f"| {s} | ¥{m['total_pnl']:+,} | {m['days_pos']} | {m['days_neg']} | {m['sharpe']:.3f} |\n")
    lines.append("\n### 日次 PnL 相関行列\n\n")
    corr = ana5['correlation']
    cols = list(corr.columns)
    header = "| 戦略 | " + " | ".join(cols) + " |\n"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|\n"
    lines.append(header)
    lines.append(sep)
    for s in cols:
        row_vals = " | ".join(f"{corr.loc[s, c]:.2f}" for c in cols)
        lines.append(f"| {s} | {row_vals} |\n")
    lines.append("\n### アンサンブル試算\n\n")
    matrix = ana5['matrix']
    # ROI 等は元戦略の bets 数集計が必要 (matrix は日次 PnL のみ)。簡易には matrix から相対 PnL のみ
    for col in ['ensemble_equal', 'oracle_best', 'rule_prev_day']:
        m = ana5['summary'][col]
        lines.append(f"- **{col}**: total ¥{m['total_pnl']:+,} (days+ {m['days_pos']}, days- {m['days_neg']}, sharpe {m['sharpe']:.3f})\n")
    # 解釈
    other_strategies = [c for c in cols if c not in ('ensemble_equal', 'oracle_best', 'rule_prev_day')]
    best_single = max(other_strategies, key=lambda s: ana5['summary'][s]['total_pnl'])
    best_single_pnl = ana5['summary'][best_single]['total_pnl']
    oracle_pnl = ana5['summary']['oracle_best']['total_pnl']
    ensemble_pnl = ana5['summary']['ensemble_equal']['total_pnl']
    rule_pnl = ana5['summary']['rule_prev_day']['total_pnl']
    lines.append(f"\n### 解釈\n\n")
    lines.append(f"- 単独最強: **{best_single}** ¥{best_single_pnl:+,}\n")
    lines.append(f"- 均等アンサンブル: ¥{ensemble_pnl:+,} (差 {ensemble_pnl - best_single_pnl:+,})\n")
    lines.append(f"- 後知恵 oracle: ¥{oracle_pnl:+,} (差 {oracle_pnl - best_single_pnl:+,}) — 理論上限、実現不可\n")
    lines.append(f"- 前日ルール: ¥{rule_pnl:+,} (差 {rule_pnl - best_single_pnl:+,})\n\n")
    # 相関平均 (対角除く)
    mean_corr = (corr.sum().sum() - corr.values.trace()) / (corr.size - len(corr))
    lines.append(f"- 戦略間平均相関 (対角除): **{mean_corr:.2f}**\n")
    if mean_corr > 0.7:
        lines.append("- → 相関高、明暗の差はノイズ。**アンサンブルしても変わらず、P6 単独で正解**\n")
    elif mean_corr > 0.4:
        lines.append("- → 中程度相関、ある程度の補完性あり。均等アンサンブルで分散低下狙えるが ROI 上振れは限定的\n")
    else:
        lines.append("- → 相関低、戦略は真に補完的。**アンサンブル採用候補**\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


def main():
    if not CACHE_PATH.exists():
        raise SystemExit(f"cache 不在: {CACHE_PATH}")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f'cache 読込: {len(cache)} レース')

    kb = KellyBettingStrategy(initial_bankroll=INITIAL_BANKROLL)

    logger.info("[1/5] P6 baseline + R番号別")
    p6_bets = run_strategy(cache, 'mc3_venue_focus_r4', kb)
    p6_metrics = metrics(p6_bets)
    logger.info(f"P6: {p6_metrics}")
    ana1 = analysis1_race_number(p6_bets)

    logger.info("[2/5] 会場別 (include_venues 解除)")
    ana2 = analysis2_venue(cache, kb)

    logger.info("[3/5] max_recommended_bets 変動")
    ana3 = analysis3_max_bets(cache, kb)

    logger.info("[4/5] 外し時 payout")
    ana4 = analysis4_miss_payout(p6_bets)

    logger.info("[5/5] 戦略間相関 + アンサンブル")
    strategies = ['mc3_venue_focus_r4', 'mc3_early_race', 'mc_early_race_filtered', 'mc3_venue_focus_r2']
    ana5 = analysis5_strategy_correlation(cache, kb, strategies)

    write_report(p6_bets, p6_metrics, ana1, ana2, ana3, ana4, ana5)
    logger.info("=== 完了 ===")


if __name__ == '__main__':
    main()
