"""本番 bets テーブル regime 別多角分析

git log で抽出した config 変更日を regime 境界とし、各戦略を fair に評価。

Regime 境界 (git log config/betting_config.json):
  pre_mcq         : ~ 2026-03-30
  mcq_only        : 2026-03-31 - 2026-04-06 (MCQ 単独)
  mc8             : 2026-04-07 - 2026-04-09
  mc10_qmc        : 2026-04-10 - 2026-04-18
  max_odds_40     : 2026-04-19 - 2026-04-27
  v10_2_shadow    : 2026-04-23 - 2026-04-28 (V10.2 並走、4/28 撤退)
  of_active       : 2026-04-28 - 2026-04-30
  p_whitelist_v2  : 2026-04-29 - 2026-04-30
  p456_current    : 2026-05-01 - 現在 (P4/P5/P6 追加、本番運用中)

入力: 本番 DB の bets + races テーブル (READ-ONLY)
出力: analysis/reports/bets_regime.md
"""
import os
import sys
import logging
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'bets_regime.md'

REGIMES = [
    ('pre_mcq', None, date(2026, 3, 30)),
    ('mcq_only', date(2026, 3, 31), date(2026, 4, 6)),
    ('mc8', date(2026, 4, 7), date(2026, 4, 9)),
    ('mc10_qmc', date(2026, 4, 10), date(2026, 4, 18)),
    ('max_odds_40', date(2026, 4, 19), date(2026, 4, 27)),
    ('of_active', date(2026, 4, 28), date(2026, 4, 30)),
    ('p456_current', date(2026, 5, 1), None),
]

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川', 6: '浜名湖',
    7: '蒲郡', 8: '常滑', 9: '津', 10: '三国', 11: 'びわこ', 12: '住之江',
    13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島', 17: '宮島', 18: '徳山',
    19: '下関', 20: '若松', 21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}


def assign_regime(d):
    for name, start, end in REGIMES:
        s_ok = start is None or d >= start
        e_ok = end is None or d <= end
        if s_ok and e_ok:
            return name
    return 'unknown'


def fetch_bets():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT b.id, b.race_id, b.strategy_type, b.combination,
               b.amount, b.odds, b.expected_value, b.payout, b.is_hit,
               b.return_amount, b.closing_odds, b.clv,
               b.created_at,
               r.venue_id, r.race_number, r.race_date, r.payout_sanrentan
        FROM bets b
        JOIN races r ON b.race_id = r.id
        ORDER BY r.race_date, b.id
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def compute_metrics(df):
    if len(df) == 0:
        return {'n': 0}
    stake = float(df['amount'].sum())
    payout = float(df['return_amount'].sum())
    hits = int(df['is_hit'].sum())
    return {
        'n': len(df),
        'hits': hits,
        'hit_rate': hits / len(df) * 100,
        'stake': stake,
        'payout': payout,
        'pnl': payout - stake,
        'roi': payout / stake * 100 if stake else 0,
    }


def main():
    logger.info("本番 bets fetch")
    rows = fetch_bets()
    logger.info(f'  bets: {len(rows)} 件')
    df = pd.DataFrame(rows)
    # 型整形
    df['race_date'] = pd.to_datetime(df['race_date']).dt.date
    df['amount'] = df['amount'].astype(float)
    df['return_amount'] = df['return_amount'].fillna(0).astype(float)
    df['payout_sanrentan'] = df['payout_sanrentan'].fillna(0).astype(float)
    df['regime'] = df['race_date'].apply(assign_regime)
    df['venue_name'] = df['venue_id'].map(VENUE_NAMES)

    lines = []
    lines.append("# 本番 bets regime 別多角分析\n\n")
    lines.append(f"生成日時: {pd.Timestamp.now().isoformat()}\n")
    lines.append(f"対象: bets {len(df)} 件 ({df['race_date'].min()} 〜 {df['race_date'].max()})\n\n")

    # 1. regime × 全体
    lines.append("## 1. Regime 別 全体パフォーマンス\n\n")
    lines.append("| regime | 期間 | bets | hit_rate | ROI | PnL |\n|---|---|---|---|---|---|\n")
    for name, start, end in REGIMES:
        sub = df[df['regime'] == name]
        m = compute_metrics(sub)
        period_str = f"{start or '-'} 〜 {end or 'now'}"
        if m['n'] == 0:
            lines.append(f"| {name} | {period_str} | 0 | - | - | - |\n")
        else:
            lines.append(f"| {name} | {period_str} | {m['n']} | {m['hit_rate']:.1f}% | "
                         f"{m['roi']:.1f}% | ¥{m['pnl']:+,.0f} |\n")

    # 2. MCQ 前後比較
    lines.append("\n## 2. MCQ 導入前後 (2026-03-31)\n\n")
    pre = df[df['race_date'] < date(2026, 3, 31)]
    post = df[df['race_date'] >= date(2026, 3, 31)]
    pre_m = compute_metrics(pre)
    post_m = compute_metrics(post)
    lines.append("| 期間 | bets | hit_rate | ROI | PnL |\n|---|---|---|---|---|\n")
    if pre_m['n']:
        lines.append(f"| pre-MCQ (〜3/30) | {pre_m['n']} | {pre_m['hit_rate']:.1f}% | "
                     f"{pre_m['roi']:.1f}% | ¥{pre_m['pnl']:+,.0f} |\n")
    if post_m['n']:
        lines.append(f"| post-MCQ (3/31〜) | {post_m['n']} | {post_m['hit_rate']:.1f}% | "
                     f"{post_m['roi']:.1f}% | ¥{post_m['pnl']:+,.0f} |\n")

    # 3. 直近 2 週 (5/1〜) × 戦略 × 会場 pivot
    lines.append("\n## 3. p456_current 期間 (5/1〜) × 戦略 × 会場 ROI\n\n")
    recent = df[df['race_date'] >= date(2026, 5, 1)]
    if len(recent) == 0:
        lines.append("p456_current 期間 bets なし\n")
    else:
        pv_stake = recent.pivot_table(index='strategy_type', columns='venue_name',
                                       values='amount', aggfunc='sum', fill_value=0)
        pv_pay = recent.pivot_table(index='strategy_type', columns='venue_name',
                                     values='return_amount', aggfunc='sum', fill_value=0)
        pv_n = recent.pivot_table(index='strategy_type', columns='venue_name',
                                   values='id', aggfunc='count', fill_value=0)
        pv_roi = (pv_pay / pv_stake.replace(0, pd.NA) * 100).round(0)
        # 戦略別合計を先に
        s_metrics = recent.groupby('strategy_type').apply(lambda g: pd.Series(compute_metrics(g)))
        s_metrics = s_metrics.sort_values('roi', ascending=False)
        lines.append("### 3-1. 戦略別 全体 (p456_current)\n\n")
        lines.append("| 戦略 | bets | hit | ROI | PnL |\n|---|---|---|---|---|\n")
        for s, row in s_metrics.iterrows():
            if row['n'] == 0:
                continue
            lines.append(f"| {s} | {int(row['n'])} | {row['hit_rate']:.1f}% | "
                         f"{row['roi']:.1f}% | ¥{row['pnl']:+,.0f} |\n")

        lines.append("\n### 3-2. 戦略 × 会場 ROI ヒートマップ (空白=bets無)\n\n")
        # 上位 5 戦略のみ抜粋 (見やすさ)
        top_strategies = s_metrics.head(5).index.tolist()
        cols = sorted(pv_roi.columns)
        lines.append("| 戦略 | " + " | ".join(cols) + " |\n")
        lines.append("|---|" + "|".join(["---"] * len(cols)) + "|\n")
        for s in top_strategies:
            cells = []
            for v in cols:
                try:
                    roi = pv_roi.loc[s, v]
                    n = pv_n.loc[s, v]
                    if pd.isna(roi) or n == 0:
                        cells.append('-')
                    else:
                        cells.append(f"{int(roi)}%({int(n)})")
                except KeyError:
                    cells.append('-')
            lines.append(f"| {s} | " + " | ".join(cells) + " |\n")

    # 4. 戦略別 自己内 期間別 ROI 推移
    lines.append("\n## 4. 戦略別 期間別 ROI (自己内 regime 推移)\n\n")
    main_strategies = df.groupby('strategy_type').size().sort_values(ascending=False).head(15).index.tolist()
    lines.append("| 戦略 | " + " | ".join(r[0] for r in REGIMES) + " |\n")
    lines.append("|---|" + "|".join(["---"] * len(REGIMES)) + "|\n")
    for s in main_strategies:
        cells = []
        for name, _, _ in REGIMES:
            sub = df[(df['strategy_type'] == s) & (df['regime'] == name)]
            m = compute_metrics(sub)
            if m['n'] == 0:
                cells.append('-')
            else:
                cells.append(f"{m['roi']:.0f}%({m['n']})")
        lines.append(f"| {s} | " + " | ".join(cells) + " |\n")

    # 5. CLV 分析 (戦略別)
    lines.append("\n## 5. CLV 分析 (戦略別 平均 CLV、p456_current)\n\n")
    recent_clv = recent[recent['clv'].notna()] if len(recent) else pd.DataFrame()
    if len(recent_clv) == 0:
        # CLV 全期間で
        recent_clv = df[df['clv'].notna()]
        lines.append("(p456_current で CLV 記録なし、全期間で表示)\n\n")
    if len(recent_clv) > 0:
        clv_g = recent_clv.groupby('strategy_type').agg(
            n=('id', 'count'),
            mean_clv=('clv', 'mean'),
            median_clv=('clv', 'median'),
        ).sort_values('mean_clv', ascending=False)
        lines.append("| 戦略 | n | 平均 CLV | 中央値 CLV |\n|---|---|---|---|\n")
        for s, row in clv_g.iterrows():
            if row['n'] < 5:
                continue
            lines.append(f"| {s} | {int(row['n'])} | {row['mean_clv']:+.3f} | {row['median_clv']:+.3f} |\n")
        lines.append("\n*CLV > 0: 締切前 odds より良い条件で約定 (有利)。CLV < 0: 遅延約定で不利。*\n")

    # 6. 配当帯別 hit rate
    lines.append("\n## 6. 配当帯別 hit_rate (戦略別、p456_current)\n\n")
    if len(recent) > 0:
        # actual payout を 4 bin に分割
        recent_c = recent.copy()
        recent_c['payout_bin'] = pd.cut(recent_c['payout_sanrentan'],
                                         bins=[0, 3000, 7000, 15000, 1e9],
                                         labels=['<3k', '3-7k', '7-15k', '15k+'])
        pv = recent_c.pivot_table(index='strategy_type', columns='payout_bin',
                                   values='is_hit', aggfunc='mean', observed=True) * 100
        pv_n = recent_c.pivot_table(index='strategy_type', columns='payout_bin',
                                     values='id', aggfunc='count', observed=True)
        for s in main_strategies:
            if s not in pv.index:
                continue
            sub = recent_c[recent_c['strategy_type'] == s]
            if len(sub) < 5:
                continue
            lines.append(f"\n### {s}\n\n| 配当帯 | bets | hit_rate |\n|---|---|---|\n")
            for b in pv.columns:
                try:
                    rate = pv.loc[s, b]
                    n = pv_n.loc[s, b]
                    if pd.isna(rate):
                        continue
                    lines.append(f"| {b} | {int(n)} | {rate:.1f}% |\n")
                except KeyError:
                    continue

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
