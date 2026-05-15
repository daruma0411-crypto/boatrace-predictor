"""O3 (mc3_early_race) 徹底解剖

直近 14 日 (p456_current 期間、2026-05-01 以降) で O3 を全角度分析:
  1. 期間 全体パフォーマンス + 累積 PnL 推移
  2. 会場別 ROI (vs P6)
  3. R 番号別 (R1-R4)
  4. 日次パフォーマンス + 連続損益
  5. 配当帯別 hit_rate
  6. CLV 詳細 (約定優位性)
  7. P6 との同日同レース差分 (どのレースで差がついたか)
  8. O3 採用なら期待 PnL (vs 現状 P6 単独)
  9. リスク評価 (Sharpe / MDD / 連敗数)

入力: 本番 DB bets + races (READ-ONLY)
出力: analysis/reports/o3_deep_dive.md
"""
import os
import sys
import logging
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'o3_deep_dive.md'

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川', 6: '浜名湖',
    7: '蒲郡', 8: '常滑', 9: '津', 10: '三国', 11: 'びわこ', 12: '住之江',
    13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島', 17: '宮島', 18: '徳山',
    19: '下関', 20: '若松', 21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

PERIOD_START = date(2026, 5, 1)
PERIOD_END = date(2026, 5, 15)
P6_INCLUDE_VENUES = [2, 4, 5, 6, 9, 10, 12, 13, 17, 23]


def fetch_bets(strategy_list):
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT b.id, b.race_id, b.strategy_type, b.combination,
               b.amount::float AS amount, b.odds::float AS odds,
               b.expected_value::float AS ev,
               b.return_amount, b.is_hit, b.clv, b.created_at,
               r.venue_id, r.race_number, r.race_date, r.payout_sanrentan
        FROM bets b
        JOIN races r ON b.race_id = r.id
        WHERE b.strategy_type = ANY(%s)
          AND r.race_date BETWEEN %s AND %s
        ORDER BY r.race_date, b.id
    """, (strategy_list, PERIOD_START, PERIOD_END))
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows)
    if len(df):
        df['return_amount'] = df['return_amount'].fillna(0).astype(float)
        df['payout_sanrentan'] = df['payout_sanrentan'].fillna(0).astype(float)
        df['pnl'] = df['return_amount'] - df['amount']
        df['venue_name'] = df['venue_id'].map(VENUE_NAMES)
        df['race_date'] = pd.to_datetime(df['race_date']).dt.date
    return df


def metrics(df, kpi_prefix=''):
    if len(df) == 0:
        return {'n': 0}
    stake = df['amount'].sum()
    payout = df['return_amount'].sum()
    hits = int(df['is_hit'].sum())
    daily = df.groupby('race_date')['pnl'].sum()
    sharpe = daily.mean() / daily.std() if daily.std() > 0 else 0
    cum = df['pnl'].cumsum()
    mdd = (cum.cummax() - cum).max()
    # 最大連敗
    hit_seq = df['is_hit'].astype(int).values
    losses = 0
    max_losses = 0
    for h in hit_seq:
        if h == 0:
            losses += 1
            max_losses = max(max_losses, losses)
        else:
            losses = 0
    return {
        'n': len(df), 'hits': hits, 'hit_rate': hits/len(df)*100,
        'stake': float(stake), 'payout': float(payout),
        'pnl': float(payout - stake), 'roi': float(payout/stake*100) if stake else 0,
        'sharpe': float(sharpe), 'mdd': float(mdd),
        'max_losses': int(max_losses),
        'days': int(daily.shape[0]),
    }


def main():
    logger.info(f"O3 徹底解剖: {PERIOD_START} 〜 {PERIOD_END}")
    o3 = fetch_bets(['mc3_early_race'])
    p6 = fetch_bets(['mc3_venue_focus_r4'])
    logger.info(f"O3 bets: {len(o3)}, P6 bets: {len(p6)}")

    lines = []
    lines.append("# O3 (mc3_early_race) 徹底解剖\n\n")
    lines.append(f"対象期間: {PERIOD_START} 〜 {PERIOD_END} (p456_current regime)\n\n")

    # 1. 全体パフォーマンス
    o3m = metrics(o3)
    p6m = metrics(p6)
    lines.append("## 1. O3 vs P6 全体パフォーマンス\n\n")
    lines.append("| 指標 | O3 | P6 (本番) | 差 |\n|---|---|---|---|\n")
    lines.append(f"| 対象日数 | {o3m['days']} | {p6m['days']} | {o3m['days']-p6m['days']:+d} |\n")
    lines.append(f"| bets | {o3m['n']} | {p6m['n']} | {o3m['n']-p6m['n']:+d} |\n")
    lines.append(f"| hit_rate | {o3m['hit_rate']:.1f}% | {p6m['hit_rate']:.1f}% | {o3m['hit_rate']-p6m['hit_rate']:+.1f}pt |\n")
    lines.append(f"| ROI | **{o3m['roi']:.1f}%** | {p6m['roi']:.1f}% | **{o3m['roi']-p6m['roi']:+.1f}pt** |\n")
    lines.append(f"| stake | ¥{o3m['stake']:,.0f} | ¥{p6m['stake']:,.0f} | ¥{o3m['stake']-p6m['stake']:+,.0f} |\n")
    lines.append(f"| PnL | **¥{o3m['pnl']:+,.0f}** | ¥{p6m['pnl']:+,.0f} | **¥{o3m['pnl']-p6m['pnl']:+,.0f}** |\n")
    lines.append(f"| Sharpe | {o3m['sharpe']:.3f} | {p6m['sharpe']:.3f} | {o3m['sharpe']-p6m['sharpe']:+.3f} |\n")
    lines.append(f"| MDD | ¥{o3m['mdd']:,.0f} | ¥{p6m['mdd']:,.0f} | ¥{o3m['mdd']-p6m['mdd']:+,.0f} |\n")
    lines.append(f"| 最大連敗 | {o3m['max_losses']} | {p6m['max_losses']} | {o3m['max_losses']-p6m['max_losses']:+d} |\n")

    # 2. 累積 PnL 推移 (日次)
    lines.append("\n## 2. 累積 PnL 推移 (日次)\n\n")
    o3_daily = o3.groupby('race_date').agg(bets=('id','count'), pnl=('pnl','sum')).reset_index()
    p6_daily = p6.groupby('race_date').agg(bets=('id','count'), pnl=('pnl','sum')).reset_index()
    # マージ
    merged = pd.merge(o3_daily, p6_daily, on='race_date', how='outer',
                      suffixes=('_o3', '_p6')).fillna(0).sort_values('race_date')
    merged['cum_o3'] = merged['pnl_o3'].cumsum()
    merged['cum_p6'] = merged['pnl_p6'].cumsum()
    lines.append("| date | O3 bets | O3 日PnL | O3 累計 | P6 bets | P6 日PnL | P6 累計 | 差 |\n|---|---|---|---|---|---|---|---|\n")
    for _, r in merged.iterrows():
        lines.append(f"| {r['race_date']} | {int(r['bets_o3'])} | ¥{r['pnl_o3']:+,.0f} | ¥{r['cum_o3']:+,.0f} | "
                     f"{int(r['bets_p6'])} | ¥{r['pnl_p6']:+,.0f} | ¥{r['cum_p6']:+,.0f} | ¥{r['cum_o3']-r['cum_p6']:+,.0f} |\n")

    # 3. 会場別 ROI (O3 vs P6 重ね)
    lines.append("\n## 3. 会場別 ROI 詳細 (O3 vs P6)\n\n")
    def venue_agg(d):
        g = d.groupby('venue_id').agg(bets=('id','count'),
                                       stake=('amount','sum'),
                                       payout=('return_amount','sum'),
                                       hits=('is_hit','sum'),
                                       pnl=('pnl','sum'))
        g['roi'] = g['payout'] / g['stake'] * 100
        g['hit_rate'] = g['hits'] / g['bets'] * 100
        return g.reset_index()
    o3v = venue_agg(o3).rename(columns={'bets':'o3_bets','roi':'o3_roi','pnl':'o3_pnl','hit_rate':'o3_hit'})
    p6v = venue_agg(p6).rename(columns={'bets':'p6_bets','roi':'p6_roi','pnl':'p6_pnl','hit_rate':'p6_hit'})
    vall = pd.merge(o3v[['venue_id','o3_bets','o3_hit','o3_roi','o3_pnl']],
                    p6v[['venue_id','p6_bets','p6_hit','p6_roi','p6_pnl']],
                    on='venue_id', how='outer').fillna(0)
    vall['name'] = vall['venue_id'].map(VENUE_NAMES)
    vall['p6_inc'] = vall['venue_id'].isin(P6_INCLUDE_VENUES)
    vall = vall.sort_values('o3_pnl', ascending=False)
    lines.append("| 会場 | P6含 | O3 bets | O3 hit | O3 ROI | O3 PnL | P6 bets | P6 hit | P6 ROI | P6 PnL |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in vall.iterrows():
        if r['o3_bets'] == 0 and r['p6_bets'] == 0:
            continue
        mark = '✓' if r['p6_inc'] else ''
        lines.append(f"| V{int(r['venue_id'])} {r['name']} | {mark} | "
                     f"{int(r['o3_bets'])} | {r['o3_hit']:.1f}% | {r['o3_roi']:.0f}% | ¥{r['o3_pnl']:+,.0f} | "
                     f"{int(r['p6_bets'])} | {r['p6_hit']:.1f}% | {r['p6_roi']:.0f}% | ¥{r['p6_pnl']:+,.0f} |\n")

    # O3 が稼いで P6 が逃した会場
    o3_only = vall[(vall['p6_bets'] == 0) & (vall['o3_pnl'] > 0)]
    lines.append("\n### O3 が稼いで P6 が機会逸失した会場\n\n")
    if len(o3_only) == 0:
        lines.append("該当なし\n")
    else:
        for _, r in o3_only.iterrows():
            lines.append(f"- V{int(r['venue_id'])} {r['name']}: O3 ROI {r['o3_roi']:.0f}% ({int(r['o3_bets'])} bets, ¥{r['o3_pnl']:+,.0f}) — **P6 に未含**\n")

    # 4. R 番号別
    lines.append("\n## 4. R 番号別 (O3 / P6)\n\n")
    lines.append("| R | O3 bets | O3 ROI | O3 PnL | P6 bets | P6 ROI | P6 PnL |\n|---|---|---|---|---|---|---|\n")
    for rn in [1, 2, 3, 4]:
        oo = o3[o3['race_number'] == rn]
        pp = p6[p6['race_number'] == rn]
        om = metrics(oo); pm = metrics(pp)
        lines.append(f"| R{rn} | {om['n']} | {om.get('roi',0):.1f}% | ¥{om.get('pnl',0):+,.0f} | "
                     f"{pm['n']} | {pm.get('roi',0):.1f}% | ¥{pm.get('pnl',0):+,.0f} |\n")

    # 5. 配当帯別 hit_rate
    lines.append("\n## 5. 配当帯別 hit_rate / ROI (O3)\n\n")
    if len(o3):
        o3c = o3.copy()
        o3c['payout_bin'] = pd.cut(o3c['payout_sanrentan'],
                                    bins=[0, 3000, 7000, 15000, 30000, 1e9],
                                    labels=['<3k','3-7k','7-15k','15-30k','30k+'])
        by_b = o3c.groupby('payout_bin', observed=True).agg(
            bets=('id','count'), hits=('is_hit','sum'),
            stake=('amount','sum'), payout=('return_amount','sum'))
        by_b['hit_rate'] = by_b['hits']/by_b['bets']*100
        by_b['roi'] = by_b['payout']/by_b['stake']*100
        lines.append("| 配当帯 | bets | hit_rate | ROI |\n|---|---|---|---|\n")
        for b, row in by_b.iterrows():
            lines.append(f"| {b} | {int(row['bets'])} | {row['hit_rate']:.1f}% | {row['roi']:.1f}% |\n")

    # 6. CLV
    lines.append("\n## 6. CLV (約定優位性、O3 単独)\n\n")
    o3_clv = o3[o3['clv'].notna()]
    if len(o3_clv):
        lines.append(f"- CLV 記録 bets: {len(o3_clv)}\n")
        lines.append(f"- 平均 CLV: **{o3_clv['clv'].mean():+.3f}**\n")
        lines.append(f"- 中央値 CLV: {o3_clv['clv'].median():+.3f}\n")
        lines.append(f"- CLV>0 (有利約定): {int((o3_clv['clv']>0).sum())} ({(o3_clv['clv']>0).mean()*100:.1f}%)\n")
        lines.append(f"- CLV<0 (不利約定): {int((o3_clv['clv']<0).sum())} ({(o3_clv['clv']<0).mean()*100:.1f}%)\n")

    # 7. 同日同レース O3 vs P6 差分
    lines.append("\n## 7. 同日同レース O3 vs P6 差分 (O3 取り、P6 逃し or 逆)\n\n")
    # races の集合
    o3_races = set(o3['race_id'].unique())
    p6_races = set(p6['race_id'].unique())
    only_o3 = o3_races - p6_races
    only_p6 = p6_races - o3_races
    both = o3_races & p6_races
    lines.append(f"- O3 のみ取り (P6 逃し): {len(only_o3)} races\n")
    lines.append(f"- P6 のみ取り (O3 逃し): {len(only_p6)} races\n")
    lines.append(f"- 両方取り: {len(both)} races\n\n")
    only_o3_df = o3[o3['race_id'].isin(only_o3)]
    only_o3_m = metrics(only_o3_df)
    only_p6_df = p6[p6['race_id'].isin(only_p6)]
    only_p6_m = metrics(only_p6_df)
    lines.append("| 区分 | bets | ROI | PnL |\n|---|---|---|---|\n")
    if only_o3_m.get('n', 0):
        lines.append(f"| O3 のみ取り | {only_o3_m['n']} | {only_o3_m['roi']:.1f}% | ¥{only_o3_m['pnl']:+,.0f} |\n")
    if only_p6_m.get('n', 0):
        lines.append(f"| P6 のみ取り | {only_p6_m['n']} | {only_p6_m['roi']:.1f}% | ¥{only_p6_m['pnl']:+,.0f} |\n")

    # 8. O3 採用 vs P6 単独の試算
    lines.append("\n## 8. O3 採用シナリオ試算\n\n")
    lines.append("もし本番を **O3 単独** に切替えたら (実際の O3 shadow bets を流用):\n\n")
    lines.append(f"- O3 単独 PnL: **¥{o3m['pnl']:+,.0f}** (14日)\n")
    lines.append(f"- 現状 P6 単独 PnL: ¥{p6m['pnl']:+,.0f}\n")
    lines.append(f"- 差: **¥{o3m['pnl']-p6m['pnl']:+,.0f}** ({(o3m['pnl']-p6m['pnl'])/abs(p6m['pnl'])*100 if p6m['pnl'] else 0:+.0f}%)\n\n")
    lines.append(f"年率換算 (×26): O3 ¥{o3m['pnl']*26:+,.0f} vs P6 ¥{p6m['pnl']*26:+,.0f}\n")

    # 9. 注意点
    lines.append("\n## 9. 結論と注意点\n\n")
    o3_better = o3m['pnl'] > p6m['pnl']
    o3_higher_roi = o3m['roi'] > p6m['roi']
    if o3_better and o3_higher_roi:
        lines.append("**✅ O3 が PnL/ROI 両指標で P6 を上回る**。本番切替の候補。\n\n")
    elif o3_higher_roi:
        lines.append("🟡 ROI は O3 が上回るが PnL 絶対値は要確認\n\n")
    else:
        lines.append("❌ O3 は P6 を上回らない\n\n")
    lines.append("**留意事項:**\n\n")
    lines.append(f"- データ期間 14 日 ({o3m['n']} bets) は **統計的に薄い**。shadow 並走 2-4 週間で検証してから本番反映が安全\n")
    lines.append("- O3 は会場制限なし (P6 は 10 会場限定) → O3 は会場分散効果あり、リスク分散としても優位\n")
    lines.append(f"- O3 の MDD ¥{o3m['mdd']:,.0f} は P6 の ¥{p6m['mdd']:,.0f} に対し {(o3m['mdd']/p6m['mdd']*100 if p6m['mdd'] else 0):.0f}% — リスク許容度確認必要\n")
    lines.append(f"- 最大連敗: O3 {o3m['max_losses']} / P6 {p6m['max_losses']} — メンタル耐性確認\n")
    lines.append("- 本番反映するなら: 1) shadow 並走 → 2) 半額切替 (P6:O3 = 50:50) → 3) full 切替\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
