"""戸田 (venue 2) baseline audit (案 X Phase 0)

戸田特化モデルを着手する前に、既存 V10+QMC が戸田でどれだけ負けているか確認。
cache の戸田 races (n=189) を全国平均と比較。

検証する 3 質問:
  Q1: 戸田で QMC top-1/top-3 hit 率は他会場と比べてどれだけ低いか
  Q2: 戸田で QMC は何を過大評価しているか (1-X-X 過大の度合い)
  Q3: 戸田の中穴 (3-X-X / 4-X-X) がどれだけ来ているか、QMC は拾えているか

これらで「戸田特化モデルが必要な data 根拠」を data 化。
backtest ROI 110% 超を目指す前段階の現状把握。

入力: cache (`analysis/qmc_predictions_cache.pkl`)
出力: analysis/reports/58_toda_baseline.md
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'qmc_predictions_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '58_toda_baseline.md'

TODA_VENUE_ID = 2


def main():
    logger.info("戸田 baseline audit (Phase 0)")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"cache: {len(cache)} races")

    # 戸田 / 他会場で分割
    toda = []
    other = []
    for rid, c in cache.items():
        if not c.get('actual') or not c.get('result_1st'):
            continue
        qmc_probs = c.get('qmc_probs', {})
        if not qmc_probs:
            continue
        top_pick = max(qmc_probs.items(), key=lambda x: x[1])
        top3 = sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]
        # 1-X-X 系合計推定確率
        sum_1xx = sum(p for k, p in qmc_probs.items() if k.startswith('1-'))
        sum_3xx = sum(p for k, p in qmc_probs.items() if k.startswith('3-'))
        sum_4xx = sum(p for k, p in qmc_probs.items() if k.startswith('4-'))
        entry = {
            'rid': rid,
            'top_pick': top_pick[0],
            'top_prob': top_pick[1],
            'top3': [t[0] for t in top3],
            'actual': c['actual'],
            'result_1st': c['result_1st'],
            'payout': c.get('payout', 0) or 0,
            'sum_1xx_pred': sum_1xx,
            'sum_3xx_pred': sum_3xx,
            'sum_4xx_pred': sum_4xx,
        }
        if c.get('venue_id') == TODA_VENUE_ID:
            toda.append(entry)
        else:
            other.append(entry)

    logger.info(f"戸田: {len(toda)} races, 他会場: {len(other)} races")

    def stats(rs, label):
        n = len(rs)
        if n == 0:
            return None
        top1_hit = sum(1 for r in rs if r['top_pick'] == r['actual'])
        top3_hit = sum(1 for r in rs if r['actual'] in r['top3'])
        b1_win = sum(1 for r in rs if r['result_1st'] == 1)
        b4_win = sum(1 for r in rs if r['result_1st'] == 4)
        b5_win = sum(1 for r in rs if r['result_1st'] == 5)
        b6_win = sum(1 for r in rs if r['result_1st'] == 6)
        actual_1xx = sum(1 for r in rs if r['actual'].startswith('1-'))
        actual_3xx = sum(1 for r in rs if r['actual'].startswith('3-'))
        actual_4xx = sum(1 for r in rs if r['actual'].startswith('4-'))
        # QMC top-3 picks 購入時の ROI 試算
        invested = n * 3 * 100  # top-3 を 100 円ずつ
        returned = sum(r['payout'] for r in rs if r['actual'] in r['top3'])
        roi = (returned - invested) / invested * 100 if invested else 0
        return {
            'label': label,
            'n': n,
            'top1_hit': top1_hit,
            'top1_rate': top1_hit / n * 100,
            'top3_hit': top3_hit,
            'top3_rate': top3_hit / n * 100,
            'b1_win_rate': b1_win / n * 100,
            'b4_win_rate': b4_win / n * 100,
            'b5_win_rate': b5_win / n * 100,
            'b6_win_rate': b6_win / n * 100,
            'pred_1xx': float(np.mean([r['sum_1xx_pred'] for r in rs])) * 100,
            'actual_1xx': actual_1xx / n * 100,
            'pred_3xx': float(np.mean([r['sum_3xx_pred'] for r in rs])) * 100,
            'actual_3xx': actual_3xx / n * 100,
            'pred_4xx': float(np.mean([r['sum_4xx_pred'] for r in rs])) * 100,
            'actual_4xx': actual_4xx / n * 100,
            'roi_top3_proxy': roi,
        }

    s_toda = stats(toda, '戸田 (venue 2)')
    s_other = stats(other, '他 23 会場')

    lines = []
    lines.append("# 戸田 baseline audit (案 X Phase 0)\n\n")
    lines.append(f"対象: cache {len(cache)} races のうち戸田 {len(toda)} / 他会場 {len(other)}\n\n")

    lines.append("## Q1: hit 率比較 (戸田 vs 他会場)\n\n")
    lines.append("| 指標 | 戸田 | 他会場 | 差 |\n|---|---|---|---|\n")
    for key, label in [
        ('top1_rate', 'QMC top-1 hit率'),
        ('top3_rate', 'QMC top-3 hit率'),
        ('b1_win_rate', '1号艇 1着率'),
        ('b4_win_rate', '4号艇 1着率'),
        ('b5_win_rate', '5号艇 1着率'),
        ('b6_win_rate', '6号艇 1着率'),
    ]:
        diff = s_toda[key] - s_other[key]
        lines.append(f"| {label} | {s_toda[key]:.2f}% | {s_other[key]:.2f}% | {diff:+.2f}pt |\n")

    lines.append("\n## Q2: QMC 過大/過小評価 (戸田特化の有効性)\n\n")
    lines.append("予測 vs 実頻度の bias を、戸田 vs 他会場で比較。bias 大きい = 戸田特化モデルの余地大。\n\n")
    lines.append("| 系 | 戸田 予測% | 戸田 実% | 戸田 bias | 他会場 予測% | 他会場 実% | 他会場 bias |\n|---|---|---|---|---|---|---|\n")
    for series in ['1xx', '3xx', '4xx']:
        toda_pred = s_toda[f'pred_{series}']
        toda_actual = s_toda[f'actual_{series}']
        toda_bias = toda_pred - toda_actual
        other_pred = s_other[f'pred_{series}']
        other_actual = s_other[f'actual_{series}']
        other_bias = other_pred - other_actual
        lines.append(f"| {series.upper()} | {toda_pred:.2f}% | {toda_actual:.2f}% | "
                     f"{toda_bias:+.2f}pt | {other_pred:.2f}% | {other_actual:.2f}% | "
                     f"{other_bias:+.2f}pt |\n")

    lines.append("\n## Q3: top-3 購入 proxy ROI\n\n")
    lines.append("|  | n | top-3 hit率 | 投資 (¥) | 回収 (¥) | ROI |\n|---|---|---|---|---|---|\n")
    for s in [s_toda, s_other]:
        invested = s['n'] * 3 * 100
        returned = invested * (1 + s['roi_top3_proxy'] / 100)
        lines.append(f"| {s['label']} | {s['n']} | {s['top3_rate']:.2f}% | "
                     f"¥{invested:,} | ¥{returned:,.0f} | {s['roi_top3_proxy']:+.2f}% |\n")

    lines.append("\n## 自動判定 (戸田特化モデル着手の妥当性)\n\n")
    # 戸田で 1xx 過大評価 が他会場より大きいか
    toda_1xx_bias = s_toda['pred_1xx'] - s_toda['actual_1xx']
    other_1xx_bias = s_other['pred_1xx'] - s_other['actual_1xx']
    diff_bias = toda_1xx_bias - other_1xx_bias
    lines.append(f"- 戸田の 1-X-X bias: {toda_1xx_bias:+.2f}pt (他会場 {other_1xx_bias:+.2f}pt、差 {diff_bias:+.2f}pt)\n")
    if diff_bias > 5.0:
        lines.append(f"  → 🟢 戸田は他会場より顕著に 1号艇本命の過大評価 (差 {diff_bias:+.2f}pt)、特化モデル余地大\n")
    elif diff_bias > 2.0:
        lines.append(f"  → 🟡 戸田は他会場と類似、効果限定の可能性\n")
    else:
        lines.append(f"  → 🔴 戸田と他会場の bias 差小、特化モデル意味薄\n")

    # 戸田 ROI proxy が他会場より低いか
    roi_diff = s_toda['roi_top3_proxy'] - s_other['roi_top3_proxy']
    lines.append(f"- top-3 購入 proxy ROI 差: 戸田 {s_toda['roi_top3_proxy']:+.2f}% vs 他会場 {s_other['roi_top3_proxy']:+.2f}% ({roi_diff:+.2f}pt)\n")
    if roi_diff < -10.0:
        lines.append(f"  → 🟢 戸田は他会場より明確に negative、特化モデルで取り戻す余地大\n")
    elif roi_diff < -5.0:
        lines.append(f"  → 🟡 戸田は他会場よりやや劣勢\n")
    else:
        lines.append(f"  → 🔴 戸田 ROI は他会場と同等、特化モデル必要性低\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- cache 期間 (2026-04-11〜05-18) のみ、戸田 n=189 で短期\n")
    lines.append("- top-3 購入 proxy は mc3 系 Kelly/EV filter 不含、相対比較指標\n")
    lines.append("- 戸田 1号艇 1着率 全期間 (n=2272) では 43.22%、cache 期間 (n=189) では 34.39%\n")
    lines.append("  → サンプル期間で揺れあり、長期 forward の data 必須\n")
    lines.append("- 結論は出さず、岩下さんの判断 (Phase 1 着手 / STAY / 別 phase) を待つ\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
