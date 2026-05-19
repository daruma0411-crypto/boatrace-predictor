"""P6 venue list 変更時の影響シミュレーション

User の問い: 「P6 を場変更したらどれくらい変わる?」

評価する 4 シナリオ (test 2026-05 hold-out、top-3 picks proxy):
  A. P6 current (10 venues) × V10 baseline: 現状の近似
  B. P6 current (10 venues) × shadow strategy (重複 6 で改善、4 で V10 維持)
  C. Functional 13 venues × shadow strategy: 我々分析 best
  D. Union 15 venues × shadow strategy: 全部入り

出力: analysis/reports/81_p6_venue_change.md
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import lightgbm as lgb

from src.features import FeatureEngineer
from src.monte_carlo import qmc_sanrentan_v3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
VENUE_PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
SPEC_76 = ROOT / 'models' / 'specialists'
SPEC_82 = ROOT / 'models' / 'specialists_82'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '81_p6_venue_change.md'

TEST_START = date(2026, 5, 1)
N_SIM = 8192
SEED = 42

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

# 各 venue test (2026-05) の top-3 proxy ROI (各 approach、80 結果から)
V10_TEST_ROI = {
    1:  -8.28, 2: -42.22, 3: -16.15, 4: -37.05, 5: -19.31, 6: +36.28,
    7:  -16.55, 8: +3.64, 9: -34.20, 10: -39.55, 11: -16.25, 12: -2.93,
    13: -43.68, 14: -45.31, 15: +1.09, 16: -58.04, 17: -19.16, 18: -20.79,
    19: -13.53, 20: -18.72, 21: -8.28, 22: -12.31, 23: -2.58, 24: -56.62,
}

SHADOW_TEST_ROI = {  # 80 結果 (各 venue の best approach 適用後)
    1: +9.34, 2: -5.43, 3: +15.09, 4: -1.59, 7: +10.65, 10: -28.13,
    12: +17.68, 13: +24.46, 14: -11.34, 16: -29.36, 22: +36.64, 23: +66.72,
    24: -41.20,
}

# venue ごとの test races 数 (top-3 proxy 投資は 300 yen/race)
TEST_N = {
    1: 122, 2: 135, 3: 78, 4: 130, 5: 96, 6: 113, 7: 138, 8: 140,
    9: 127, 10: 132, 11: 128, 12: 115, 13: 77, 14: 167, 15: 92, 16: 151,
    17: 99, 18: 156, 19: 117, 20: 151, 21: 116, 22: 120, 23: 132, 24: 72,
}

P6_CURRENT = [2, 4, 5, 6, 9, 10, 12, 13, 17, 23]
FUNCTIONAL_13 = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24]
UNION_15 = sorted(set(P6_CURRENT) | set(FUNCTIONAL_13))


def scenario_roi(venue_list, use_shadow):
    """venue_list で top-3 proxy 統合 ROI を算出.
    use_shadow=True なら functional venues は shadow、それ以外は V10.
    use_shadow=False なら 全て V10."""
    total_invested = 0
    total_returned = 0
    venue_details = []
    for vid in venue_list:
        n = TEST_N.get(vid, 0)
        invested = n * 300  # top-3 × ¥100
        if use_shadow and vid in SHADOW_TEST_ROI:
            roi = SHADOW_TEST_ROI[vid]
        else:
            roi = V10_TEST_ROI.get(vid, 0)
        returned = invested * (1 + roi / 100)
        total_invested += invested
        total_returned += returned
        venue_details.append({
            'venue': vid, 'name': VENUE_NAMES[vid], 'n': n,
            'roi': roi, 'source': 'shadow' if (use_shadow and vid in SHADOW_TEST_ROI) else 'V10',
        })
    total_roi = (total_returned - total_invested) / total_invested * 100 if total_invested else 0
    return {
        'total_n': sum(TEST_N.get(v, 0) for v in venue_list),
        'invested': total_invested, 'returned': total_returned,
        'pnl': total_returned - total_invested, 'roi': total_roi,
        'venues': venue_details,
    }


def main():
    logger.info("P6 venue 変更影響シミュレーション")
    scenarios = {
        'A. P6 current × V10': scenario_roi(P6_CURRENT, use_shadow=False),
        'B. P6 current × shadow': scenario_roi(P6_CURRENT, use_shadow=True),
        'C. Functional 13 × shadow': scenario_roi(FUNCTIONAL_13, use_shadow=True),
        'D. Union 15 × shadow': scenario_roi(UNION_15, use_shadow=True),
        '参考: 全 24 venues × V10': scenario_roi(list(VENUE_NAMES.keys()), use_shadow=False),
        '参考: 全 24 venues × shadow (13 で shadow、他 V10)': scenario_roi(list(VENUE_NAMES.keys()), use_shadow=True),
    }

    lines = []
    lines.append("# P6 venue list 変更時の影響シミュレーション\n\n")
    lines.append("**評価**: test 2026-05 hold-out、top-3 picks proxy ROI (300¥/race 投資)\n")
    lines.append("**shadow strategy**: 80 で確定した venue 別 best approach (recipe/pool/specialist)\n")
    lines.append("**V10 baseline**: 各 venue で V10 NN raw probs + QMC top-3 picks\n\n")

    lines.append("## venue list 比較\n\n")
    lines.append("| set | venues |\n|---|---|\n")
    lines.append(f"| **P6 current (10)** | {','.join(f'v{v}' for v in P6_CURRENT)} |\n")
    lines.append(f"| Functional 13 | {','.join(f'v{v}' for v in FUNCTIONAL_13)} |\n")
    lines.append(f"| Union 15 (P6 ∪ Functional) | {','.join(f'v{v}' for v in UNION_15)} |\n")

    common = set(P6_CURRENT) & set(FUNCTIONAL_13)
    p6_only = set(P6_CURRENT) - set(FUNCTIONAL_13)
    func_only = set(FUNCTIONAL_13) - set(P6_CURRENT)
    lines.append(f"\n- 重複: {','.join(f'v{v}' for v in sorted(common))} ({len(common)} venues)\n")
    lines.append(f"- P6 のみ: {','.join(f'v{v}' for v in sorted(p6_only))} ({len(p6_only)} venues)\n")
    lines.append(f"- Functional のみ: {','.join(f'v{v}' for v in sorted(func_only))} ({len(func_only)} venues)\n")

    lines.append("\n## 4 シナリオ 統合 ROI 比較\n\n")
    lines.append("| シナリオ | n_races | 投資 ¥ | 回収 ¥ | PnL | ROI |\n|---|---|---|---|---|---|\n")
    for label, s in scenarios.items():
        lines.append(f"| {label} | {s['total_n']} | ¥{s['invested']:,} | ¥{s['returned']:,.0f} | "
                     f"¥{s['pnl']:+,.0f} | **{s['roi']:+.2f}%** |\n")

    # 各シナリオの venue 内訳
    for label, s in scenarios.items():
        if '参考' in label:
            continue
        lines.append(f"\n### {label} venue 内訳\n\n")
        lines.append("| venue | name | n | ROI | source |\n|---|---|---|---|---|\n")
        for v in s['venues']:
            lines.append(f"| {v['venue']} | {v['name']} | {v['n']} | {v['roi']:+.2f}% | {v['source']} |\n")

    # 比較サマリ
    lines.append("\n## 比較サマリ (場変更の影響)\n\n")
    a = scenarios['A. P6 current × V10']['roi']
    b = scenarios['B. P6 current × shadow']['roi']
    c = scenarios['C. Functional 13 × shadow']['roi']
    d = scenarios['D. Union 15 × shadow']['roi']
    lines.append(f"- **A → B (P6 維持、内部 shadow 化)**: {a:+.2f}% → {b:+.2f}% ({b-a:+.2f}pt)\n")
    lines.append(f"- **A → C (venue list を functional 13 に変更 + shadow)**: {a:+.2f}% → {c:+.2f}% ({c-a:+.2f}pt)\n")
    lines.append(f"- **A → D (venue list を union 15 に変更 + shadow)**: {a:+.2f}% → {d:+.2f}% ({d-a:+.2f}pt)\n")
    lines.append(f"- **B → C (venue list 変更の純粋効果)**: {b:+.2f}% → {c:+.2f}% ({c-b:+.2f}pt)\n")

    # 注意
    lines.append("\n## 注意事項\n\n")
    lines.append("- top-3 picks proxy ROI は **Kelly/EV filter なし**、P6 実 production ROI とは別物\n")
    lines.append("- P6 実 production は max_odds=80, skip_56=True, kelly_fraction=0.0625 等で picks 絞り\n")
    lines.append("- proxy ROI は venue list + 予測精度の比較指標、絶対 ROI ではない\n")
    lines.append("- 真の影響は production 投入 + actual purchase data で検証必須\n")

    # 推奨判定
    lines.append("\n## 推奨判定\n\n")
    best_scenario = max([(label, s['roi']) for label, s in scenarios.items() if '参考' not in label],
                       key=lambda x: x[1])
    lines.append(f"- proxy ROI 最大: **{best_scenario[0]}** ({best_scenario[1]:+.2f}%)\n")
    if c > a + 5:
        lines.append(f"- **venue list 変更 + shadow** は明確な改善 ({c-a:+.2f}pt)\n")
    if d > c:
        lines.append(f"- Union 15 が Functional 13 より良い ({d-c:+.2f}pt) → venue 拡張余地あり\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
