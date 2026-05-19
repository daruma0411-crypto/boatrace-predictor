"""戸田 LightGBM + QMC backtest (案 X Option A 改、最終判定)

64 で訓練した戸田 LightGBM (1着) を QMC に投入、戸田 2026-05 hold-out で
ROI を V10 baseline と比較。

評価:
  baseline: V10 pkl probs (calibrator 適用済、production 状態) + QMC → top-3 picks
  challenger: 戸田 LightGBM raw probs + QMC → top-3 picks

判定:
  🟢 ROI 改善 > +5pt → 採用 candidate (shadow 並走へ)
  🟡 0〜+5pt → 効果限定、別検討
  🔴 < 0 → 凍結

出力: analysis/reports/65_toda_lgb_qmc_backtest.md
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
from scipy.stats import qmc, norm

from src.features import FeatureEngineer
from src.monte_carlo import qmc_sanrentan_v3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PRED_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
LGB_PATH = ROOT / 'models' / 'lightgbm_toda_1st.txt'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '65_toda_lgb_qmc_backtest.md'

TEST_START = date(2026, 5, 1)
N_SIM = 8192
SEED = 42


def main():
    logger.info("戸田 LightGBM + QMC backtest")
    with open(PRED_PATH, 'rb') as f:
        predictions = pickle.load(f)
    logger.info(f"戸田 predictions: {len(predictions)}")

    # test (2026-05)
    test_data = []
    for rid, p in predictions.items():
        d = date.fromisoformat(p['race_date'])
        if d >= TEST_START:
            test_data.append(p)
    logger.info(f"Test (2026-05): {len(test_data)} races")

    # LightGBM 1着 model
    lgb_model = lgb.Booster(model_file=str(LGB_PATH))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

    # Backtest 2 scenarios
    def backtest(probs_fn, label):
        n_total = 0
        n_top1 = 0
        n_top3 = 0
        invested = 0
        returned = 0
        hits = []
        for p in test_data:
            try:
                probs_1st = probs_fn(p)
            except Exception:
                continue
            try:
                qmc_probs = qmc_sanrentan_v3(
                    probs_1st, boats_data=p['boats'],
                    race_data=p['race_data'], race_number=p['race_number'],
                    n_simulations=N_SIM, seed=SEED,
                )
            except Exception:
                continue
            top3 = sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]
            top3_combos = [t[0] for t in top3]
            actual = p['actual']
            if top3_combos[0] == actual:
                n_top1 += 1
            if actual in top3_combos:
                n_top3 += 1
                returned += p['payout'] or 0
                hits.append({'race_id': p['race_id'], 'combo': actual, 'payout': p['payout']})
            invested += 300
            n_total += 1
        roi = (returned - invested) / invested * 100 if invested else 0
        return {
            'label': label, 'n': n_total,
            'top1': n_top1, 'top1_rate': n_top1 / n_total * 100,
            'top3': n_top3, 'top3_rate': n_top3 / n_total * 100,
            'invested': invested, 'returned': returned,
            'pnl': returned - invested, 'roi': roi,
            'hits': hits,
        }

    # V10 baseline (pkl probs)
    logger.info("V10 baseline backtest")
    v10_fn = lambda p: p['probs_1st']
    s_v10 = backtest(v10_fn, 'V10 (pkl probs + QMC)')

    # LightGBM Toda
    logger.info("戸田 LightGBM backtest")
    def lgb_fn(p):
        features = fe.transform(p['race_data'], p['boats'])
        features = scaler.transform(features.reshape(1, -1)).flatten()
        probs = lgb_model.predict(features.reshape(1, -1), num_iteration=lgb_model.best_iteration)
        return probs[0].tolist()
    s_lgb = backtest(lgb_fn, '戸田 LightGBM raw + QMC')

    # Report
    lines = []
    lines.append("# 戸田 LightGBM + QMC backtest (最終判定)\n\n")
    lines.append(f"Test (hold-out 2026-05): {s_v10['n']} races\n")
    lines.append("各 race で top-3 picks (¥100 × 3 = ¥300/race) を購入する proxy ROI\n\n")

    lines.append("## 結果\n\n")
    lines.append("| 戦略 | n | top-1 hit% | top-3 hit% | 投資 ¥ | 回収 ¥ | PnL | ROI |\n|---|---|---|---|---|---|---|---|\n")
    for s in [s_v10, s_lgb]:
        lines.append(f"| {s['label']} | {s['n']} | {s['top1_rate']:.2f}% | {s['top3_rate']:.2f}% | "
                     f"¥{s['invested']:,} | ¥{s['returned']:,.0f} | ¥{s['pnl']:+,.0f} | {s['roi']:+.2f}% |\n")

    roi_diff = s_lgb['roi'] - s_v10['roi']
    top3_diff = s_lgb['top3_rate'] - s_v10['top3_rate']
    lines.append(f"\n**改善幅 (LightGBM - V10)**: top-3 hit {top3_diff:+.2f}pt, ROI {roi_diff:+.2f}pt\n")

    # 自動判定
    lines.append("\n## 自動判定 (CLAUDE.md 採用基準)\n\n")
    if roi_diff > 5.0:
        lines.append(f"- 🟢 **改善 +{roi_diff:.2f}pt > 撤退ライン (+5pt)** → 採用 candidate、shadow 並走\n")
    elif roi_diff > 0:
        lines.append(f"- 🟡 **改善 +{roi_diff:.2f}pt (撤退ライン未達)** → 効果限定\n")
    else:
        lines.append(f"- 🔴 **悪化 {roi_diff:.2f}pt** → 凍結\n")

    # Hits 詳細
    lines.append("\n## Hits 詳細 (top-3 当選)\n\n")
    for s in [s_v10, s_lgb]:
        lines.append(f"\n### {s['label']} ({len(s['hits'])} hits)\n\n")
        if not s['hits']:
            lines.append("(なし)\n")
            continue
        lines.append("| race_id | combo | payout |\n|---|---|---|\n")
        for h in sorted(s['hits'], key=lambda x: -x['payout']):
            lines.append(f"| {h['race_id']} | {h['combo']} | ¥{h['payout']:,} |\n")

    # 留意
    lines.append("\n## 留意事項\n\n")
    lines.append("- top-3 全部購入 proxy は Kelly/EV filter 不含、本番 ROI とは別\n")
    lines.append("- Test n=135 races (2026-05) は検出力ぎりぎり、forward 1-2 ヶ月で再検証推奨\n")
    lines.append("- 2026-05 戸田 1号艇 1着率 30% (平均 43% から -13pt) は極端期、再現性不明\n")
    lines.append("- 採用候補なら shadow 並走 2 週間必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
