"""24 venue 単独 LightGBM specialists (Phase B + C)

各会場の data だけで LightGBM 訓練 → その会場の hold-out で評価。
24 個の「調味料」モデル完成。

設計:
  各 venue で:
    Train: 2025-06 〜 2026-03 (全月)
    Val:   2026-04 (early stopping)
    Test (hold-out): 2026-05
  V10 baseline vs 単独 specialist で ROI 比較

出力: models/lightgbm_v{NN}_1st.txt × 24
      analysis/reports/70_specialists_summary.md
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
PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
MODEL_DIR = ROOT / 'models' / 'specialists'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '70_specialists_summary.md'

VAL_START = date(2026, 4, 1)
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


def build_features(predictions, fe, scaler):
    out = []
    for rid, p in predictions.items():
        try:
            features = fe.transform(p['race_data'], p['boats'])
            features = scaler.transform(features.reshape(1, -1)).flatten()
            out.append({
                'rid': rid, 'X': features,
                'y1': p['result_1st'] - 1,
                'date': date.fromisoformat(p['race_date']),
                'prediction': p,
            })
        except Exception:
            continue
    return out


def train_specialist(train_data, val_data, venue_id):
    Xtr = np.array([r['X'] for r in train_data], dtype=np.float32)
    y1tr = np.array([r['y1'] for r in train_data], dtype=np.int32)
    Xv = np.array([r['X'] for r in val_data], dtype=np.float32)
    y1v = np.array([r['y1'] for r in val_data], dtype=np.int32)
    params = {
        'objective': 'multiclass', 'num_class': 6, 'metric': 'multi_logloss',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 20, 'lambda_l2': 1.0,
        'verbose': -1, 'seed': SEED,
    }
    lgb_train = lgb.Dataset(Xtr, y1tr)
    lgb_val = lgb.Dataset(Xv, y1v, reference=lgb_train)
    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)],
    )
    return model


def backtest_qmc(probs_list, predictions_list):
    n_total = n_top1 = n_top3 = 0
    invested = returned = 0
    for probs, r in zip(probs_list, predictions_list):
        p = r['prediction']
        try:
            qp = qmc_sanrentan_v3(
                probs.tolist() if hasattr(probs, 'tolist') else list(probs),
                boats_data=p['boats'],
                race_data=p['race_data'], race_number=p['race_number'],
                n_simulations=N_SIM, seed=SEED,
            )
        except Exception:
            continue
        top3 = sorted(qp.items(), key=lambda x: -x[1])[:3]
        top3_combos = [t[0] for t in top3]
        if top3_combos[0] == p['actual']:
            n_top1 += 1
        if p['actual'] in top3_combos:
            n_top3 += 1
            returned += p['payout'] or 0
        invested += 300
        n_total += 1
    roi = (returned - invested) / invested * 100 if invested else 0
    return {
        'n': n_total, 'top1_rate': n_top1/n_total*100 if n_total else 0,
        'top3_rate': n_top3/n_total*100 if n_total else 0,
        'invested': invested, 'returned': returned,
        'pnl': returned - invested, 'roi': roi,
    }


def main():
    logger.info("24 venue specialists 訓練 + 評価")
    venue_preds = pickle.load(open(PRED_PATH, 'rb'))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for vid in sorted(venue_preds.keys()):
        venue_name = VENUE_NAMES.get(vid, str(vid))
        preds = venue_preds[vid]
        if len(preds) < 500:
            logger.warning(f"venue {vid} ({venue_name}) skip (n={len(preds)})")
            continue
        records = build_features(preds, fe, scaler)
        train = [r for r in records if r['date'] < VAL_START]
        val   = [r for r in records if VAL_START <= r['date'] < TEST_START]
        test  = [r for r in records if r['date'] >= TEST_START]
        if not train or not val or not test:
            logger.warning(f"venue {vid} split 不足 (tr={len(train)}, v={len(val)}, te={len(test)})")
            continue

        logger.info(f"venue {vid} ({venue_name}): tr={len(train)}, v={len(val)}, te={len(test)}")
        model = train_specialist(train, val, vid)
        model_path = MODEL_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        model.save_model(str(model_path))

        Xte = np.array([r['X'] for r in test], dtype=np.float32)
        p1_lgb = model.predict(Xte, num_iteration=model.best_iteration)
        p1_v10 = np.array([r['prediction']['probs_1st'] for r in test])

        # Backtest
        s_v10 = backtest_qmc(p1_v10, test)
        s_lgb = backtest_qmc(p1_lgb, test)

        roi_diff = s_lgb['roi'] - s_v10['roi']
        top3_diff = s_lgb['top3_rate'] - s_v10['top3_rate']
        results.append({
            'venue': vid, 'name': venue_name,
            'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
            'best_iter': model.best_iteration,
            'val_logloss': float(model.best_score['val']['multi_logloss']),
            'v10_top3': s_v10['top3_rate'], 'v10_roi': s_v10['roi'],
            'lgb_top3': s_lgb['top3_rate'], 'lgb_roi': s_lgb['roi'],
            'roi_diff': roi_diff, 'top3_diff': top3_diff,
        })
        logger.info(f"  v10 ROI={s_v10['roi']:+.2f}%, lgb ROI={s_lgb['roi']:+.2f}%, diff {roi_diff:+.2f}pt")

    # Report
    lines = []
    lines.append("# 24 venue specialists training + evaluation (Phase B+C)\n\n")
    lines.append("各 venue を **その venue data だけで** LightGBM 訓練 (調味料モデル)。\n")
    lines.append("Test (hold-out 2026-05) で V10 baseline と top-3 + QMC backtest 比較。\n\n")

    # 改善幅でソート
    results_sorted = sorted(results, key=lambda x: -x['roi_diff'])
    lines.append("## venue 別改善幅 ranking (specialist vs V10)\n\n")
    lines.append("| venue | name | n_test | V10 ROI | specialist ROI | ROI diff | top-3 diff | val_logloss |\n|---|---|---|---|---|---|---|---|\n")
    for r in results_sorted:
        flag = '🟢' if r['roi_diff'] > 5 else ('🟡' if r['roi_diff'] > 0 else '🔴')
        lines.append(f"| {r['venue']} | {r['name']} | {r['n_test']} | "
                     f"{r['v10_roi']:+.2f}% | {r['lgb_roi']:+.2f}% | "
                     f"**{r['roi_diff']:+.2f}pt {flag}** | {r['top3_diff']:+.2f}pt | "
                     f"{r['val_logloss']:.4f} |\n")

    # サマリ統計
    n_improve = sum(1 for r in results if r['roi_diff'] > 5)
    n_mid = sum(1 for r in results if 0 < r['roi_diff'] <= 5)
    n_worse = sum(1 for r in results if r['roi_diff'] <= 0)
    avg_diff = float(np.mean([r['roi_diff'] for r in results]))
    median_diff = float(np.median([r['roi_diff'] for r in results]))
    lines.append(f"\n## サマリ ({len(results)} venues)\n\n")
    lines.append(f"- 🟢 +5pt 以上改善: **{n_improve} venues**\n")
    lines.append(f"- 🟡 0〜+5pt 改善: {n_mid} venues\n")
    lines.append(f"- 🔴 悪化: {n_worse} venues\n")
    lines.append(f"- 平均改善幅: {avg_diff:+.2f}pt\n")
    lines.append(f"- 中央値: {median_diff:+.2f}pt\n")

    # 戸田 highlight
    toda_r = next((r for r in results if r['venue'] == 2), None)
    if toda_r:
        lines.append(f"\n戸田: V10 ROI {toda_r['v10_roi']:+.2f}% → 特化 {toda_r['lgb_roi']:+.2f}% (改善 {toda_r['roi_diff']:+.2f}pt)\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
