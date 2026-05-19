"""戸田 + 桐生 LightGBM (sequential venue add Step 1)

戸田単独 LightGBM (64/65、ROI -35.85%) に **桐生 を pool 追加**。
data 量: 戸田 2271 + 桐生 2252 = 4523 races。
Val/Test は 戸田単独 で評価 (clear comparison)。

訓練:
  Train: 戸田 2025-06〜2026-03 (1968) + 桐生 全期間 (2252、ただし 2026-05 除く)
  Val:   戸田 2026-04 のみ (168)
  Test (hold-out): 戸田 2026-05 のみ (135)

判定:
  🟢 戸田 hold-out ROI > 65 (-35.85%) + 5pt → 桐生 追加効果あり、次は + 平和島
  🟡 0〜+5pt → 効果限定
  🔴 < 0 → 桐生 ノイズ、戸田単独に戻す

出力: models/lightgbm_toda_kiryu_*.txt + analysis/reports/68_toda_kiryu_lightgbm.md
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
TODA_PRED_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
KIRYU_PRED_PATH = ROOT / 'analysis' / 'kiryu_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
MODEL_PREFIX = ROOT / 'models' / 'lightgbm_toda_kiryu'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '68_toda_kiryu_lightgbm.md'

TODA_TEST_START = date(2026, 5, 1)
TODA_VAL_START = date(2026, 4, 1)
N_SIM = 8192
SEED = 42


def build_features_dataset(predictions, venue_label):
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    out = []
    for rid, p in predictions.items():
        try:
            features = fe.transform(p['race_data'], p['boats'])
            features = scaler.transform(features.reshape(1, -1)).flatten()
            out.append({
                'rid': rid,
                'venue': venue_label,
                'X': features,
                'y1': p['result_1st'] - 1,
                'y2': p['result_2nd'] - 1,
                'y3': p['result_3rd'] - 1,
                'date': date.fromisoformat(p['race_date']),
                'prediction': p,
            })
        except Exception:
            continue
    return out


def main():
    logger.info("戸田 + 桐生 LightGBM")
    toda = build_features_dataset(pickle.load(open(TODA_PRED_PATH, 'rb')), 'toda')
    kiryu = build_features_dataset(pickle.load(open(KIRYU_PRED_PATH, 'rb')), 'kiryu')
    logger.info(f"戸田: {len(toda)}, 桐生: {len(kiryu)}")

    # 戸田 split
    toda_train = [r for r in toda if r['date'] < TODA_VAL_START]
    toda_val   = [r for r in toda if TODA_VAL_START <= r['date'] < TODA_TEST_START]
    toda_test  = [r for r in toda if r['date'] >= TODA_TEST_START]
    # 桐生 は 2026-05 を除外 (戸田 hold-out 期間と整合、leakage 防止)
    kiryu_train = [r for r in kiryu if r['date'] < TODA_TEST_START]

    logger.info(f"戸田 train: {len(toda_train)}, val: {len(toda_val)}, test: {len(toda_test)}")
    logger.info(f"桐生 train: {len(kiryu_train)}")

    Xtr = np.array([r['X'] for r in toda_train + kiryu_train], dtype=np.float32)
    y1tr = np.array([r['y1'] for r in toda_train + kiryu_train], dtype=np.int32)
    y2tr = np.array([r['y2'] for r in toda_train + kiryu_train], dtype=np.int32)
    y3tr = np.array([r['y3'] for r in toda_train + kiryu_train], dtype=np.int32)
    Xv = np.array([r['X'] for r in toda_val], dtype=np.float32)
    y1v = np.array([r['y1'] for r in toda_val], dtype=np.int32)
    Xte = np.array([r['X'] for r in toda_test], dtype=np.float32)
    y1te = np.array([r['y1'] for r in toda_test], dtype=np.int32)
    logger.info(f"Train total: {len(Xtr)}, Val (戸田): {len(Xv)}, Test (戸田): {len(Xte)}")

    # LightGBM 訓練 (1着のみで十分、2/3 着は省略)
    params = {
        'objective': 'multiclass', 'num_class': 6, 'metric': 'multi_logloss',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 20, 'lambda_l2': 1.0,
        'verbose': -1, 'seed': SEED,
    }
    lgb_train = lgb.Dataset(Xtr, y1tr)
    lgb_val = lgb.Dataset(Xv, y1v, reference=lgb_train)
    m1 = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)],
    )
    logger.info(f"1着 best iter: {m1.best_iteration}, val multi_logloss: {m1.best_score['val']['multi_logloss']:.4f}")
    m1.save_model(f"{MODEL_PREFIX}_1st.txt")

    # 戸田 hold-out 評価 (NN-only + QMC backtest)
    p1_lgb = m1.predict(Xte, num_iteration=m1.best_iteration)
    p1_top = p1_lgb.argmax(axis=1)
    nn_acc = (p1_top == y1te).mean()
    eps = 1e-8
    nn_logloss = -np.log(np.clip(p1_lgb[np.arange(len(y1te)), y1te], eps, 1)).mean()

    # boat-level calibration
    actual_by_boat = [(y1te == b).mean() * 100 for b in range(6)]
    lgb_pred_by_boat = [p1_lgb[:, b].mean() * 100 for b in range(6)]

    # QMC backtest (戸田 hold-out)
    logger.info("戸田 hold-out QMC backtest")
    def backtest_qmc(probs_arr, predictions_list):
        n_total = n_top1 = n_top3 = 0
        invested = returned = 0
        for probs, r in zip(probs_arr, predictions_list):
            p = r['prediction']
            try:
                qp = qmc_sanrentan_v3(
                    probs.tolist(), boats_data=p['boats'],
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
            'n': n_total, 'top1': n_top1, 'top1_rate': n_top1/n_total*100,
            'top3': n_top3, 'top3_rate': n_top3/n_total*100,
            'invested': invested, 'returned': returned,
            'pnl': returned - invested, 'roi': roi,
        }

    s_tk = backtest_qmc(p1_lgb, toda_test)

    # 比較: 65 の戸田 単独 LightGBM (Phase 64 model)
    # 単独 model load
    lgb_toda_solo = lgb.Booster(model_file=str(ROOT / 'models' / 'lightgbm_toda_1st.txt'))
    p1_solo = lgb_toda_solo.predict(Xte, num_iteration=lgb_toda_solo.best_iteration)
    s_solo = backtest_qmc(p1_solo, toda_test)

    # V10 baseline (pkl)
    v10_probs = np.array([r['prediction']['probs_1st'] for r in toda_test])
    s_v10 = backtest_qmc(v10_probs, toda_test)

    # Report
    lines = []
    lines.append("# 戸田 + 桐生 LightGBM (sequential venue add Step 1)\n\n")
    lines.append(f"Train: 戸田 {len(toda_train)} + 桐生 {len(kiryu_train)} = {len(Xtr)} races\n")
    lines.append(f"Val: 戸田 {len(Xv)}, Test (hold-out): 戸田 {len(Xte)}\n\n")

    lines.append("## 訓練結果\n\n")
    lines.append(f"- 1着 best iter: **{m1.best_iteration}**\n")
    lines.append(f"- val multi_logloss: **{m1.best_score['val']['multi_logloss']:.4f}**\n\n")
    lines.append("(参考 64 戸田単独: best iter 23, val logloss 1.4065)\n\n")

    lines.append("## NN-only metrics on 戸田 hold-out (2026-05)\n\n")
    lines.append(f"- top-1 acc: {nn_acc:.4f}\n")
    lines.append(f"- 1着 log-loss: {nn_logloss:.4f}\n\n")

    lines.append("## Boat-level calibration (戸田 hold-out)\n\n")
    lines.append("| boat | actual | LightGBM (戸田+桐生) pred | bias |\n|---|---|---|---|\n")
    for b in range(6):
        bias = lgb_pred_by_boat[b] - actual_by_boat[b]
        lines.append(f"| {b+1} | {actual_by_boat[b]:.2f}% | {lgb_pred_by_boat[b]:.2f}% | {bias:+.2f}pt |\n")

    lines.append("\n## QMC backtest 比較 (戸田 hold-out 2026-05)\n\n")
    lines.append("| 戦略 | n | top-3 hit% | 投資 | 回収 | PnL | ROI |\n|---|---|---|---|---|---|---|\n")
    for label, s in [
        ('V10 (pkl)', s_v10),
        ('LightGBM 戸田単独 (65)', s_solo),
        ('**LightGBM 戸田+桐生**', s_tk),
    ]:
        lines.append(f"| {label} | {s['n']} | {s['top3_rate']:.2f}% | "
                     f"¥{s['invested']:,} | ¥{s['returned']:,.0f} | ¥{s['pnl']:+,.0f} | {s['roi']:+.2f}% |\n")

    roi_vs_solo = s_tk['roi'] - s_solo['roi']
    roi_vs_v10 = s_tk['roi'] - s_v10['roi']
    lines.append(f"\n**改善幅 (戸田+桐生 vs 戸田単独)**: ROI {roi_vs_solo:+.2f}pt\n")
    lines.append(f"**改善幅 (戸田+桐生 vs V10)**: ROI {roi_vs_v10:+.2f}pt\n")

    # 自動判定
    lines.append("\n## 自動判定\n\n")
    if roi_vs_solo > 5.0:
        lines.append(f"- 🟢 **桐生 追加で +{roi_vs_solo:.2f}pt 改善** → 効果あり、次は + 平和島\n")
    elif roi_vs_solo > 0:
        lines.append(f"- 🟡 **桐生 追加で +{roi_vs_solo:.2f}pt** (撤退ライン +5pt 未達) → 効果限定\n")
    else:
        lines.append(f"- 🔴 **桐生 追加で {roi_vs_solo:+.2f}pt 悪化** → 戸田単独 (65) に戻す\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- Test n=135 races は検出力ぎりぎり、改善幅は noise の可能性\n")
    lines.append("- 桐生 train 全期間 (2025-06〜2026-04) は test 期間 (2026-05) を含まず、leakage なし\n")
    lines.append("- ROI 改善は top-3 全部購入 proxy、本番 Kelly/EV filter とは別\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
