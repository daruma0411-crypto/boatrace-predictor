"""戸田特化 LightGBM (案 X Option A 改、NN を LightGBM に置換)

NN warm-start fine-tune (63) は 1号艇 bias +12pt 悪化 → overfit 傾向。
LightGBM は小データ (2000+) で安定、戸田 2271 races でより適合。
過去 Phase C は全国 data で V10 タイ、今回は戸田単独で signal 集中。

設計:
- 76dim features (V10 と同じ、特徴量蓄積活用)
- LightGBM multi-class (6 boats 1着分類)
- 1着 / 2着 / 3着 で 3 model 並列訓練
- num_leaves=31, lr=0.05, early stopping (val multi_logloss)
- Train: 2025-06〜2026-03 (n=1968)
- Val:   2026-04 (n=168)
- Test (hold-out): 2026-05 (n=135)

出力: models/lightgbm_toda_*.pkl + analysis/reports/64_toda_lightgbm.md
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PRED_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
MODEL_PREFIX = ROOT / 'models' / 'lightgbm_toda'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '64_toda_lightgbm.md'

VAL_START = date(2026, 4, 1)
TEST_START = date(2026, 5, 1)

SEED = 42
np.random.seed(SEED)


def build_features(predictions):
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    Xs = []
    y1, y2, y3 = [], [], []
    dates = []
    for rid, p in predictions.items():
        try:
            features = fe.transform(p['race_data'], p['boats'])
            features = scaler.transform(features.reshape(1, -1)).flatten()
            Xs.append(features)
            y1.append(p['result_1st'] - 1)
            y2.append(p['result_2nd'] - 1)
            y3.append(p['result_3rd'] - 1)
            dates.append(date.fromisoformat(p['race_date']))
        except Exception:
            continue
    return (np.array(Xs, dtype=np.float32),
            np.array(y1, dtype=np.int32),
            np.array(y2, dtype=np.int32),
            np.array(y3, dtype=np.int32),
            dates)


def split_by_date(X, y1, y2, y3, dates):
    tr = [i for i, d in enumerate(dates) if d < VAL_START]
    va = [i for i, d in enumerate(dates) if VAL_START <= d < TEST_START]
    te = [i for i, d in enumerate(dates) if d >= TEST_START]
    return {
        'train': (X[tr], y1[tr], y2[tr], y3[tr]),
        'val':   (X[va], y1[va], y2[va], y3[va]),
        'test':  (X[te], y1[te], y2[te], y3[te]),
    }


def train_lgb(X_train, y_train, X_val, y_val, task_name='1st'):
    """LightGBM multi-class (6 boats) 訓練."""
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': SEED,
    }
    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)],
    )
    logger.info(f"[{task_name}] best iter: {model.best_iteration}, best score: {model.best_score['val']['multi_logloss']:.4f}")
    return model


def main():
    logger.info("戸田 LightGBM (case Option A 改)")
    with open(PRED_PATH, 'rb') as f:
        predictions = pickle.load(f)
    logger.info(f"戸田 predictions: {len(predictions)}")

    X, y1, y2, y3, dates = build_features(predictions)
    logger.info(f"X shape: {X.shape}")

    splits = split_by_date(X, y1, y2, y3, dates)
    for k, v in splits.items():
        logger.info(f"{k}: n={len(v[0])}")

    Xtr, y1tr, y2tr, y3tr = splits['train']
    Xv, y1v, y2v, y3v = splits['val']
    Xte, y1te, y2te, y3te = splits['test']

    # 訓練 3 モデル (1着 / 2着 / 3着)
    logger.info("=== 1着 LightGBM 訓練 ===")
    m1 = train_lgb(Xtr, y1tr, Xv, y1v, '1st')
    logger.info("=== 2着 LightGBM 訓練 ===")
    m2 = train_lgb(Xtr, y2tr, Xv, y2v, '2nd')
    logger.info("=== 3着 LightGBM 訓練 ===")
    m3 = train_lgb(Xtr, y3tr, Xv, y3v, '3rd')

    # 保存
    for name, model in [('1st', m1), ('2nd', m2), ('3rd', m3)]:
        path = f"{MODEL_PREFIX}_{name}.txt"
        model.save_model(path)
        logger.info(f"保存: {path}")

    # Test predictions
    p1_lgb = m1.predict(Xte, num_iteration=m1.best_iteration)
    p1_lgb_top = p1_lgb.argmax(axis=1)
    lgb_acc = (p1_lgb_top == y1te).mean()
    eps = 1e-8
    lgb_logloss = -np.log(np.clip(p1_lgb[np.arange(len(y1te)), y1te], eps, 1)).mean()

    # V10 baseline (戸田 test) — pkl 既存 probs
    test_rids = [rid for rid, p in predictions.items()
                 if date.fromisoformat(p['race_date']) >= TEST_START]
    v10_probs = []
    v10_y1 = []
    for rid in test_rids:
        p = predictions[rid]
        v10_probs.append(p['probs_1st'])
        v10_y1.append(p['result_1st'] - 1)
    v10_probs = np.array(v10_probs)
    v10_y1 = np.array(v10_y1)
    v10_top = v10_probs.argmax(axis=1)
    v10_acc = (v10_top == v10_y1).mean()
    v10_logloss = -np.log(np.clip(v10_probs[np.arange(len(v10_y1)), v10_y1], eps, 1)).mean()

    # Boat-level mean
    actual_by_boat = [(y1te == b).mean() * 100 for b in range(6)]
    v10_pred_by_boat = [v10_probs[:, b].mean() * 100 for b in range(6)]
    lgb_pred_by_boat = [p1_lgb[:, b].mean() * 100 for b in range(6)]

    # Feature importance (top 15)
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    try:
        names = FeatureEngineer().get_feature_names()
        if len(names) == X.shape[1]:
            feature_names = names
    except Exception:
        pass
    imp = m1.feature_importance(importance_type='gain')
    top_imp = sorted(enumerate(imp), key=lambda x: -x[1])[:15]

    # Report
    lines = []
    lines.append("# 戸田 LightGBM (案 X Option A 改、NN を LightGBM に置換)\n\n")
    lines.append(f"Train: 2025-06〜2026-03 (n={len(Xtr)}), Val: 2026-04 (n={len(Xv)}), "
                 f"Test (hold-out): 2026-05 (n={len(Xte)})\n\n")
    lines.append("LightGBM multi-class (6 boats), num_leaves=31, lr=0.05, early stopping (val multi_logloss)\n\n")

    lines.append("## 訓練結果\n\n")
    lines.append("| task | best_iter | val multi_logloss |\n|---|---|---|\n")
    for name, m in [('1着', m1), ('2着', m2), ('3着', m3)]:
        lines.append(f"| {name} | {m.best_iteration} | {m.best_score['val']['multi_logloss']:.4f} |\n")

    lines.append("\n## Test (hold-out 2026-05) NN-only vs LightGBM 比較\n\n")
    lines.append("| model | 1着 top-1 acc | 1着 log-loss |\n|---|---|---|\n")
    lines.append(f"| V10 baseline | {v10_acc:.4f} | {v10_logloss:.4f} |\n")
    lines.append(f"| **戸田 LightGBM** | **{lgb_acc:.4f}** | **{lgb_logloss:.4f}** |\n")
    lines.append(f"\n**改善幅**: acc {(lgb_acc-v10_acc)*100:+.2f}pt, log-loss {lgb_logloss-v10_logloss:+.4f}\n")

    lines.append("\n## Test 期間 boat-level calibration\n\n")
    lines.append("| boat | actual | V10 pred | LightGBM pred | V10 bias | LightGBM bias |\n|---|---|---|---|---|---|\n")
    for b in range(6):
        v10_bias = v10_pred_by_boat[b] - actual_by_boat[b]
        lgb_bias = lgb_pred_by_boat[b] - actual_by_boat[b]
        lines.append(f"| {b+1} | {actual_by_boat[b]:.2f}% | {v10_pred_by_boat[b]:.2f}% | "
                     f"{lgb_pred_by_boat[b]:.2f}% | {v10_bias:+.2f}pt | {lgb_bias:+.2f}pt |\n")

    lines.append("\n## Feature importance (1着 model top 15)\n\n")
    lines.append("| rank | feature | gain |\n|---|---|---|\n")
    for i, (idx, gain) in enumerate(top_imp):
        fname = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
        lines.append(f"| {i+1} | {fname} | {gain:.0f} |\n")

    # 比較: NN (63 result) と LightGBM
    lines.append("\n## V10 NN warm-start (63) vs LightGBM 比較 (Test)\n\n")
    lines.append("| model | 1着 top-1 acc | 1着 log-loss |\n|---|---|---|\n")
    lines.append(f"| V10 baseline | 0.3556 | 1.5060 |\n")
    lines.append(f"| V10 + Toda fine-tune (NN) | 0.3778 | 1.5102 |\n")
    lines.append(f"| **戸田 LightGBM** | **{lgb_acc:.4f}** | **{lgb_logloss:.4f}** |\n")

    lines.append("\n## 判定 ヒント\n\n")
    if lgb_acc > v10_acc + 0.03:
        lines.append("- 🟢 LightGBM が V10 を top-1 acc で +3pt 以上上回る、次は QMC backtest\n")
    elif lgb_acc > v10_acc:
        lines.append("- 🟡 LightGBM が V10 を僅かに上回る、効果限定\n")
    else:
        lines.append("- 🔴 LightGBM が V10 を上回らない、戸田特化アプローチ限界\n")
    lines.append("- 次は LightGBM + QMC で backtest ROI 算出 (analysis/65)\n")
    lines.append("- 結論は岩下さんに委ね、shadow 並走必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
