"""V11 モデル訓練: LightGBM + Miss Analysis サンプル重み

特徴:
  - 1st/2nd/3rd それぞれ独立した multi-class (6 classes) 分類器
  - sample_weight = Miss Analysisベースの重み × 配当log加重
  - 時系列 80/20 split (train/val)
  - early_stopping with multi_logloss

V10 (PyTorch NN, models/boatrace_model.pth) は一切触らない。
出力は analysis/models_v11/ 下のみ。
"""
import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models_v11"
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def train_position(X_tr, y_tr, w_tr, X_val, y_val, w_val, position_label):
    """特定順位の6-class分類器を訓練"""
    logger.info(f"  {position_label} 訓練中...")
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'seed': 42,
    }
    train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    val_ds = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_ds)
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    # 正解率
    probs = model.predict(X_val, num_iteration=model.best_iteration)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y_val).mean()
    logger.info(f"    {position_label} best_iter={model.best_iteration} "
                f"val_acc={acc*100:.2f}%")
    return model, acc


def main():
    logger.info("V11 訓練 開始")

    # データ読込
    data_path = MODELS_DIR / "train_data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y_1st = data['y_1st']
    y_2nd = data['y_2nd']
    y_3rd = data['y_3rd']
    weights = data['weights']
    feature_names = data['feature_names']

    logger.info(f"データ: {len(X)}件, {X.shape[1]}特徴量")

    # 時系列80/20 split
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_val = X[:split], X[split:]
    w_tr, w_val = weights[:split], weights[split:]
    y1_tr, y1_val = y_1st[:split], y_1st[split:]
    y2_tr, y2_val = y_2nd[:split], y_2nd[split:]
    y3_tr, y3_val = y_3rd[:split], y_3rd[split:]
    logger.info(f"train={len(X_tr)} val={len(X_val)}")

    # 3ポジション独立訓練
    results = {}
    models = {}
    for pos, y_tr, y_val in [('1st', y1_tr, y1_val),
                              ('2nd', y2_tr, y2_val),
                              ('3rd', y3_tr, y3_val)]:
        model, acc = train_position(X_tr, y_tr, w_tr, X_val, y_val, w_val, pos)
        models[pos] = model
        results[pos] = {
            'best_iteration': model.best_iteration,
            'val_accuracy': float(acc),
        }
        # 保存
        model_path = MODELS_DIR / f"boatrace_v11_{pos}.txt"
        model.save_model(str(model_path))
        logger.info(f"  saved: {model_path.name}")

    # 特徴量重要度 (1st分類器ベース)
    importance = models['1st'].feature_importance(importance_type='gain')
    imp_pairs = sorted(zip(feature_names, importance.tolist()),
                       key=lambda x: -x[1])
    importance_report = {
        'trained_at': datetime.now().isoformat(),
        'total_samples': n,
        'train_samples': split,
        'val_samples': n - split,
        'results': results,
        'top20_features': [{'name': n_, 'importance': float(v)}
                           for n_, v in imp_pairs[:20]],
        'bottom10_features': [{'name': n_, 'importance': float(v)}
                              for n_, v in imp_pairs[-10:]],
    }

    imp_path = REPORT_DIR / "03_v11_feature_importance.json"
    with open(imp_path, 'w', encoding='utf-8') as f:
        json.dump(importance_report, f, ensure_ascii=False, indent=2)
    logger.info(f"importance: {imp_path}")

    # 結果サマリ
    logger.info("")
    logger.info("=== 訓練結果 ===")
    for pos in ['1st', '2nd', '3rd']:
        r = results[pos]
        logger.info(f"  {pos}: val_acc={r['val_accuracy']*100:.2f}% "
                    f"best_iter={r['best_iteration']}")

    logger.info("")
    logger.info("=== Top 10 特徴量 ===")
    for n_, v in imp_pairs[:10]:
        logger.info(f"  {n_:30s} {v:>10.1f}")


if __name__ == '__main__':
    main()
