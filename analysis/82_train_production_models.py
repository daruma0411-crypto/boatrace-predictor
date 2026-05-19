"""Production 投入用 pool models + shadow config 生成

shadow_recipe_v1 strategy で必要な production artifacts:
  1. 3 つの Pool LightGBM model (v14, v16, v23 の L3 strict pool)
  2. shadow_config.json:
     - venue 別 strategy 定義 (13 venues)
     - venue 距離行列 (recipe blend に必要)
     - functional venues list

出力:
  models/pool_models/pool_v??_L3_strict.txt × 3
  models/shadow_config.json
"""
import os
import sys
import json
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
VENUE_PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
POOL_DIR = ROOT / 'models' / 'pool_models'
CONFIG_PATH = ROOT / 'models' / 'shadow_config.json'

# 注: production では train+val 全期間使用 (test 期間関係なく、最新まで)
# 解析時点 (2026-05-20) では cache の最新 (2026-05-19) まで含めて訓練
TRAIN_END_PROD = date(2026, 5, 19)
SEED = 42

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

POOL_DEFS = [
    (14, 'L3_海水_strict', [14, 4, 16, 17, 22]),
    (16, 'L3_海水_strict', [16, 14, 17, 22, 4]),
    (23, 'L3_淡水_strict', [23, 12, 13, 21, 5, 11]),
]

# Functional 13 venues with best approach
VENUE_STRATEGIES = {
    1:  {'type': 'recipe_v10_own', 'v10_weight': 0.6, 'own_weight': 0.4},
    2:  {'type': 'specialist_82', 'venue': 2},
    3:  {'type': 'specialist_82', 'venue': 3},
    4:  {'type': 'recipe_top_K_sim', 'target': 4, 'K': 5},
    7:  {'type': 'specialist_76', 'venue': 7},
    10: {'type': 'recipe_top_K_sim', 'target': 10, 'K': 2},
    12: {'type': 'specialist_76', 'venue': 12},
    13: {'type': 'recipe_75_sub', 'target': 13, 'K': 2, 'own_w': 1.0, 'sub_alpha': 0.3},
    14: {'type': 'pool', 'pool_id': 'pool_v14_L3_海水_strict'},
    16: {'type': 'pool', 'pool_id': 'pool_v16_L3_海水_strict'},
    22: {'type': 'specialist_82', 'venue': 22},
    23: {'type': 'pool', 'pool_id': 'pool_v23_L3_淡水_strict'},
    24: {'type': 'recipe_own_functional', 'target': 24, 'own_w': 2.0,
         'functional_others': [2, 3, 7, 12, 13, 14, 16]},
}


def build_features_82(p, fe, scaler):
    features = fe.transform(p['race_data'], p['boats'])
    features = scaler.transform(features.reshape(1, -1)).flatten()
    local_adv = []
    for b in p['boats']:
        lr = b.get('local_win_rate_2', 0) or 0
        gr = b.get('win_rate_2', 0) or 0
        local_adv.append((lr - gr) / 100.0)
    return np.concatenate([features, np.array(local_adv, dtype=np.float32)])


def main():
    logger.info("Production 投入用モデル + config 生成")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

    # 全 venue records (cache の最新まで全部 train)
    logger.info("All records build (train period = 2025-06 〜 2026-05-19)")
    all_records = {}
    for vid, preds in venue_preds.items():
        recs = []
        for rid, p in preds.items():
            try:
                features_82 = build_features_82(p, fe, scaler)
                d = date.fromisoformat(p['race_date'])
                if d <= TRAIN_END_PROD:
                    recs.append({'features_82': features_82, 'y1': p['result_1st'] - 1, 'date': d})
            except Exception:
                continue
        all_records[vid] = recs

    # Pool training (3 venues、各 L3 strict)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for target_vid, level_name, members in POOL_DEFS:
        logger.info(f"Pool v{target_vid} ({level_name}) members={members}")
        pool_train = []
        target_val = []
        for v in members:
            for r in all_records.get(v, []):
                if v == target_vid:
                    if r['date'] < date(2026, 5, 1):
                        pool_train.append(r)
                    else:
                        target_val.append(r)
                else:
                    if r['date'] < date(2026, 5, 1):
                        pool_train.append(r)
        # train (val 2026-05 を early stop に使う production-final)
        # 注: production 用は val なしで full data train が筋だが、early stop で過学習防ぐ
        Xtr = np.array([r['features_82'][:76] for r in pool_train], dtype=np.float32)
        y1tr = np.array([r['y1'] for r in pool_train], dtype=np.int32)
        Xv = np.array([r['features_82'][:76] for r in target_val], dtype=np.float32)
        y1v = np.array([r['y1'] for r in target_val], dtype=np.int32)
        params = {
            'objective': 'multiclass', 'num_class': 6, 'metric': 'multi_logloss',
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'min_data_in_leaf': 20, 'lambda_l2': 1.0, 'verbose': -1, 'seed': SEED,
        }
        lgb_train = lgb.Dataset(Xtr, y1tr)
        lgb_val = lgb.Dataset(Xv, y1v, reference=lgb_train)
        model = lgb.train(params, lgb_train,
                          valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'],
                          num_boost_round=500,
                          callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)])
        path = POOL_DIR / f'pool_v{target_vid:02d}_L3_strict.txt'
        model.save_model(str(path))
        logger.info(f"  saved: {path} (best_iter={model.best_iteration}, val_logloss={model.best_score['val']['multi_logloss']:.4f})")

    # 距離行列計算 (recipe blend に必要)
    feats = {}
    for vid, preds in venue_preds.items():
        if not preds:
            continue
        b1 = np.zeros(6)
        for p in preds.values():
            b1[p['result_1st'] - 1] += 1
        feats[vid] = (b1 / len(preds) * 100).tolist()
    distances = {}
    for v1 in feats:
        sorted_d = sorted(
            [(v2, float(np.linalg.norm(np.array(feats[v1]) - np.array(feats[v2]))))
             for v2 in feats if v2 != v1],
            key=lambda x: x[1]
        )
        distances[str(v1)] = [{'venue_id': v2, 'distance': d} for v2, d in sorted_d]

    # Config 保存
    config = {
        'version': 'v1',
        'created_at': '2026-05-20',
        'description': 'Shadow Recipe v1 strategy (13 functional venues) — Recipe blend + Pool + Specialist 全部入り',
        'venue_strategies': {str(k): v for k, v in VENUE_STRATEGIES.items()},
        'venue_distances': distances,  # 各 venue の類似度ランキング
        'functional_venues': sorted(VENUE_STRATEGIES.keys()),
        'pool_models': {
            'pool_v14_L3_海水_strict': {
                'members': [14, 4, 16, 17, 22],
                'file': 'models/pool_models/pool_v14_L3_strict.txt',
            },
            'pool_v16_L3_海水_strict': {
                'members': [16, 14, 17, 22, 4],
                'file': 'models/pool_models/pool_v16_L3_strict.txt',
            },
            'pool_v23_L3_淡水_strict': {
                'members': [23, 12, 13, 21, 5, 11],
                'file': 'models/pool_models/pool_v23_L3_strict.txt',
            },
        },
        'specialists_76_dir': 'models/specialists',
        'specialists_82_dir': 'models/specialists_82',
        'scaler_path': 'models/feature_scaler.pkl',
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Config saved: {CONFIG_PATH}")
    logger.info(f"Pool models saved: {POOL_DIR}")


if __name__ == '__main__':
    main()
