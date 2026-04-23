"""V11 グリッド探索: 15変種を一括訓練

軸:
  weight_preset × kelly_fraction × filter_profile
  ※ feature軸は90dim固定（V10の76dim再現は実装コスト高のため別途）

出力:
  analysis/models_v11/grid/{variant_name}/{1st,2nd,3rd}.txt
  analysis/reports/05_grid_training.json
"""
import os
import sys
import json
import pickle
import math
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models_v11"
GRID_DIR = MODELS_DIR / "grid"
GRID_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# === Weight プリセット ===
WEIGHT_PRESETS = {
    'none':    {'weak_venue': 1.0, 'r1': 1.0, 'weak_wind': 1.0, 'strong_venue': 1.0, 'strong_wind': 1.0},
    'mild':    {'weak_venue': 0.7, 'r1': 0.5, 'weak_wind': 1.2, 'strong_venue': 1.3, 'strong_wind': 1.2},
    'strong':  {'weak_venue': 0.3, 'r1': 0.3, 'weak_wind': 1.8, 'strong_venue': 2.0, 'strong_wind': 1.5},
    'extreme': {'weak_venue': 0.1, 'r1': 0.0, 'weak_wind': 2.5, 'strong_venue': 3.0, 'strong_wind': 2.0},
}

# Miss Analysis 定義
WEAK_VENUES = [4, 3, 6, 5, 2, 1]
STRONG_VENUES = [12, 23, 9]
WEAK_R_NUMBER = 1
WEAK_WIND_RANGE = (0, 2)
STRONG_WIND_RANGE = (3, 5)


def build_weights(races_info, preset_name):
    """レース情報から preset に従って sample_weight を計算"""
    preset = WEIGHT_PRESETS[preset_name]
    weights = []
    for r in races_info:
        w = 1.0
        vid = r['venue_id']
        if vid in WEAK_VENUES:
            w *= preset['weak_venue']
        elif vid in STRONG_VENUES:
            w *= preset['strong_venue']

        if r['race_number'] == WEAK_R_NUMBER:
            w *= preset['r1']

        wind = r.get('wind_speed')
        if wind is not None:
            if STRONG_WIND_RANGE[0] <= wind <= STRONG_WIND_RANGE[1]:
                w *= preset['strong_wind']
            elif WEAK_WIND_RANGE[0] <= wind <= WEAK_WIND_RANGE[1]:
                w *= preset['weak_wind']

        # 配当加重（全プリセット共通: profit最適化の基礎）
        payout = r.get('payout_sanrentan') or 0
        if payout > 0:
            w *= math.log1p(float(payout) / 1000.0)

        weights.append(max(w, 0.01))
    return np.array(weights, dtype=np.float32)


# === 訓練ターゲット変種（15個）===
VARIANTS = [
    # === baseline ===
    {'name': 'v11_baseline', 'weight_preset': 'none', 'num_leaves': 63, 'learning_rate': 0.05},

    # === weight軸（kelly=0.0625, filter=V10 は共通）===
    {'name': 'v11_w_mild',    'weight_preset': 'mild',    'num_leaves': 63, 'learning_rate': 0.05},
    {'name': 'v11_w_strong',  'weight_preset': 'strong',  'num_leaves': 63, 'learning_rate': 0.05},
    {'name': 'v11_w_extreme', 'weight_preset': 'extreme', 'num_leaves': 63, 'learning_rate': 0.05},

    # === LightGBM hyperparams軸（weight_strong固定で複雑度を変える）===
    {'name': 'v11_hp_shallow', 'weight_preset': 'strong', 'num_leaves': 31,  'learning_rate': 0.05},
    {'name': 'v11_hp_deep',    'weight_preset': 'strong', 'num_leaves': 127, 'learning_rate': 0.05},
    {'name': 'v11_hp_fast_lr', 'weight_preset': 'strong', 'num_leaves': 63,  'learning_rate': 0.1 },
    {'name': 'v11_hp_slow_lr', 'weight_preset': 'strong', 'num_leaves': 63,  'learning_rate': 0.02},

    # === 組合せ（有望そうな組）===
    {'name': 'v11_w_mild_deep',     'weight_preset': 'mild',    'num_leaves': 127, 'learning_rate': 0.05},
    {'name': 'v11_w_mild_fast',     'weight_preset': 'mild',    'num_leaves': 63,  'learning_rate': 0.1 },
    {'name': 'v11_w_strong_shallow','weight_preset': 'strong',  'num_leaves': 31,  'learning_rate': 0.05},
    {'name': 'v11_w_extreme_deep',  'weight_preset': 'extreme', 'num_leaves': 127, 'learning_rate': 0.05},

    # === 正則化強めバリエーション ===
    {'name': 'v11_reg_strong',  'weight_preset': 'strong', 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 50},
    {'name': 'v11_reg_extreme', 'weight_preset': 'strong', 'num_leaves': 15, 'learning_rate': 0.05, 'min_child_samples': 100},

    # === 重み付けなし + 複雑モデル（モデル能力限界テスト） ===
    {'name': 'v11_raw_deep', 'weight_preset': 'none', 'num_leaves': 127, 'learning_rate': 0.05},
]


def train_one_variant(variant, data, race_info_list):
    """1変種訓練"""
    name = variant['name']
    logger.info(f"\n=== {name} ===")

    # weight構築
    weights = build_weights(race_info_list, variant['weight_preset'])

    X = data['X']
    y_1st = data['y_1st']
    y_2nd = data['y_2nd']
    y_3rd = data['y_3rd']

    # 時系列80/20
    n = len(X)
    split = int(n * 0.8)

    X_tr, X_val = X[:split], X[split:]
    w_tr, w_val = weights[:split], weights[split:]

    variant_dir = GRID_DIR / name
    variant_dir.mkdir(exist_ok=True)

    # LightGBM params
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'learning_rate': variant.get('learning_rate', 0.05),
        'num_leaves': variant.get('num_leaves', 63),
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': variant.get('min_child_samples', 20),
        'verbose': -1,
        'seed': 42,
    }

    results = {}
    for pos, y_all in [('1st', y_1st), ('2nd', y_2nd), ('3rd', y_3rd)]:
        y_tr, y_val = y_all[:split], y_all[split:]
        train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        val_ds = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_ds)
        model = lgb.train(
            params, train_ds, num_boost_round=500,
            valid_sets=[val_ds], valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        probs = model.predict(X_val, num_iteration=model.best_iteration)
        pred = np.argmax(probs, axis=1)
        acc = float((pred == y_val).mean())
        model.save_model(str(variant_dir / f"{pos}.txt"))
        results[pos] = {'best_iter': model.best_iteration, 'val_acc': acc}

    logger.info(f"  1st={results['1st']['val_acc']*100:.1f}% "
                f"2nd={results['2nd']['val_acc']*100:.1f}% "
                f"3rd={results['3rd']['val_acc']*100:.1f}%")
    return results


def main():
    logger.info("V11 グリッド訓練 開始")
    # データロード
    with open(MODELS_DIR / "train_data.pkl", 'rb') as f:
        data = pickle.load(f)
    logger.info(f"学習データ: {len(data['X'])}件, {data['X'].shape[1]}次元")

    # race info を DB から再取得（weight構築用のraw情報）
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from dotenv import load_dotenv
    load_dotenv()
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    race_ids = [int(r) for r in data['race_ids']]
    cur.execute("""
        SELECT id, venue_id, race_number, wind_speed, wave_height, payout_sanrentan
        FROM races WHERE id = ANY(%s)
    """, (race_ids,))
    race_info_map = {r['id']: r for r in cur.fetchall()}
    race_info_list = [race_info_map[rid] for rid in race_ids]
    conn.close()
    logger.info(f"race_info取得: {len(race_info_list)}件")

    # 全変種訓練
    all_results = {}
    for variant in VARIANTS:
        try:
            r = train_one_variant(variant, data, race_info_list)
            all_results[variant['name']] = {
                'config': variant,
                'results': r,
                'status': 'success',
            }
        except Exception as e:
            logger.error(f"  {variant['name']} 失敗: {e}")
            all_results[variant['name']] = {
                'config': variant,
                'status': 'failed',
                'error': str(e),
            }

    # レポート保存
    report = {
        'trained_at': datetime.now().isoformat(),
        'n_variants': len(VARIANTS),
        'variants': all_results,
    }
    report_path = REPORT_DIR / "05_grid_training.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=== 全変種訓練 完了 ===")
    logger.info("変種                     1st    2nd    3rd")
    logger.info("-" * 55)
    for name, r in all_results.items():
        if r['status'] == 'success':
            res = r['results']
            logger.info(f"{name:26s} "
                        f"{res['1st']['val_acc']*100:5.2f}% "
                        f"{res['2nd']['val_acc']*100:5.2f}% "
                        f"{res['3rd']['val_acc']*100:5.2f}%")
    logger.info(f"\nレポート: {report_path}")


if __name__ == '__main__':
    main()
