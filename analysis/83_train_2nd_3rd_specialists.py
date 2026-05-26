"""V11.5: 24 venue 単独 LightGBM specialists for 2着 + 3着

V11 (1着 specialist のみ訓練 + 2/3着 V10 baseline) の 1号艇軸偏重 (88.9%) を解消するため、
2着・3着も venue 別 specialist 化して全 boat の prob を対称に上振れさせ、
正規化後の相対比を実 1着率 (59%) に近づける。

設計:
  既存 70_all_venues_specialists.py を完全踏襲 (76dim feature, multiclass 6, 同 split)
  target だけ変更: result_2nd-1 / result_3rd-1

出力:
  models/specialists/lightgbm_v??_2nd.txt × 24
  models/specialists/lightgbm_v??_3rd.txt × 24
  analysis/reports/83_2nd_3rd_specialists.md
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
PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
MODEL_DIR = ROOT / 'models' / 'specialists'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '83_2nd_3rd_specialists.md'

VAL_START = date(2026, 4, 1)
TEST_START = date(2026, 5, 1)
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
            r2 = p.get('result_2nd')
            r3 = p.get('result_3rd')
            if r2 is None or r3 is None:
                continue
            out.append({
                'rid': rid, 'X': features,
                'y1': p['result_1st'] - 1,
                'y2': r2 - 1,
                'y3': r3 - 1,
                'date': date.fromisoformat(p['race_date']),
                'prediction': p,
            })
        except Exception:
            continue
    return out


def train_specialist(train_data, val_data, y_key):
    Xtr = np.array([r['X'] for r in train_data], dtype=np.float32)
    ytr = np.array([r[y_key] for r in train_data], dtype=np.int32)
    Xv = np.array([r['X'] for r in val_data], dtype=np.float32)
    yv = np.array([r[y_key] for r in val_data], dtype=np.int32)
    params = {
        'objective': 'multiclass', 'num_class': 6, 'metric': 'multi_logloss',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 20, 'lambda_l2': 1.0,
        'verbose': -1, 'seed': SEED,
    }
    lgb_train = lgb.Dataset(Xtr, ytr)
    lgb_val = lgb.Dataset(Xv, yv, reference=lgb_train)
    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)],
    )
    return model


def evaluate_topk(model, test_data, y_key, k=3):
    Xte = np.array([r['X'] for r in test_data], dtype=np.float32)
    yte = np.array([r[y_key] for r in test_data], dtype=np.int32)
    probs = model.predict(Xte, num_iteration=model.best_iteration)
    top1_hits = 0
    topk_hits = 0
    for p, y in zip(probs, yte):
        order = np.argsort(-p)
        if order[0] == y:
            top1_hits += 1
        if y in order[:k]:
            topk_hits += 1
    n = len(yte)
    return {
        'n': n,
        'top1': top1_hits / n if n else 0,
        f'top{k}': topk_hits / n if n else 0,
    }


def main():
    logger.info("V11.5: 2着 + 3着 specialists 訓練開始")
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

        logger.info(f"--- venue {vid} ({venue_name}): tr={len(train)}, v={len(val)}, te={len(test)} ---")

        # 2着 specialist
        m2 = train_specialist(train, val, 'y2')
        m2_path = MODEL_DIR / f'lightgbm_v{vid:02d}_2nd.txt'
        m2.save_model(str(m2_path))
        eval2 = evaluate_topk(m2, test, 'y2', k=3)
        logger.info(f"  2着 specialist: best_iter={m2.best_iteration} val_ll={m2.best_score['val']['multi_logloss']:.4f} test top1={eval2['top1']:.3f} top3={eval2['top3']:.3f}")

        # 3着 specialist
        m3 = train_specialist(train, val, 'y3')
        m3_path = MODEL_DIR / f'lightgbm_v{vid:02d}_3rd.txt'
        m3.save_model(str(m3_path))
        eval3 = evaluate_topk(m3, test, 'y3', k=3)
        logger.info(f"  3着 specialist: best_iter={m3.best_iteration} val_ll={m3.best_score['val']['multi_logloss']:.4f} test top1={eval3['top1']:.3f} top3={eval3['top3']:.3f}")

        results.append({
            'venue': vid, 'name': venue_name,
            'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
            'm2_iter': m2.best_iteration,
            'm2_val_ll': float(m2.best_score['val']['multi_logloss']),
            'm2_test_top1': eval2['top1'], 'm2_test_top3': eval2['top3'],
            'm3_iter': m3.best_iteration,
            'm3_val_ll': float(m3.best_score['val']['multi_logloss']),
            'm3_test_top1': eval3['top1'], 'm3_test_top3': eval3['top3'],
        })

    # Report
    lines = []
    lines.append("# V11.5: 2着 + 3着 specialists training (Phase D 起点)\n\n")
    lines.append("V11 (1着 specialist のみ) の 1号艇軸偏重 (88.9%) 解消のため、\n")
    lines.append("2着・3着も venue 別 specialist 化。target 以外は 70 script と同設計。\n\n")
    lines.append("Train: 〜2026-03、Val: 2026-04 (early stopping)、Test: 2026-05\n\n")
    lines.append("## venue 別結果\n\n")
    lines.append("| venue | name | n_train | n_val | n_test | 2着 val_ll | 2着 test top1 | 2着 test top3 | 3着 val_ll | 3着 test top1 | 3着 test top3 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in results:
        lines.append(f"| {r['venue']} | {r['name']} | {r['n_train']} | {r['n_val']} | {r['n_test']} | "
                     f"{r['m2_val_ll']:.4f} | {r['m2_test_top1']:.3f} | {r['m2_test_top3']:.3f} | "
                     f"{r['m3_val_ll']:.4f} | {r['m3_test_top1']:.3f} | {r['m3_test_top3']:.3f} |\n")

    avg_2_top1 = float(np.mean([r['m2_test_top1'] for r in results]))
    avg_3_top1 = float(np.mean([r['m3_test_top1'] for r in results]))
    avg_2_top3 = float(np.mean([r['m2_test_top3'] for r in results]))
    avg_3_top3 = float(np.mean([r['m3_test_top3'] for r in results]))
    lines.append(f"\n## 集計\n\n")
    lines.append(f"- venue 数: {len(results)}\n")
    lines.append(f"- 平均 2着 top1: {avg_2_top1:.3f} (random 1/6 = 0.167)\n")
    lines.append(f"- 平均 2着 top3: {avg_2_top3:.3f} (random 3/6 = 0.500)\n")
    lines.append(f"- 平均 3着 top1: {avg_3_top1:.3f}\n")
    lines.append(f"- 平均 3着 top3: {avg_3_top3:.3f}\n")
    lines.append(f"\n## 次ステップ\n\n")
    lines.append(f"V11VAR13Predictor を V11.5 化し、venue 別 2着/3着 specialist で probs_2nd/3rd を出力。\n")
    lines.append(f"forward 2026-04 で V11 vs V11.5 picks 分布 + ROI 比較。\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(''.join(lines), encoding='utf-8')
    logger.info(f"Report saved: {REPORT_PATH}")
    logger.info(f"\n=== Summary ===")
    logger.info(f"venues trained: {len(results)}")
    logger.info(f"avg 2着 top1: {avg_2_top1:.3f}")
    logger.info(f"avg 3着 top1: {avg_3_top1:.3f}")


if __name__ == '__main__':
    main()
