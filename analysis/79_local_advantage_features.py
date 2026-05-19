"""Local advantage feature 拡張 specialists 再訓練 (Path A)

78 で未取り込み edge を data 化:
  - 全体 Q5 帯 (local rate >50) で V10 1号艇 7.59pt 過小評価
  - 5 venues で bias 差 >10pt
  - B1+B2+B3 で local advantage +3-4.5pt 効果

→ 新規 6 features を追加し 24 specialists を 82dim で再訓練:
  local_advantage_B{1-6} = (local_win_rate_2 - win_rate_2) / 100

各 venue で:
  1. 拡張 specialist の test ROI (own venue 単独)
  2. 76dim original specialist (70) との比較
  3. 改善幅 算出

評価指標:
  V10 baseline → 76dim specialist (70) → 82dim specialist (79) の累積改善
  さらに Pool / Recipe approach にも拡張 features 投入可能 (後回し)

出力: models/specialists_82/lightgbm_v??_1st.txt × 24
      analysis/reports/79_local_advantage.md
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
SPEC_OLD = ROOT / 'models' / 'specialists'
SPEC_NEW = ROOT / 'models' / 'specialists_82'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '79_local_advantage.md'

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
FUNCTIONAL = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24]


def build_features_82(p, fe, scaler):
    """76dim (scaled) + 6dim local advantage = 82dim"""
    features = fe.transform(p['race_data'], p['boats'])
    features = scaler.transform(features.reshape(1, -1)).flatten()
    local_adv = []
    for b in p['boats']:
        lr = b.get('local_win_rate_2', 0) or 0
        gr = b.get('win_rate_2', 0) or 0
        local_adv.append((lr - gr) / 100.0)
    return np.concatenate([features, np.array(local_adv, dtype=np.float32)])


def build_records(venue_preds, fe, scaler):
    by_venue = {}
    for vid, preds in venue_preds.items():
        records = []
        for rid, p in preds.items():
            try:
                features_82 = build_features_82(p, fe, scaler)
                records.append({
                    'rid': rid, 'venue_id': vid,
                    'features_82': features_82,
                    'y1': p['result_1st'] - 1,
                    'date': date.fromisoformat(p['race_date']),
                    'prediction': p,
                })
            except Exception:
                continue
        by_venue[vid] = records
    return by_venue


def train_specialist_82(train_records, val_records):
    Xtr = np.array([r['features_82'] for r in train_records], dtype=np.float32)
    y1tr = np.array([r['y1'] for r in train_records], dtype=np.int32)
    Xv = np.array([r['features_82'] for r in val_records], dtype=np.float32)
    y1v = np.array([r['y1'] for r in val_records], dtype=np.int32)
    params = {
        'objective': 'multiclass', 'num_class': 6, 'metric': 'multi_logloss',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 20, 'lambda_l2': 1.0, 'verbose': -1, 'seed': SEED,
    }
    lgb_train = lgb.Dataset(Xtr, y1tr)
    lgb_val = lgb.Dataset(Xv, y1v, reference=lgb_train)
    return lgb.train(params, lgb_train,
                     valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'],
                     num_boost_round=500,
                     callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)])


def qmc_backtest(probs_list, target_test):
    n_top1 = n_top3 = 0
    invested = returned = 0
    for probs, r in zip(probs_list, target_test):
        p = r['prediction']
        try:
            qp = qmc_sanrentan_v3(
                probs.tolist() if hasattr(probs, 'tolist') else list(probs),
                boats_data=p['boats'], race_data=p['race_data'],
                race_number=p['race_number'], n_simulations=N_SIM, seed=SEED,
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
    n = len(target_test)
    return {'n': n, 'top1_rate': n_top1/n*100 if n else 0, 'top3_rate': n_top3/n*100 if n else 0,
            'roi': (returned-invested)/invested*100 if invested else 0}


def main():
    logger.info("Path A: 82dim 拡張 specialists 再訓練")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    logger.info("82dim features pre-compute")
    by_venue = build_records(venue_preds, fe, scaler)
    SPEC_NEW.mkdir(parents=True, exist_ok=True)

    # 訓練: 全 24 venues で specialist (82dim) 訓練
    new_specialists = {}
    for vid in VENUE_NAMES:
        records = by_venue.get(vid, [])
        train = [r for r in records if r['date'] < VAL_START]
        val   = [r for r in records if VAL_START <= r['date'] < TEST_START]
        if len(train) < 100 or len(val) < 30:
            logger.warning(f"venue {vid} skip (train={len(train)}, val={len(val)})")
            continue
        logger.info(f"venue {vid} ({VENUE_NAMES[vid]}): train={len(train)}, val={len(val)}")
        model = train_specialist_82(train, val)
        path = SPEC_NEW / f'lightgbm_v{vid:02d}_1st.txt'
        model.save_model(str(path))
        new_specialists[vid] = model

    # 評価: 13 functional venues で test 評価
    results = {}
    # 76dim original specialists (70 の戸田単独 ROI 比較用)
    old_specialists = {vid: lgb.Booster(model_file=str(SPEC_OLD / f'lightgbm_v{vid:02d}_1st.txt'))
                       for vid in FUNCTIONAL if (SPEC_OLD / f'lightgbm_v{vid:02d}_1st.txt').exists()}

    for vid in FUNCTIONAL:
        records = by_venue.get(vid, [])
        test = [r for r in records if r['date'] >= TEST_START]
        if len(test) < 30:
            continue
        Xte_82 = np.array([r['features_82'] for r in test], dtype=np.float32)
        Xte_76 = Xte_82[:, :76]  # 76dim 部分のみ抽出 (old specialist 用)

        # V10 baseline
        v10_probs = np.array([r['prediction']['probs_1st'] for r in test])
        v10_qmc = qmc_backtest(v10_probs, test)

        # 76dim old specialist
        old_qmc = None
        if vid in old_specialists:
            p1_old = old_specialists[vid].predict(Xte_76, num_iteration=old_specialists[vid].best_iteration)
            old_qmc = qmc_backtest(p1_old, test)

        # 82dim new specialist
        p1_new = new_specialists[vid].predict(Xte_82, num_iteration=new_specialists[vid].best_iteration)
        new_qmc = qmc_backtest(p1_new, test)

        # feature importance (top 15) の中に local advantage features があるか
        importance = new_specialists[vid].feature_importance(importance_type='gain')
        feature_names = [f'orig_{i}' for i in range(76)] + [f'local_adv_B{i+1}' for i in range(6)]
        top_features = sorted(enumerate(importance), key=lambda x: -x[1])[:15]
        local_adv_in_top15 = [feature_names[i] for i, _ in top_features if 'local_adv' in feature_names[i]]

        results[vid] = {
            'name': VENUE_NAMES[vid],
            'n_test': len(test),
            'v10_roi': v10_qmc['roi'],
            'old_76dim_roi': old_qmc['roi'] if old_qmc else None,
            'new_82dim_roi': new_qmc['roi'],
            'local_adv_in_top15': local_adv_in_top15,
            'best_iter': new_specialists[vid].best_iteration,
            'val_logloss': float(new_specialists[vid].best_score['val']['multi_logloss']),
        }
        logger.info(f"  venue {vid} V10={v10_qmc['roi']:+.2f}%, 76dim={old_qmc['roi'] if old_qmc else 0:+.2f}%, 82dim={new_qmc['roi']:+.2f}%")

    # Report
    lines = []
    lines.append("# Local advantage feature 拡張 specialists (Path A)\n\n")
    lines.append("76dim → 82dim (`local_advantage_B{1-6}` 追加)。24 specialists を再訓練。\n")
    lines.append("Test (2026-05) で 76dim vs 82dim で比較。\n\n")

    lines.append("## 13 functional venues 結果\n\n")
    lines.append("| venue | name | n | V10 ROI | 76dim ROI | **82dim ROI** | 82-76 改善 | local_adv top15? | val_logloss |\n|---|---|---|---|---|---|---|---|---|\n")
    diffs = []
    for vid in FUNCTIONAL:
        r = results.get(vid)
        if not r:
            continue
        diff = r['new_82dim_roi'] - (r['old_76dim_roi'] or 0)
        diffs.append(diff)
        flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
        old_str = f"{r['old_76dim_roi']:+.2f}%" if r['old_76dim_roi'] is not None else 'N/A'
        local_in_top = ','.join(r['local_adv_in_top15']) if r['local_adv_in_top15'] else 'なし'
        lines.append(f"| {vid} | {r['name']} | {r['n_test']} | {r['v10_roi']:+.2f}% | "
                     f"{old_str} | **{r['new_82dim_roi']:+.2f}%** | **{diff:+.2f}pt {flag}** | "
                     f"{local_in_top} | {r['val_logloss']:.4f} |\n")

    if diffs:
        n_strong = sum(1 for d in diffs if d > 5)
        n_mid = sum(1 for d in diffs if 0 < d <= 5)
        n_worse = sum(1 for d in diffs if d <= 0)
        lines.append(f"\n## 統計 (82dim vs 76dim)\n\n")
        lines.append(f"- 🟢 +5pt 以上改善: **{n_strong} venues**\n")
        lines.append(f"- 🟡 0〜+5pt: {n_mid} venues\n")
        lines.append(f"- 🔴 悪化: {n_worse} venues\n")
        lines.append(f"- 平均改善: {float(np.mean(diffs)):+.2f}pt\n")
        lines.append(f"- 中央値: {float(np.median(diffs)):+.2f}pt\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- 82dim features: 76dim (scaled) + 6dim local advantage (normalized /100)\n")
    lines.append("- 24 specialists 全て 82dim で再訓練、評価は 13 functional venues のみ\n")
    lines.append("- 次は Pool / Recipe approach も 82dim 化して再評価する余地\n")
    lines.append("- 採用候補は shadow 並走 2 週間で forward 検証必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
