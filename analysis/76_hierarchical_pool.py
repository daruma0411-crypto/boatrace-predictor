"""階層的 pool で訓練データ統合の効果検証 (Phase D 追加)

User の指摘: 「淡水 group から海水/汽水抜く」「同 group 内でも異質な venue 除外」
6 段階の purity level で pool を定義し、各 target で LightGBM 訓練 → hold-out 評価。

Target venues: 戸田(2), 平和島(4), 尼崎(13), 唐津(23)
Pool levels (per target):
  L1: 全国 24 venues (V10 相当)
  L2: 同水質 ALL
  L3: 同水質 strict (異質除外)
  L4: 66 距離 top-5 類似
  L5: 66 距離 top-3 類似
  L6: target 単独 (= 70 specialist)

各 (target, level) で LightGBM を訓練し、target test (2026-05) で QMC + ROI 評価。
73/74/75 の現状 best と比較し、Pool 路線 vs Recipe Blend 路線の優位性を判定。

出力: analysis/reports/76_hierarchical_pool.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '76_hierarchical_pool.md'

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
WATER = {  # 1=淡水, 2=海水, 3=汽水
    1:1, 2:1, 3:3, 4:2, 5:1, 6:3, 7:3, 8:2, 9:3, 10:1, 11:1, 12:1,
    13:1, 14:2, 15:2, 16:2, 17:2, 18:2, 19:2, 20:2, 21:1, 22:2, 23:1, 24:2,
}
WATER_NAMES = {1: '淡水', 2: '海水', 3: '汽水'}
ALL_VENUES = list(VENUE_NAMES.keys())

# 現状 best (73/74/75 の test ROI)
CURRENT_BEST = {
    2:  {'name': '65 own specialist alone', 'test_roi': -35.85},
    4:  {'name': '73 R_top5_sim_avg', 'test_roi': -1.59},
    13: {'name': '75 K2_own1.0_sub_opp3x0.3', 'test_roi': +24.46},
    23: {'name': '73 R_top5_sim_avg', 'test_roi': +50.98},
}

# Pool 定義 (target venue ごとに 6 段階)
# 戸田 (淡水・狭水面・横風・1号艇 43%)
# 同水質 strict: 淡水 minus 芦屋(1号艇強)・住之江(屋根)・多摩川(静水)
POOLS = {
    2: {  # 戸田
        'water': 1,
        'L3_strict': [1, 2, 10, 11, 13, 23],  # 淡水 minus 芦屋(21)、住之江(12)、多摩川(5)
    },
    4: {  # 平和島
        'water': 2,
        'L3_strict': [4, 14, 16, 17, 22, 20, 8, 15],  # 海水 minus 大村(24)、徳山(18)、下関(19)
    },
    13: {  # 尼崎
        'water': 1,
        'L3_strict': [13, 5, 12, 21, 23],  # 淡水 strict: 静水寄り (戸田・桐生・三国の風強系を除外)
    },
    23: {  # 唐津
        'water': 1,
        'L3_strict': [23, 12, 13, 21, 5, 11],  # 淡水 strict: 静水寄り
    },
}


def load_venue_preds():
    return pickle.load(open(VENUE_PRED_PATH, 'rb'))


def get_distances(venue_preds):
    feats = {}
    for vid, preds in venue_preds.items():
        if not preds:
            continue
        b1 = np.zeros(6)
        for p in preds.values():
            b1[p['result_1st'] - 1] += 1
        feats[vid] = b1 / len(preds) * 100
    out = {}
    for v1 in feats:
        out[v1] = sorted(
            [(v2, float(np.linalg.norm(feats[v1] - feats[v2]))) for v2 in feats if v2 != v1],
            key=lambda x: x[1]
        )
    return out


def build_all_records(venue_preds, fe, scaler):
    """全 venue の records (features + label + date) を生成."""
    by_venue = {}
    for vid, preds in venue_preds.items():
        records = []
        for rid, p in preds.items():
            try:
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                records.append({
                    'rid': rid, 'venue_id': vid,
                    'features': features,
                    'y1': p['result_1st'] - 1,
                    'y2': p['result_2nd'] - 1,
                    'y3': p['result_3rd'] - 1,
                    'date': date.fromisoformat(p['race_date']),
                    'prediction': p,
                })
            except Exception:
                continue
        by_venue[vid] = records
    return by_venue


def gather_pool_data(target_vid, pool_venues, all_records):
    """pool data: 各 venue から train+val 期間 (2026-04-30 以前) を pool として集約。
    target は train (2026-03-31 以前) のみ採用 (target の val は target val 評価用)."""
    pool_train = []
    target_val = []
    target_test = []
    for vid in pool_venues:
        recs = all_records.get(vid, [])
        if vid == target_vid:
            for r in recs:
                if r['date'] < VAL_START:
                    pool_train.append(r)
                elif r['date'] < TEST_START:
                    target_val.append(r)
                else:
                    target_test.append(r)
        else:
            for r in recs:
                if r['date'] < TEST_START:
                    pool_train.append(r)
                # 他 venue の test 期間 data は混入させない (時間整合)
    return pool_train, target_val, target_test


def train_lgb_pool(pool_train, target_val):
    """pool data で LightGBM 訓練、target val で early stopping."""
    if not pool_train:
        return None
    Xtr = np.array([r['features'] for r in pool_train], dtype=np.float32)
    y1tr = np.array([r['y1'] for r in pool_train], dtype=np.int32)
    Xv = np.array([r['features'] for r in target_val], dtype=np.float32)
    y1v = np.array([r['y1'] for r in target_val], dtype=np.int32)
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


def qmc_backtest(probs_list, target_test):
    n_top1 = n_top3 = 0
    invested = returned = 0
    for probs, r in zip(probs_list, target_test):
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
    n = len(target_test)
    return {'n': n, 'top1_rate': n_top1/n*100 if n else 0, 'top3_rate': n_top3/n*100 if n else 0,
            'invested': invested, 'returned': returned,
            'pnl': returned-invested, 'roi': (returned-invested)/invested*100 if invested else 0}


def main():
    logger.info("階層的 pool 検証 (Phase D 追加)")
    venue_preds = load_venue_preds()
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    logger.info("Features pre-compute (全 24 venues)")
    all_records = build_all_records(venue_preds, fe, scaler)
    distances = get_distances(venue_preds)

    results = {}
    for target_vid in POOLS.keys():
        target_name = VENUE_NAMES[target_vid]
        water_id = POOLS[target_vid]['water']
        water_venues = [v for v in ALL_VENUES if WATER[v] == water_id]
        l3_strict = POOLS[target_vid]['L3_strict']
        top5_sim = [v for v, _ in distances[target_vid][:5]] + [target_vid]
        top3_sim = [v for v, _ in distances[target_vid][:3]] + [target_vid]
        # dedup
        top5_sim = list(set(top5_sim))
        top3_sim = list(set(top3_sim))

        levels = [
            ('L1_全国24', ALL_VENUES),
            (f'L2_同{WATER_NAMES[water_id]}_ALL', water_venues),
            (f'L3_同{WATER_NAMES[water_id]}_strict', l3_strict),
            ('L4_類似top5', top5_sim),
            ('L5_類似top3', top3_sim),
            ('L6_target単独', [target_vid]),
        ]

        target_records = all_records.get(target_vid, [])
        # 共通 val / test を最初に確保
        target_test = [r for r in target_records if r['date'] >= TEST_START]
        target_val = [r for r in target_records if VAL_START <= r['date'] < TEST_START]
        if len(target_test) < 30 or len(target_val) < 30:
            continue

        logger.info(f"\n=== Target {target_vid} ({target_name}) val={len(target_val)}, test={len(target_test)} ===")
        target_results = []
        for level_name, pool_venues in levels:
            pool_train, _, _ = gather_pool_data(target_vid, pool_venues, all_records)
            model = train_lgb_pool(pool_train, target_val)
            if not model:
                continue
            # Test prediction
            Xte = np.array([r['features'] for r in target_test], dtype=np.float32)
            p1 = model.predict(Xte, num_iteration=model.best_iteration)
            # QMC backtest
            qmc_result = qmc_backtest(p1, target_test)
            target_results.append({
                'level': level_name,
                'n_pool_train': len(pool_train),
                'pool_venues': pool_venues,
                'best_iter': model.best_iteration,
                'val_logloss': float(model.best_score['val']['multi_logloss']),
                'test_top3_rate': qmc_result['top3_rate'],
                'test_roi': qmc_result['roi'],
                'pnl': qmc_result['pnl'],
            })
            logger.info(f"  {level_name}: pool_n={len(pool_train)}, val_logloss={model.best_score['val']['multi_logloss']:.4f}, test_roi={qmc_result['roi']:+.2f}%")

        results[target_vid] = {
            'name': target_name,
            'water': WATER_NAMES[water_id],
            'n_val': len(target_val), 'n_test': len(target_test),
            'levels': target_results,
        }

    # Report
    lines = []
    lines.append("# 階層的 pool 検証 (Phase D 追加)\n\n")
    lines.append("「淡水 group から海水/汽水抜く」「同 group 内でも異質除外」を階層的に検証。\n")
    lines.append("各 (target × level) で LightGBM 訓練 → target test (2026-05) hold-out 評価。\n\n")

    for target_vid in POOLS.keys():
        r = results.get(target_vid)
        if not r:
            continue
        cb = CURRENT_BEST.get(target_vid, {})
        cb_roi = cb.get('test_roi', 0)
        lines.append(f"\n## venue {target_vid} ({r['name']}, {r['water']})\n\n")
        lines.append(f"val: {r['n_val']} races, test: {r['n_test']} races\n")
        lines.append(f"現状 best ({cb.get('name', 'N/A')}): test ROI {cb_roi:+.2f}%\n\n")
        lines.append("### Pool レベル別結果\n\n")
        lines.append("| level | pool venues | pool n | best iter | val logloss | test top-3% | test ROI | 現状比 |\n|---|---|---|---|---|---|---|---|\n")
        for lvl in r['levels']:
            diff = lvl['test_roi'] - cb_roi
            flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
            pv = ','.join(f'v{v}' for v in lvl['pool_venues'][:6])
            if len(lvl['pool_venues']) > 6:
                pv += f',+{len(lvl["pool_venues"])-6}'
            lines.append(f"| {lvl['level']} | {pv} | {lvl['n_pool_train']:,} | {lvl['best_iter']} | "
                         f"{lvl['val_logloss']:.4f} | {lvl['test_top3_rate']:.2f}% | "
                         f"{lvl['test_roi']:+.2f}% | **{diff:+.2f}pt {flag}** |\n")

        # 最良 level
        best_level = max(r['levels'], key=lambda x: x['test_roi'])
        lines.append(f"\n**Pool best**: {best_level['level']} (test ROI {best_level['test_roi']:+.2f}%, 現状比 {best_level['test_roi']-cb_roi:+.2f}pt)\n")

    # サマリ: Pool vs Recipe
    lines.append("\n## 全体: Pool 路線 vs Recipe Blend 路線\n\n")
    lines.append("| venue | name | Recipe best (現状) | Pool best | 勝者 |\n|---|---|---|---|---|\n")
    pool_wins = 0
    recipe_wins = 0
    for target_vid in POOLS.keys():
        r = results.get(target_vid)
        if not r or not r['levels']:
            continue
        cb_roi = CURRENT_BEST.get(target_vid, {}).get('test_roi', 0)
        best_level = max(r['levels'], key=lambda x: x['test_roi'])
        pool_roi = best_level['test_roi']
        if pool_roi > cb_roi:
            winner = f"Pool ({best_level['level']})"
            pool_wins += 1
        else:
            winner = "Recipe"
            recipe_wins += 1
        lines.append(f"| {target_vid} | {r['name']} | {cb_roi:+.2f}% | {pool_roi:+.2f}% | **{winner}** |\n")
    lines.append(f"\n- Pool 路線勝ち: {pool_wins} venues\n")
    lines.append(f"- Recipe 路線勝ち: {recipe_wins} venues\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- target val/test 分離維持 (target val=2026-04, target test=2026-05)\n")
    lines.append("- pool 他 venue の data は train+val 期間 (2026-04-30 以前) のみ使用\n")
    lines.append("- L1 全国 ≠ V10 完全相当 (V10 は 34,712 races で訓練、L1 は ~50k で再訓練)\n")
    lines.append("- 採用候補は shadow 並走 2 週間で forward 検証必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
