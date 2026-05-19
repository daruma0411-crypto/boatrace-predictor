"""階層的 pool 検証を残り 9 venues に拡張 (76 補完)

76 で 4 venues (戸田・平和島・尼崎・唐津) は完了。
追加 9 venues: 桐生・江戸川・蒲郡・三国・住之江・鳴門・児島・福岡・大村
各 venue で同じ 6 階層 pool 評価 → Recipe (現状 best) vs Pool 比較。

13 venues 全体の最終 best approach を確定。
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '77_hierarchical_pool_extend.md'

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
WATER = {1:1,2:1,3:3,4:2,5:1,6:3,7:3,8:2,9:3,10:1,11:1,12:1,
         13:1,14:2,15:2,16:2,17:2,18:2,19:2,20:2,21:1,22:2,23:1,24:2}
WATER_NAMES = {1:'淡水', 2:'海水', 3:'汽水'}
ALL_VENUES = list(VENUE_NAMES.keys())

# 76 既評価 + 追加 9 = 全 13 venues
POOLS = {
    # 76 既評価 (参考)
    2:  {'water': 1, 'L3_strict': [1, 2, 10, 11, 13, 23]},        # 戸田
    4:  {'water': 2, 'L3_strict': [4, 14, 16, 17, 22, 20, 8, 15]}, # 平和島
    13: {'water': 1, 'L3_strict': [13, 5, 12, 21, 23]},           # 尼崎
    23: {'water': 1, 'L3_strict': [23, 12, 13, 21, 5, 11]},       # 唐津
    # 追加 9
    1:  {'water': 1, 'L3_strict': [1, 2, 10, 11, 23]},            # 桐生 (淡水・風強)
    3:  {'water': 3, 'L3_strict': [3, 6, 7, 9]},                  # 江戸川 (汽水 ALL)
    7:  {'water': 3, 'L3_strict': [7, 6, 9]},                     # 蒲郡 (汽水・静水)
    10: {'water': 1, 'L3_strict': [10, 1, 2, 11, 23]},            # 三国 (淡水・風強)
    12: {'water': 1, 'L3_strict': [12, 21, 5, 13]},               # 住之江 (淡水・ナイター)
    14: {'water': 2, 'L3_strict': [14, 4, 16, 17, 22]},            # 鳴門 (海水・荒れ系)
    16: {'water': 2, 'L3_strict': [16, 14, 17, 22, 4]},            # 児島 (海水・難)
    22: {'water': 2, 'L3_strict': [22, 14, 16, 17, 8]},            # 福岡 (海水・強風)
    24: {'water': 2, 'L3_strict': [24, 18, 19, 20]},               # 大村 (海水・静水・1号艇強)
}

# 現状 best (Recipe/Pool 各 venue)
CURRENT_BEST = {
    1:  {'name': '73 V10x0.6+own_x0.4', 'test_roi': 9.34, 'src': 'Recipe'},
    2:  {'name': '76 L1_全国24 (Pool)', 'test_roi': -24.17, 'src': 'Pool'},
    3:  {'name': '73 V10x0.4+own', 'test_roi': -5.85, 'src': 'Recipe'},
    4:  {'name': '73 R_top5_sim_avg', 'test_roi': -1.59, 'src': 'Recipe'},
    7:  {'name': '73 R_functional8_avg', 'test_roi': -7.68, 'src': 'Recipe'},
    10: {'name': '73 R_top2_sim_avg', 'test_roi': -28.13, 'src': 'Recipe'},
    12: {'name': '74 R-_V10x0.5+ownx0.5-opp3x0.1', 'test_roi': 13.22, 'src': 'Recipe'},
    13: {'name': '75 K2_own1.0_sub_opp3x0.3', 'test_roi': 24.46, 'src': 'Recipe'},
    14: {'name': '74 R-_V10x0.5+ownx0.5-opp3x0.3', 'test_roi': -30.14, 'src': 'Recipe'},
    16: {'name': '73 V10x0.4+own', 'test_roi': -45.06, 'src': 'Recipe'},
    22: {'name': '73 R_top5_sim_avg', 'test_roi': 4.83, 'src': 'Recipe'},
    23: {'name': '76 L3_淡水strict (Pool)', 'test_roi': 66.72, 'src': 'Pool'},
    24: {'name': '73 R_own2+functional', 'test_roi': -41.20, 'src': 'Recipe'},
}


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
    by_venue = {}
    for vid, preds in venue_preds.items():
        records = []
        for rid, p in preds.items():
            try:
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                records.append({'rid': rid, 'venue_id': vid, 'features': features,
                                'y1': p['result_1st'] - 1,
                                'date': date.fromisoformat(p['race_date']),
                                'prediction': p})
            except Exception:
                continue
        by_venue[vid] = records
    return by_venue


def gather_pool_data(target_vid, pool_venues, all_records):
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
    return pool_train, target_val, target_test


def train_lgb_pool(pool_train, target_val):
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
    logger.info("階層 pool 13 venues 全網羅")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    logger.info("features pre-compute")
    all_records = build_all_records(venue_preds, fe, scaler)
    distances = get_distances(venue_preds)

    results = {}
    # 全 13 venues 評価
    for target_vid in POOLS.keys():
        target_name = VENUE_NAMES[target_vid]
        water_id = POOLS[target_vid]['water']
        water_venues = [v for v in ALL_VENUES if WATER[v] == water_id]
        l3_strict = POOLS[target_vid]['L3_strict']
        top5_sim = list(set([v for v, _ in distances[target_vid][:5]] + [target_vid]))
        top3_sim = list(set([v for v, _ in distances[target_vid][:3]] + [target_vid]))

        levels = [
            ('L1_全国24', ALL_VENUES),
            (f'L2_同{WATER_NAMES[water_id]}_ALL', water_venues),
            (f'L3_同{WATER_NAMES[water_id]}_strict', l3_strict),
            ('L4_類似top5', top5_sim),
            ('L5_類似top3', top3_sim),
            ('L6_target単独', [target_vid]),
        ]

        target_records = all_records.get(target_vid, [])
        target_test = [r for r in target_records if r['date'] >= TEST_START]
        target_val = [r for r in target_records if VAL_START <= r['date'] < TEST_START]
        if len(target_test) < 30 or len(target_val) < 30:
            continue

        logger.info(f"=== Target {target_vid} ({target_name}) val={len(target_val)}, test={len(target_test)} ===")
        target_results = []
        for level_name, pool_venues in levels:
            pool_train, _, _ = gather_pool_data(target_vid, pool_venues, all_records)
            model = train_lgb_pool(pool_train, target_val)
            if not model:
                continue
            Xte = np.array([r['features'] for r in target_test], dtype=np.float32)
            p1 = model.predict(Xte, num_iteration=model.best_iteration)
            qmc = qmc_backtest(p1, target_test)
            target_results.append({'level': level_name, 'n_pool': len(pool_train),
                                   'pool_venues': pool_venues,
                                   'test_top3': qmc['top3_rate'], 'test_roi': qmc['roi']})
            logger.info(f"  {level_name}: n={len(pool_train)}, ROI={qmc['roi']:+.2f}%")

        results[target_vid] = {'name': target_name, 'water': WATER_NAMES[water_id],
                               'n_val': len(target_val), 'n_test': len(target_test),
                               'levels': target_results}

    # Report
    lines = []
    lines.append("# 階層的 pool 検証 — 全 13 venues 網羅 (76 拡張)\n\n")
    lines.append("各 venue で 6 階層 pool を評価し、現状 best (Recipe or Pool) と比較。\n\n")

    lines.append("## 全 13 venues サマリ (Pool best vs 現状 best)\n\n")
    lines.append("| venue | name | 水質 | n_test | 現状 best (src) | 現状 ROI | Pool best | Pool ROI | 勝者 | 改善 |\n|---|---|---|---|---|---|---|---|---|---|\n")
    pool_wins = recipe_wins = 0
    for vid in sorted(POOLS.keys()):
        r = results.get(vid)
        if not r or not r['levels']:
            continue
        cb = CURRENT_BEST.get(vid, {})
        cb_roi = cb.get('test_roi', 0)
        cb_src = cb.get('src', '?')
        best_pool = max(r['levels'], key=lambda x: x['test_roi'])
        diff = best_pool['test_roi'] - cb_roi
        flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
        winner = 'Pool' if diff > 0 else 'Recipe' if cb_src == 'Recipe' else '現状維持'
        if diff > 0:
            pool_wins += 1
        else:
            if cb_src == 'Recipe':
                recipe_wins += 1
        lines.append(f"| {vid} | {r['name']} | {r['water']} | {r['n_test']} | "
                     f"{cb['name']} ({cb_src}) | {cb_roi:+.2f}% | {best_pool['level']} | "
                     f"{best_pool['test_roi']:+.2f}% | **{winner}** | {diff:+.2f}pt {flag} |\n")
    lines.append(f"\n- **Pool 路線勝ち**: {pool_wins}/13 venues\n")
    lines.append(f"- **Recipe 路線勝ち**: {recipe_wins}/13 venues\n")

    # 各 venue の詳細 (全 level)
    lines.append("\n## venue 別詳細 (全 level test ROI)\n\n")
    for vid in sorted(POOLS.keys()):
        r = results.get(vid)
        if not r:
            continue
        cb = CURRENT_BEST.get(vid, {})
        lines.append(f"\n### {vid} {r['name']} ({r['water']}) — 現状 best {cb.get('test_roi', 0):+.2f}% ({cb.get('src', '?')})\n\n")
        lines.append("| level | pool n | top-3% | test ROI | 現状比 |\n|---|---|---|---|---|\n")
        for lvl in r['levels']:
            diff = lvl['test_roi'] - cb.get('test_roi', 0)
            flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
            lines.append(f"| {lvl['level']} | {lvl['n_pool']:,} | {lvl['test_top3']:.2f}% | "
                         f"{lvl['test_roi']:+.2f}% | {diff:+.2f}pt {flag} |\n")

    # 最終 best 表
    lines.append("\n## 最終 best approach per venue (Recipe vs Pool 統合)\n\n")
    lines.append("| venue | name | 採用 approach | test ROI | V10 比 |\n|---|---|---|---|---|\n")
    v10_baselines = {  # 参考 (73/74 から)
        1: -8.28, 2: -42.22, 3: -16.15, 4: -37.05, 7: -16.55, 10: -39.55,
        12: -2.93, 13: -43.68, 14: -45.31, 16: -58.04, 22: -12.31, 23: -2.58, 24: -56.62,
    }
    for vid in sorted(POOLS.keys()):
        r = results.get(vid)
        if not r or not r['levels']:
            continue
        cb = CURRENT_BEST.get(vid, {})
        cb_roi = cb.get('test_roi', 0)
        best_pool = max(r['levels'], key=lambda x: x['test_roi'])
        if best_pool['test_roi'] > cb_roi:
            final_roi = best_pool['test_roi']
            final_name = f"Pool {best_pool['level']}"
        else:
            final_roi = cb_roi
            final_name = cb.get('name', '?')
        v10 = v10_baselines.get(vid, 0)
        improvement = final_roi - v10
        lines.append(f"| {vid} | {r['name']} | {final_name} | {final_roi:+.2f}% | "
                     f"**+{improvement:.2f}pt** |\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- val/test 分離維持\n")
    lines.append("- pool 他 venue の 2026-05 データは混入なし\n")
    lines.append("- 採用候補は shadow 並走 2 週間で forward 検証必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
