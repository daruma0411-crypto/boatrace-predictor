"""Shadow 並走 戦略 v1 設計 + 2026-05 hold-out 再現確認

13 venues の venue 別 best approach を統合した shadow strategy を設計。
2026-05 test set 上で「全 venue 統合 ROI」を算出し、各 venue best ROI が
忠実に再現されるか確認。

最終的に shadow_recipe_v1 として production に投入予定。

Strategy mapping (venue → best approach):
  venue 1 (桐生):    Recipe blend R+_V10x0.6+own_x0.4
  venue 2 (戸田):    82dim Specialist
  venue 3 (江戸川):  82dim Specialist
  venue 4 (平和島):  Recipe blend R+_top5_sim_avg
  venue 7 (蒲郡):    76dim Specialist (Pool L6)
  venue 10 (三国):   Recipe blend R+_top2_sim_avg
  venue 12 (住之江): 76dim Specialist (Pool L6)
  venue 13 (尼崎):   Recipe blend 75 K2_own1.0_sub_opp3x0.3
  venue 14 (鳴門):   Pool L3 海水 strict
  venue 16 (児島):   Pool L3 海水 strict
  venue 22 (福岡):   82dim Specialist
  venue 23 (唐津):   Pool L3 淡水 strict
  venue 24 (大村):   Recipe blend R+_own_x2+functional

非 13 venues (11 venues): V10 baseline 維持

出力:
  analysis/shadow_strategy_v1_config.json
  analysis/reports/80_shadow_design.md
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
from src.monte_carlo import qmc_sanrentan_v3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
VENUE_PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
SPEC_76 = ROOT / 'models' / 'specialists'
SPEC_82 = ROOT / 'models' / 'specialists_82'
CONFIG_PATH = ROOT / 'analysis' / 'shadow_strategy_v1_config.json'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '80_shadow_design.md'

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

# 全 venue 戦略 mapping
VENUE_STRATEGY = {
    1:  {'type': 'recipe', 'name': '73 R+_V10x0.6+own_x0.4',
         'weights': [('v10', 0.6), (1, 0.4)]},
    2:  {'type': 'specialist_82', 'venue': 2},
    3:  {'type': 'specialist_82', 'venue': 3},
    4:  {'type': 'recipe', 'name': '73 R+_top5_sim_avg',
         'weights_top5': True, 'target': 4},
    7:  {'type': 'specialist_76', 'venue': 7},
    10: {'type': 'recipe', 'name': '73 R+_top2_sim_avg',
         'weights_top2': True, 'target': 10},
    12: {'type': 'specialist_76', 'venue': 12},
    13: {'type': 'recipe_75', 'name': '75 K2_own1.0_sub_opp3x0.3',
         'target': 13, 'K': 2, 'own_w': 1.0, 'sub_alpha': 0.3},
    14: {'type': 'pool', 'level': 'L3_海水_strict', 'members': [14, 4, 16, 17, 22]},
    16: {'type': 'pool', 'level': 'L3_海水_strict', 'members': [16, 14, 17, 22, 4]},
    22: {'type': 'specialist_82', 'venue': 22},
    23: {'type': 'pool', 'level': 'L3_淡水_strict', 'members': [23, 12, 13, 21, 5, 11]},
    24: {'type': 'recipe', 'name': '73 R+_own_x2+functional',
         'target': 24, 'own_w': 2.0,
         'functional_others': [2, 3, 7, 12, 13, 14, 16]},
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


def build_features_82(p, fe, scaler):
    features = fe.transform(p['race_data'], p['boats'])
    features = scaler.transform(features.reshape(1, -1)).flatten()
    local_adv = []
    for b in p['boats']:
        lr = b.get('local_win_rate_2', 0) or 0
        gr = b.get('win_rate_2', 0) or 0
        local_adv.append((lr - gr) / 100.0)
    return np.concatenate([features, np.array(local_adv, dtype=np.float32)])


def train_pool_model(pool_train, target_val, dim):
    if not pool_train:
        return None
    Xtr = np.array([r['features_82'][:dim] for r in pool_train], dtype=np.float32)
    y1tr = np.array([r['y1'] for r in pool_train], dtype=np.int32)
    Xv = np.array([r['features_82'][:dim] for r in target_val], dtype=np.float32)
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


def predict_shadow(record, venue_id, all_specialists_76, all_specialists_82,
                   pool_models, distances):
    """venue 別 best approach で probs_1st を返す."""
    strategy = VENUE_STRATEGY.get(venue_id)
    if not strategy:
        # 非 functional venue: V10 baseline
        return np.array(record['prediction']['probs_1st'])

    features_82 = record['features_82']
    features_76 = features_82[:76]
    p = record['prediction']

    if strategy['type'] == 'specialist_76':
        spec = all_specialists_76.get(strategy['venue'])
        if spec:
            return spec.predict(features_76.reshape(1, -1),
                                num_iteration=spec.best_iteration)[0]
    elif strategy['type'] == 'specialist_82':
        spec = all_specialists_82.get(strategy['venue'])
        if spec:
            return spec.predict(features_82.reshape(1, -1),
                                num_iteration=spec.best_iteration)[0]
    elif strategy['type'] == 'pool':
        pool_model = pool_models.get((venue_id, strategy['level']))
        if pool_model:
            return pool_model.predict(features_76.reshape(1, -1),
                                       num_iteration=pool_model.best_iteration)[0]
    elif strategy['type'] == 'recipe':
        return apply_recipe(strategy, record, all_specialists_76, distances)
    elif strategy['type'] == 'recipe_75':
        return apply_recipe_75(strategy, record, all_specialists_76, distances)

    # fallback: V10
    return np.array(p['probs_1st'])


def apply_recipe(strategy, record, all_specialists_76, distances):
    """Recipe blend (足し算 / 引き算)."""
    probs = np.zeros(6)
    target = strategy.get('target')
    if strategy.get('weights_top5'):
        # target + top-5 類似 venue specialist 均等平均
        top5 = [v for v, _ in distances[target][:5]]
        members = [target] + top5
        for v in members:
            spec = all_specialists_76.get(v)
            if spec:
                p = spec.predict(record['features_82'][:76].reshape(1, -1),
                                 num_iteration=spec.best_iteration)[0]
                probs += p
        probs /= len(members)
        return probs
    if strategy.get('weights_top2'):
        top2 = [v for v, _ in distances[target][:2]]
        members = [target] + top2
        for v in members:
            spec = all_specialists_76.get(v)
            if spec:
                p = spec.predict(record['features_82'][:76].reshape(1, -1),
                                 num_iteration=spec.best_iteration)[0]
                probs += p
        probs /= len(members)
        return probs
    if 'functional_others' in strategy:
        # own × own_w + functional × 1
        own_w = strategy.get('own_w', 2.0)
        members_w = [(target, own_w)] + [(v, 1.0) for v in strategy['functional_others']]
        total_w = sum(w for _, w in members_w)
        for v, w in members_w:
            spec = all_specialists_76.get(v)
            if spec:
                p = spec.predict(record['features_82'][:76].reshape(1, -1),
                                 num_iteration=spec.best_iteration)[0]
                probs += (w / total_w) * p
        return probs
    if 'weights' in strategy:
        # explicit weights: (source, weight)
        total_w = sum(w for _, w in strategy['weights'])
        for source, w in strategy['weights']:
            wnorm = w / total_w
            if source == 'v10':
                probs += wnorm * np.array(record['prediction']['probs_1st'])
            else:
                spec = all_specialists_76.get(source)
                if spec:
                    p = spec.predict(record['features_82'][:76].reshape(1, -1),
                                     num_iteration=spec.best_iteration)[0]
                    probs += wnorm * p
        return probs
    return np.array(record['prediction']['probs_1st'])


def apply_recipe_75(strategy, record, all_specialists_76, distances):
    """75 K2_own1.0_sub_opp3x0.3 形式."""
    target = strategy['target']
    K = strategy['K']
    own_w = strategy['own_w']
    sub_alpha = strategy['sub_alpha']
    top_K = [v for v, _ in distances[target][:K]]
    opp3 = [v for v, _ in sorted(distances[target], key=lambda x: -x[1])[:3]]
    probs = np.zeros(6)
    members = [(target, own_w)] + [(v, 1.0) for v in top_K] + [(v, -sub_alpha/3) for v in opp3]
    for source, w in members:
        spec = all_specialists_76.get(source)
        if spec:
            p = spec.predict(record['features_82'][:76].reshape(1, -1),
                             num_iteration=spec.best_iteration)[0]
            probs += w * p
    probs = np.clip(probs, 0.001, None)
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


def qmc_backtest(records, predict_fn):
    n_top1 = n_top3 = 0
    invested = returned = 0
    for r in records:
        probs = predict_fn(r)
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
    n = len(records)
    return {'n': n, 'top1_rate': n_top1/n*100 if n else 0, 'top3_rate': n_top3/n*100 if n else 0,
            'invested': invested, 'returned': returned, 'pnl': returned-invested,
            'roi': (returned-invested)/invested*100 if invested else 0}


def main():
    logger.info("Shadow 戦略 v1 設計 + 再現確認")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    distances = get_distances(venue_preds)

    # 全 venue records (test 期間のみ in-scope)
    logger.info("Records (82dim) pre-compute")
    by_venue = {}
    for vid, preds in venue_preds.items():
        records = []
        for rid, p in preds.items():
            try:
                features_82 = build_features_82(p, fe, scaler)
                records.append({
                    'features_82': features_82,
                    'y1': p['result_1st'] - 1,
                    'date': date.fromisoformat(p['race_date']),
                    'prediction': p,
                    'venue_id': vid,
                })
            except Exception:
                continue
        by_venue[vid] = records

    # 76dim / 82dim specialists load
    spec_76 = {vid: lgb.Booster(model_file=str(SPEC_76 / f'lightgbm_v{vid:02d}_1st.txt'))
               for vid in VENUE_NAMES if (SPEC_76 / f'lightgbm_v{vid:02d}_1st.txt').exists()}
    spec_82 = {vid: lgb.Booster(model_file=str(SPEC_82 / f'lightgbm_v{vid:02d}_1st.txt'))
               for vid in VENUE_NAMES if (SPEC_82 / f'lightgbm_v{vid:02d}_1st.txt').exists()}
    logger.info(f"specialists loaded: 76dim {len(spec_76)}, 82dim {len(spec_82)}")

    # Pool 訓練 (14: L3_海水_strict, 16: L3_海水_strict, 23: L3_淡水_strict)
    pool_models = {}
    pool_configs = [
        (14, 'L3_海水_strict', [14, 4, 16, 17, 22]),
        (16, 'L3_海水_strict', [16, 14, 17, 22, 4]),
        (23, 'L3_淡水_strict', [23, 12, 13, 21, 5, 11]),
    ]
    for target_vid, level_name, members in pool_configs:
        logger.info(f"Pool 訓練 v{target_vid} ({level_name}) members={members}")
        pool_train = []
        target_val = []
        for v in members:
            for r in by_venue.get(v, []):
                if v == target_vid:
                    if r['date'] < date(2026, 4, 1):
                        pool_train.append(r)
                    elif r['date'] < TEST_START:
                        target_val.append(r)
                else:
                    if r['date'] < TEST_START:
                        pool_train.append(r)
        model = train_pool_model(pool_train, target_val, dim=76)
        if model:
            pool_models[(target_vid, level_name)] = model

    # Shadow backtest on 2026-05 (13 functional venues)
    results = {}
    for vid in FUNCTIONAL:
        recs = [r for r in by_venue.get(vid, []) if r['date'] >= TEST_START]
        if len(recs) < 30:
            continue
        shadow_predict = lambda r: predict_shadow(r, vid, spec_76, spec_82, pool_models, distances)
        v10_predict = lambda r: np.array(r['prediction']['probs_1st'])
        shadow_qmc = qmc_backtest(recs, shadow_predict)
        v10_qmc = qmc_backtest(recs, v10_predict)
        results[vid] = {
            'name': VENUE_NAMES[vid],
            'n': len(recs),
            'strategy': VENUE_STRATEGY[vid],
            'shadow_roi': shadow_qmc['roi'],
            'shadow_top3': shadow_qmc['top3_rate'],
            'v10_roi': v10_qmc['roi'],
            'shadow_pnl': shadow_qmc['pnl'],
            'v10_pnl': v10_qmc['pnl'],
            'shadow_invested': shadow_qmc['invested'],
            'shadow_returned': shadow_qmc['returned'],
        }
        logger.info(f"  v{vid} ({VENUE_NAMES[vid]}): shadow ROI={shadow_qmc['roi']:+.2f}%, V10={v10_qmc['roi']:+.2f}%")

    # 全体集計: 13 venues 統合 ROI
    total_invested = sum(r['shadow_invested'] for r in results.values())
    total_returned = sum(r['shadow_returned'] for r in results.values())
    total_roi = (total_returned - total_invested) / total_invested * 100 if total_invested else 0
    v10_total_invested = sum(r['v10_pnl'] + r['shadow_invested'] - r['shadow_returned'] +
                             r['shadow_invested'] for r in results.values()) // 1  # rough
    # 正しい V10 total
    v10_total_invested = sum(r['shadow_invested'] for r in results.values())  # same n_races
    v10_total_returned = 0
    for vid, r in results.items():
        # V10 returned は v10_pnl + invested から逆算
        v10_returned = r['v10_pnl'] + r['shadow_invested']
        v10_total_returned += v10_returned
    v10_total_roi = (v10_total_returned - v10_total_invested) / v10_total_invested * 100 if v10_total_invested else 0

    # Report
    lines = []
    lines.append("# Shadow 戦略 v1 設計 + 2026-05 再現確認\n\n")
    lines.append("13 functional venues の venue 別 best approach を統合した shadow strategy。\n\n")

    lines.append("## venue 別 strategy mapping\n\n")
    lines.append("| venue | name | approach | type | ROI 期待値 (期待: 累積分析より) |\n|---|---|---|---|---|\n")
    expected_rois = {1:9.34, 2:-5.43, 3:15.09, 4:-1.59, 7:10.65, 10:-28.13, 12:17.68,
                     13:24.46, 14:-11.34, 16:-29.36, 22:36.64, 23:66.72, 24:-41.20}
    for vid in FUNCTIONAL:
        strat = VENUE_STRATEGY.get(vid, {})
        lines.append(f"| {vid} | {VENUE_NAMES[vid]} | {strat.get('name', strat.get('type'))} | "
                     f"{strat.get('type')} | {expected_rois.get(vid, 0):+.2f}% |\n")

    lines.append("\n## 2026-05 再現確認 (各 venue で shadow ROI vs 期待値)\n\n")
    lines.append("| venue | name | n | V10 ROI | shadow ROI | 期待 ROI | 再現性 |\n|---|---|---|---|---|---|---|\n")
    reproductions = []
    for vid in FUNCTIONAL:
        r = results.get(vid)
        if not r:
            continue
        expected = expected_rois.get(vid, 0)
        actual = r['shadow_roi']
        diff = actual - expected
        flag = '✅' if abs(diff) < 2 else '⚠️'
        reproductions.append(diff)
        lines.append(f"| {vid} | {r['name']} | {r['n']} | {r['v10_roi']:+.2f}% | "
                     f"{actual:+.2f}% | {expected:+.2f}% | {diff:+.2f}pt {flag} |\n")

    lines.append(f"\n## 13 venues 統合集計\n\n")
    total_n = sum(r['n'] for r in results.values())
    lines.append(f"- 統合 n: {total_n}\n")
    lines.append(f"- shadow 投資合計: ¥{total_invested:,}\n")
    lines.append(f"- shadow 回収合計: ¥{total_returned:,.0f}\n")
    lines.append(f"- shadow ROI: **{total_roi:+.2f}%**\n")
    lines.append(f"- V10 baseline ROI (同 races): **{v10_total_roi:+.2f}%**\n")
    lines.append(f"- shadow vs V10 差: **{total_roi - v10_total_roi:+.2f}pt**\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- 13 venues のみ shadow 戦略採用、11 venues は V10 baseline 維持\n")
    lines.append("- 各 venue の再現性 (shadow ROI vs 期待) は実装の整合性チェック\n")
    lines.append("- 2026-05 単月 hold-out、forward (2026-06+) で再検証必須\n")
    lines.append("- 採用候補は production scheduler 統合 → shadow テーブル記録 → 2 週間 forward 検証\n")

    # Config 出力
    config = {
        'version': 'v1',
        'fitted_at': '2026-05-19',
        'venue_strategies': {str(k): v for k, v in VENUE_STRATEGY.items()},
        'expected_rois': {str(k): v for k, v in expected_rois.items()},
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Config: {CONFIG_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
