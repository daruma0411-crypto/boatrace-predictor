"""唐津 / 平和島 / 尼崎 限定の recipe 細密化 (1 ラウンド限定)

73/74 で各 venue の best recipe (改善 +35〜+53pt) が出たが、
重み付け / K / 引き算混合の細密 grid は未探索。
val 上で 1 ラウンドだけ探索、現状 best を超えれば更新、超えなければ維持。

Recipe 空間 (per target):
  - K (similar venues 数): 2, 3, 4, 5, 6, 7
  - own_weight: 1.0, 1.5, 2.0, 3.0
  - 引き算 component: none / opp3 × 0.1 / opp3 × 0.2 / opp3 × 0.3
  - 合計 6 × 4 × 4 = 96 recipes per venue

判定:
  val best → test 評価
  test ROI が 73/74 の現状 best を超えれば 🟢、超えなければ 🔴 (現状維持)

出力: analysis/reports/75_refine_top3.md
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
SPECIALISTS_DIR = ROOT / 'models' / 'specialists'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '75_refine_top3.md'

VAL_START = date(2026, 4, 1)
TEST_START = date(2026, 5, 1)
N_SIM = 8192
SEED = 42

TARGET_VENUES = {
    23: '唐津',
    4: '平和島',
    13: '尼崎',
}

# 73/74 の現状 best test ROI
CURRENT_BEST = {
    23: {'recipe_name': '73 R_top5_sim_avg', 'test_roi': 50.98},
    4:  {'recipe_name': '73 R_top5_sim_avg', 'test_roi': -1.59},
    13: {'recipe_name': '74 R-_own-opp3x0.2', 'test_roi': -5.41},
}

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}


def load_specialists():
    return {vid: lgb.Booster(model_file=str(SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt'))
            for vid in VENUE_NAMES if (SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt').exists()}


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


def build_records(venue_preds, specialists, fe, scaler, target_vids):
    by_venue = {}
    for tvid in target_vids:
        preds = venue_preds.get(tvid, {})
        val_recs, test_recs = [], []
        for rid, p in preds.items():
            d = date.fromisoformat(p['race_date'])
            if d < VAL_START:
                continue
            try:
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                spec_probs = {sid: sm.predict(features.reshape(1, -1), num_iteration=sm.best_iteration)[0]
                              for sid, sm in specialists.items()}
                v10_probs = np.array(p['probs_1st'])
                rec = {'v10_probs': v10_probs, 'spec_probs': spec_probs,
                       'actual': p['actual'], 'result_1st': p['result_1st'],
                       'payout': p['payout'] or 0, 'prediction': p}
                (val_recs if d < TEST_START else test_recs).append(rec)
            except Exception:
                continue
        by_venue[tvid] = {'val': val_recs, 'test': test_recs}
    return by_venue


def blend_probs(recipe, record):
    probs = np.zeros(6)
    for source, w in recipe:
        if source == 'v10':
            probs += w * record['v10_probs']
        else:
            probs += w * record['spec_probs'][source]
    probs = np.clip(probs, 0.001, None)
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


def qmc_score(recipe, records):
    if not records:
        return None
    n_top1 = n_top3 = 0
    invested = returned = 0
    for r in records:
        probs = blend_probs(recipe, r)
        try:
            qp = qmc_sanrentan_v3(
                probs.tolist(),
                boats_data=r['prediction']['boats'],
                race_data=r['prediction']['race_data'],
                race_number=r['prediction']['race_number'],
                n_simulations=N_SIM, seed=SEED,
            )
        except Exception:
            continue
        top3 = sorted(qp.items(), key=lambda x: -x[1])[:3]
        top3_combos = [t[0] for t in top3]
        if top3_combos[0] == r['actual']:
            n_top1 += 1
        if r['actual'] in top3_combos:
            n_top3 += 1
            returned += r['payout']
        invested += 300
    n = len(records)
    return {'n': n, 'top1_rate': n_top1/n*100, 'top3_rate': n_top3/n*100,
            'invested': invested, 'returned': returned,
            'pnl': returned-invested, 'roi': (returned-invested)/invested*100 if invested else 0}


def generate_refined_recipes(target_vid, similar_vids, opposite_vids):
    recipes = []
    opp3 = opposite_vids[:3]
    K_list = [2, 3, 4, 5, 6, 7]
    own_weights = [1.0, 1.5, 2.0, 3.0]
    sub_options = [
        ('none', 0.0),
        ('opp3x0.1', 0.1),
        ('opp3x0.2', 0.2),
        ('opp3x0.3', 0.3),
    ]
    for K in K_list:
        if len(similar_vids) < K:
            continue
        sim = similar_vids[:K]
        for ow in own_weights:
            for sub_name, sub_alpha in sub_options:
                rec = [(target_vid, ow)] + [(v, 1.0) for v in sim]
                if sub_alpha > 0:
                    rec += [(v, -sub_alpha / 3) for v in opp3]
                name = f"K{K}_own{ow:.1f}_sub_{sub_name}"
                recipes.append((name, rec))
    return recipes


def main():
    logger.info("唐津 / 平和島 / 尼崎 recipe 細密化")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    specialists = load_specialists()
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    distances = get_distances(venue_preds)
    by_venue = build_records(venue_preds, specialists, fe, scaler, TARGET_VENUES.keys())

    results = {}
    for tvid in TARGET_VENUES.keys():
        recs = by_venue.get(tvid, {})
        val_recs, test_recs = recs.get('val', []), recs.get('test', [])
        sims = [v for v, _ in distances.get(tvid, [])][:10]
        opps = [v for v, _ in sorted(distances.get(tvid, []), key=lambda x: -x[1])][:5]
        recipes = generate_refined_recipes(tvid, sims, opps)
        logger.info(f"venue {tvid} ({VENUE_NAMES[tvid]}): val={len(val_recs)}, test={len(test_recs)}, recipes={len(recipes)}")

        # Phase 1: val 全 recipe 評価
        val_scores = []
        for name, recipe in recipes:
            s = qmc_score(recipe, val_recs)
            if s:
                val_scores.append({'name': name, 'recipe': recipe, 'val_qmc': s})
        val_scores.sort(key=lambda x: -x['val_qmc']['roi'])

        # Phase 2: top 5 を test 評価
        top5 = val_scores[:5]
        for r in top5:
            r['test_qmc'] = qmc_score(r['recipe'], test_recs)

        # V10 baseline test
        v10_test = qmc_score([('v10', 1.0)], test_recs)

        results[tvid] = {
            'name': VENUE_NAMES[tvid],
            'n_val': len(val_recs), 'n_test': len(test_recs),
            'top5': top5,
            'v10_test_roi': v10_test['roi'] if v10_test else None,
            'sims': sims[:7], 'opps': opps[:3],
        }

    # Report
    lines = []
    lines.append("# 唐津 / 平和島 / 尼崎 recipe 細密化 (1 ラウンド)\n\n")
    lines.append("各 target で K × own_weight × 引き算 component の 96 recipes を val 上で評価。\n")
    lines.append("Phase 1 val top 5 → Phase 2 test hold-out 評価。\n\n")

    for tvid in TARGET_VENUES.keys():
        r = results.get(tvid)
        if not r:
            continue
        cb = CURRENT_BEST.get(tvid, {})
        cb_roi = cb.get('test_roi', 0)
        lines.append(f"\n## venue {tvid} ({r['name']})\n\n")
        lines.append(f"val: {r['n_val']} races, test: {r['n_test']} races\n")
        lines.append(f"V10 test ROI: {r['v10_test_roi']:+.2f}%\n")
        lines.append(f"現状 best ({cb.get('recipe_name', 'N/A')}): test ROI {cb_roi:+.2f}%\n\n")
        lines.append("### val top 5 → test ROI\n\n")
        lines.append("| rank | recipe | val ROI | test ROI | 現状比 |\n|---|---|---|---|---|\n")
        for i, rec in enumerate(r['top5']):
            test_roi = rec['test_qmc']['roi'] if rec.get('test_qmc') else None
            diff = (test_roi - cb_roi) if test_roi is not None else None
            flag = '🟢' if diff is not None and diff > 0 else '🔴' if diff is not None and diff < -5 else '🟡'
            test_str = f"{test_roi:+.2f}%" if test_roi is not None else 'N/A'
            diff_str = f"{diff:+.2f}pt {flag}" if diff is not None else 'N/A'
            lines.append(f"| {i+1} | {rec['name']} | {rec['val_qmc']['roi']:+.2f}% | {test_str} | {diff_str} |\n")

        # val best の判定
        val_best = r['top5'][0] if r['top5'] else None
        if val_best and val_best.get('test_qmc'):
            test_at_best = val_best['test_qmc']['roi']
            improvement = test_at_best - cb_roi
            lines.append(f"\n**val best ({val_best['name']}) 細密化結果**:\n")
            lines.append(f"- test ROI: {test_at_best:+.2f}%\n")
            lines.append(f"- 現状比改善: {improvement:+.2f}pt\n")
            if improvement > 5:
                lines.append(f"- 🟢 **採用更新候補** (+5pt 以上)\n")
            elif improvement > 0:
                lines.append(f"- 🟡 微改善、updating の価値判定要\n")
            else:
                lines.append(f"- 🔴 **現状維持** (改善なし)\n")

    lines.append("\n## 全体判定\n\n")
    updated = 0
    for tvid in TARGET_VENUES.keys():
        r = results.get(tvid)
        if not r or not r['top5']:
            continue
        val_best = r['top5'][0]
        if val_best.get('test_qmc'):
            cb_roi = CURRENT_BEST.get(tvid, {}).get('test_roi', 0)
            if val_best['test_qmc']['roi'] > cb_roi + 5:
                updated += 1
    lines.append(f"- 細密化で +5pt 以上更新: **{updated}/3 venues**\n")
    lines.append(f"- これで「改善余地探索」は完了、これ以上は forward (2026-06+) 待ち\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- val/test 分離方式 (73/74 と同じ厳密性)\n")
    lines.append("- 1 ラウンドのみ、これ以上の細密化は test set 上の過剰最適化\n")
    lines.append("- 採用候補は shadow 並走 2 週間で forward 検証必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
