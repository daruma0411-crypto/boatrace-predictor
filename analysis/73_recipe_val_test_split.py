"""val/test 分離による 71 足し算 recipe の overfit 検証 (第 1 段階)

71 は test set 上で recipe 選定 → test ROI 報告 (二重バイアス)。
73 は recipe 選定を **val (2026-04)** のみで行い、test (2026-05) で hold-out 評価。

判定:
  val best recipe を test に適用したときの ROI:
  - V10 baseline 比 +5pt 以上改善 → 真の signal
  - 0〜+5pt → noise の可能性
  - 悪化 → 71 の +22pt は overfit、72 (引き算) も進めない

各 venue で:
  Phase 1: val (2026-04) 上で 71 の全 recipe 候補を QMC + ROI 評価
  Phase 2: val best recipe を抽出
  Phase 3: その recipe を test (2026-05) に適用 → hold-out ROI

71 best (test 上選定) との比較で「test 上 best が val 上 best と同じか」を確認。

出力: analysis/reports/73_recipe_val_test_split.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '73_recipe_val_test_split.md'

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
FUNCTIONAL_VENUES = [2, 3, 7, 12, 13, 14, 16, 24]


def load_specialists():
    models = {}
    for vid in VENUE_NAMES:
        path = SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        if path.exists():
            models[vid] = lgb.Booster(model_file=str(path))
    return models


def get_similarity(target_vid, venue_preds, top_k=10):
    feats = {}
    for vid, preds in venue_preds.items():
        if not preds:
            continue
        boat_1st = np.zeros(6)
        for p in preds.values():
            boat_1st[p['result_1st'] - 1] += 1
        feats[vid] = boat_1st / len(preds) * 100
    if target_vid not in feats:
        return []
    target = feats[target_vid]
    sims = []
    for vid, f in feats.items():
        if vid == target_vid:
            continue
        sims.append((vid, float(np.linalg.norm(target - f))))
    sims.sort(key=lambda x: x[1])
    return [vid for vid, _ in sims[:top_k]]


def build_records(venue_preds, specialists, fe, scaler):
    """venue_id -> {val: [...], test: [...]}"""
    by_venue = {}
    for target_vid, preds in venue_preds.items():
        val_recs = []
        test_recs = []
        for rid, p in preds.items():
            d = date.fromisoformat(p['race_date'])
            if d < VAL_START:
                continue
            try:
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                spec_probs = {}
                for sid, sm in specialists.items():
                    pred = sm.predict(features.reshape(1, -1), num_iteration=sm.best_iteration)
                    spec_probs[sid] = pred[0]
                v10_probs = np.array(p['probs_1st'])
                record = {
                    'rid': rid,
                    'v10_probs': v10_probs, 'spec_probs': spec_probs,
                    'actual': p['actual'], 'result_1st': p['result_1st'],
                    'payout': p['payout'] or 0,
                    'prediction': p,
                }
                if d < TEST_START:
                    val_recs.append(record)
                else:
                    test_recs.append(record)
            except Exception:
                continue
        by_venue[target_vid] = {'val': val_recs, 'test': test_recs}
    return by_venue


def blend_probs(recipe, record):
    total_w = sum(w for _, w in recipe)
    if total_w == 0:
        return None
    probs = np.zeros(6)
    for source, w in recipe:
        wnorm = w / total_w
        if source == 'v10':
            probs += wnorm * record['v10_probs']
        else:
            probs += wnorm * record['spec_probs'][source]
    return probs


def qmc_score(recipe, records):
    if not records:
        return None
    n_top1 = n_top3 = 0
    invested = returned = 0
    for r in records:
        probs = blend_probs(recipe, r)
        if probs is None:
            continue
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
    roi = (returned - invested) / invested * 100 if invested else 0
    return {
        'n': n, 'top1_rate': n_top1/n*100, 'top3_rate': n_top3/n*100,
        'invested': invested, 'returned': returned,
        'pnl': returned-invested, 'roi': roi,
    }


def generate_recipes(target_vid, similar_vids):
    recipes = []
    recipes.append(('R0_V10_only', [('v10', 1.0)]))
    recipes.append(('R1_own_specialist', [(target_vid, 1.0)]))
    for i, alpha in enumerate(np.linspace(0.1, 0.9, 9)):
        recipes.append((f'R{2+i}_V10x{1-alpha:.1f}+own_x{alpha:.1f}',
                        [('v10', 1-alpha), (target_vid, alpha)]))
    for K in [2, 3, 5]:
        if len(similar_vids) >= K:
            top_k = similar_vids[:K]
            recipes.append((f'R_top{K}_sim_avg',
                            [(target_vid, 1.0)] + [(v, 1.0) for v in top_k]))
    recipes.append(('R_all24_avg', [(v, 1.0) for v in VENUE_NAMES.keys()]))
    recipes.append(('R_functional8_avg', [(v, 1.0) for v in FUNCTIONAL_VENUES]))
    recipes.append(('R_own_x2+functional',
                    [(target_vid, 2.0)] + [(v, 1.0) for v in FUNCTIONAL_VENUES if v != target_vid]))
    return recipes


def main():
    logger.info("Recipe val/test 分離検証 (第 1 段階)")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    specialists = load_specialists()
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    logger.info("Records pre-compute")
    by_venue = build_records(venue_preds, specialists, fe, scaler)

    results = {}
    for target_vid in sorted(VENUE_NAMES.keys()):
        recs = by_venue.get(target_vid, {})
        val_recs = recs.get('val', [])
        test_recs = recs.get('test', [])
        if len(val_recs) < 30 or len(test_recs) < 30:
            logger.warning(f"venue {target_vid} skip (val={len(val_recs)}, test={len(test_recs)})")
            continue
        similar_vids = get_similarity(target_vid, venue_preds, top_k=10)
        recipes = generate_recipes(target_vid, similar_vids)
        logger.info(f"venue {target_vid} ({VENUE_NAMES[target_vid]}): val={len(val_recs)}, test={len(test_recs)}, recipes={len(recipes)}")

        # Phase 1: val で全 recipe 評価
        val_scores = []
        for name, recipe in recipes:
            s = qmc_score(recipe, val_recs)
            if s:
                val_scores.append({'name': name, 'recipe': recipe, 'val_qmc': s})
        val_scores.sort(key=lambda x: -x['val_qmc']['roi'])

        # Phase 2: val best
        val_best = val_scores[0]
        v10_recipe_obj = next(r for r in val_scores if r['name'] == 'R0_V10_only')

        # Phase 3: val best を test に適用
        test_at_val_best = qmc_score(val_best['recipe'], test_recs)
        test_at_v10 = qmc_score(v10_recipe_obj['recipe'], test_recs)
        # 71 で test 上に選んだ best と比較するため、test 上 best も計算
        test_scores = []
        for r in val_scores:
            ts = qmc_score(r['recipe'], test_recs)
            test_scores.append({'name': r['name'], 'recipe': r['recipe'],
                                'val_roi': r['val_qmc']['roi'],
                                'test_roi': ts['roi'] if ts else None,
                                'test_top3_rate': ts['top3_rate'] if ts else 0})
        test_best = max([r for r in test_scores if r['test_roi'] is not None],
                        key=lambda x: x['test_roi'])

        results[target_vid] = {
            'name': VENUE_NAMES[target_vid],
            'n_val': len(val_recs), 'n_test': len(test_recs),
            'val_best': val_best,
            'test_at_val_best': test_at_val_best,
            'test_at_v10': test_at_v10,
            'test_best': test_best,
            'val_scores': val_scores[:5],
        }

    # Report
    lines = []
    lines.append("# val/test 分離による recipe overfit 検証 (第 1 段階)\n\n")
    lines.append(f"val=2026-04, test=2026-05 (hold-out)\n")
    lines.append(f"val 上で recipe 選定 → test に適用、71 (test 上選定) との overfit gap を測定\n\n")

    lines.append("## venue 別: val best recipe を test に適用した結果\n\n")
    lines.append("| venue | name | n_val | n_test | V10 test ROI | val best 名 | val best ROI | val best at test | 真の改善 | test 上 best ROI (71 相当) | overfit gap |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    real_improvements = []
    for vid in sorted(results.keys()):
        r = results[vid]
        v10_test_roi = r['test_at_v10']['roi'] if r['test_at_v10'] else None
        val_best_test_roi = r['test_at_val_best']['roi'] if r['test_at_val_best'] else None
        test_best_roi = r['test_best']['test_roi']
        if v10_test_roi is None or val_best_test_roi is None:
            continue
        real_diff = val_best_test_roi - v10_test_roi
        overfit_gap = test_best_roi - val_best_test_roi
        real_improvements.append(real_diff)
        flag = '🟢' if real_diff > 5 else ('🟡' if real_diff > 0 else '🔴')
        lines.append(f"| {vid} | {r['name']} | {r['n_val']} | {r['n_test']} | "
                     f"{v10_test_roi:+.2f}% | {r['val_best']['name']} | "
                     f"{r['val_best']['val_qmc']['roi']:+.2f}% | {val_best_test_roi:+.2f}% | "
                     f"**{real_diff:+.2f}pt {flag}** | {test_best_roi:+.2f}% | {overfit_gap:+.2f}pt |\n")

    # 統計
    if real_improvements:
        n_strong = sum(1 for d in real_improvements if d > 5)
        n_mid = sum(1 for d in real_improvements if 0 < d <= 5)
        n_worse = sum(1 for d in real_improvements if d <= 0)
        lines.append(f"\n## 統計 (val→test 検証、真の signal)\n\n")
        lines.append(f"- 🟢 +5pt 以上: **{n_strong} venues**\n")
        lines.append(f"- 🟡 0〜+5pt: {n_mid} venues\n")
        lines.append(f"- 🔴 悪化: {n_worse} venues\n")
        lines.append(f"- 平均改善: {float(np.mean(real_improvements)):+.2f}pt\n")
        lines.append(f"- 中央値: {float(np.median(real_improvements)):+.2f}pt\n")
        lines.append(f"- 最大: {max(real_improvements):+.2f}pt\n")
        lines.append(f"\n参考: 71 (test 上で選定、二重バイアス) は平均 +20.92pt、中央値 +17.23pt\n")
        lines.append(f"差: 71 平均 vs 73 平均 = overfit 度合い\n")

    # 戸田 詳細
    lines.append("\n## 戸田 (target 2) 詳細\n\n")
    toda_r = results.get(2)
    if toda_r:
        lines.append(f"V10 test ROI: {toda_r['test_at_v10']['roi']:+.2f}%\n")
        lines.append(f"val best recipe: {toda_r['val_best']['name']}\n")
        lines.append(f"  - val ROI: {toda_r['val_best']['val_qmc']['roi']:+.2f}%\n")
        lines.append(f"  - test ROI (hold-out): {toda_r['test_at_val_best']['roi']:+.2f}%\n")
        lines.append(f"  - 真の改善: {toda_r['test_at_val_best']['roi'] - toda_r['test_at_v10']['roi']:+.2f}pt\n\n")
        lines.append("### val top 5 recipes と test ROI\n\n")
        lines.append("| rank | recipe | val ROI | test ROI |\n|---|---|---|---|\n")
        for i, r in enumerate(toda_r['val_scores']):
            test_match = next((t for t in [toda_r['test_at_val_best']] if t), None)
            test_roi = qmc_score(r['recipe'], by_venue[2]['test'])
            lines.append(f"| {i+1} | {r['name']} | {r['val_qmc']['roi']:+.2f}% | {test_roi['roi']:+.2f}% |\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- val=2026-04 はちょうど GW 直前期、特異性ある可能性\n")
    lines.append("- test=2026-05 は GW 直後で 1号艇率低下期、特異性ある可能性\n")
    lines.append("- 厳密な forward 検証は最低 3 ヶ月の連続 hold-out 必要\n")
    lines.append("- 結果が positive でも shadow 並走 2 週間必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
