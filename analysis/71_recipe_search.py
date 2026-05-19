"""24 venue specialist recipe 探索 (Phase D)

「料理の調味料」型 recipe 探索。
24 venue specialist + V10 を blend、各 target venue で best recipe を決定。

2 段階:
  Phase 1 (高速): NN-only top-3 hit 率で recipe 候補を絞る (~1000 recipes 評価)
  Phase 2 (詳細): top-5 recipe を QMC + ROI で本評価

Recipe 候補 (per target venue):
  R0: V10 baseline
  R1: own specialist alone
  R2-R11: V10 × own blend (α 0.0〜1.0、10 段)
  R12-R15: top-K 類似 venues 平均 (K=2/3/5)
  R16: all 24 specialists average
  R17: 機能 venues (8 venues) average
  R18-R...: greedy forward selection per venue

入力: models/specialists/lightgbm_v??_1st.txt × 24
出力: analysis/reports/71_recipe_search.md
"""
import os
import sys
import pickle
import logging
import itertools
from pathlib import Path
from datetime import date
from collections import defaultdict

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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '71_recipe_search.md'

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

# 70 で機能した venues (ROI 改善 > +5pt)
FUNCTIONAL_VENUES = [2, 3, 7, 12, 13, 14, 16, 24]


def load_specialists():
    """全 specialist model 読み込み."""
    models = {}
    for vid in VENUE_NAMES:
        path = SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        if path.exists():
            models[vid] = lgb.Booster(model_file=str(path))
    return models


def build_test_predictions(venue_preds, specialists, fe, scaler):
    """各 target venue の test races (2026-05) で全 specialist の予測を pre-compute."""
    test_by_venue = {}
    for target_vid, preds in venue_preds.items():
        records = []
        for rid, p in preds.items():
            d = date.fromisoformat(p['race_date'])
            if d < TEST_START:
                continue
            try:
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                # 全 specialist の予測
                spec_probs = {}
                for sid, sm in specialists.items():
                    pred = sm.predict(features.reshape(1, -1), num_iteration=sm.best_iteration)
                    spec_probs[sid] = pred[0]
                v10_probs = np.array(p['probs_1st'])
                records.append({
                    'rid': rid,
                    'features': features,
                    'v10_probs': v10_probs,
                    'spec_probs': spec_probs,  # dict[sid -> probs_array_6]
                    'actual': p['actual'],
                    'result_1st': p['result_1st'],
                    'payout': p['payout'] or 0,
                    'prediction': p,
                })
            except Exception:
                continue
        test_by_venue[target_vid] = records
    return test_by_venue


def blend_probs(recipe, record):
    """recipe = list of (source, weight)、source ∈ {'v10', specialist_id (int)}
    weight は自動 normalize."""
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


def fast_score(recipe, records):
    """NN-only top-1 hit 率 (1着 argmax と actual 一致率)."""
    if not records:
        return 0.0
    n_hit = 0
    n_total = 0
    for r in records:
        probs = blend_probs(recipe, r)
        if probs is None:
            continue
        top_boat = int(np.argmax(probs)) + 1
        if top_boat == r['result_1st']:
            n_hit += 1
        n_total += 1
    return n_hit / n_total * 100 if n_total else 0


def qmc_score(recipe, records):
    """QMC + top-3 + ROI 詳細評価."""
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
        'pnl': returned - invested, 'roi': roi,
    }


def get_similarity(target_vid, venue_preds, top_k=5):
    """各 target_venue の上位 K 類似 venue を返す (66 と同じ距離計算)."""
    feats = {}
    for vid, preds in venue_preds.items():
        if not preds:
            continue
        boat_1st = np.zeros(6)
        for p in preds.values():
            boat_1st[p['result_1st'] - 1] += 1
        boat_1st /= len(preds)
        feats[vid] = boat_1st * 100
    if target_vid not in feats:
        return []
    target = feats[target_vid]
    sims = []
    for vid, f in feats.items():
        if vid == target_vid:
            continue
        d = np.linalg.norm(target - f)
        sims.append((vid, d))
    sims.sort(key=lambda x: x[1])
    return [vid for vid, _ in sims[:top_k]]


def generate_recipes(target_vid, similar_vids):
    """target venue 向け recipe 候補を生成."""
    recipes = []
    # R0: V10
    recipes.append(('R0_V10_only', [('v10', 1.0)]))
    # R1: own specialist
    recipes.append((f'R1_own_specialist', [(target_vid, 1.0)]))
    # R2-R11: V10 + own blend (α 0.0-1.0、10 段)
    for i, alpha in enumerate(np.linspace(0.1, 0.9, 9)):
        recipes.append((f'R{2+i}_V10x{1-alpha:.1f}+own_x{alpha:.1f}',
                        [('v10', 1-alpha), (target_vid, alpha)]))
    # top-K 類似 specialist averages
    for K in [2, 3, 5]:
        if len(similar_vids) >= K:
            top_k_vids = similar_vids[:K]
            recipes.append((f'R_top{K}_sim_avg',
                            [(target_vid, 1.0)] + [(v, 1.0) for v in top_k_vids]))
    # all specialists avg
    all_vids = list(VENUE_NAMES.keys())
    recipes.append(('R_all24_avg', [(v, 1.0) for v in all_vids]))
    # functional specialists avg
    recipes.append(('R_functional8_avg', [(v, 1.0) for v in FUNCTIONAL_VENUES]))
    # functional + own emphasis
    recipes.append(('R_own_x2+functional', [(target_vid, 2.0)] +
                    [(v, 1.0) for v in FUNCTIONAL_VENUES if v != target_vid]))
    return recipes


def greedy_forward(target_vid, records, similar_vids, max_add=5):
    """Greedy forward: target venue から類似順に 1 つずつ追加、ROI 改善する限り続行."""
    best_recipe = [(target_vid, 1.0)]
    best_score = fast_score(best_recipe, records)
    history = [('start', target_vid, best_score)]
    used = {target_vid}
    for step in range(max_add):
        candidates = [v for v in similar_vids if v not in used]
        if not candidates:
            break
        improved = False
        best_cand = None
        best_cand_score = best_score
        for cand in candidates:
            test_recipe = best_recipe + [(cand, 1.0)]
            s = fast_score(test_recipe, records)
            if s > best_cand_score + 0.5:  # 0.5pt 以上改善で採用
                best_cand = cand
                best_cand_score = s
        if best_cand is None:
            break
        best_recipe.append((best_cand, 1.0))
        best_score = best_cand_score
        used.add(best_cand)
        history.append((f'add_v{best_cand}', best_cand, best_score))
    return best_recipe, history


def main():
    logger.info("Recipe 探索 (Phase D)")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    specialists = load_specialists()
    logger.info(f"specialists loaded: {len(specialists)}")
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

    logger.info("Test predictions pre-compute (全 venue × 全 specialist)")
    test_by_venue = build_test_predictions(venue_preds, specialists, fe, scaler)

    results = {}
    for target_vid in sorted(VENUE_NAMES.keys()):
        if target_vid not in test_by_venue or not test_by_venue[target_vid]:
            continue
        records = test_by_venue[target_vid]
        if len(records) < 50:
            continue
        similar_vids = get_similarity(target_vid, venue_preds, top_k=10)
        logger.info(f"venue {target_vid} ({VENUE_NAMES[target_vid]}): n={len(records)}, similar={similar_vids[:5]}")

        # Phase 1: 全 recipe 候補を NN-only で screening
        recipes = generate_recipes(target_vid, similar_vids)
        # + greedy forward
        greedy_recipe, greedy_hist = greedy_forward(target_vid, records, similar_vids, max_add=5)
        recipes.append(('R_greedy_forward', greedy_recipe))

        recipe_scores = []
        for name, recipe in recipes:
            top1_rate = fast_score(recipe, records)
            recipe_scores.append({'name': name, 'recipe': recipe, 'fast_top1': top1_rate})
        recipe_scores.sort(key=lambda x: -x['fast_top1'])

        # Phase 2: top 8 を QMC + ROI 詳細評価
        top_recipes = recipe_scores[:8]
        # V10 baseline (always include) and own specialist
        baseline_names = {'R0_V10_only', 'R1_own_specialist'}
        for r in recipe_scores:
            if r['name'] in baseline_names and r not in top_recipes:
                top_recipes.append(r)

        for r in top_recipes:
            r['qmc'] = qmc_score(r['recipe'], records)

        results[target_vid] = {
            'name': VENUE_NAMES[target_vid],
            'n_test': len(records),
            'similar': similar_vids[:5],
            'recipes': top_recipes,
            'greedy_history': greedy_hist,
        }

    # Report
    lines = []
    lines.append("# 24 venue 別 best recipe 探索 (Phase D)\n\n")
    lines.append("各 target venue で 24 specialist + V10 を blend、QMC + ROI で best recipe を探索。\n")
    lines.append("**機能 venues** (70 で +5pt 以上改善): 戸田(2), 江戸川(3), 蒲郡(7), 住之江(12), 尼崎(13), 鳴門(14), 児島(16), 大村(24)\n\n")

    # サマリ: 各 venue で best recipe + ROI
    lines.append("## サマリ: 各 venue の best recipe\n\n")
    lines.append("| venue | name | n | V10 ROI | best recipe ROI | best recipe | 改善 |\n|---|---|---|---|---|---|---|\n")
    summary_rows = []
    for vid in sorted(results.keys()):
        res = results[vid]
        # V10 baseline ROI
        v10_recipe = next((r for r in res['recipes'] if r['name'] == 'R0_V10_only'), None)
        v10_roi = v10_recipe['qmc']['roi'] if v10_recipe and v10_recipe.get('qmc') else None
        # best recipe (QMC ROI 最大)
        evaluated = [r for r in res['recipes'] if r.get('qmc')]
        if not evaluated or v10_roi is None:
            continue
        best = max(evaluated, key=lambda r: r['qmc']['roi'])
        diff = best['qmc']['roi'] - v10_roi
        flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
        summary_rows.append((vid, res['name'], res['n_test'], v10_roi, best['qmc']['roi'],
                             best['name'], diff, flag))
        recipe_str = ', '.join(f'v{s}×{w:.1f}' for s, w in best['recipe'])
        lines.append(f"| {vid} | {res['name']} | {res['n_test']} | "
                     f"{v10_roi:+.2f}% | {best['qmc']['roi']:+.2f}% | {best['name']} "
                     f"({recipe_str[:60]}) | **{diff:+.2f}pt {flag}** |\n")

    # 統計
    if summary_rows:
        improvements = [row[6] for row in summary_rows]
        n_strong = sum(1 for d in improvements if d > 5)
        n_mid = sum(1 for d in improvements if 0 < d <= 5)
        n_worse = sum(1 for d in improvements if d <= 0)
        lines.append(f"\n## 統計\n\n")
        lines.append(f"- 🟢 +5pt 以上: **{n_strong} venues**\n")
        lines.append(f"- 🟡 0〜+5pt: {n_mid} venues\n")
        lines.append(f"- 🔴 悪化: {n_worse} venues\n")
        lines.append(f"- 平均: {float(np.mean(improvements)):+.2f}pt\n")
        lines.append(f"- 中央値: {float(np.median(improvements)):+.2f}pt\n")
        lines.append(f"- 最大改善: {max(improvements):+.2f}pt\n")

    # 詳細: 戸田の recipe 一覧
    lines.append("\n## 戸田 (target venue 2) 全 recipe 評価\n\n")
    toda_res = results.get(2)
    if toda_res:
        lines.append("| rank | recipe name | fast top-1% | QMC top-3% | ROI |\n|---|---|---|---|---|\n")
        sorted_recipes = sorted(toda_res['recipes'],
                                key=lambda r: -(r['qmc']['roi'] if r.get('qmc') else -999))
        for i, r in enumerate(sorted_recipes):
            roi = r['qmc']['roi'] if r.get('qmc') else 'N/A'
            top3 = r['qmc']['top3_rate'] if r.get('qmc') else 'N/A'
            roi_str = f"{roi:+.2f}%" if isinstance(roi, float) else 'N/A'
            top3_str = f"{top3:.2f}%" if isinstance(top3, float) else 'N/A'
            lines.append(f"| {i+1} | {r['name']} | {r['fast_top1']:.2f}% | {top3_str} | {roi_str} |\n")

        # Greedy history
        lines.append("\n### 戸田 greedy forward 履歴\n\n")
        for step, vid, score in toda_res['greedy_history']:
            name = VENUE_NAMES.get(vid, str(vid))
            lines.append(f"- {step}: venue {vid} ({name}) → fast top-1 {score:.2f}%\n")

    # 留意
    lines.append("\n## 留意事項\n\n")
    lines.append("- Phase 1 (fast score) は NN-only top-1、Phase 2 (QMC) と必ずしも一致しない\n")
    lines.append("- 各 venue n=70-170 races で検出力限界、改善幅は noise の可能性\n")
    lines.append("- 2026-05 単月 hold-out のみ、forward 月跨ぎ再現性は別途検証\n")
    lines.append("- 採用候補は shadow 並走 2 週間必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
