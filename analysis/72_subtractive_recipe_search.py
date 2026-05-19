"""引き算 recipe 探索 (Phase D 追加、anti-specialist 形態 1)

71 は足し算のみ。今回は **負の重み (引き算)** を recipe に許可。
opposite venues (66 距離 max) を「ノイズ源」として引く。

Recipe 候補 (per target venue):
  既存 71 の recipes (足し算)
  + R_sub_V10_minus_opp(α): V10 - α × opposite_top3_avg
  + R_sub_V10+own_minus_opp: V10 + 0.3 × own - α × opp
  + R_sub_own_minus_opp: own - α × opp
  + R_sub_full: V10 + own - opp + similar (フル形)

負の重みは blended_probs を一部負にする可能性 → clip(0) + 再 normalize で扱う。

出力: analysis/reports/72_subtractive_recipe.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '72_subtractive_recipe.md'

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


def load_specialists():
    models = {}
    for vid in VENUE_NAMES:
        path = SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        if path.exists():
            models[vid] = lgb.Booster(model_file=str(path))
    return models


def get_venue_distances(venue_preds):
    """各 venue の outcome feature 距離行列."""
    feats = {}
    for vid, preds in venue_preds.items():
        if not preds:
            continue
        boat_1st = np.zeros(6)
        for p in preds.values():
            boat_1st[p['result_1st'] - 1] += 1
        boat_1st /= len(preds)
        feats[vid] = boat_1st * 100
    distances = {}
    for vid1 in feats:
        d = {}
        for vid2 in feats:
            if vid1 == vid2:
                continue
            d[vid2] = float(np.linalg.norm(feats[vid1] - feats[vid2]))
        distances[vid1] = d
    return distances


def build_test_predictions(venue_preds, specialists, fe, scaler):
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
                spec_probs = {}
                for sid, sm in specialists.items():
                    pred = sm.predict(features.reshape(1, -1), num_iteration=sm.best_iteration)
                    spec_probs[sid] = pred[0]
                v10_probs = np.array(p['probs_1st'])
                records.append({
                    'rid': rid, 'features': features,
                    'v10_probs': v10_probs, 'spec_probs': spec_probs,
                    'actual': p['actual'], 'result_1st': p['result_1st'],
                    'payout': p['payout'] or 0,
                    'prediction': p,
                })
            except Exception:
                continue
        test_by_venue[target_vid] = records
    return test_by_venue


def blend_probs(recipe, record):
    """recipe = list of (source, weight)、weight 負も許可."""
    probs = np.zeros(6)
    for source, w in recipe:
        if source == 'v10':
            probs += w * record['v10_probs']
        else:
            probs += w * record['spec_probs'][source]
    # negative clip + normalize
    probs = np.clip(probs, 0.001, None)
    s = probs.sum()
    if s > 0:
        probs = probs / s
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
        if top3[0][0] == r['actual']:
            n_top1 += 1
        if r['actual'] in [t[0] for t in top3]:
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


def generate_subtractive_recipes(target_vid, distances):
    """target venue の opposite top-3/5 を引く recipe を生成."""
    # opposite top-3/5 (距離 max)
    sorted_opp = sorted(distances[target_vid].items(), key=lambda x: -x[1])
    opp3 = [vid for vid, _ in sorted_opp[:3]]
    opp5 = [vid for vid, _ in sorted_opp[:5]]
    recipes = []
    # R_sub_V10_minus_opp(α): V10 - α × opp_avg
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        rec = [('v10', 1.0)] + [(vid, -alpha / len(opp3)) for vid in opp3]
        recipes.append((f'R_sub_V10_minus_opp3x{alpha}', rec))
    # R_sub_V10+own_minus_opp
    for alpha in [0.1, 0.2, 0.3]:
        for beta in [0.3, 0.5]:
            rec = [('v10', 1.0 - beta), (target_vid, beta)] + \
                  [(vid, -alpha / len(opp3)) for vid in opp3]
            recipes.append((f'R_sub_V10_x{1-beta:.1f}+own_x{beta:.1f}-opp3x{alpha}', rec))
    # R_sub_own_minus_opp
    for alpha in [0.1, 0.2, 0.3]:
        rec = [(target_vid, 1.0)] + [(vid, -alpha / len(opp3)) for vid in opp3]
        recipes.append((f'R_sub_own-opp3x{alpha}', rec))
    # R_sub_V10_minus_opp5
    for alpha in [0.2, 0.3]:
        rec = [('v10', 1.0)] + [(vid, -alpha / len(opp5)) for vid in opp5]
        recipes.append((f'R_sub_V10_minus_opp5x{alpha}', rec))
    return recipes, opp3, opp5


def main():
    logger.info("引き算 recipe 探索 (Phase D Subtractive)")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    specialists = load_specialists()
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

    logger.info("Test predictions pre-compute")
    test_by_venue = build_test_predictions(venue_preds, specialists, fe, scaler)
    distances = get_venue_distances(venue_preds)

    results = {}
    for target_vid in sorted(VENUE_NAMES.keys()):
        records = test_by_venue.get(target_vid, [])
        if len(records) < 50:
            continue
        logger.info(f"venue {target_vid} ({VENUE_NAMES[target_vid]}) n={len(records)}")
        recipes, opp3, opp5 = generate_subtractive_recipes(target_vid, distances)

        # V10 baseline
        baseline_recipe = [('v10', 1.0)]
        baseline_qmc = qmc_score(baseline_recipe, records)

        # 評価
        evaluated = []
        for name, recipe in recipes:
            s = qmc_score(recipe, records)
            if s:
                evaluated.append({'name': name, 'recipe': recipe, 'qmc': s})

        evaluated.sort(key=lambda x: -x['qmc']['roi'])
        best = evaluated[0] if evaluated else None
        results[target_vid] = {
            'name': VENUE_NAMES[target_vid],
            'n': len(records),
            'opp3': opp3, 'opp5': opp5,
            'baseline_roi': baseline_qmc['roi'] if baseline_qmc else None,
            'recipes': evaluated[:5],  # top 5
            'best': best,
        }

    # Report
    lines = []
    lines.append("# 引き算 recipe 探索 (Phase D Subtractive)\n\n")
    lines.append("opposite top-3/5 venues を **負の重み** で引いて V10/own のノイズ除去。\n\n")

    lines.append("## venue 別 best 引き算 recipe\n\n")
    lines.append("| venue | name | n | V10 ROI | best sub recipe ROI | 改善 | recipe |\n|---|---|---|---|---|---|---|\n")
    improvements = []
    for vid in sorted(results.keys()):
        r = results[vid]
        if not r['best']:
            continue
        v10_roi = r['baseline_roi']
        best_roi = r['best']['qmc']['roi']
        diff = best_roi - v10_roi
        improvements.append(diff)
        flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
        opp_str = ','.join(f'v{v}' for v in r['opp3'])
        recipe_str = r['best']['name']
        lines.append(f"| {vid} | {r['name']} | {r['n']} | {v10_roi:+.2f}% | "
                     f"{best_roi:+.2f}% | **{diff:+.2f}pt {flag}** | {recipe_str} (opp3={opp_str}) |\n")

    if improvements:
        n_strong = sum(1 for d in improvements if d > 5)
        n_mid = sum(1 for d in improvements if 0 < d <= 5)
        n_worse = sum(1 for d in improvements if d <= 0)
        lines.append(f"\n## 統計 (subtractive recipes only)\n\n")
        lines.append(f"- 🟢 +5pt 以上: **{n_strong} venues**\n")
        lines.append(f"- 🟡 0〜+5pt: {n_mid} venues\n")
        lines.append(f"- 🔴 悪化: {n_worse} venues\n")
        lines.append(f"- 平均改善: {float(np.mean(improvements)):+.2f}pt\n")
        lines.append(f"- 中央値: {float(np.median(improvements)):+.2f}pt\n")
        lines.append(f"- 最大: {max(improvements):+.2f}pt\n")

    # 戸田 詳細
    lines.append("\n## 戸田 (target 2) 引き算 recipe top 5\n\n")
    toda_r = results.get(2)
    if toda_r:
        opp3_names = [VENUE_NAMES.get(v, str(v)) for v in toda_r['opp3']]
        lines.append(f"opposite top-3 (戸田から最遠): {', '.join(f'v{v} ({n})' for v, n in zip(toda_r['opp3'], opp3_names))}\n\n")
        lines.append(f"V10 baseline: ROI {toda_r['baseline_roi']:+.2f}%\n\n")
        lines.append("| rank | recipe | top-3% | ROI |\n|---|---|---|---|\n")
        for i, r in enumerate(toda_r['recipes']):
            lines.append(f"| {i+1} | {r['name']} | {r['qmc']['top3_rate']:.2f}% | {r['qmc']['roi']:+.2f}% |\n")

    # 71 (足し算) との比較
    lines.append("\n## 71 (足し算 best) vs 72 (引き算 best) 比較\n\n")
    # 71 の結果手動入力 (代表値):
    additive_results = {
        2:  ('R6_V10x0.5+own_x0.5', -20.12),
        4:  ('R_greedy_forward', +9.72),
        13: ('R_top2_sim_avg', +24.46),
        23: ('R_top5_sim_avg', +53.56),
    }
    lines.append("| venue | name | V10 | 足し算 best | 引き算 best | 差 |\n|---|---|---|---|---|---|\n")
    for vid in sorted(additive_results.keys()):
        if vid not in results or not results[vid]['best']:
            continue
        add_name, add_roi = additive_results[vid]
        sub_roi = results[vid]['best']['qmc']['roi']
        v10 = results[vid]['baseline_roi']
        better = sub_roi if sub_roi > add_roi else add_roi
        kind = '引き算' if sub_roi > add_roi else '足し算'
        lines.append(f"| {vid} | {VENUE_NAMES[vid]} | {v10:+.2f}% | "
                     f"{add_roi:+.2f}% | {sub_roi:+.2f}% | {kind} 勝ち |\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- 負の重みは blended_probs の clip(0) + 再 normalize で扱う\n")
    lines.append("- 71 と同じく test set 上で recipe 選定、overfitting バイアス含む\n")
    lines.append("- forward 検証 (val/test split) は必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
