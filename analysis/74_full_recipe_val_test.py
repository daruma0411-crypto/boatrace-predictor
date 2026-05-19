"""足し算 + 引き算 全 recipe を val/test 分離で評価

73 (足し算のみ、test 全 recipe 評価) を拡張:
  - 引き算 recipes (negative weight) を追加
  - test eval は val-best 1 + V10 baseline のみで高速化
  - 足し算 best vs 引き算 best vs V10 を比較

Recipe 候補 (per target venue):
  足し算: V10、own、V10+own blend (10段)、top-K sim avg、all24、functional、own+functional
  引き算: V10 - α × opp_avg、V10+own - α × opp、own - α × opp

判定: val→test で真の signal:
  - V10 baseline と比較した改善幅
  - 足し算 vs 引き算 で大きい方が best
  - +5pt 以上 → shadow 候補

出力: analysis/reports/74_full_recipe_val_test.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '74_full_recipe_val_test.md'

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
FUNCTIONAL = [2, 3, 7, 12, 13, 14, 16, 24]


def load_specialists():
    return {vid: lgb.Booster(model_file=str(SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt'))
            for vid in VENUE_NAMES if (SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt').exists()}


def get_venue_distances(venue_preds):
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


def build_records(venue_preds, specialists, fe, scaler):
    by_venue = {}
    for tvid, preds in venue_preds.items():
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


def generate_all_recipes(target_vid, similar_vids, opposite_vids):
    """足し算 + 引き算 全 recipe."""
    recipes = []
    # 足し算 (71 と同じ)
    recipes.append(('R0_V10', [('v10', 1.0)]))
    recipes.append(('R1_own', [(target_vid, 1.0)]))
    for i, alpha in enumerate(np.linspace(0.1, 0.9, 9)):
        recipes.append((f'R+_V10x{1-alpha:.1f}+own_x{alpha:.1f}',
                        [('v10', 1-alpha), (target_vid, alpha)]))
    for K in [2, 3, 5]:
        if len(similar_vids) >= K:
            recipes.append((f'R+_top{K}_sim',
                            [(target_vid, 1.0)] + [(v, 1.0) for v in similar_vids[:K]]))
    recipes.append(('R+_all24', [(v, 1.0) for v in VENUE_NAMES.keys()]))
    recipes.append(('R+_functional', [(v, 1.0) for v in FUNCTIONAL]))
    recipes.append(('R+_own2+functional',
                    [(target_vid, 2.0)] + [(v, 1.0) for v in FUNCTIONAL if v != target_vid]))
    # 引き算
    opp3 = opposite_vids[:3]
    opp5 = opposite_vids[:5]
    # V10 - α × opp3
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        rec = [('v10', 1.0)] + [(v, -alpha/3) for v in opp3]
        recipes.append((f'R-_V10-opp3x{alpha}', rec))
    # V10 + own - opp3
    for alpha in [0.1, 0.2, 0.3]:
        for beta in [0.3, 0.5]:
            rec = [('v10', 1-beta), (target_vid, beta)] + [(v, -alpha/3) for v in opp3]
            recipes.append((f'R-_V10x{1-beta:.1f}+ownx{beta:.1f}-opp3x{alpha}', rec))
    # own - opp3
    for alpha in [0.1, 0.2, 0.3]:
        rec = [(target_vid, 1.0)] + [(v, -alpha/3) for v in opp3]
        recipes.append((f'R-_own-opp3x{alpha}', rec))
    # V10 - α × opp5
    for alpha in [0.2, 0.3]:
        rec = [('v10', 1.0)] + [(v, -alpha/5) for v in opp5]
        recipes.append((f'R-_V10-opp5x{alpha}', rec))
    return recipes


def main():
    logger.info("足し算 + 引き算 val/test 分離評価")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    specialists = load_specialists()
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    logger.info("records pre-compute")
    by_venue = build_records(venue_preds, specialists, fe, scaler)
    distances = get_venue_distances(venue_preds)

    results = {}
    for tvid in sorted(VENUE_NAMES.keys()):
        recs = by_venue.get(tvid, {})
        val_recs, test_recs = recs.get('val', []), recs.get('test', [])
        if len(val_recs) < 30 or len(test_recs) < 30:
            continue
        sims = [v for v, _ in distances.get(tvid, [])][:10]
        opps = [v for v, _ in sorted(distances.get(tvid, []), key=lambda x: -x[1])][:5]
        recipes = generate_all_recipes(tvid, sims, opps)
        logger.info(f"venue {tvid} ({VENUE_NAMES[tvid]}): val={len(val_recs)}, test={len(test_recs)}, recipes={len(recipes)}")

        # Phase 1: val 全 recipe 評価
        val_scores = []
        for name, recipe in recipes:
            s = qmc_score(recipe, val_recs)
            if s:
                kind = '足し算' if not name.startswith('R-_') else '引き算'
                if name in ('R0_V10', 'R1_own'):
                    kind = 'baseline'
                val_scores.append({'name': name, 'kind': kind, 'recipe': recipe, 'val_qmc': s})
        val_scores.sort(key=lambda x: -x['val_qmc']['roi'])

        # Phase 2: val-best (overall, additive, subtractive) を test 評価
        val_best = val_scores[0]
        val_add_best = next((r for r in val_scores if r['kind'] == '足し算'), None)
        val_sub_best = next((r for r in val_scores if r['kind'] == '引き算'), None)
        v10_obj = next(r for r in val_scores if r['name'] == 'R0_V10')

        test_v10 = qmc_score(v10_obj['recipe'], test_recs)
        test_best = qmc_score(val_best['recipe'], test_recs)
        test_add_best = qmc_score(val_add_best['recipe'], test_recs) if val_add_best else None
        test_sub_best = qmc_score(val_sub_best['recipe'], test_recs) if val_sub_best else None

        results[tvid] = {
            'name': VENUE_NAMES[tvid],
            'n_val': len(val_recs), 'n_test': len(test_recs),
            'val_best': val_best,
            'val_add_best': val_add_best, 'val_sub_best': val_sub_best,
            'test_v10': test_v10, 'test_best': test_best,
            'test_add_best': test_add_best, 'test_sub_best': test_sub_best,
        }

    # Report
    lines = []
    lines.append("# 足し算 + 引き算 recipe val/test 分離評価\n\n")
    lines.append("val=2026-04 で recipe 選定 → test=2026-05 hold-out で評価\n\n")
    lines.append("## venue 別: 足し算 vs 引き算 best 比較\n\n")
    lines.append("| venue | name | V10 test ROI | 足し算 best test ROI | 引き算 best test ROI | overall best | 真の改善 |\n|---|---|---|---|---|---|---|\n")
    real_improvements_overall = []
    real_improvements_add = []
    real_improvements_sub = []
    for vid in sorted(results.keys()):
        r = results[vid]
        v10 = r['test_v10']['roi']
        add = r['test_add_best']['roi'] if r['test_add_best'] else None
        sub = r['test_sub_best']['roi'] if r['test_sub_best'] else None
        overall = r['test_best']['roi']
        diff = overall - v10
        diff_add = (add - v10) if add is not None else None
        diff_sub = (sub - v10) if sub is not None else None
        real_improvements_overall.append(diff)
        if diff_add is not None:
            real_improvements_add.append(diff_add)
        if diff_sub is not None:
            real_improvements_sub.append(diff_sub)
        flag = '🟢' if diff > 5 else ('🟡' if diff > 0 else '🔴')
        best_kind = r['val_best']['kind']
        lines.append(f"| {vid} | {r['name']} | {v10:+.2f}% | "
                     f"{add:+.2f}% (val: {r['val_add_best']['val_qmc']['roi']:+.2f}%)" if add else 'N/A')
        lines.append(f" | {sub:+.2f}% (val: {r['val_sub_best']['val_qmc']['roi']:+.2f}%)" if sub else ' | N/A')
        lines.append(f" | **{overall:+.2f}% ({best_kind})** | **{diff:+.2f}pt {flag}** |\n")

    # 統計
    def stats(arr):
        return (np.mean(arr), np.median(arr), sum(1 for x in arr if x > 5),
                sum(1 for x in arr if 0 < x <= 5), sum(1 for x in arr if x <= 0))

    if real_improvements_overall:
        mo, md, ns, nm, nw = stats(real_improvements_overall)
        lines.append(f"\n## 統計 (overall best、val→test 真の signal)\n\n")
        lines.append(f"- 🟢 +5pt 以上: **{ns} venues**\n")
        lines.append(f"- 🟡 0〜+5pt: {nm} venues\n")
        lines.append(f"- 🔴 悪化: {nw} venues\n")
        lines.append(f"- 平均: {mo:+.2f}pt / 中央値: {md:+.2f}pt\n")

    if real_improvements_add and real_improvements_sub:
        ma, _, _, _, _ = stats(real_improvements_add)
        ms, _, _, _, _ = stats(real_improvements_sub)
        lines.append(f"\n## 足し算 vs 引き算\n\n")
        lines.append(f"- 足し算 best 平均改善: {ma:+.2f}pt\n")
        lines.append(f"- 引き算 best 平均改善: {ms:+.2f}pt\n")
        lines.append(f"- 差: 引き算 {ms - ma:+.2f}pt {'勝ち' if ms > ma else '負け'}\n")

    # 引き算 best が overall best だった venues
    sub_winners = [vid for vid, r in results.items()
                   if r['val_best']['kind'] == '引き算']
    lines.append(f"\n## 引き算が overall best だった venues ({len(sub_winners)}/24)\n\n")
    for vid in sub_winners:
        r = results[vid]
        lines.append(f"- {r['name']} (v{vid}): {r['val_best']['name']}, "
                     f"test ROI {r['test_best']['roi']:+.2f}% (vs V10 {r['test_v10']['roi']:+.2f}%, "
                     f"差 {r['test_best']['roi']-r['test_v10']['roi']:+.2f}pt)\n")

    # 戸田 詳細
    lines.append("\n## 戸田 (target 2) 詳細\n\n")
    toda = results.get(2)
    if toda:
        lines.append(f"V10 test ROI: {toda['test_v10']['roi']:+.2f}%\n\n")
        lines.append("| kind | recipe | val ROI | test ROI |\n|---|---|---|---|\n")
        for label, obj_v, obj_t in [
            ('足し算 best', toda['val_add_best'], toda['test_add_best']),
            ('引き算 best', toda['val_sub_best'], toda['test_sub_best']),
            ('overall best', toda['val_best'], toda['test_best']),
        ]:
            if obj_v and obj_t:
                lines.append(f"| {label} | {obj_v['name']} | {obj_v['val_qmc']['roi']:+.2f}% | {obj_t['roi']:+.2f}% |\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- val=2026-04 / test=2026-05 共に GW 前後の特異期、forward 月跨ぎ再現性は別途必要\n")
    lines.append("- recipe 選定は val のみ、test は純粋 hold-out\n")
    lines.append("- 採用候補は shadow 並走 2 週間必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
