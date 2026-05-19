"""戸田 QMC 係数 1D grid (A1 係数のみ) (案 X Phase 1 Step 2.1)

戸田 2271 races で compute_ratings_early の A1 係数を grid search。
他は heuristic 据え置き。
baseline (A1=0.75) vs 拡大 (0.85/0.95/1.05) で hit 率 / ROI を比較。

仮説: 戸田の A1 は他会場より -10.9pt 弱い → A1 係数 0.75 は過小、std 拡大すべき。
ただし B' で全会場 grid は失敗、戸田単独で意味ある signal が出るか data で確認。

入力: analysis/toda_v10_predictions.pkl (Step 1 で作成)
出力: analysis/reports/62_toda_qmc_grid_a1.md
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
from scipy.stats import qmc, norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PRED_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '62_toda_qmc_grid_a1.md'

# Train / Test 分割
TRAIN_END = date(2026, 4, 30)  # 2026-04-30 まで train、2026-05 が hold-out
N_SIM = 8192
SEED = 42


def compute_ratings_with_a1(probs_1st, boats_data, race_data, race_number, a1_coef):
    """A1 係数のみ可変、他は compute_ratings_early のまま。"""
    probs = np.array(probs_1st, dtype=np.float64)
    probs = np.clip(probs, 0.01, 0.99)
    ratings = np.log(probs / (1.0 - probs))

    base_std = 0.8
    stds = np.full(6, base_std)

    if race_data:
        wind = race_data.get('wind_speed') or 0
        wave = race_data.get('wave_height') or 0
        wf = 1.0
        if wind >= 5: wf += 0.15
        elif wind >= 3: wf += 0.05
        if wave >= 5: wf += 0.10
        elif wave >= 3: wf += 0.05
        stds *= wf

    if boats_data and len(boats_data) == 6:
        class_values = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        classes = [class_values.get(b.get('player_class', 'B1'), 2) for b in boats_data]
        class_std = float(np.std(classes))
        csf = np.clip(1.10 - 0.17 * class_std, 0.85, 1.10)
        stds *= csf

    ex_times = [b.get('exhibition_time') for b in boats_data if b.get('exhibition_time') and b['exhibition_time'] > 0]
    if len(ex_times) >= 4:
        ex_range = max(ex_times) - min(ex_times)
        stds *= (1.05 - 0.5 * min(ex_range, 0.3))
    avg_exhibition = sum(ex_times) / len(ex_times) if ex_times else None

    if boats_data and len(boats_data) == 6:
        for i, b in enumerate(boats_data):
            pc = b.get('player_class', 'B1')
            cf = {'A1': a1_coef, 'A2': 0.90, 'B1': 1.10, 'B2': 1.35}.get(pc, 1.0)
            stds[i] *= cf
            mwr = b.get('motor_win_rate_2', 30.0) or 30.0
            if mwr > 50.0 or mwr < 15.0: stds[i] *= 1.1
            if b.get('parts_changed', False): stds[i] *= 1.15
            et = b.get('exhibition_time')
            if et and et > 0 and avg_exhibition:
                d = et - avg_exhibition
                if d < -0.05: stds[i] *= 0.88
                elif d > 0.10: stds[i] *= 1.12
            ast = b.get('avg_st')
            if ast is not None:
                if ast > 0.20: stds[i] *= 1.10
                elif ast < 0.10: stds[i] *= 1.08
            crs = b.get('approach_course')
            if crs is not None:
                if crs >= 5: stds[i] *= 1.15
                elif crs >= 4: stds[i] *= 1.05
            lwr = b.get('local_win_rate_2') or b.get('local_win_rate')
            if lwr is not None and lwr > 0:
                if lwr > 40.0: stds[i] *= 0.90
                elif lwr < 15.0: stds[i] *= 1.10
    return ratings, stds


def qmc_run(probs, boats, race_data, race_number, a1_coef, n=N_SIM, seed=SEED):
    ratings, stds = compute_ratings_with_a1(probs, boats, race_data, race_number, a1_coef)
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n, 64))))
    n_actual = 2 ** m
    us = sampler.random(n_actual)
    perfs = norm.ppf(us, loc=ratings, scale=stds)
    orderings = np.argsort(-perfs, axis=1)
    top3 = orderings[:, :3] + 1
    keys = [f"{t[0]}-{t[1]}-{t[2]}" for t in top3]
    cnt = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    return {k: v / n_actual for k, v in cnt.items() if v > 0}


def eval_a1(races, a1_coef, label):
    """全 races で a1_coef で QMC 計算、top-3 hit率 + ROI proxy."""
    n = len(races)
    top1_hit = 0
    top3_hit = 0
    invested = 0
    returned = 0
    for r in races:
        qmc_probs = qmc_run(r['probs_1st'], r['boats'], r['race_data'], r['race_number'], a1_coef)
        top3 = sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]
        top3_combos = [t[0] for t in top3]
        actual = r['actual']
        if top3_combos[0] == actual:
            top1_hit += 1
        if actual in top3_combos:
            top3_hit += 1
            returned += r['payout'] or 0
        invested += 300  # top-3 × ¥100
    pnl = returned - invested
    roi = pnl / invested * 100 if invested else 0
    return {
        'label': label, 'a1': a1_coef, 'n': n,
        'top1_rate': top1_hit / n * 100,
        'top3_rate': top3_hit / n * 100,
        'invested': invested, 'returned': returned, 'pnl': pnl, 'roi': roi,
    }


def main():
    logger.info("戸田 QMC 1D grid (A1)")
    with open(PRED_PATH, 'rb') as f:
        predictions = pickle.load(f)
    logger.info(f"戸田 predictions: {len(predictions)}")

    train = []
    test = []
    for rid, p in predictions.items():
        d = date.fromisoformat(p['race_date'])
        if d <= TRAIN_END:
            train.append(p)
        else:
            test.append(p)
    logger.info(f"Train: {len(train)}, Test: {len(test)}")

    a1_grid = [0.75, 0.85, 0.95, 1.05, 1.15]

    lines = []
    lines.append("# 戸田 QMC 1D grid (A1 係数) (案 X Phase 1 Step 2.1)\n\n")
    lines.append(f"Train: 2025-06〜2026-04 (n={len(train)}), Test: 2026-05 (n={len(test)})\n")
    lines.append("compute_ratings_early の A1 係数のみ可変。他は全国 heuristic 据え置き。\n")
    lines.append("仮説: 戸田 A1 は他会場より -10.9pt 弱い → A1 係数 0.75 (std 縮小) は過小。\n\n")

    lines.append("## Train 期間 grid search\n\n")
    lines.append("| A1 係数 | n | top-1 hit% | top-3 hit% | 投資 ¥ | 回収 ¥ | PnL | ROI |\n|---|---|---|---|---|---|---|---|\n")
    train_results = []
    for a1 in a1_grid:
        logger.info(f"[Train] A1={a1}")
        s = eval_a1(train, a1, f'A1={a1}')
        train_results.append(s)
        lines.append(f"| {a1} | {s['n']} | {s['top1_rate']:.2f}% | {s['top3_rate']:.2f}% | "
                     f"¥{s['invested']:,} | ¥{s['returned']:,.0f} | ¥{s['pnl']:+,.0f} | {s['roi']:+.2f}% |\n")
    # best train
    best_train = max(train_results, key=lambda x: x['roi'])
    lines.append(f"\n**Train best A1**: {best_train['a1']} (ROI {best_train['roi']:+.2f}%)\n")

    lines.append("\n## Test 期間 (hold-out) 検証\n\n")
    lines.append("| A1 係数 | n | top-1 hit% | top-3 hit% | 投資 ¥ | 回収 ¥ | PnL | ROI |\n|---|---|---|---|---|---|---|---|\n")
    test_results = []
    for a1 in a1_grid:
        logger.info(f"[Test] A1={a1}")
        s = eval_a1(test, a1, f'A1={a1}')
        test_results.append(s)
        lines.append(f"| {a1} | {s['n']} | {s['top1_rate']:.2f}% | {s['top3_rate']:.2f}% | "
                     f"¥{s['invested']:,} | ¥{s['returned']:,.0f} | ¥{s['pnl']:+,.0f} | {s['roi']:+.2f}% |\n")
    best_test = max(test_results, key=lambda x: x['roi'])
    lines.append(f"\n**Test best A1**: {best_test['a1']} (ROI {best_test['roi']:+.2f}%)\n")

    # Train baseline (A1=0.75) vs Test best comparison
    train_baseline = next(s for s in train_results if s['a1'] == 0.75)
    test_baseline = next(s for s in test_results if s['a1'] == 0.75)
    train_best_idx = train_results.index(best_train)
    test_at_train_best = test_results[train_best_idx]

    lines.append("\n## 自動判定 (CLAUDE.md 採用基準)\n\n")
    lines.append(f"- Train baseline (A1=0.75) ROI {train_baseline['roi']:+.2f}% vs Train best (A1={best_train['a1']}) ROI {best_train['roi']:+.2f}%\n")
    lines.append(f"- Test baseline (A1=0.75) ROI {test_baseline['roi']:+.2f}% vs Test at Train best A1={best_train['a1']} ROI {test_at_train_best['roi']:+.2f}%\n")

    train_improvement = best_train['roi'] - train_baseline['roi']
    test_improvement = test_at_train_best['roi'] - test_baseline['roi']
    lines.append(f"- Train 改善幅: {train_improvement:+.2f}pt\n")
    lines.append(f"- **Test 改善幅 (forward)**: {test_improvement:+.2f}pt\n")

    if test_improvement > 5.0:
        lines.append(f"  → 🟢 forward でも改善、**A1 係数 {best_train['a1']} 採用候補**\n")
        lines.append(f"  → Phase 2 (展示偏差係数 grid) に進む\n")
    elif test_improvement > 0:
        lines.append(f"  → 🟡 forward で微改善、効果限定\n")
    else:
        lines.append(f"  → 🔴 forward で改善なし、A1 grid は機能せず\n")

    # Train-Test gap (overfit)
    overfit = train_improvement - test_improvement
    lines.append(f"- overfit gap (Train 改善 - Test 改善): {overfit:+.2f}pt\n")
    if overfit > 10:
        lines.append("  - ⚠️ Train で改善するが Test で消える = overfit 疑い\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- A1 係数のみ grid (1D)、他 10 項目は全国 heuristic\n")
    lines.append("- top-3 全部購入 proxy は Kelly/EV filter 不含、相対比較指標\n")
    lines.append("- Test n=136 races はサンプル少、forward 検出力ぎりぎり\n")
    lines.append("- 効果あれば Phase 2 (展示偏差) に進む、なければ凍結\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
