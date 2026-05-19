"""戸田特化 calibrator PoC (案 X Phase 1 minimum viable)

戸田 V10 NN 出力に IsotonicRegression × 6 boats の calibrator を fit、
test 期間で QMC + top-3 hit率 / ROI を比較。

Train: 2026-03 (n=133)
Test:  2026-04 + 2026-05 (n=304)

判定:
  🟢 calibrator あり ROI > calibrator なし ROI + 5pt → Phase 2 (QMC 係数調整) に進む
  🟡 ROI 差 0〜5pt → 効果限定、別アプローチ検討
  🔴 ROI 差 < 0 → 凍結

入力: DB predictions + boats + races (戸田 venue 2)
出力: analysis/reports/59_toda_calibrator_poc.md + models/toda_calibrator.pkl
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
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.isotonic import IsotonicRegression

from src.monte_carlo import qmc_sanrentan_v3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / 'analysis' / 'reports' / '59_toda_calibrator_poc.md'
CALIBRATOR_PATH = ROOT / 'models' / 'toda_calibrator.pkl'

TODA_VENUE_ID = 2
TRAIN_END = date(2026, 3, 31)


def fetch_toda_data():
    """戸田 races の V10 predictions + boats + race_data + actual."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    # races + predictions
    cur.execute("""
        SELECT
            r.id AS race_id,
            r.race_date,
            r.race_number,
            r.result_1st,
            r.result_2nd,
            r.result_3rd,
            r.actual_result_trifecta,
            r.payout_sanrentan,
            r.wind_speed,
            r.wave_height,
            p.probabilities_1st
        FROM races r
        JOIN predictions p ON p.race_id = r.id
        WHERE r.venue_id = %s
          AND r.result_1st IS NOT NULL
          AND p.probabilities_1st IS NOT NULL
          AND p.id = (SELECT MAX(id) FROM predictions WHERE race_id = r.id)
        ORDER BY r.race_date, r.race_number
    """, (TODA_VENUE_ID,))
    races = cur.fetchall()

    race_ids = [r['race_id'] for r in races]
    cur.execute("""
        SELECT race_id, boat_number, player_class, win_rate_2, local_win_rate_2,
               avg_st, motor_win_rate_2, exhibition_time, approach_course,
               is_new_motor, tilt, parts_changed, weight
        FROM boats WHERE race_id = ANY(%s)
        ORDER BY race_id, boat_number
    """, (race_ids,))
    boats_map = {}
    for r in cur.fetchall():
        boats_map.setdefault(r['race_id'], []).append(dict(r))
    conn.close()
    # boats を 6 艇順序で sort して races に attach
    result = []
    for r in races:
        boats = sorted(boats_map.get(r['race_id'], []), key=lambda x: x['boat_number'])
        if len(boats) != 6:
            continue
        result.append({
            **dict(r),
            'boats': boats,
            'race_data': {
                'wind_speed': r['wind_speed'],
                'wave_height': r['wave_height'],
            },
        })
    return result


def fit_calibrator(train_races):
    """戸田 train data で IsotonicRegression × 6 boats fit."""
    boat_preds = [[] for _ in range(6)]
    boat_actuals = [[] for _ in range(6)]
    for r in train_races:
        probs = r['probabilities_1st']
        actual_1st = r['result_1st']
        for boat in range(6):
            boat_preds[boat].append(float(probs[boat]))
            boat_actuals[boat].append(1 if actual_1st == boat + 1 else 0)
    calibrators = []
    for boat in range(6):
        cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        cal.fit(boat_preds[boat], boat_actuals[boat])
        calibrators.append(cal)
        logger.info(f"calibrator boat {boat+1}: train n={len(boat_preds[boat])}, "
                    f"raw mean pred={np.mean(boat_preds[boat]):.4f}, "
                    f"actual rate={np.mean(boat_actuals[boat]):.4f}")
    return calibrators


def apply_calibrator(probs, calibrators):
    """Calibrator 適用 + 正規化."""
    calibrated = np.array([float(calibrators[i].predict([probs[i]])[0]) for i in range(6)])
    s = calibrated.sum()
    if s > 0:
        calibrated = calibrated / s
    return calibrated.tolist()


def backtest_race(r, probs):
    """1 race で QMC 計算 → top-3 picks → ROI 試算."""
    try:
        qmc_probs = qmc_sanrentan_v3(
            probs, boats_data=r['boats'],
            race_data=r['race_data'], race_number=r['race_number'],
            n_simulations=8192, seed=42,
        )
    except Exception as e:
        logger.warning(f"QMC failed race {r['race_id']}: {e}")
        return None
    top3 = sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]
    top3_combos = [t[0] for t in top3]
    actual = r['actual_result_trifecta']
    if not actual:
        actual = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
    top1_hit = top3_combos[0] == actual
    top3_hit = actual in top3_combos
    # top-3 全 picks ¥100 ずつ購入
    invested = 300
    returned = float(r['payout_sanrentan'] or 0) if top3_hit else 0
    return {
        'race_id': r['race_id'],
        'top_pick': top3_combos[0],
        'top3': top3_combos,
        'actual': actual,
        'top1_hit': top1_hit,
        'top3_hit': top3_hit,
        'invested': invested,
        'returned': returned,
        'pnl': returned - invested,
    }


def evaluate(races, probs_fn, label):
    """races 全体で backtest 集計."""
    results = []
    for r in races:
        probs = probs_fn(r)
        if probs is None:
            continue
        out = backtest_race(r, probs)
        if out:
            results.append(out)
    n = len(results)
    if n == 0:
        return None
    top1 = sum(1 for x in results if x['top1_hit'])
    top3 = sum(1 for x in results if x['top3_hit'])
    invested = sum(x['invested'] for x in results)
    returned = sum(x['returned'] for x in results)
    pnl = returned - invested
    roi = pnl / invested * 100 if invested else 0
    return {
        'label': label,
        'n': n,
        'top1': top1,
        'top1_rate': top1 / n * 100,
        'top3': top3,
        'top3_rate': top3 / n * 100,
        'invested': invested,
        'returned': returned,
        'pnl': pnl,
        'roi': roi,
    }


def main():
    logger.info("戸田 calibrator PoC (案 X Phase 1)")
    races = fetch_toda_data()
    logger.info(f"戸田 races (V10 予測 + boats あり): {len(races)}")

    # Train (2026-03) / Test (2026-04 + 05) 分割
    train = [r for r in races if r['race_date'] <= TRAIN_END]
    test = [r for r in races if r['race_date'] > TRAIN_END]
    logger.info(f"Train: {len(train)} races, Test: {len(test)} races")

    if len(train) < 100 or len(test) < 50:
        logger.warning("Train/Test 不足")

    # Calibrator fit
    logger.info("Calibrator fit (戸田 train)")
    calibrators = fit_calibrator(train)

    # 保存
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATOR_PATH, 'wb') as f:
        pickle.dump({
            'venue_id': TODA_VENUE_ID,
            'venue_name': 'Toda',
            '1st': calibrators,
            'fitted_at': '2026-05-19',
            'n_samples': len(train),
            'train_period': '2026-03',
        }, f)
    logger.info(f"calibrator 保存: {CALIBRATOR_PATH}")

    # Test backtest: baseline (V10 raw) vs calibrated
    logger.info("Test backtest (calibrator なし)")
    raw_fn = lambda r: r['probabilities_1st']
    s_raw = evaluate(test, raw_fn, 'V10 raw')

    logger.info("Test backtest (calibrator あり)")
    cal_fn = lambda r: apply_calibrator(r['probabilities_1st'], calibrators)
    s_cal = evaluate(test, cal_fn, 'V10 + 戸田 calibrator')

    # Train backtest も比較 (overfitting check)
    logger.info("Train backtest (cross check)")
    s_train_raw = evaluate(train, raw_fn, 'Train V10 raw')
    s_train_cal = evaluate(train, cal_fn, 'Train V10 + cal')

    # レポート
    lines = []
    lines.append("# 戸田 calibrator PoC (案 X Phase 1)\n\n")
    lines.append(f"Train: 2026-03 (n={len(train)}), Test: 2026-04 + 2026-05 (n={len(test)})\n\n")

    # サマリ
    lines.append("## Test 期間 backtest (forward 検証)\n\n")
    lines.append("| 戦略 | n | top-1 hit% | top-3 hit% | 投資 (¥) | 回収 (¥) | PnL | ROI |\n|---|---|---|---|---|---|---|---|\n")
    for s in [s_raw, s_cal]:
        if not s: continue
        lines.append(f"| {s['label']} | {s['n']} | {s['top1_rate']:.2f}% | {s['top3_rate']:.2f}% | "
                     f"¥{s['invested']:,} | ¥{s['returned']:,.0f} | ¥{s['pnl']:+,.0f} | {s['roi']:+.2f}% |\n")

    # 改善幅
    if s_raw and s_cal:
        diff_roi = s_cal['roi'] - s_raw['roi']
        diff_hit = s_cal['top3_rate'] - s_raw['top3_rate']
        lines.append(f"\n**改善幅 (Test)**: top-3 hit {diff_hit:+.2f}pt, ROI {diff_roi:+.2f}pt\n")

    # Train 比較
    lines.append("\n## Train 期間 backtest (overfit check)\n\n")
    lines.append("| 戦略 | n | top-3 hit% | ROI |\n|---|---|---|---|\n")
    for s in [s_train_raw, s_train_cal]:
        if not s: continue
        lines.append(f"| {s['label']} | {s['n']} | {s['top3_rate']:.2f}% | {s['roi']:+.2f}% |\n")

    # Calibrator 動作の可視化
    lines.append("\n## Calibrator が NN 出力をどう変えたか\n\n")
    lines.append("Test 期間で boat-1 raw prob と calibrated prob の差を集計:\n\n")
    boat_stats = []
    for boat in range(6):
        raws = [float(r['probabilities_1st'][boat]) for r in test]
        cals = [float(calibrators[boat].predict([raw])[0]) for raw in raws]
        actual_rate = sum(1 for r in test if r['result_1st'] == boat + 1) / len(test) * 100
        boat_stats.append({
            'boat': boat + 1,
            'raw_mean': float(np.mean(raws)) * 100,
            'cal_mean': float(np.mean(cals)) * 100,
            'actual_rate': actual_rate,
        })
    lines.append("| boat | raw mean% | calibrated mean% | actual rate% | raw bias | cal bias |\n|---|---|---|---|---|---|\n")
    for bs in boat_stats:
        raw_bias = bs['raw_mean'] - bs['actual_rate']
        cal_bias = bs['cal_mean'] - bs['actual_rate']
        lines.append(f"| {bs['boat']} | {bs['raw_mean']:.2f}% | {bs['cal_mean']:.2f}% | "
                     f"{bs['actual_rate']:.2f}% | {raw_bias:+.2f}pt | {cal_bias:+.2f}pt |\n")

    # 自動判定
    lines.append("\n## 自動判定 (CLAUDE.md 採用基準)\n\n")
    if s_raw and s_cal:
        diff_roi = s_cal['roi'] - s_raw['roi']
        if diff_roi > 5.0:
            lines.append(f"- 🟢 **calibrator 効果 大** (ROI {diff_roi:+.2f}pt) → **Phase 2 (戸田 QMC 係数調整)** に進む\n")
        elif diff_roi > 0:
            lines.append(f"- 🟡 **calibrator 効果 小** (ROI {diff_roi:+.2f}pt) → 効果限定、別アプローチ検討\n")
        else:
            lines.append(f"- 🔴 **calibrator 効果なし or 悪化** (ROI {diff_roi:+.2f}pt) → 凍結\n")

        # Train との overfit check
        if s_train_cal and s_cal:
            overfit_gap = s_train_cal['roi'] - s_cal['roi']
            lines.append(f"- overfit check: Train ROI {s_train_cal['roi']:+.2f}% vs Test ROI {s_cal['roi']:+.2f}% (gap {overfit_gap:+.2f}pt)\n")
            if overfit_gap > 20:
                lines.append("  - ⚠️ Train-Test gap 大、overfit 疑い\n")

    # 留意
    lines.append("\n## 留意事項\n\n")
    lines.append("- Train n=133 races は IsotonicRegression にギリギリ充足、calibrator 精度に限界\n")
    lines.append("- Test n=304 races (forward) で +5pt 以上の ROI 改善が安定して出れば実用候補\n")
    lines.append("- top-3 全部購入 proxy は Kelly/EV filter 不含、本番 ROI とは別\n")
    lines.append("- 効果ありなら、次に Phase 2 (QMC 係数調整) と組み合わせ\n")
    lines.append("- 結論は岩下さん判断、shadow 並走必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
