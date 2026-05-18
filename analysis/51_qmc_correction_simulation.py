"""QMC 修正シナリオ 6 種 シミュレーション (B' task)

C task (50_qmc_overestimation_root_cause) で判明した H2 (ロジット往復 +12.58pt 増幅) と
H3 (B1 std 0.86x) の関係性を、修正シナリオの data simulation で深掘りする。

CLAUDE.md 批判プロトコル準拠:
  各シナリオに Pro / Con / Unknowns を必ず書き、結論は出さず岩下さんの判断に委ねる。
  「明らかに悪化」「改善するが副作用大」「改善+副作用小」の 3 値フラグで事実ラベル。

検証シナリオ:
  S0 baseline                  — 現状 (cache そのまま)
  S1 H2 単独 (linear)          — ロジット → 線形変換に置換
  S2 H3 単独 (A1 0.75→0.85)    — A1 クラス係数のみ拡大
  S3 H1+H2 (cal v2 + linear)   — calibrator v2 + 線形変換
  S4 H3 grid (A1 0.80/85/90/95) — A1 係数 grid search
  S5 H1+H3 (cal v2 + A1 0.85)  — calibrator v2 + A1 係数

入力: cache (`analysis/qmc_predictions_cache.pkl`) + DB NN probs + calibrators_v2.pkl
出力: analysis/reports/51_qmc_correction_scenarios.md
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy.stats import qmc, norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'qmc_predictions_cache.pkl'
CAL_PATH = ROOT / 'models' / 'calibrators_v2.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '51_qmc_correction_scenarios.md'

N_SIM = 8192  # 同条件比較のため固定 (H5 で 8192 と上位試行は同結果と判明)
SEED = 42      # 再現性


# ============================================================
# QMC 再計算のコア (compute_ratings_early を toggle 化)
# ============================================================

def compute_ratings_modified(
    probs_1st, boats_data, race_data, race_number,
    use_linear: bool = False,
    a1_coef: float = 0.75,
):
    """compute_ratings_early の改造版.

    Args:
      use_linear: True なら logit を線形変換 (probs - mean) * 8.0 に置換 (H2 単独修正)
      a1_coef: A1 クラス係数 (H3 修正用、デフォルト 0.75)
    """
    probs = np.array(probs_1st, dtype=np.float64)
    probs = np.clip(probs, 0.01, 0.99)

    if use_linear:
        # 線形 rating: (prob - mean) * scale
        # scale=8.0 は baseline ロジットの rating spread (1号艇 +0.17 / 6号艇 -3.74) と
        # おおよそ同じレンジになるよう調整。線形は単調変換だが、高 prob 帯で増幅しない。
        ratings = (probs - probs.mean()) * 8.0
    else:
        ratings = np.log(probs / (1.0 - probs))

    base_std = 0.8
    stds = np.full(6, base_std)

    # === 全艇共通の環境要因 ===
    weather_factor = 1.0
    if race_data:
        wind = race_data.get('wind_speed') or 0
        wave = race_data.get('wave_height') or 0
        if wind >= 5:
            weather_factor += 0.15
        elif wind >= 3:
            weather_factor += 0.05
        if wave >= 5:
            weather_factor += 0.10
        elif wave >= 3:
            weather_factor += 0.05
    stds *= weather_factor

    # === クラス分散 ===
    if boats_data and len(boats_data) == 6:
        class_values = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        classes = [class_values.get(b.get('player_class', 'B1'), 2)
                   for b in boats_data]
        class_std = np.std(classes)
        class_spread_factor = 1.10 - 0.17 * class_std
        class_spread_factor = np.clip(class_spread_factor, 0.85, 1.10)
        stds *= class_spread_factor

    # === 展示タイム差 ===
    ex_times = []
    if boats_data and len(boats_data) == 6:
        for boat in boats_data:
            et = boat.get('exhibition_time')
            if et and et > 0:
                ex_times.append(et)
    if len(ex_times) >= 4:
        ex_range = max(ex_times) - min(ex_times)
        ex_factor = 1.05 - 0.5 * min(ex_range, 0.3)
        stds *= ex_factor

    # === 艇別の要因 ===
    avg_exhibition = sum(ex_times) / len(ex_times) if ex_times else None

    if boats_data and len(boats_data) == 6:
        for i, boat in enumerate(boats_data):
            # ① クラス係数 (H3 修正で a1_coef を可変化)
            player_class = boat.get('player_class', 'B1')
            class_factor = {
                'A1': a1_coef, 'A2': 0.90, 'B1': 1.10, 'B2': 1.35
            }.get(player_class, 1.0)
            stds[i] *= class_factor

            # ② モーター勝率
            motor_wr2 = boat.get('motor_win_rate_2', 30.0) or 30.0
            if motor_wr2 > 50.0 or motor_wr2 < 15.0:
                stds[i] *= 1.1

            # ③ 部品交換
            if boat.get('parts_changed', False):
                stds[i] *= 1.15

            # ④ 展示タイム偏差
            ex_time = boat.get('exhibition_time')
            if ex_time and ex_time > 0 and avg_exhibition:
                diff = ex_time - avg_exhibition
                if diff < -0.05:
                    stds[i] *= 0.88
                elif diff > 0.10:
                    stds[i] *= 1.12

            # ⑤ 平均ST
            avg_st = boat.get('avg_st')
            if avg_st is not None:
                if avg_st > 0.20:
                    stds[i] *= 1.10
                elif avg_st < 0.10:
                    stds[i] *= 1.08

            # ⑥ 進入コース
            course = boat.get('approach_course')
            if course is not None:
                if course >= 5:
                    stds[i] *= 1.15
                elif course >= 4:
                    stds[i] *= 1.05

            # ⑦ 当地勝率
            local_wr = boat.get('local_win_rate_2') or boat.get('local_win_rate')
            if local_wr is not None and local_wr > 0:
                if local_wr > 40.0:
                    stds[i] *= 0.90
                elif local_wr < 15.0:
                    stds[i] *= 1.10

    return ratings, stds


def qmc_modified(probs_1st, boats_data, race_data, race_number,
                 use_linear=False, a1_coef=0.75, n_simulations=N_SIM, seed=SEED):
    """qmc_sanrentan_v3 改造版."""
    ratings, stds = compute_ratings_modified(
        probs_1st, boats_data, race_data, race_number,
        use_linear=use_linear, a1_coef=a1_coef,
    )
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_simulations, 64))))
    n_actual = 2 ** m
    uniform_samples = sampler.random(n_actual)
    performances = norm.ppf(uniform_samples, loc=ratings, scale=stds)
    orderings = np.argsort(-performances, axis=1)
    top3 = orderings[:, :3] + 1
    keys = [f"{t[0]}-{t[1]}-{t[2]}" for t in top3]
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return {k: v / n_actual for k, v in counts.items() if v > 0}, ratings, stds


# ============================================================
# Calibrator v2 適用
# ============================================================

def apply_calibrator(probs, calibrators_1st):
    """艇別 IsotonicRegression を適用 + 正規化."""
    calibrated = np.array([
        float(calibrators_1st[i].predict(np.array([probs[i]]))[0])
        for i in range(6)
    ])
    s = calibrated.sum()
    if s > 0:
        calibrated = calibrated / s
    return calibrated


# ============================================================
# 評価メトリクス
# ============================================================

def evaluate_scenario(name, cache, nn_probs, qmc_fn):
    """シナリオを全 races に適用 → メトリクス集計.

    qmc_fn(probs, boats_data, race_data, race_number) → qmc_probs dict
    """
    b1_pred = []
    b1_actual = []
    b1_pred_hi = []   # B1 prob >= 0.5 のみ
    b1_actual_hi = []
    combo_pred = defaultdict(float)
    combo_actual = defaultdict(int)
    n_races = 0

    # ROI proxy: top-3 picks per race
    total_invested = 0
    total_returns = 0
    n_bets = 0
    n_hits = 0

    for rid, c in cache.items():
        probs = nn_probs.get(rid)
        if not probs or len(probs) != 6 or not c.get('boats'):
            continue
        try:
            qmc_probs = qmc_fn(probs, c['boats'], c['race_data'], c['race_number'])
        except Exception:
            continue
        # B1 推定 prob = QMC 中 1着が B1 の合計
        b1_pred_prob = sum(p for k, p in qmc_probs.items() if k.startswith('1-'))
        is_b1_win = 1 if c['result_1st'] == 1 else 0
        b1_pred.append(b1_pred_prob)
        b1_actual.append(is_b1_win)
        if b1_pred_prob >= 0.5:
            b1_pred_hi.append(b1_pred_prob)
            b1_actual_hi.append(is_b1_win)
        # 全 combo aggregate
        for k, p in qmc_probs.items():
            combo_pred[k] += p
        actual = c['actual']
        if actual:
            combo_actual[actual] += 1
        # ROI proxy: 上位3 combo を購入 (100 yen each)
        top3 = sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]
        for combo, _ in top3:
            total_invested += 100
            n_bets += 1
            if combo == actual:
                total_returns += c.get('payout', 0) or 0
                n_hits += 1
        n_races += 1

    if n_races == 0:
        return None

    # 正規化
    combo_pred_norm = {k: v / n_races for k, v in combo_pred.items()}
    combo_actual_norm = {k: v / n_races for k, v in combo_actual.items()}

    # KL divergence (QMC || actual)
    all_keys = set(list(combo_pred_norm.keys()) + list(combo_actual_norm.keys()))
    kl = 0.0
    for k in all_keys:
        q = combo_pred_norm.get(k, 1e-9)
        p = combo_actual_norm.get(k, 1e-9)
        if q > 1e-9 and p > 1e-9:
            kl += q * np.log(q / p)

    # 4-X-X 系合計
    pred_4xx = sum(v for k, v in combo_pred_norm.items() if k.startswith('4-'))
    actual_4xx = sum(v for k, v in combo_actual_norm.items() if k.startswith('4-'))
    # 3-X-X 系合計
    pred_3xx = sum(v for k, v in combo_pred_norm.items() if k.startswith('3-'))
    actual_3xx = sum(v for k, v in combo_actual_norm.items() if k.startswith('3-'))
    # 1-X-X 系合計
    pred_1xx = sum(v for k, v in combo_pred_norm.items() if k.startswith('1-'))
    actual_1xx = sum(v for k, v in combo_actual_norm.items() if k.startswith('1-'))

    return {
        'name': name,
        'n_races': n_races,
        'b1_mean_pred': float(np.mean(b1_pred)) * 100,
        'b1_mean_actual': float(np.mean(b1_actual)) * 100,
        'b1_bias': (float(np.mean(b1_pred)) - float(np.mean(b1_actual))) * 100,
        'b1_hi_n': len(b1_pred_hi),
        'b1_hi_pred': float(np.mean(b1_pred_hi)) * 100 if b1_pred_hi else None,
        'b1_hi_actual': float(np.mean(b1_actual_hi)) * 100 if b1_actual_hi else None,
        'b1_hi_bias': (float(np.mean(b1_pred_hi)) - float(np.mean(b1_actual_hi))) * 100 if b1_pred_hi else None,
        'top10_pred': sorted(combo_pred_norm.items(), key=lambda x: -x[1])[:10],
        'top10_actual': sorted(combo_actual_norm.items(), key=lambda x: -x[1])[:10],
        'combo_132_pred': combo_pred_norm.get('1-3-2', 0) * 100,
        'combo_132_actual': combo_actual_norm.get('1-3-2', 0) * 100,
        'combo_315_pred': combo_pred_norm.get('3-1-5', 0) * 100,
        'combo_315_actual': combo_actual_norm.get('3-1-5', 0) * 100,
        'pred_4xx': pred_4xx * 100,
        'actual_4xx': actual_4xx * 100,
        'pred_3xx': pred_3xx * 100,
        'actual_3xx': actual_3xx * 100,
        'pred_1xx': pred_1xx * 100,
        'actual_1xx': actual_1xx * 100,
        'kl_qmc_to_actual': float(kl),
        'roi_proxy': (total_returns - total_invested) / total_invested * 100 if total_invested else 0,
        'hit_rate': n_hits / n_bets * 100 if n_bets else 0,
        'n_bets': n_bets,
        'n_hits': n_hits,
    }


def fetch_nn_probs(race_ids):
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (race_id) race_id, probabilities_1st
        FROM predictions
        WHERE race_id = ANY(%s) AND probabilities_1st IS NOT NULL
        ORDER BY race_id, id DESC
    """, (race_ids,))
    probs = {r['race_id']: r['probabilities_1st'] for r in cur.fetchall()}
    conn.close()
    return probs


# ============================================================
# 結果フラグ判定 (緑 / 黄 / 赤)
# ============================================================

def judge_flag(baseline, scenario):
    """シナリオ vs baseline で 3 値フラグ.

    赤: ROI proxy 悪化 OR B1 bias 絶対値悪化 5pt 以上
    黄: B1 bias 改善するが他艇 (3-X-X / 4-X-X) で逆方向過大評価 (1pt 以上)
    緑: B1 bias 改善 + 他艇副作用 1pt 未満 + ROI 悪化 1pt 未満
    """
    b1_improvement = abs(baseline['b1_bias']) - abs(scenario['b1_bias'])
    roi_diff = scenario['roi_proxy'] - baseline['roi_proxy']
    bias_3xx_diff = abs(scenario['pred_3xx'] - scenario['actual_3xx']) - \
                    abs(baseline['pred_3xx'] - baseline['actual_3xx'])
    bias_4xx_diff = abs(scenario['pred_4xx'] - scenario['actual_4xx']) - \
                    abs(baseline['pred_4xx'] - baseline['actual_4xx'])

    if roi_diff < -5.0 or abs(scenario['b1_bias']) > abs(baseline['b1_bias']) + 5.0:
        return '🔴 赤', 'baseline より明らかに悪化'
    if b1_improvement > 0 and (bias_3xx_diff > 1.0 or bias_4xx_diff > 1.0):
        return '🟡 黄', '1号艇 bias 改善するが他艇副作用あり'
    if b1_improvement > 0 and roi_diff > -1.0:
        return '🟢 緑', '1号艇 bias 改善 + 副作用小'
    return '⚪ 中立', '判定不能 / 大きな変化なし'


# ============================================================
# Main
# ============================================================

def main():
    logger.info("QMC 修正シナリオ 6 種 simulation (B' task)")

    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"cache: {len(cache)} races")

    with open(CAL_PATH, 'rb') as f:
        cal = pickle.load(f)
    calibrators_1st = cal['1st']
    logger.info(f"calibrators_v2 loaded: fit_period={cal['fit_period']}")

    race_ids = list(cache.keys())
    logger.info("NN probs fetch")
    nn_probs = fetch_nn_probs(race_ids)
    logger.info(f"NN probs: {len(nn_probs)} races")

    # キャリブレ済み NN probs を事前計算
    nn_probs_cal = {
        rid: apply_calibrator(np.array(p), calibrators_1st).tolist()
        for rid, p in nn_probs.items()
    }

    scenarios = []

    # S0 baseline
    logger.info("[S0] baseline (現状)")
    s0_fn = lambda p, b, r, n: qmc_modified(p, b, r, n, use_linear=False, a1_coef=0.75)[0]
    s0 = evaluate_scenario('S0 baseline', cache, nn_probs, s0_fn)
    s0['_use_linear'] = False
    s0['_a1_coef'] = 0.75
    s0['_use_cal'] = False
    scenarios.append(s0)

    # S1 H2 単独 (linear)
    logger.info("[S1] H2 単独 (linear)")
    s1_fn = lambda p, b, r, n: qmc_modified(p, b, r, n, use_linear=True, a1_coef=0.75)[0]
    s1 = evaluate_scenario('S1 H2 単独 (linear)', cache, nn_probs, s1_fn)
    s1['_use_linear'] = True
    s1['_a1_coef'] = 0.75
    s1['_use_cal'] = False
    scenarios.append(s1)

    # S2 H3 単独 (A1 0.85)
    logger.info("[S2] H3 単独 (A1 0.85)")
    s2_fn = lambda p, b, r, n: qmc_modified(p, b, r, n, use_linear=False, a1_coef=0.85)[0]
    s2 = evaluate_scenario('S2 H3 単独 (A1 0.85)', cache, nn_probs, s2_fn)
    s2['_use_linear'] = False
    s2['_a1_coef'] = 0.85
    s2['_use_cal'] = False
    scenarios.append(s2)

    # S3 H1+H2 (cal + linear)
    logger.info("[S3] H1+H2 (cal v2 + linear)")
    s3 = _evaluate_with_cal(cache, nn_probs_cal, use_linear=True, a1_coef=0.75)
    s3['name'] = 'S3 H1+H2 (cal v2 + linear)'
    s3['_use_linear'] = True
    s3['_a1_coef'] = 0.75
    s3['_use_cal'] = True
    scenarios.append(s3)

    # S4 H3 grid (0.80, 0.85, 0.90, 0.95)
    s4_subs = []
    for a1 in [0.80, 0.85, 0.90, 0.95]:
        logger.info(f"[S4] H3 grid A1={a1}")
        fn = lambda p, b, r, n, a1=a1: qmc_modified(p, b, r, n, use_linear=False, a1_coef=a1)[0]
        sub = evaluate_scenario(f'S4 A1={a1}', cache, nn_probs, fn)
        sub['_use_linear'] = False
        sub['_a1_coef'] = a1
        sub['_use_cal'] = False
        s4_subs.append(sub)
    scenarios.extend(s4_subs)

    # S5 H1+H3 (cal + A1 0.85)
    logger.info("[S5] H1+H3 (cal v2 + A1 0.85)")
    s5 = _evaluate_with_cal(cache, nn_probs_cal, use_linear=False, a1_coef=0.85)
    s5['name'] = 'S5 H1+H3 (cal v2 + A1 0.85)'
    s5['_use_linear'] = False
    s5['_a1_coef'] = 0.85
    s5['_use_cal'] = True
    scenarios.append(s5)

    # =============== レポート出力 ===============
    write_report(scenarios)
    logger.info(f"レポート出力: {REPORT_PATH}")


def _evaluate_with_cal(cache, nn_probs_cal, use_linear, a1_coef):
    """calibrated probs を使った評価 (apply_calibrator 済み)."""
    def fn(p, b, r, n):
        return qmc_modified(p, b, r, n, use_linear=use_linear, a1_coef=a1_coef)[0]
    return evaluate_scenario('S_cal', cache, nn_probs_cal, fn)


def write_report(scenarios):
    baseline = scenarios[0]
    lines = []
    lines.append("# QMC 修正シナリオ 6 種 simulation (B' task)\n\n")
    lines.append("C task (50_qmc_overestimation_root_cause) の H2/H3 知見を踏まえ、\n")
    lines.append("修正シナリオを data simulation で比較。**結論は出さず、岩下さんの判断に委ねる**。\n\n")
    lines.append("対象: cache 5828 races (49_qmc_vs_empirical.py で生成)、QMC 試行 8192 固定、seed=42\n\n")

    lines.append("## シナリオ間メトリクス比較\n\n")
    lines.append("| シナリオ | B1 推定 | B1 実 | B1 bias | B1 hi 推定 | B1 hi 実 | B1 hi bias | 1-3-2 P | 1-3-2 A | 3-1-5 P | 3-1-5 A | 4-X-X P | 4-X-X A | KL | ROI% | hit% | flag |\n")
    lines.append("|" + "---|" * 16 + "\n")
    for s in scenarios:
        if s is None:
            continue
        flag, _ = judge_flag(baseline, s)
        b1_hi_pred = f"{s['b1_hi_pred']:.2f}" if s['b1_hi_pred'] is not None else 'NA'
        b1_hi_actual = f"{s['b1_hi_actual']:.2f}" if s['b1_hi_actual'] is not None else 'NA'
        b1_hi_bias = f"{s['b1_hi_bias']:+.2f}" if s['b1_hi_bias'] is not None else 'NA'
        lines.append(
            f"| {s['name']} | "
            f"{s['b1_mean_pred']:.2f} | {s['b1_mean_actual']:.2f} | {s['b1_bias']:+.2f} | "
            f"{b1_hi_pred} | {b1_hi_actual} | {b1_hi_bias} | "
            f"{s['combo_132_pred']:.2f} | {s['combo_132_actual']:.2f} | "
            f"{s['combo_315_pred']:.2f} | {s['combo_315_actual']:.2f} | "
            f"{s['pred_4xx']:.2f} | {s['actual_4xx']:.2f} | "
            f"{s['kl_qmc_to_actual']:.3f} | {s['roi_proxy']:+.2f} | {s['hit_rate']:.2f} | {flag} |\n"
        )

    # ROI proxy の補足
    lines.append("\n**ROI proxy 注記**: 各 race で QMC top-3 combo を 100¥ ずつ購入 (300¥/race) → 当選なら payout 取得、外れなら損失 100¥。\n")
    lines.append("**mc3 戦略の Kelly / EV filter は含まない**ため、絶対 ROI ではなくシナリオ間相対比較として読むこと。\n\n")

    # 各シナリオの top10 比較
    for s in scenarios:
        if s is None:
            continue
        lines.append(f"### {s['name']} — TOP10 3 連単買い目\n\n")
        lines.append("| rank | QMC 予測 | 実頻度 |\n|---|---|---|\n")
        for i, (qp, ap) in enumerate(zip(s['top10_pred'], s['top10_actual'])):
            lines.append(f"| {i+1} | {qp[0]} ({qp[1]*100:.2f}%) | {ap[0]} ({ap[1]*100:.2f}%) |\n")
        lines.append("\n")

    # 批判プロトコル
    lines.append("\n## 批判プロトコル (各シナリオの Pro / Con / Unknowns)\n\n")
    protocols = scenario_protocols()
    for s in scenarios:
        if s is None:
            continue
        name = s['name']
        p = protocols.get(name.split(' ')[0], protocols.get('S4_grid'))  # S4 grid は共通
        flag, flag_note = judge_flag(baseline, s)
        lines.append(f"### {name}\n\n")
        lines.append(f"**自動フラグ**: {flag} ({flag_note})\n\n")
        lines.append(f"**擁護論 (Pro)**:\n{p['pro']}\n\n")
        lines.append(f"**批判論 (Con)**:\n{p['con']}\n\n")
        lines.append(f"**未検証論点 (Unknowns)**:\n{p['unknowns']}\n\n")

    # 留意事項
    lines.append("\n## 留意 (CLAUDE.md 批判プロトコル準拠)\n\n")
    lines.append("- フラグ (緑/黄/赤) は heuristic 閾値判定であり、絶対基準ではない\n")
    lines.append("- ROI proxy は top-3 単純購入で mc3 の Kelly/EV filter 不含、相対比較指標\n")
    lines.append("- 全シナリオ (S0 含む) を同 seed (42) で再計算しているため、シナリオ差は純粋に rating/std/calibrator 由来\n")
    lines.append("- 本 cache の元 QMC は別 seed で生成されているため、49_qmc_vs_empirical.py の数値と S0 は微差あり\n")
    lines.append("- KL は対称性なし、QMC → actual 一方向のみ\n")
    lines.append("- 結論は出さず、岩下さんの判断 (どのシナリオを B 本実装で採用するか) を待つ\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def scenario_protocols():
    """各シナリオの Pro / Con / Unknowns."""
    return {
        'S0': {
            'pro': '- 現状の挙動を参照する基準点として価値あり\n- 変更による悪化リスクなし',
            'con': '- C で判明した H2 (+12.58pt 増幅) と H3 (B1 std 0.86x) の bias を抱えたまま\n- mc3 系の根本問題が解消されない',
            'unknowns': '- このまま forward でどれだけ bias が累積するか',
        },
        'S1': {
            'pro': '- 線形変換は単調かつ等比、高 prob 帯の異常増幅が原理的に生じない\n- H2 が真の原因なら最も直接的な治療\n- 1号艇 prob 60% が rating 上で他艇との不当な差を生むのを防ぐ',
            'con': '- compute_ratings 全体の挙動が変わるため、他の補正項 (展示タイム差/クラス係数等) との整合性が崩れる可能性\n- scale=8.0 は heuristic、過去の logit base 設計の前提が崩れる\n- 2 号艇・3 号艇の rating 順位が予期せず逆転する可能性\n- 線形は確率の上下端でも線形なので、p=0.01 と p=0.99 の差が圧縮される',
            'unknowns': '- scale 値の感度 (8.0 vs 6.0 vs 10.0 で結果がどう変わるか)\n- 他の std 補正項と組み合わせた時の最終 rating 分布形状\n- forward 期間で勝率に乗るか',
        },
        'S2': {
            'pro': '- 単一係数変更で済むため副作用が局所化\n- A1 = 0.85 は他艇 (A2: 0.90) との差を縮め、設計の手触りが自然\n- C の H3 が「A1 0.75 で std systematic に小さい」と示した直接の応答',
            'con': '- A1 0.75 は v10.3-10.6 開発時の集中チューニング結果で、外す根拠が薄い\n- B1 std が大きくなると 1着率も低下 → 戦略全体の Kelly 計算に影響\n- H3 が真の主因かは C では「部分該当 🟡」止まり、効果限定の可能性\n- 1号艇以外の艇クラスでも A1 選手の挙動が変わる',
            'unknowns': '- 0.85 が最適か 0.80 か 0.90 か (S4 grid で別検証)\n- 旧 v10 開発時に A1=0.75 を選んだ data 根拠が現存するか',
        },
        'S3': {
            'pro': '- H1 (高確率帯 -4.41pt 過小評価) と H2 (logit +12.58pt 増幅) の相殺関係を両方絶つ\n- calibrator v2 は 2 月-3 月 OOS で fit 済み、独立の補正レイヤー\n- 線形化と組み合わせると bias の発生源が二重に消える',
            'con': '- 二箇所同時変更は副作用切り分け困難\n- calibrator v2 は本番未投入 (Phase A A1 撤退済み)、forward 性能未検証\n- H1 と H2 が相殺していたなら、両方直すと逆に過小評価に転じる可能性\n- 線形化と calibrator の交互作用が予測しづらい',
            'unknowns': '- calibrator v2 単独 (S3a 相当) の効果\n- 相殺バランスが崩れた時の系統的方向',
        },
        'S4_grid': {
            'pro': '- A1 係数の最適値を data で特定できる\n- 単一係数の段階変動なので副作用が予測しやすい\n- 0.80-0.95 のレンジは設計の自然な範囲',
            'con': '- grid search 自体は事後選択バイアスを生む (5828 races 上でフィットする値は本番でも最適とは限らない)\n- 過去 data だけで最適化すると forward overfitting\n- A1 のみ動かすので H2 logit 問題は残る',
            'unknowns': '- forward 期間 (2026-05-12 以降) で同じ最適値か\n- A1 以外のクラス係数 (A2/B1/B2) も連動調整が必要か',
        },
        'S5': {
            'pro': '- calibrator v2 で NN 自体のキャリブを直し、std で過大評価を緩和\n- logit 構造を温存するので既存の compute_ratings の他補正と整合\n- 二箇所変更だが効果原理が異なる (確率補正 + 不確実性補正)',
            'con': '- calibrator v2 単独投入は Phase A で「旧版優位」で撤退済み (2026-05-12)、再投入の論拠が必要\n- A1 0.85 が最適値とは限らない (S4 結果次第)\n- H2 logit 問題は残るので、forward で似た現象が再発する可能性',
            'unknowns': '- calibrator v2 撤退の根拠が「全戦略 ROI」だったか「単一指標」だったかの再確認\n- forward 期間で logit 問題が顕在化するシナリオの有無',
        },
    }


if __name__ == '__main__':
    main()
