"""QMC 1号艇過大評価 / 中穴過小評価 の原因特定 (5 仮説検証)

CLAUDE.md 批判プロトコル準拠:
  各仮説に Pro/Con/Unknowns を必ず書き、data 検証は「該当 / 該当しない / 判定不能」
  の 3 値判定で結論は出さず、最終判断は岩下さんに委ねる。

検証対象 (5 仮説):
  H1: NN softmax 自体が 1号艇過大バイアス
  H2: ロジット変換でバイアス増幅
  H3: compute_ratings_early の std 補正不足
  H4: 正規分布仮定がボートレース実分布と乖離
  H5: QMC 試行回数 8192 では中穴推定精度不足

入力: cache (`analysis/qmc_predictions_cache.pkl`) + predictions DB + boats
出力: analysis/reports/50_qmc_root_cause.md
"""
import os
import sys
import pickle
import logging
from datetime import date
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from src.monte_carlo import qmc_sanrentan_v3, compute_ratings_early

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'qmc_predictions_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '50_qmc_root_cause.md'


def fetch_nn_probs(race_ids):
    """各 race の NN probs_1st (最新 prediction)"""
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


def h1_nn_calibration(cache, nn_probs):
    """H1: NN softmax の 1着確率は systematic に 1号艇過大評価か"""
    # 各 race で probs_1st の 1号艇確率と実 1着率を集計
    boat_probs = defaultdict(list)  # boat -> list of (predicted_prob, actual_win)
    for rid, c in cache.items():
        probs = nn_probs.get(rid)
        if not probs or len(probs) != 6:
            continue
        actual_1st = c['result_1st']
        for boat in range(1, 7):
            predicted = float(probs[boat - 1])
            actual_win = 1 if actual_1st == boat else 0
            boat_probs[boat].append((predicted, actual_win))

    # 各艇 × 確率帯で実 hit 率
    rows = []
    for boat in range(1, 7):
        data = boat_probs[boat]
        if not data:
            continue
        preds, actuals = zip(*data)
        preds = np.array(preds)
        actuals = np.array(actuals)
        # 全体 calibration
        rows.append({
            'boat': boat,
            'n': len(data),
            'mean_pred': float(preds.mean()) * 100,
            'mean_actual': float(actuals.mean()) * 100,
            'calibration_bias': (float(preds.mean()) - float(actuals.mean())) * 100,
        })
        # 高確率帯 (>=50%)
        mask_hi = preds >= 0.5
        if mask_hi.sum() >= 30:
            rows.append({
                'boat': boat,
                'n': int(mask_hi.sum()),
                'mean_pred': float(preds[mask_hi].mean()) * 100,
                'mean_actual': float(actuals[mask_hi].mean()) * 100,
                'calibration_bias': (float(preds[mask_hi].mean()) - float(actuals[mask_hi].mean())) * 100,
                'note': '高確率帯 (>=50%)',
            })
    return rows


def h2_logit_amplification(cache, nn_probs):
    """H2: ロジット変換でバイアス増幅か"""
    # NN probs vs rating の関係性、再構築確率との差
    samples = []
    for rid, c in cache.items():
        probs = nn_probs.get(rid)
        if not probs or len(probs) != 6:
            continue
        probs_arr = np.array(probs, dtype=np.float64)
        probs_clip = np.clip(probs_arr, 0.01, 0.99)
        ratings = np.log(probs_clip / (1.0 - probs_clip))
        # 再構築 (rating から softmax 確率)
        exp_r = np.exp(ratings - ratings.max())
        reconstructed = exp_r / exp_r.sum()
        # 元 prob と再構築 prob の差
        for boat in range(6):
            samples.append({
                'boat': boat + 1,
                'orig_prob': float(probs_arr[boat]),
                'reconstructed_prob': float(reconstructed[boat]),
                'diff': float(reconstructed[boat] - probs_arr[boat]),
                'rating': float(ratings[boat]),
            })
    df = pd.DataFrame(samples)
    if df.empty:
        return None
    # 艇別の rating 分布と再構築誤差
    summary = df.groupby('boat').agg(
        n=('orig_prob', 'count'),
        orig_mean=('orig_prob', 'mean'),
        recon_mean=('reconstructed_prob', 'mean'),
        diff=('diff', 'mean'),
        rating_mean=('rating', 'mean'),
        rating_std=('rating', 'std'),
    ).reset_index()
    return summary


def h3_std_insufficient(cache, nn_probs):
    """H3: std 補正 11 項目後の std が systematic に小さすぎないか"""
    rows = []
    for rid, c in cache.items():
        probs = nn_probs.get(rid)
        if not probs or len(probs) != 6 or not c.get('boats'):
            continue
        try:
            ratings, stds = compute_ratings_early(
                probs, boats_data=c['boats'], race_data=c['race_data'],
                race_number=c['race_number']
            )
        except Exception:
            continue
        actual_1st = c['result_1st']
        for boat in range(6):
            rows.append({
                'race_id': rid,
                'boat': boat + 1,
                'rating': float(ratings[boat]),
                'std': float(stds[boat]),
                'prob': float(probs[boat]),
                'is_1st': 1 if actual_1st == boat + 1 else 0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    # 艇別 std 統計 + 実 1着率
    summary = df.groupby('boat').agg(
        n=('rating', 'count'),
        mean_std=('std', 'mean'),
        median_std=('std', 'median'),
        mean_rating=('rating', 'mean'),
        actual_1st_rate=('is_1st', 'mean'),
    ).reset_index()
    summary['actual_1st_rate'] = summary['actual_1st_rate'] * 100
    # 1号艇 std が他艇より明確に小さいか
    # rating の単位は logit、std が同じなら確率分布の形は艇毎に同じ
    return summary, df


def h4_normal_distribution_check(cache, nn_probs, n_sample_races=200):
    """H4: 正規分布仮定 vs 実データ着順分布の乖離

    各 race で QMC のサンプリング (n=10000) の着順分布と実データの着順分布を比較。
    KL divergence で測定。
    """
    sampled = list(cache.items())[:n_sample_races]
    kl_divs = []
    actual_pattern_freq = defaultdict(int)
    qmc_pattern_freq = defaultdict(float)
    total_races = 0
    for rid, c in sampled:
        probs = nn_probs.get(rid)
        if not probs or len(probs) != 6 or not c.get('boats'):
            continue
        # QMC 予測分布
        qmc_probs = c['qmc_probs']
        actual = c['actual']
        if not actual:
            continue
        # QMC 予測の上位 N、実頻度の上位 N
        # 全 race にわたって aggregate
        for combo, p in qmc_probs.items():
            qmc_pattern_freq[combo] += p
        actual_pattern_freq[actual] += 1
        total_races += 1
    # 正規化
    qmc_total = sum(qmc_pattern_freq.values())
    qmc_norm = {k: v / qmc_total for k, v in qmc_pattern_freq.items()}
    actual_norm = {k: v / total_races for k, v in actual_pattern_freq.items()}
    # KL divergence (Q || P) — Q = QMC, P = actual
    all_combos = set(list(qmc_norm.keys()) + list(actual_norm.keys()))
    kl_q_p = 0.0
    for combo in all_combos:
        q = qmc_norm.get(combo, 1e-9)
        p = actual_norm.get(combo, 1e-9)
        if q > 1e-9 and p > 1e-9:
            kl_q_p += q * np.log(q / p)
    return {
        'n_races': total_races,
        'kl_qmc_to_actual': float(kl_q_p),
        'qmc_top10': sorted(qmc_norm.items(), key=lambda x: -x[1])[:10],
        'actual_top10': sorted(actual_norm.items(), key=lambda x: -x[1])[:10],
    }


def h5_qmc_simulation_count(cache, nn_probs, n_test_races=10):
    """H5: QMC 試行回数を増やすと中穴買い目の確率推定はどう変化するか"""
    test_rids = list(cache.keys())[:n_test_races]
    n_options = [8192, 32768, 131072]
    results = defaultdict(dict)
    for rid in test_rids:
        c = cache[rid]
        probs = nn_probs.get(rid)
        if not probs or len(probs) != 6 or not c.get('boats'):
            continue
        for n in n_options:
            try:
                qmc_probs = qmc_sanrentan_v3(
                    probs, boats_data=c['boats'],
                    race_data=c['race_data'],
                    race_number=c['race_number'],
                    n_simulations=n,
                )
            except Exception:
                continue
            # 中穴帯 (個別 combo 確率 0.005-0.02 = 0.5-2%) の数と平均
            mid = [p for p in qmc_probs.values() if 0.005 <= p <= 0.02]
            results[n].setdefault('n_mid_combos', []).append(len(mid))
            results[n].setdefault('total_combos', []).append(len(qmc_probs))
            results[n].setdefault('total_mid_prob', []).append(sum(mid))
    summary = []
    for n, data in sorted(results.items()):
        if 'n_mid_combos' not in data:
            continue
        summary.append({
            'n_simulations': n,
            'mean_total_combos': float(np.mean(data['total_combos'])),
            'mean_n_mid_combos': float(np.mean(data['n_mid_combos'])),
            'mean_total_mid_prob': float(np.mean(data['total_mid_prob'])),
        })
    return summary


def main():
    logger.info("QMC 1号艇過大評価 原因特定 (5 仮説検証)")
    if not CACHE_PATH.exists():
        raise SystemExit(f"cache 不在: {CACHE_PATH} (先に 49_qmc_vs_empirical.py を実行)")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"cache: {len(cache)} races")

    race_ids = list(cache.keys())
    logger.info("NN probs fetch")
    nn_probs = fetch_nn_probs(race_ids)
    logger.info(f"NN probs: {len(nn_probs)} races")

    lines = []
    lines.append("# QMC 1号艇過大評価 / 中穴過小評価 原因特定 (5 仮説検証)\n\n")
    lines.append(f"対象: cache {len(cache)} races (49_qmc_vs_empirical.py で生成)\n")
    lines.append("CLAUDE.md 批判プロトコル準拠、結論は出さず岩下さんに判断委ねる\n\n")
    lines.append("## 検証対象の現象 (49_qmc_vs_empirical.py より)\n\n")
    lines.append("- 1-3-2: QMC 予測 10.21% vs 実 5.65% (bias **-4.57pt**)\n")
    lines.append("- 1-2-3: QMC 10.62% vs 実 7.64% (bias **-2.98pt**)\n")
    lines.append("- 3-1-5: QMC 0.49% vs 実 1.65% (3.4 倍頻出)\n")
    lines.append("- 4-5-6: QMC 0.004% vs 実 0.60% (150 倍頻出)\n\n")

    # H1
    logger.info("[H1/5] NN softmax 1号艇バイアス検証")
    h1_rows = h1_nn_calibration(cache, nn_probs)
    lines.append("## 仮説 H1: NN softmax 自体が 1号艇過大バイアスを持つ\n\n")
    lines.append("### 擁護論 (Pro)\n")
    lines.append("- NN は訓練データの 1号艇 1着率に強くフィットしている可能性\n")
    lines.append("- 76dim 特徴量の多くが 1号艇有利を補強する設計\n")
    lines.append("- memory に「キャリブレーション崩壊 (0.7-0.8 帯で -10.6pt 過小評価)」記録あり\n\n")
    lines.append("### 批判論 (Con)\n")
    lines.append("- 過去 calibrator_v2 検証で 1号艇 1着率は P=0.62 でほぼ正しく予測されていた (A1 結果)\n")
    lines.append("- もし NN が systematic に 1号艇過大なら、全戦略がもっと早く赤字化していたはず\n")
    lines.append("- 「1号艇軸スキップ」が calculate_all_strategies で実装されているのは、NN が 1号艇強さを\n")
    lines.append("  正しく見えている前提\n\n")
    lines.append("### 検証結果\n\n")
    lines.append("各艇 × 全体平均予測 vs 実 1着率:\n\n")
    lines.append("| boat | n | mean_pred | mean_actual | bias |\n|---|---|---|---|---|\n")
    for r in h1_rows:
        note = r.get('note', '')
        lines.append(f"| B{r['boat']} {note} | {r['n']} | {r['mean_pred']:.2f}% | {r['mean_actual']:.2f}% | **{r['calibration_bias']:+.2f}** |\n")
    # 自動判定
    overall = [r for r in h1_rows if 'note' not in r]
    if any(abs(r['calibration_bias']) > 5 for r in overall):
        verdict_h1 = '🔴 該当 (NN 段階で systematic bias ≥5pt)'
    elif any(abs(r['calibration_bias']) > 2 for r in overall):
        verdict_h1 = '🟡 部分該当 (NN 段階で 2-5pt の bias、QMC が増幅している可能性)'
    else:
        verdict_h1 = '🟢 該当しない (NN の calibration は概ね正しい)'
    lines.append(f"\n**自動判定**: {verdict_h1}\n")

    # H2
    logger.info("[H2/5] ロジット変換バイアス増幅検証")
    h2 = h2_logit_amplification(cache, nn_probs)
    lines.append("\n## 仮説 H2: ロジット変換でバイアス増幅\n\n")
    lines.append("### 擁護論 (Pro)\n")
    lines.append("- log(p/(1-p)) は p=0.5 付近で線形だが、p>0.6 で非線形に増幅\n")
    lines.append("- 1号艇 prob ~0.5 が rating で他艇との差が拡大される可能性\n\n")
    lines.append("### 批判論 (Con)\n")
    lines.append("- ロジット変換は単調、確率の順位は保存される\n")
    lines.append("- compute_ratings は rating の絶対値ではなく相対比較が機能要件\n\n")
    lines.append("### 検証結果\n\n")
    if h2 is not None:
        lines.append("艇別 (元 prob vs 再構築 prob、ロジット → softmax の往復):\n\n")
        lines.append("| boat | n | orig_mean | recon_mean | diff | rating_mean | rating_std |\n|---|---|---|---|---|---|---|\n")
        for _, r in h2.iterrows():
            lines.append(f"| B{int(r['boat'])} | {int(r['n'])} | {r['orig_mean']*100:.2f}% | "
                         f"{r['recon_mean']*100:.2f}% | {r['diff']*100:+.3f}pt | "
                         f"{r['rating_mean']:+.3f} | {r['rating_std']:.3f} |\n")
        max_diff = h2['diff'].abs().max() * 100
        if max_diff > 1.0:
            verdict_h2 = f'🔴 該当 (ロジット往復で {max_diff:.2f}pt の歪み)'
        elif max_diff > 0.1:
            verdict_h2 = f'🟡 部分該当 (微小歪み {max_diff:.2f}pt あり)'
        else:
            verdict_h2 = '🟢 該当しない (ロジット変換は順位保存、bias 増幅していない)'
        lines.append(f"\n**自動判定**: {verdict_h2}\n")

    # H3
    logger.info("[H3/5] std 補正不足検証")
    h3 = h3_std_insufficient(cache, nn_probs)
    lines.append("\n## 仮説 H3: compute_ratings_early の std 補正が不十分\n\n")
    lines.append("### 擁護論 (Pro)\n")
    lines.append("- 11 項目マニュアルチューニング、根拠不明の係数 (A1:0.75 等)\n")
    lines.append("- v10.3-10.6 (2026-04-08〜10) で集中開発、個別検証なし\n")
    lines.append("- 1号艇は A1 係数 0.75 で std が systematic に小さくなる設計\n\n")
    lines.append("### 批判論 (Con)\n")
    lines.append("- 1号艇 A1 が他艇より安定するのは合理的 (A1 は実力上位)\n")
    lines.append("- 係数 0.75 が「過小」とは限らない、実データで決まる\n\n")
    lines.append("### 検証結果\n\n")
    if h3 is not None:
        summary, full_df = h3
        lines.append("艇別 std と実 1着率:\n\n")
        lines.append("| boat | n | mean_std | median_std | mean_rating | 実1着率 |\n|---|---|---|---|---|---|\n")
        for _, r in summary.iterrows():
            lines.append(f"| B{int(r['boat'])} | {int(r['n'])} | {r['mean_std']:.3f} | "
                         f"{r['median_std']:.3f} | {r['mean_rating']:+.3f} | {r['actual_1st_rate']:.2f}% |\n")
        # 1号艇 std が他艇より大幅小さいか?
        b1_std = summary[summary['boat'] == 1]['mean_std'].iloc[0]
        others_std = summary[summary['boat'] != 1]['mean_std'].mean()
        ratio = b1_std / others_std
        if ratio < 0.85:
            verdict_h3 = f'🔴 該当 (1号艇 std が他平均の {ratio:.2f} 倍と systematic に小さい)'
        elif ratio < 0.95:
            verdict_h3 = f'🟡 部分該当 (1号艇 std やや小さい {ratio:.2f}x)'
        else:
            verdict_h3 = f'🟢 該当しない (1号艇 std は他艇と同程度 {ratio:.2f}x)'
        lines.append(f"\n**自動判定**: {verdict_h3} (1号艇 std / 他艇 std 平均 = {ratio:.3f})\n")

    # H4
    logger.info("[H4/5] 正規分布仮定 vs 実分布検証")
    h4 = h4_normal_distribution_check(cache, nn_probs, n_sample_races=500)
    lines.append("\n## 仮説 H4: 正規分布仮定がボートレース実分布と乖離\n\n")
    lines.append("### 擁護論 (Pro)\n")
    lines.append("- ボートレースの実分布は heavy tail (たまの大波乱) を持つ可能性\n")
    lines.append("- 正規分布だと extreme outcomes (6号艇 1着等) を確率付与できない\n\n")
    lines.append("### 批判論 (Con)\n")
    lines.append("- compute_ratings の std 補正で実分布の特性を間接的に表現している\n")
    lines.append("- 正規分布の標準的な扱い、ボートレースに特化した分布の root cause は QMC ではなく rating 設計の問題\n\n")
    lines.append("### 検証結果\n\n")
    if h4:
        lines.append(f"対象 {h4['n_races']} races、KL divergence (QMC || actual) = **{h4['kl_qmc_to_actual']:.4f}**\n\n")
        lines.append("**QMC 予測 TOP 10** vs **実頻度 TOP 10**:\n\n")
        lines.append("| rank | QMC predicted | 実頻度 |\n|---|---|---|\n")
        for i, ((qc, qp), (ac, ap)) in enumerate(zip(h4['qmc_top10'], h4['actual_top10']), 1):
            lines.append(f"| {i} | {qc} ({qp*100:.2f}%) | {ac} ({ap*100:.2f}%) |\n")
        if h4['kl_qmc_to_actual'] > 0.10:
            verdict_h4 = f"🔴 該当 (KL = {h4['kl_qmc_to_actual']:.3f} > 0.10、分布が大きく乖離)"
        elif h4['kl_qmc_to_actual'] > 0.03:
            verdict_h4 = f"🟡 部分該当 (KL = {h4['kl_qmc_to_actual']:.3f}、微妙な乖離)"
        else:
            verdict_h4 = f"🟢 該当しない (KL = {h4['kl_qmc_to_actual']:.3f}、分布は概ね一致)"
        lines.append(f"\n**自動判定**: {verdict_h4}\n")

    # H5
    logger.info("[H5/5] QMC 試行回数検証")
    h5 = h5_qmc_simulation_count(cache, nn_probs, n_test_races=10)
    lines.append("\n## 仮説 H5: QMC 試行 8192 では中穴推定精度不足\n\n")
    lines.append("### 擁護論 (Pro)\n")
    lines.append("- 中穴 (個別 combo 確率 0.5-2%) は 8192 試行で 40-160 hit、相対誤差大\n")
    lines.append("- 試行不足で確率 0 または過小に表示される可能性\n\n")
    lines.append("### 批判論 (Con)\n")
    lines.append("- Sobol scrambled で MC より収束速度 O(1/N)、8192 で十分なはず\n")
    lines.append("- 実データ突合で systematic に過小評価されているのは 8192 vs 32768 で同方向\n\n")
    lines.append("### 検証結果\n\n")
    if h5:
        lines.append("試行回数別 中穴 (0.5-2%) combo 統計 (10 races サンプル):\n\n")
        lines.append("| n_sim | total_combos | mid (0.5-2%) combos | total mid prob |\n|---|---|---|---|\n")
        for r in h5:
            lines.append(f"| {r['n_simulations']} | {r['mean_total_combos']:.1f} | "
                         f"{r['mean_n_mid_combos']:.1f} | {r['mean_total_mid_prob']*100:.2f}% |\n")
        if len(h5) >= 2:
            ratio = h5[-1]['mean_n_mid_combos'] / max(h5[0]['mean_n_mid_combos'], 0.1)
            if ratio > 1.5:
                verdict_h5 = f'🔴 該当 (試行 8192 → 131072 で中穴 combos {ratio:.1f} 倍、精度不足)'
            elif ratio > 1.15:
                verdict_h5 = f'🟡 部分該当 (中穴 combos {ratio:.2f} 倍)'
            else:
                verdict_h5 = f'🟢 該当しない (試行回数を増やしても変化なし {ratio:.2f}x、別原因)'
            lines.append(f"\n**自動判定**: {verdict_h5}\n")

    lines.append("\n## 総合判定 (5 仮説サマリ)\n\n")
    lines.append(f"- H1 NN 1号艇バイアス: 上記参照\n")
    lines.append(f"- H2 ロジット増幅: 上記参照\n")
    lines.append(f"- H3 std 補正不足: 上記参照\n")
    lines.append(f"- H4 正規分布乖離: 上記参照\n")
    lines.append(f"- H5 試行不足: 上記参照\n\n")
    lines.append("## 留意 (CLAUDE.md 批判プロトコル準拠)\n\n")
    lines.append("- 各仮説の自動判定は厳密な statistical test ではなく heuristic な閾値判定\n")
    lines.append("- 複数仮説が同時に該当する場合、主従関係の特定は岩下さんの判断が必要\n")
    lines.append("- 反証された仮説でも、特定条件下で再該当する可能性は残る\n")
    lines.append("- 結論は出さず、岩下さんの判断 (どの仮説を優先するか、係数調整 vs 再設計) を待つ\n")
    lines.append("- 次の B (compute_ratings_early 係数調整) は、最有力仮説の data に基づいて優先順位を決める\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
