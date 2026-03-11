"""バックテスト: Kelly パラメータ最適化

DB内の過去レース結果を使い、最適な賭けパラメータをシミュレーションで特定する。

■ オッズ問題と解決策
  - 全120通りの市場オッズは記録なし（勝ち組の払戻金のみ）
  - 合成オッズ = 0.75 / model_prob を max_odds フィルタ（確率閾値）に使用
  - 的中時の払戻は実データ(payout_sanrentan)を使用
  - NOTE: EV = P × (0.75/P × discount) = 0.75×discount (定数) のため
    EVフィルタは合成オッズでは機能しない → top_n（レース内上位N点）で代替
  - NOTE: divergenceフィルタは合成オッズだと定数になるためテスト対象外

■ 2段階アプローチ
  Stage A (高速スクリーニング): 均等100円ベット × パラメータグリッド → ROI順
  Stage B (精密シミュレーション): Stage A上位20 × kelly 6段階 → bankrollシミュレーション

■ パラメータグリッド
  max_odds:       [10, 15, 20, 25, 30, 40, 50]  (7) ← 確率閾値 P > 0.75/max_odds
  top_n:          [1, 2, 3, 5, 8]                (5) ← レース内上位N点
  filter_type:    [none, entropy, ensemble]       (3)
  max_entropy:    [1.8, 2.0, 2.3, 2.5]           (4) ← entropy時のみ
  kelly_fraction: [0.05, 0.10, 0.125, 0.15, 0.20, 0.25] (6) ← Stage Bのみ
  Stage A: 7×5×1 + 7×5×4 + 7×5×1 = 210
  Stage B: 20×6 = 120

Usage:
    DATABASE_URL=xxx python scripts/backtest_kelly_params.py
"""
import sys
import os
import time
import logging
import numpy as np
import torch
from itertools import permutations, product
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load_model
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- 3連単120通りのインデックスマッピング ---
SANRENTAN_COMBOS = list(permutations(range(6), 3))  # 120通り (0-indexed)
COMBO_TO_IDX = {combo: idx for idx, combo in enumerate(SANRENTAN_COMBOS)}

TAKEOUT_RATE = 0.25


# =========================================================================
# Stage 0: データ取得 + 特徴量生成 + モデル推論
# =========================================================================

def load_all_race_data(years=3):
    """DB一括取得 → (races_meta, X_features)

    Returns:
        races_meta: list of dict (venue_id, race_number, result_combo_idx, payout)
        X: np.ndarray (N, 208)
    """
    feature_engineer = FeatureEngineer()
    cutoff_date = datetime.now() - timedelta(days=365 * years)

    logger.info("=== データ一括取得中 ===")

    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT r.id, r.venue_id, r.race_number, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.payout_sanrentan,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.race_date >= %s AND r.status = 'finished'
              AND r.result_1st IS NOT NULL
              AND r.payout_sanrentan IS NOT NULL
              AND r.payout_sanrentan > 0
            ORDER BY r.race_date, r.venue_id, r.race_number
        """, (cutoff_date.date(),))
        races = cur.fetchall()
        logger.info(f"レース取得: {len(races):,}件")

        race_ids = [r['id'] for r in races]

        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()
        logger.info(f"ボート取得: {len(all_boats):,}件")

    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    logger.info("特徴量生成中...")
    X_list = []
    races_meta = []

    for race in races:
        boats = boats_by_race.get(race['id'], [])
        if len(boats) != 6:
            continue

        race_data = {
            'venue_id': race['venue_id'],
            'month': race['race_date'].month,
            'distance': 1800,
            'wind_speed': race.get('wind_speed') or 0,
            'wind_direction': race.get('wind_direction') or 'calm',
            'temperature': race.get('temperature') or 20,
            'wave_height': race.get('wave_height') or 0,
            'water_temperature': race.get('water_temperature') or 20,
        }

        try:
            features = feature_engineer.transform(race_data, boats)
        except Exception:
            continue

        r1 = race['result_1st'] - 1
        r2 = race['result_2nd'] - 1
        r3 = race['result_3rd'] - 1
        result_combo = (r1, r2, r3)
        if result_combo not in COMBO_TO_IDX:
            continue

        X_list.append(features)
        races_meta.append({
            'venue_id': race['venue_id'],
            'race_number': race['race_number'],
            'result_combo_idx': COMBO_TO_IDX[result_combo],
            'payout': race['payout_sanrentan'],
        })

    X = np.array(X_list, dtype=np.float32)
    logger.info(f"有効レース: {len(X):,}件, 特徴量次元: {X.shape[1]}")
    return races_meta, X


def run_model_inference(X, model_paths=None, batch_size=4096):
    """全レースをバッチ推論 → probs (N, 6) × 3ヘッド × 4モデル"""
    if model_paths is None:
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_paths = [
            os.path.join(base, 'boatrace_model.pth'),
            os.path.join(base, 'boatrace_model_s05.pth'),
            os.path.join(base, 'boatrace_model_s07.pth'),
            os.path.join(base, 'boatrace_model_s085.pth'),
        ]
        model_paths = [p for p in model_paths if os.path.exists(p)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"推論デバイス: {device}, モデル数: {len(model_paths)}")

    X_tensor = torch.FloatTensor(X).to(device)
    N = len(X)
    softmax = torch.nn.Softmax(dim=1)

    all_probs_1st = []
    all_probs_2nd = []
    all_probs_3rd = []

    for path in model_paths:
        model = load_model(path, device=device)
        model.eval()

        p1_list, p2_list, p3_list = [], [], []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = X_tensor[start:end]
                out1, out2, out3 = model(batch)
                p1_list.append(softmax(out1).cpu().numpy())
                p2_list.append(softmax(out2).cpu().numpy())
                p3_list.append(softmax(out3).cpu().numpy())

        all_probs_1st.append(np.concatenate(p1_list, axis=0))
        all_probs_2nd.append(np.concatenate(p2_list, axis=0))
        all_probs_3rd.append(np.concatenate(p3_list, axis=0))

    probs_1st = all_probs_1st[0]
    probs_2nd = all_probs_2nd[0]
    probs_3rd = all_probs_3rd[0]

    logger.info(f"推論完了: {N:,}件 × {len(model_paths)}モデル")
    return probs_1st, probs_2nd, probs_3rd, all_probs_1st


def compute_sanrentan_probs_batch(probs_1st, probs_2nd, probs_3rd):
    """条件付き確率で3連単120通りの確率を一括計算 (N, 120)

    P(i,j,k) = P(1st=i) * P(2nd=j|1st!=i) * P(3rd=k|1st!=i,2nd!=j)
    """
    N = probs_1st.shape[0]
    sanrentan = np.zeros((N, 120), dtype=np.float32)

    for idx, (i, j, k) in enumerate(SANRENTAN_COMBOS):
        p1 = probs_1st[:, i]

        mask_2nd = [x for x in range(6) if x != i]
        sum_2nd = probs_2nd[:, mask_2nd].sum(axis=1)
        sum_2nd = np.maximum(sum_2nd, 1e-10)
        p2_given_1 = probs_2nd[:, j] / sum_2nd

        mask_3rd = [x for x in range(6) if x != i and x != j]
        sum_3rd = probs_3rd[:, mask_3rd].sum(axis=1)
        sum_3rd = np.maximum(sum_3rd, 1e-10)
        p3_given_12 = probs_3rd[:, k] / sum_3rd

        sanrentan[:, idx] = p1 * p2_given_1 * p3_given_12

    return sanrentan


def compute_entropy_batch(probs_1st):
    """Shannon entropy H = -sum(p_i * log2(p_i)) ベクトル化版 (N,)"""
    safe = np.maximum(probs_1st, 1e-10)
    return -np.sum(safe * np.log2(safe), axis=1)


def check_ensemble_agreement_batch(all_probs_1st, min_agreement=3):
    """アンサンブル合議バッチ版 → (N,) bool mask"""
    N = all_probs_1st[0].shape[0]
    top_boats = np.stack([p.argmax(axis=1) for p in all_probs_1st], axis=0)

    agreed = np.zeros(N, dtype=bool)
    for n in range(N):
        counts = np.bincount(top_boats[:, n], minlength=6)
        if counts.max() >= min_agreement:
            agreed[n] = True

    return agreed


# =========================================================================
# Stage A: 高速スクリーニング（均等100円ベット）
# =========================================================================

def stage_a_screening(races_meta, sanrentan_probs, entropy, ensemble_mask,
                      skip_56_mask):
    """Stage A: パラメータグリッド × 均等100円ベット → ROI計算

    合成オッズ(0.75/P)を max_odds フィルタ（確率閾値）として使用。
    EVフィルタは合成オッズでは定数(0.75×discount)になるため使用不可。
    代わりに top_n（レース内上位N点）で選択性を制御する。

    的中時は実データ payout_sanrentan を使用。
    """
    max_odds_list = [10, 15, 20, 25, 30, 40, 50]
    top_n_list = [1, 2, 3, 5, 8]
    filter_configs = [
        ('none', [None]),
        ('entropy', [1.8, 2.0, 2.3, 2.5]),
        ('ensemble', [None]),
    ]

    N = len(races_meta)
    result_combo_idxs = np.array([r['result_combo_idx'] for r in races_meta])
    payouts = np.array([r['payout'] for r in races_meta], dtype=np.float64)

    # 合成オッズ (N, 120)
    synthetic_odds = np.where(sanrentan_probs > 1e-10,
                              (1.0 - TAKEOUT_RATE) / sanrentan_probs,
                              9999.0)

    # 確率の降順ソートインデックス (N, 120)
    prob_rank_indices = np.argsort(-sanrentan_probs, axis=1)

    results = []
    param_count = 0

    for filter_type, entropy_thresholds in filter_configs:
        for max_entropy in entropy_thresholds:
            # レースマスク
            if filter_type == 'none':
                race_mask = ~skip_56_mask
            elif filter_type == 'entropy':
                race_mask = ~skip_56_mask & (entropy < max_entropy)
            elif filter_type == 'ensemble':
                race_mask = ~skip_56_mask & ensemble_mask
            else:
                continue

            n_active = int(race_mask.sum())
            if n_active == 0:
                continue

            active_probs = sanrentan_probs[race_mask]
            active_odds = synthetic_odds[race_mask]
            active_results = result_combo_idxs[race_mask]
            active_payouts = payouts[race_mask]
            active_ranks = prob_rank_indices[race_mask]

            for max_odds, top_n in product(max_odds_list, top_n_list):
                # min_prob implied by max_odds
                min_prob = (1.0 - TAKEOUT_RATE) / max_odds

                # 各レースで: 確率上位top_n点 かつ 確率 >= min_prob
                total_bets = 0
                total_payout = 0.0
                hits = 0

                for m in range(n_active):
                    bet_count = 0
                    for rank in range(min(top_n, 120)):
                        combo_idx = active_ranks[m, rank]
                        prob = active_probs[m, combo_idx]
                        if prob < min_prob:
                            break
                        bet_count += 1

                        if combo_idx == active_results[m]:
                            total_payout += active_payouts[m]
                            hits += 1

                    total_bets += bet_count

                if total_bets == 0:
                    continue

                total_invested = total_bets * 100
                roi = total_payout / total_invested

                results.append({
                    'filter_type': filter_type,
                    'max_odds': max_odds,
                    'max_entropy': max_entropy,
                    'top_n': top_n,
                    'roi': roi,
                    'total_bets': total_bets,
                    'total_invested': total_invested,
                    'total_payout': total_payout,
                    'hits': hits,
                    'hit_rate': hits / n_active,
                    'avg_bets_per_race': total_bets / n_active,
                    'n_races': n_active,
                })
                param_count += 1

    results.sort(key=lambda x: x['roi'], reverse=True)
    logger.info(f"Stage A: {param_count}パラメータセット評価完了")
    return results


# =========================================================================
# Stage B: 精密シミュレーション（bankroll連動ベットサイジング）
# =========================================================================

def stage_b_kelly_simulation(top_params, races_meta, sanrentan_probs,
                             entropy, ensemble_mask, skip_56_mask,
                             initial_bankroll=200000):
    """Stage B: 上位パラメータセットで逐次bankroll更新シミュレーション

    ベットサイジング:
      bet = bankroll × kelly_fraction × (prob / sum_qualifying_probs)
      → 高確率の組み合わせに大きくベット、合計を kelly_fraction × bankroll に制御
      → 100円単位に丸め、1レース合計上限 = bankroll × 3%

    的中時は実データ payout_sanrentan を使用。
    """
    kelly_fractions = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]

    N = len(races_meta)
    result_combo_idxs = np.array([r['result_combo_idx'] for r in races_meta])
    payouts = np.array([r['payout'] for r in races_meta], dtype=np.float64)

    prob_rank_indices = np.argsort(-sanrentan_probs, axis=1)

    results = []
    total_sims = len(top_params) * len(kelly_fractions)
    logger.info(f"Stage B: {total_sims}シミュレーション × {N:,}レース")

    for params in top_params:
        filter_type = params['filter_type']
        max_odds = params['max_odds']
        max_entropy = params.get('max_entropy')
        top_n = params['top_n']
        min_prob = (1.0 - TAKEOUT_RATE) / max_odds

        # レースマスク
        if filter_type == 'none':
            race_mask = ~skip_56_mask
        elif filter_type == 'entropy':
            race_mask = ~skip_56_mask & (entropy < max_entropy)
        elif filter_type == 'ensemble':
            race_mask = ~skip_56_mask & ensemble_mask
        else:
            continue

        active_indices = np.where(race_mask)[0]
        n_active = len(active_indices)
        if n_active == 0:
            continue

        for kelly_frac in kelly_fractions:
            bankroll = float(initial_bankroll)
            peak_bankroll = bankroll
            max_drawdown = 0.0
            total_invested = 0.0
            total_payout = 0.0
            hits = 0
            total_bets = 0
            bust = False

            for race_idx in active_indices:
                if bankroll < 100:
                    bust = True
                    break

                probs_120 = sanrentan_probs[race_idx]
                ranks = prob_rank_indices[race_idx]
                result_idx = result_combo_idxs[race_idx]
                payout = payouts[race_idx]

                # 対象組み合わせの選定
                qualifying = []
                for rank in range(min(top_n, 120)):
                    combo_idx = ranks[rank]
                    prob = probs_120[combo_idx]
                    if prob < min_prob:
                        break
                    qualifying.append((combo_idx, prob))

                if not qualifying:
                    continue

                # ベットサイジング: 確率比例、合計 = bankroll × kelly_fraction
                sum_probs = sum(p for _, p in qualifying)
                race_budget = bankroll * kelly_frac
                max_race_total = bankroll * 0.03

                race_bets = []
                for combo_idx, prob in qualifying:
                    bet = race_budget * (prob / sum_probs)
                    bet = int(round(bet / 100) * 100)
                    bet = max(100, bet)
                    race_bets.append((combo_idx, bet))

                # 合計上限カット
                race_total = sum(b for _, b in race_bets)
                if race_total > max_race_total:
                    ratio = max_race_total / race_total
                    race_bets = [
                        (idx, max(100, int(round(amt * ratio / 100) * 100)))
                        for idx, amt in race_bets
                    ]

                # ベット実行
                race_invested = sum(b for _, b in race_bets)
                bankroll -= race_invested
                total_invested += race_invested
                total_bets += len(race_bets)

                # 的中判定
                for combo_idx, bet_amt in race_bets:
                    if combo_idx == result_idx:
                        win_amount = payout * (bet_amt / 100.0)
                        bankroll += win_amount
                        total_payout += win_amount
                        hits += 1

                # ドローダウン
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

            roi = total_payout / total_invested if total_invested > 0 else 0.0

            results.append({
                'filter_type': filter_type,
                'max_odds': max_odds,
                'max_entropy': max_entropy,
                'top_n': top_n,
                'kelly_fraction': kelly_frac,
                'roi': roi,
                'final_bankroll': bankroll,
                'initial_bankroll': initial_bankroll,
                'profit': bankroll - initial_bankroll,
                'profit_pct': (bankroll - initial_bankroll) / initial_bankroll * 100,
                'max_drawdown': max_drawdown,
                'total_invested': total_invested,
                'total_payout': total_payout,
                'hits': hits,
                'total_bets': total_bets,
                'hit_rate': hits / n_active if n_active > 0 else 0,
                'n_races': n_active,
                'bust': bust,
            })

    results.sort(key=lambda x: x['profit'], reverse=True)
    return results


# =========================================================================
# メイン
# =========================================================================

def main():
    t0 = time.time()

    # --- Stage 0: データ準備 ---
    logger.info("=" * 60)
    logger.info("Stage 0: データ取得 + 特徴量生成 + モデル推論")
    logger.info("=" * 60)

    races_meta, X = load_all_race_data(years=3)
    N = len(races_meta)

    if N == 0:
        logger.error("有効なレースデータが0件です")
        return

    probs_1st, probs_2nd, probs_3rd, all_probs_1st = run_model_inference(X)

    logger.info("3連単120通り確率計算中...")
    sanrentan_probs = compute_sanrentan_probs_batch(probs_1st, probs_2nd, probs_3rd)
    logger.info(f"3連単確率: shape={sanrentan_probs.shape}")

    # 統計情報
    top1_probs = sanrentan_probs.max(axis=1)
    logger.info(f"top1確率: mean={top1_probs.mean():.4f}, max={top1_probs.max():.4f}")

    # エントロピー
    entropy = compute_entropy_batch(probs_1st)
    logger.info(f"エントロピー: mean={entropy.mean():.3f}, median={np.median(entropy):.3f}")

    # アンサンブル合議
    if len(all_probs_1st) >= 2:
        ensemble_mask = check_ensemble_agreement_batch(all_probs_1st, min_agreement=3)
        logger.info(f"アンサンブル合議(3/4): {ensemble_mask.sum():,}/{N:,} ({ensemble_mask.mean()*100:.1f}%)")
    else:
        ensemble_mask = np.ones(N, dtype=bool)

    # 5-6号艇スキップ
    top_boats = probs_1st.argmax(axis=1)
    skip_56_mask = top_boats >= 4
    logger.info(f"5-6号艇軸スキップ: {skip_56_mask.sum():,}/{N:,} ({skip_56_mask.mean()*100:.1f}%)")

    # 的中組み合わせのモデル確率分布
    result_combo_idxs = np.array([r['result_combo_idx'] for r in races_meta])
    hit_probs = sanrentan_probs[np.arange(N), result_combo_idxs]
    logger.info(f"的中組のモデル確率: mean={hit_probs.mean():.4f}, median={np.median(hit_probs):.4f}")

    # 的中組の確率ランク
    prob_ranks = np.argsort(-sanrentan_probs, axis=1)
    hit_ranks = np.zeros(N, dtype=int)
    for n in range(N):
        hit_ranks[n] = np.where(prob_ranks[n] == result_combo_idxs[n])[0][0]
    for topk in [1, 3, 5, 10, 20]:
        rate = (hit_ranks < topk).mean() * 100
        logger.info(f"  的中がtop{topk:2d}に入る率: {rate:.1f}%")

    t1 = time.time()
    logger.info(f"Stage 0 完了: {t1 - t0:.1f}秒")

    # --- Stage A: 高速スクリーニング ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Stage A: 高速スクリーニング（均等100円ベット）")
    logger.info("=" * 60)

    stage_a_results = stage_a_screening(
        races_meta, sanrentan_probs, entropy, ensemble_mask, skip_56_mask
    )

    t2 = time.time()
    logger.info(f"Stage A 完了: {t2 - t1:.1f}秒")

    # Stage A 結果表示
    print("\n" + "=" * 90)
    print("Stage A 結果: ROI上位30 (均等100円ベット)")
    print("=" * 90)
    print(f"{'#':>3} {'filter':>10} {'maxOdds':>7} {'maxH':>5} {'topN':>4} "
          f"{'ROI':>7} {'hits':>5} {'bets':>8} {'races':>6} {'hit%':>6} {'avg/R':>5}")
    print("-" * 90)

    for i, r in enumerate(stage_a_results[:30]):
        max_h_str = f"{r['max_entropy']:.1f}" if r['max_entropy'] is not None else "  -"
        print(f"{i+1:3d} {r['filter_type']:>10} {r['max_odds']:7d} "
              f"{max_h_str:>5} {r['top_n']:4d} "
              f"{r['roi']:7.3f} {r['hits']:5d} {r['total_bets']:8d} "
              f"{r['n_races']:6d} {r['hit_rate']*100:5.1f}% "
              f"{r['avg_bets_per_race']:5.1f}")

    # --- Stage B: 精密シミュレーション ---
    top20 = stage_a_results[:20]
    if not top20:
        logger.warning("Stage Aで有効なパラメータセットが見つかりませんでした")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("Stage B: 精密シミュレーション（bankroll連動）")
    logger.info("=" * 60)

    stage_b_results = stage_b_kelly_simulation(
        top20, races_meta, sanrentan_probs, entropy, ensemble_mask, skip_56_mask
    )

    t3 = time.time()
    logger.info(f"Stage B 完了: {t3 - t2:.1f}秒, {len(stage_b_results)}シミュレーション")

    # Stage B 結果表示
    print("\n" + "=" * 110)
    print("Stage B 結果: 利益上位20 (bankroll連動, 初期=200,000円)")
    print("=" * 110)
    print(f"{'#':>3} {'filter':>10} {'mOdds':>5} {'maxH':>5} {'topN':>4} "
          f"{'kelly':>6} {'ROI':>6} {'final_BR':>10} {'profit%':>8} "
          f"{'maxDD':>6} {'hits':>5} {'bets':>6} {'bust':>4}")
    print("-" * 110)

    for i, r in enumerate(stage_b_results[:20]):
        bust_str = "YES" if r['bust'] else ""
        max_h_str = f"{r['max_entropy']:.1f}" if r['max_entropy'] is not None else "  -"
        print(f"{i+1:3d} {r['filter_type']:>10} {r['max_odds']:5d} "
              f"{max_h_str:>5} {r['top_n']:4d} "
              f"{r['kelly_fraction']:6.3f} {r['roi']:6.3f} "
              f"{r['final_bankroll']:10,.0f} {r['profit_pct']:+7.1f}% "
              f"{r['max_drawdown']*100:5.1f}% {r['hits']:5d} "
              f"{r['total_bets']:6d} {bust_str:>4}")

    # --- 推奨パラメータ ---
    print("\n" + "=" * 80)
    print("推奨パラメータ (各フィルタタイプの最良, bust除外, ROI>0.8)")
    print("=" * 80)

    best_by_filter = {}
    for r in stage_b_results:
        ft = r['filter_type']
        if ft not in best_by_filter and not r['bust'] and r['roi'] > 0.8:
            best_by_filter[ft] = r

    strategy_map = {
        'none': 'conservative/standard (A/B)',
        'entropy': 'high_confidence (D)',
        'ensemble': 'ensemble (E)',
    }

    if best_by_filter:
        for ft, r in best_by_filter.items():
            name = strategy_map.get(ft, ft)
            print(f"\n  {name}:")
            print(f"    max_odds:       {r['max_odds']}")
            if r['max_entropy'] is not None:
                print(f"    max_entropy:    {r['max_entropy']}")
            print(f"    top_n:          {r['top_n']}")
            print(f"    kelly_fraction: {r['kelly_fraction']}")
            print(f"    ROI: {r['roi']:.3f}, 利益: {r['profit']:+,.0f}円 ({r['profit_pct']:+.1f}%)")
            print(f"    最大DD: {r['max_drawdown']*100:.1f}%, 的中率: {r['hit_rate']*100:.2f}%")

        # 設定ファイルマッピング
        print("\n  → config/betting_config.json への反映例:")
        for ft, r in best_by_filter.items():
            if ft == 'none':
                for sname in ['conservative', 'standard']:
                    print(f"    {sname}: max_odds={r['max_odds']}, "
                          f"kelly_fraction={r['kelly_fraction']}")
            elif ft == 'entropy':
                print(f"    high_confidence: max_odds={r['max_odds']}, "
                      f"max_entropy={r['max_entropy']}, "
                      f"kelly_fraction={r['kelly_fraction']}")
            elif ft == 'ensemble':
                print(f"    ensemble: max_odds={r['max_odds']}, "
                      f"kelly_fraction={r['kelly_fraction']}")
    else:
        print("\n  (ROI > 0.8 かつ bust なしの設定が見つかりませんでした)")

    # NOTE
    print("\n" + "-" * 80)
    print("NOTE:")
    print("  - odds_discount, min_EV は合成オッズ(0.75/P)では定数となるため")
    print("    このバックテストでは未テスト。リアルオッズ取得後に別途検証が必要。")
    print("  - divergence/div_confidence も同様にテスト対象外。")
    print("  - 合成オッズの限界: モデル自身の確率でオッズを推定するため、")
    print("    モデルと市場の乖離（=エッジの源泉）をシミュレートできない。")
    print("    本結果は「モデルの確率ランキングの精度」を検証するもの。")
    print("-" * 80)

    total_time = time.time() - t0
    logger.info(f"\n合計実行時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")


if __name__ == '__main__':
    main()
