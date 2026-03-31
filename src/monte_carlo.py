"""モンテカルロ・シミュレーション確率算出

NNの3独立headによる条件付き確率の近似問題を回避:
  - 各艇のパフォーマンス分布（レーティング + 分散）からN回サンプリング
  - シミュレーション結果の頻度 = 三連単確率
  - P(2nd|1st), P(3rd|1st,2nd) の近似が不要

レーティング算出:
  NNの1着確率をベースに、勝率・モーター・展示タイム等で補正。
  分散は選手のクラス・レース条件（荒天等）で変動。
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_ratings(probs_1st, boats_data=None):
    """NNの1着確率 + 艇データからレーティングと分散を算出

    Args:
        probs_1st: list[6] — NNのsoftmax出力（1着確率）
        boats_data: list[dict] — 6艇の選手データ（オプション）

    Returns:
        ratings: np.array(6) — 各艇のパフォーマンス期待値
        stds: np.array(6) — 各艇のパフォーマンスばらつき
    """
    probs = np.array(probs_1st, dtype=np.float64)
    probs = np.clip(probs, 0.01, 0.99)

    # 確率をロジットスケールに変換（レーティングの基盤）
    ratings = np.log(probs / (1.0 - probs))

    # 基本分散: 全艇共通の不確実性
    base_std = 0.8

    stds = np.full(6, base_std)

    if boats_data and len(boats_data) == 6:
        for i, boat in enumerate(boats_data):
            # 選手クラスで分散調整
            # A1=安定(低分散), B2=不安定(高分散)
            player_class = boat.get('player_class', 'B1')
            class_factor = {
                'A1': 0.85, 'A2': 0.95, 'B1': 1.05, 'B2': 1.20
            }.get(player_class, 1.0)
            stds[i] *= class_factor

            # モーター勝率が極端に高い/低い → 分散拡大（予測が難しい）
            motor_wr2 = boat.get('motor_win_rate_2', 30.0) or 30.0
            if motor_wr2 > 50.0 or motor_wr2 < 15.0:
                stds[i] *= 1.1

            # 部品交換 → 分散拡大（未知数が増える）
            if boat.get('parts_changed', False):
                stds[i] *= 1.15

    return ratings, stds


def monte_carlo_sanrentan(probs_1st, boats_data=None,
                          n_simulations=10000, seed=None):
    """モンテカルロ・シミュレーションで三連単確率を直接算出

    Args:
        probs_1st: list[6] — NNのsoftmax出力（1着確率）
        boats_data: list[dict] — 6艇の選手データ
        n_simulations: シミュレーション回数
        seed: 乱数シード（再現性用）

    Returns:
        dict: {"1-2-3": prob, ...} — 三連単確率（120通り）
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    ratings, stds = compute_ratings(probs_1st, boats_data)

    # N回シミュレーション: 各艇のパフォーマンスを正規分布からサンプリング
    # shape: (n_simulations, 6)
    performances = rng.normal(
        loc=ratings,       # (6,) → broadcast to (n_sim, 6)
        scale=stds,        # (6,) → broadcast to (n_sim, 6)
        size=(n_simulations, 6)
    )

    # パフォーマンス降順 → 着順（高いほど上位）
    # argsort ascending → reverse for descending
    orderings = np.argsort(-performances, axis=1)  # (n_sim, 6)

    # 三連単カウント: 上位3艇の組み合わせ
    sanrentan_counts = {}
    for sim in range(n_simulations):
        first = orderings[sim, 0] + 1   # 0-indexed → 1-indexed
        second = orderings[sim, 1] + 1
        third = orderings[sim, 2] + 1
        key = f"{first}-{second}-{third}"
        sanrentan_counts[key] = sanrentan_counts.get(key, 0) + 1

    # カウント → 確率
    sanrentan_probs = {
        key: count / n_simulations
        for key, count in sanrentan_counts.items()
        if count > 0
    }

    return sanrentan_probs


def monte_carlo_positions(probs_1st, boats_data=None,
                          n_simulations=10000, seed=None):
    """着順別の確率分布を算出（デバッグ・分析用）

    Returns:
        np.array shape (6, 6) — [boat_idx][position_idx] = prob
        例: result[0][0] = 1号艇が1着になる確率
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    ratings, stds = compute_ratings(probs_1st, boats_data)

    performances = rng.normal(
        loc=ratings, scale=stds,
        size=(n_simulations, 6)
    )

    orderings = np.argsort(-performances, axis=1)

    # 着順カウント
    position_counts = np.zeros((6, 6), dtype=np.int64)
    for sim in range(n_simulations):
        for pos in range(6):
            boat = orderings[sim, pos]
            position_counts[boat, pos] += 1

    return position_counts / n_simulations
