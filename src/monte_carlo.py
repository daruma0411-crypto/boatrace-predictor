"""モンテカルロ・シミュレーション確率算出 v2 + QMC + v3序盤特化

NNの3独立headによる条件付き確率の近似問題を回避:
  - 各艇のパフォーマンス分布（レーティング + 分散）からN回サンプリング
  - シミュレーション結果の頻度 = 三連単確率
  - P(2nd|1st), P(3rd|1st,2nd) の近似が不要

v2 変更点 (2026-04-09):
  - ノイズモデル9変数化（風速/波高/展示タイム/平均ST/進入コース/レース番号）
  - デフォルトシミュレーション回数: 10000 → 50000

v3 QMC 変更点 (2026-04-10):
  - 準モンテカルロ法 (Quasi-Monte Carlo) を追加
  - Sobol列による均等サンプリングで収束速度 O(1/√N) → O(1/N) に改善

v3 序盤特化MC 変更点 (2026-04-10):
  - R1-R4専用ノイズモデル compute_ratings_early()
  - クラス分散・展示タイム差・当地勝率の3変数追加
  - クラス係数を序盤向けに強化（A1:0.75, B2:1.35）
"""
import logging
import numpy as np
from scipy.stats import qmc, norm

logger = logging.getLogger(__name__)


def compute_ratings(probs_1st, boats_data=None, race_data=None,
                    race_number=None):
    """NNの1着確率 + 艇データ + レース環境からレーティングと分散を算出

    Args:
        probs_1st: list[6] — NNのsoftmax出力（1着確率）
        boats_data: list[dict] — 6艇の選手データ（オプション）
        race_data: dict — レース環境データ（オプション）
            {wind_speed, wave_height, ...}
        race_number: int — レース番号 1-12（オプション）

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

    # === 全艇共通の環境要因 ===

    # ④ 風速: 強風でターン乱れ → 全艇の分散拡大
    weather_factor = 1.0
    if race_data:
        wind = race_data.get('wind_speed') or 0
        wave = race_data.get('wave_height') or 0

        if wind >= 5:
            weather_factor += 0.15
        elif wind >= 3:
            weather_factor += 0.05

        # ⑤ 波高: 高波でボート不安定
        if wave >= 5:
            weather_factor += 0.10
        elif wave >= 3:
            weather_factor += 0.05

    stds *= weather_factor

    # ⑨ レース番号: 終盤は本命決着率が高い
    if race_number is not None:
        if race_number in (11, 12):
            stds *= 0.85   # 堅いレース → 分散縮小
        elif race_number in (1, 2, 3):
            stds *= 1.10   # 序盤 → 荒れやすい

    # === 艇別の要因 ===

    # 展示タイムの平均を事前計算（⑥で使用）
    ex_times = []
    if boats_data and len(boats_data) == 6:
        for boat in boats_data:
            et = boat.get('exhibition_time')
            if et and et > 0:
                ex_times.append(et)
    avg_exhibition = sum(ex_times) / len(ex_times) if ex_times else None

    if boats_data and len(boats_data) == 6:
        for i, boat in enumerate(boats_data):
            # ① 選手クラスで分散調整 (A1=安定, B2=不安定)
            player_class = boat.get('player_class', 'B1')
            class_factor = {
                'A1': 0.85, 'A2': 0.95, 'B1': 1.05, 'B2': 1.20
            }.get(player_class, 1.0)
            stds[i] *= class_factor

            # ② モーター勝率が極端 → 分散拡大
            motor_wr2 = boat.get('motor_win_rate_2', 30.0) or 30.0
            if motor_wr2 > 50.0 or motor_wr2 < 15.0:
                stds[i] *= 1.1

            # ③ 部品交換 → 未知数が増える
            if boat.get('parts_changed', False):
                stds[i] *= 1.15

            # ⑥ 展示タイム偏差: 好タイム→安定、悪タイム→不安定
            ex_time = boat.get('exhibition_time')
            if ex_time and ex_time > 0 and avg_exhibition:
                diff = ex_time - avg_exhibition
                if diff < -0.05:       # 平均より0.05秒以上速い
                    stds[i] *= 0.90    # 安定方向
                elif diff > 0.10:      # 平均より0.10秒以上遅い
                    stds[i] *= 1.10    # 不安定方向

            # ⑦ 平均ST: スタートが不安定な選手は波乱要因
            avg_st = boat.get('avg_st')
            if avg_st is not None:
                if avg_st > 0.20:      # 遅い → 出遅れリスク
                    stds[i] *= 1.10
                elif avg_st < 0.10:    # 早い → フライングリスクで慎重になる
                    stds[i] *= 1.08

            # ⑧ 進入コース: アウトコースは展開依存
            course = boat.get('approach_course')
            if course is not None:
                if course >= 5:        # 5-6コース
                    stds[i] *= 1.15
                elif course >= 4:      # 4コース
                    stds[i] *= 1.05

    return ratings, stds


def monte_carlo_sanrentan(probs_1st, boats_data=None,
                          n_simulations=50000, seed=None,
                          race_data=None, race_number=None):
    """モンテカルロ・シミュレーションで三連単確率を直接算出

    Args:
        probs_1st: list[6] — NNのsoftmax出力（1着確率）
        boats_data: list[dict] — 6艇の選手データ
        n_simulations: シミュレーション回数（v2: 50000）
        seed: 乱数シード（再現性用）
        race_data: dict — レース環境データ（風速/波高等）
        race_number: int — レース番号 1-12

    Returns:
        dict: {"1-2-3": prob, ...} — 三連単確率（120通り）
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    ratings, stds = compute_ratings(probs_1st, boats_data,
                                    race_data=race_data,
                                    race_number=race_number)

    # N回シミュレーション: 各艇のパフォーマンスを正規分布からサンプリング
    # shape: (n_simulations, 6)
    performances = rng.normal(
        loc=ratings,       # (6,) → broadcast to (n_sim, 6)
        scale=stds,        # (6,) → broadcast to (n_sim, 6)
        size=(n_simulations, 6)
    )

    # パフォーマンス降順 → 着順（高いほど上位）
    orderings = np.argsort(-performances, axis=1)  # (n_sim, 6)

    # 三連単カウント: 上位3艇の組み合わせ
    top3 = orderings[:, :3] + 1  # 0-indexed → 1-indexed
    keys = [f"{t[0]}-{t[1]}-{t[2]}" for t in top3]

    sanrentan_counts = {}
    for key in keys:
        sanrentan_counts[key] = sanrentan_counts.get(key, 0) + 1

    # カウント → 確率
    sanrentan_probs = {
        key: count / n_simulations
        for key, count in sanrentan_counts.items()
        if count > 0
    }

    return sanrentan_probs


def monte_carlo_positions(probs_1st, boats_data=None,
                          n_simulations=50000, seed=None,
                          race_data=None, race_number=None):
    """着順別の確率分布を算出（デバッグ・分析用）

    Returns:
        np.array shape (6, 6) — [boat_idx][position_idx] = prob
        例: result[0][0] = 1号艇が1着になる確率
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    ratings, stds = compute_ratings(probs_1st, boats_data,
                                    race_data=race_data,
                                    race_number=race_number)

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


def qmc_sanrentan(probs_1st, boats_data=None,
                  n_simulations=8192, seed=None,
                  race_data=None, race_number=None):
    """準モンテカルロ法 (Sobol列) で三連単確率を算出

    通常MCの乱数をSobol列（低食い違い量列）に置換。
    空間を均等に埋めるため、少ない試行で高精度な確率推定が可能。
    収束速度: MC O(1/√N) → QMC O(1/N)

    Args:
        probs_1st: list[6] — NNのsoftmax出力（1着確率）
        boats_data: list[dict] — 6艇の選手データ
        n_simulations: シミュレーション回数（2の冪乗が最適、デフォルト8192）
        seed: 乱数シード（Sobolのスクランブル用）
        race_data: dict — レース環境データ（風速/波高等）
        race_number: int — レース番号 1-12

    Returns:
        dict: {"1-2-3": prob, ...} — 三連単確率（120通り）
    """
    ratings, stds = compute_ratings(probs_1st, boats_data,
                                    race_data=race_data,
                                    race_number=race_number)

    # Sobol列で [0,1]^6 の均等点を生成（6次元 = 6艇）
    # scramble=True でランダム化QMC（分散推定可能 + 偏り防止）
    sampler = qmc.Sobol(d=6, scramble=True,
                        seed=seed if seed is not None else np.random.randint(0, 2**31))

    # n_simulations を2の冪乗に切り上げ（Sobol列の最適条件）
    m = int(np.ceil(np.log2(max(n_simulations, 64))))
    n_actual = 2 ** m
    uniform_samples = sampler.random(n_actual)  # shape: (n_actual, 6)

    # [0,1] 均一分布 → 正規分布に変換（逆CDF変換）
    # 各艇の ratings (平均) と stds (標準偏差) を使用
    performances = norm.ppf(uniform_samples,
                            loc=ratings,    # (6,) → broadcast
                            scale=stds)     # (6,) → broadcast

    # パフォーマンス降順 → 着順
    orderings = np.argsort(-performances, axis=1)

    # 三連単カウント
    top3 = orderings[:, :3] + 1
    keys = [f"{t[0]}-{t[1]}-{t[2]}" for t in top3]

    sanrentan_counts = {}
    for key in keys:
        sanrentan_counts[key] = sanrentan_counts.get(key, 0) + 1

    sanrentan_probs = {
        key: count / n_actual
        for key, count in sanrentan_counts.items()
        if count > 0
    }

    return sanrentan_probs


def compute_ratings_early(probs_1st, boats_data=None, race_data=None,
                          race_number=None):
    """序盤R1-R4特化ノイズモデル

    汎用版 compute_ratings() との違い:
      - クラス係数を強化（A級とB級の実力差が大きいレース向け）
      - クラス分散: 6艇のクラスばらつきが大きい→予測しやすい→分散小
      - 展示タイム差: 最速-最遅の差が大きい→実力差明確→分散小
      - 当地勝率: 高い→その場に慣れている→分散小
      - レース番号の全体調整は不要（R1-R4専用のため）
    """
    probs = np.array(probs_1st, dtype=np.float64)
    probs = np.clip(probs, 0.01, 0.99)
    ratings = np.log(probs / (1.0 - probs))

    base_std = 0.8
    stds = np.full(6, base_std)

    # === 全艇共通の環境要因 ===

    # 風速・波高
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

    # === 序盤特化: クラス分散 ===
    # 6艇のクラスのばらつきが大きい → 実力差が明確 → 予測しやすい
    if boats_data and len(boats_data) == 6:
        class_values = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        classes = [class_values.get(b.get('player_class', 'B1'), 2)
                   for b in boats_data]
        class_std = np.std(classes)
        # クラス分散が大きい → 分散小（予測しやすい）
        # std=0: 全員同クラス → ×1.10, std=1.5: 大混在 → ×0.85
        class_spread_factor = 1.10 - 0.17 * class_std
        class_spread_factor = np.clip(class_spread_factor, 0.85, 1.10)
        stds *= class_spread_factor

    # === 序盤特化: 展示タイム差 ===
    # 最速と最遅の差が大きい → 実力差明確 → 分散小
    ex_times = []
    if boats_data and len(boats_data) == 6:
        for boat in boats_data:
            et = boat.get('exhibition_time')
            if et and et > 0:
                ex_times.append(et)
    if len(ex_times) >= 4:
        ex_range = max(ex_times) - min(ex_times)
        # 差0.1秒以下: 横並び → ×1.05, 差0.3秒以上: 差が明確 → ×0.90
        ex_factor = 1.05 - 0.5 * min(ex_range, 0.3)
        stds *= ex_factor

    # === 艇別の要因 ===
    avg_exhibition = sum(ex_times) / len(ex_times) if ex_times else None

    if boats_data and len(boats_data) == 6:
        for i, boat in enumerate(boats_data):
            # ① クラス係数（序盤強化版: A級の安定性をより重視）
            player_class = boat.get('player_class', 'B1')
            class_factor = {
                'A1': 0.75, 'A2': 0.90, 'B1': 1.10, 'B2': 1.35
            }.get(player_class, 1.0)
            stds[i] *= class_factor

            # ② モーター勝率が極端 → 分散拡大
            motor_wr2 = boat.get('motor_win_rate_2', 30.0) or 30.0
            if motor_wr2 > 50.0 or motor_wr2 < 15.0:
                stds[i] *= 1.1

            # ③ 部品交換 → 未知数
            if boat.get('parts_changed', False):
                stds[i] *= 1.15

            # ④ 展示タイム偏差
            ex_time = boat.get('exhibition_time')
            if ex_time and ex_time > 0 and avg_exhibition:
                diff = ex_time - avg_exhibition
                if diff < -0.05:
                    stds[i] *= 0.88   # 好タイム → より安定（v1: 0.90）
                elif diff > 0.10:
                    stds[i] *= 1.12   # 悪タイム → より不安定（v1: 1.10）

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

            # ⑦ 当地勝率（序盤特化: その場での経験値）
            local_wr = boat.get('local_win_rate_2') or boat.get('local_win_rate')
            if local_wr is not None and local_wr > 0:
                if local_wr > 40.0:
                    stds[i] *= 0.90   # 当地に強い → 安定
                elif local_wr < 15.0:
                    stds[i] *= 1.10   # 当地で弱い → 不安定

    return ratings, stds


def qmc_sanrentan_v3(probs_1st, boats_data=None,
                     n_simulations=8192, seed=None,
                     race_data=None, race_number=None):
    """序盤特化QMC (v3) で三連単確率を算出

    compute_ratings_early() + Sobol列。
    序盤R1-R4のクラス混在レース向けノイズモデル × 準モンテカルロの高精度サンプリング。
    """
    ratings, stds = compute_ratings_early(probs_1st, boats_data,
                                          race_data=race_data,
                                          race_number=race_number)

    sampler = qmc.Sobol(d=6, scramble=True,
                        seed=seed if seed is not None else np.random.randint(0, 2**31))

    m = int(np.ceil(np.log2(max(n_simulations, 64))))
    n_actual = 2 ** m
    uniform_samples = sampler.random(n_actual)

    performances = norm.ppf(uniform_samples,
                            loc=ratings,
                            scale=stds)

    orderings = np.argsort(-performances, axis=1)
    top3 = orderings[:, :3] + 1
    keys = [f"{t[0]}-{t[1]}-{t[2]}" for t in top3]

    sanrentan_counts = {}
    for key in keys:
        sanrentan_counts[key] = sanrentan_counts.get(key, 0) + 1

    sanrentan_probs = {
        key: count / n_actual
        for key, count in sanrentan_counts.items()
        if count > 0
    }

    return sanrentan_probs
