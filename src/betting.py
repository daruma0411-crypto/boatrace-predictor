"""ケリー基準ベッティング戦略（最重要モジュール）

v9.2: 定石Kelly戦略(I) 追加 (2026-03-18)

joseki_kelly (I戦略):
  43,965レース分析に基づく定石フィルタリング:
  - 本命場(b1≥58%)スキップ → オッズ圧縮で勝てない
  - 荒れ場(b1≤52%)でベット → オッズに歪みあり
  - グレー場(53-57%)はR1-R6のみ → 後半Rは本命化
  キャリブレーション ×2.3（モデル過小推定補正）
  リアルKelly: キャップ撤廃、Kelly≤0はスキップ

既存A-F+H戦略は並走継続。
"""
import json
import logging
import math
import os
from itertools import permutations
from src.database import get_db_connection, get_current_bankroll

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'config', 'betting_config.json'
)
CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'config', 'calibration.json'
)

# --- 場の荒れ度分類 (1号艇勝率ベース, 43,965レース分析) ---
# 既存A-F用（動的オッズ上限調整）
VENUE_STABLE = {18, 19, 24}  # 徳山, 下関, 大村
VENUE_CHAOTIC = {2, 3, 4, 14}  # 戸田, 江戸川, 平和島, 鳴門
RACE_STABLE = {11, 12}     # R11-R12: 1kaku 67-69%
RACE_CHAOTIC = {2, 3, 4}   # R2-R4: 1kaku 45-48%

# --- 全戦略共通: 壊滅会場除外 (Phase2分析結果) ---
# ROI壊滅の4会場は全戦略で無条件スキップ
VENUE_BLACKLIST = {7, 15, 16, 18, 19, 21}  # 蒲郡(2.2%), 丸亀(0%), 児島(27.5%), 徳山(0%), 下関(12%), 芦屋(21.3%)

# --- 定石Kelly(I戦略)用: 3層分類 ---
# 本命場 (b1≥58%): オッズ圧縮でROI 25.8% → ベットしない
VENUE_HONMEI = {8, 11, 18, 19, 21, 24}  # 常滑, びわこ, 徳山, 下関, 芦屋, 大村
# 荒れ場 (b1≤52%): オッズに歪みありROI 95.2% → ベットする
VENUE_ARE = {2, 3, 4, 5, 6, 14, 15, 20}  # 戸田, 江戸川, 平和島, 多摩川, 浜名湖, 鳴門, 丸亀, 若松
# グレー場 (53-57%): 条件付きベット（R1-R6のみ）
# 上記2つに含まれない場 = {1, 7, 9, 10, 12, 13, 16, 17, 22, 23}
# 桐生, 蒲郡, 津, 三国, 住之江, 尼崎, 児島, 宮島, 福岡, 唐津


def _load_calibration():
    """キャリブレーション係数を読み込み

    config/calibration.json から確率帯別の補正係数を取得。
    ファイルがなければデフォルト（補正なし=1.0）を返す。
    """
    try:
        with open(CALIBRATION_PATH, 'r') as f:
            data = json.load(f)
        bands = data.get('prob_bands', [])
        if bands:
            logger.info(f"キャリブレーション読み込み: {len(bands)}帯")
            return bands
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"キャリブレーション読み込みエラー: {e}")
    return []


def _apply_calibration(prob, calibration_bands):
    """確率帯別キャリブレーション係数を適用

    Args:
        prob: モデルの生確率
        calibration_bands: _load_calibration()の返値

    Returns:
        補正済み確率
    """
    if not calibration_bands:
        return prob

    for band in calibration_bands:
        if band['min'] <= prob < band['max']:
            factor = band.get('calibration_factor', 1.0)
            return prob * factor

    # どの帯にも該当しない場合はそのまま
    return prob


def _load_config():
    """ベッティング設定を読み込み"""
    defaults = {
        'initial_bankroll': 200000,
        'strategies': {
            'conservative': {
                'kelly_fraction': 0.125,
                'max_total_bet_ratio': 0.02,
                'max_ticket_bet_ratio': 0.008,
                'min_expected_value': 1.15,
                'odds_discount_factor': 0.90,
                'max_recommended_bets': 3,
                'min_bet_amount': 100,
                'max_odds': 80,
                'filter_type': 'none',
            },
            'standard': {
                'kelly_fraction': 0.25,
                'max_total_bet_ratio': 0.03,
                'max_ticket_bet_ratio': 0.012,
                'min_expected_value': 1.10,
                'odds_discount_factor': 0.95,
                'max_recommended_bets': 5,
                'min_bet_amount': 100,
                'max_odds': 150,
                'filter_type': 'none',
            },
        },
    }
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        defaults.update(config)
    except FileNotFoundError:
        pass
    return defaults


def _adjust_max_odds(base_max_odds, venue_id, race_number):
    """場とレース番号でオッズ上限を動的調整

    堅い場 × 堅いR → base × 0.7（絞る）
    荒れる場 × 荒れるR → base × 1.5（広げる）
    """
    factor = 1.0

    # 場の調整
    if venue_id in VENUE_STABLE:
        factor *= 0.8
    elif venue_id in VENUE_CHAOTIC:
        factor *= 1.3

    # レース番号の調整
    if race_number in RACE_STABLE:
        factor *= 0.8
    elif race_number in RACE_CHAOTIC:
        factor *= 1.3

    adjusted = base_max_odds * factor
    # 絶対上限: configの値を超えない（動的拡張を禁止）
    return min(adjusted, base_max_odds)


def _should_skip_by_top_boat(probs_1st):
    """発見3: モデルが5-6号艇を1着最有力と予測 → ゾーン外なのでスキップ

    5-6号艇が1着の場合、80x以下に収まるのは41%しかない。
    モデルの1着予測トップが5番or6番ならベットしない。
    """
    top_boat = max(range(6), key=lambda i: probs_1st[i])
    # 0-indexed: 4=5号艇, 5=6号艇
    return top_boat >= 4


def _calculate_entropy(probs):
    """Shannon entropy H = -Σ p_i * log2(p_i)

    低いほど確信度が高い（1つに集中）。
    6艇の一様分布だと H = log2(6) ≈ 2.585。
    """
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def _check_ensemble_agreement(ensemble_predictions, min_agreement=3):
    """モデル多数決による1着予測一致チェック

    min_agreement=3: 4モデル中3つ以上が一致すればOK（デフォルト）
    min_agreement=4: 全一致（旧動作）

    Returns:
        tuple: (agreed: bool, top_boat_idx: int or None)
    """
    if not ensemble_predictions:
        return False, None

    top_boats = []
    for pred in ensemble_predictions:
        probs = pred['probs_1st']
        top = max(range(6), key=lambda i: probs[i])
        top_boats.append(top)

    from collections import Counter
    counts = Counter(top_boats)
    most_common_boat, most_common_count = counts.most_common(1)[0]

    if most_common_count >= min_agreement:
        return True, most_common_boat
    return False, None


def _average_ensemble_probs(ensemble_predictions):
    """全モデルの確率を平均する

    Returns:
        dict: {probs_1st, probs_2nd, probs_3rd} (平均済み)
    """
    n = len(ensemble_predictions)
    avg_1st = [0.0] * 6
    avg_2nd = [0.0] * 6
    avg_3rd = [0.0] * 6

    for pred in ensemble_predictions:
        for i in range(6):
            avg_1st[i] += pred['probs_1st'][i] / n
            avg_2nd[i] += pred['probs_2nd'][i] / n
            avg_3rd[i] += pred['probs_3rd'][i] / n

    return {
        'probs_1st': avg_1st,
        'probs_2nd': avg_2nd,
        'probs_3rd': avg_3rd,
    }


def _get_dynamic_discount(raw_odds):
    """オッズ帯別の動的ディスカウントファクター

    データ分析結果:
    - 低オッズ(<25x): 確定時に大きく下落(-34%まで) → 厳しく割引
    - 中オッズ(25-40x): 中程度の変動(-11%程度) → やや割引
    - 高オッズ(40x+): 変動小(+6%〜-5%) → 軽い割引
    """
    if raw_odds < 25.0:
        return 0.85
    elif raw_odds < 40.0:
        return 0.88
    else:
        return 0.95


DAILY_LOSS_LIMIT_PER_STRATEGY = 30000  # 戦略別の1日最大損失額
DAILY_BET_LIMIT_PER_STRATEGY = 9999  # リミット無効化 (比較テスト中)

TEST_MODE = False  # Kelly有効化: 日次損失制限・ドローダウン防止ON

# v10.6: MC v1 vs QMC vs MC v3(序盤特化) A/Bテスト 11戦略
ACTIVE_STRATEGIES = {
    # MC v1 (20K/3変数) — 既存データの連続性を維持
    'mc_quarter_kelly',     # L: MC v1 基準
    'mc_early_race',        # O: MC v1 + R1-R4限定
    'mc_venue_focus',       # P: MC v1 + 得意会場
    'mc_high_ev',           # Q: MC v1 + EV≥1.0
    'mc_are_v2',            # R: MC v1 + ModelB
    # QMC (Sobol 8192回) — 準モンテカルロ法
    'mc2_quarter_kelly',    # L2: QMC 基準
    'mc2_early_race',       # O2: QMC + R1-R4限定
    'mc2_venue_focus',      # P2: QMC + 得意会場
    'mc2_high_ev',          # Q2: QMC + EV≥1.0
    'mc2_are_v2',           # R2: QMC + ModelB
    # QMC v3 (序盤特化ノイズモデル + Sobol列) — クラス分散/展示タイム差/当地勝率+風速/波高
    'mc3_early_race',       # O3: QMC v3 + R1-R4限定
    # Miss Analysis フィルタ (V10ベース + 勝ち会場ホワイトリスト + R1除外)
    'mc_early_race_filtered',  # Of: O + Miss Analysis (shadow検証中, 2026-04-28〜)
    # P4 (P2 + V3 + TOP10会場): 2026-05-01 新設
    'mc3_venue_focus',     # P4: QMC v3 + 10会場ホワイトリスト (R1-R12)
    'mc3_venue_focus_r2',  # P5: P4 + R1-R2 限定
    'mc3_venue_focus_r4',  # P6: P4 + R1-R4 限定
}


def _get_today_bet_count(strategy_type):
    """当日の指定戦略のベット件数をDBから取得"""
    if TEST_MODE:
        return 0
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) as cnt FROM bets
                WHERE strategy_type = %s
                  AND created_at >= CURRENT_DATE
            """, (strategy_type,))
            row = cur.fetchone()
            return int(row['cnt']) if row else 0
    except Exception as e:
        logger.warning(f"日次ベット数チェックDB障害: {e}")
        return 0


def _get_today_consecutive_losses(strategy_type):
    """当日の直近連敗数を取得（結果確定済みベットのみ）"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT is_hit FROM bets
                WHERE strategy_type = %s
                  AND created_at >= CURRENT_DATE
                  AND result IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 20
            """, (strategy_type,))
            rows = cur.fetchall()
            streak = 0
            for r in rows:
                if r['is_hit']:
                    break
                streak += 1
            return streak
    except Exception as e:
        logger.warning(f"連敗チェックDB障害: {e}")
        return 0


def _get_today_strategy_loss(strategy_type):
    """本日の指定戦略の損失額をDBから取得"""
    if TEST_MODE:
        return 0
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COALESCE(SUM(amount), 0) as total_wagered,
                       COALESCE(SUM(payout), 0) as total_payout
                FROM bets
                WHERE strategy_type = %s
                  AND created_at >= CURRENT_DATE
            """, (strategy_type,))
            row = cur.fetchone()
            wagered = float(row['total_wagered'])
            payout = float(row['total_payout'])
            return wagered - payout  # 正=損失, 負=利益
    except Exception as e:
        logger.warning(f"日次損失チェックDB障害: {e}")
        return 0


class KellyBettingStrategy:
    """ケリー基準 + 6戦略並列テスト ベッティング戦略"""

    def __init__(self, initial_bankroll=None):
        self.config = _load_config()
        self.calibration_bands = _load_calibration()
        self.initial_bankroll = (
            initial_bankroll or self.config['initial_bankroll']
        )

    def calculate_all_strategies(self, probs_1st, probs_2nd, probs_3rd,
                                  odds_data, bankroll=None,
                                  venue_id=None, race_number=None,
                                  ensemble_predictions=None,
                                  odds_2t=None,
                                  boats_data=None, race_data=None,
                                  are_prediction=None):
        """全戦略を計算して返す

        are_prediction: Model B (荒れ専門) の予測結果
            {probs_1st: [5], probs_2nd: [6], probs_3rd: [6]}
            probs_1st は 2-6号艇の5クラス確率
        """
        # 壊滅会場ブラックリスト: 全戦略共通で無条件スキップ
        if venue_id is not None and venue_id in VENUE_BLACKLIST:
            logger.info(
                f"壊滅会場ブラックリスト: 全戦略スキップ (場{venue_id})"
            )
            return {s: [] for s in self.config['strategies']}

        # 1号艇軸スキップ: モデルが1号艇1着と予測 → エッジなし（市場織り込み済み）
        top_boat = max(range(6), key=lambda i: probs_1st[i])
        if top_boat == 0:  # 0-indexed: 0=1号艇
            logger.info(
                f"1号艇軸スキップ: 全戦略スキップ "
                f"(場{venue_id} R{race_number}, P={probs_1st[0]:.3f})"
            )
            return {s: [] for s in self.config['strategies']}

        # 5-6号艇軸: 戦略設定の skip_56=true で有効化
        skip_56 = _should_skip_by_top_boat(probs_1st)
        if skip_56:
            logger.info(
                f"5-6号艇軸: 注意 "
                f"(モデル1着予測={top_boat+1}号艇, 場{venue_id} R{race_number})"
            )

        # 通常の3連単確率（A/B/C/D/F用）
        sanrentan_probs = self._calculate_sanrentan_bets_conditional(
            probs_1st, probs_2nd, probs_3rd
        )

        # モンテカルロ・シミュレーション確率（MC対応戦略用）
        mc_sanrentan = None   # v1用キャッシュ
        _mc_cache = {}        # v2用キャッシュ（戦略別）

        # 2連単確率（G戦略用）
        nirentan_probs = self._calculate_nirentan_probs(probs_1st, probs_2nd)

        # 市場乖離度テーブル: model_prob / market_prob
        divergence_map = {}
        if odds_data:
            for combo, prob in sanrentan_probs.items():
                raw_odds = odds_data.get(combo, 0.0)
                if raw_odds > 1.0:
                    market_prob = 1.0 / raw_odds
                    divergence_map[combo] = prob / market_prob if market_prob > 0 else 0.0

        # エントロピー計算（D/F用）
        entropy_1st = _calculate_entropy(probs_1st)

        # アンサンブル合議（E用）— configからmin_agreement取得
        ensemble_agreed = False
        ensemble_top_boat = None
        ensemble_sanrentan = None
        ens_config = self.config['strategies'].get('ensemble', {})
        min_agreement = ens_config.get('min_agreement', 3)
        if ensemble_predictions and len(ensemble_predictions) >= 2:
            ensemble_agreed, ensemble_top_boat = _check_ensemble_agreement(
                ensemble_predictions, min_agreement=min_agreement
            )
            if ensemble_agreed:
                avg = _average_ensemble_probs(ensemble_predictions)
                ensemble_sanrentan = self._calculate_sanrentan_bets_conditional(
                    avg['probs_1st'], avg['probs_2nd'], avg['probs_3rd']
                )

        results = {}
        for strategy_name, strategy_config in self.config['strategies'].items():
            # アクティブ戦略以外はスキップ
            if strategy_name not in ACTIVE_STRATEGIES:
                results[strategy_name] = []
                continue

            # 場・レース番号の除外フィルタ
            exclude_venues = strategy_config.get('exclude_venues', [])
            if venue_id is not None and venue_id in exclude_venues:
                logger.info(
                    f"場除外フィルタ: {strategy_name} スキップ "
                    f"(場{venue_id} in {exclude_venues})"
                )
                results[strategy_name] = []
                continue

            # 場ホワイトリスト（指定時はリスト内の会場のみベット）
            include_venues = strategy_config.get('include_venues', [])
            if include_venues and venue_id is not None and venue_id not in include_venues:
                logger.info(
                    f"場ホワイトリスト: {strategy_name} スキップ "
                    f"(場{venue_id} not in {include_venues})"
                )
                results[strategy_name] = []
                continue

            exclude_races = strategy_config.get('exclude_race_numbers', [])
            if race_number is not None and race_number in exclude_races:
                logger.info(
                    f"R除外フィルタ: {strategy_name} スキップ "
                    f"(R{race_number} in {exclude_races})"
                )
                results[strategy_name] = []
                continue

            # レース番号上限フィルター（序盤レース特化用）
            max_race = strategy_config.get('max_race_number', 12)
            if race_number is not None and race_number > max_race:
                logger.info(
                    f"R上限フィルタ: {strategy_name} スキップ "
                    f"(R{race_number} > max_race={max_race})"
                )
                results[strategy_name] = []
                continue

            # --- 5-6号艇スキップ (skip_56=true の戦略のみ) ---
            if strategy_config.get('skip_56', False) and skip_56:
                top_boat = max(range(6), key=lambda i: probs_1st[i]) + 1
                logger.info(
                    f"5-6号艇スキップ: {strategy_name} "
                    f"(モデル1着予測={top_boat}号艇, 場{venue_id} R{race_number})"
                )
                results[strategy_name] = []
                continue

            # --- 定石フィルタ (joseki_mode=true の戦略のみ) ---
            if strategy_config.get('joseki_mode', False) and venue_id is not None:
                if venue_id in VENUE_HONMEI:
                    logger.info(
                        f"定石フィルタ: {strategy_name} 本命場スキップ "
                        f"(場{venue_id})"
                    )
                    results[strategy_name] = []
                    continue
                # グレー場: R1-R6のみ（joseki_skip_gray_late=falseで無効化可）
                if strategy_config.get('joseki_skip_gray_late', True):
                    is_gray = venue_id not in VENUE_ARE
                    if is_gray and race_number is not None and race_number >= 7:
                        logger.info(
                            f"定石フィルタ: {strategy_name} グレー場後半Rスキップ "
                            f"(場{venue_id} R{race_number})"
                        )
                        results[strategy_name] = []
                        continue

            # 戦略別日次ベット数制限
            today_bets = _get_today_bet_count(strategy_name)
            if today_bets >= DAILY_BET_LIMIT_PER_STRATEGY:
                logger.warning(
                    f"[{strategy_name}] 日次ベット上限: "
                    f"{today_bets}件 >= {DAILY_BET_LIMIT_PER_STRATEGY}件 → スキップ"
                )
                results[strategy_name] = []
                continue

            # 戦略別日次損失制限
            strategy_loss = _get_today_strategy_loss(strategy_name)
            if strategy_loss >= DAILY_LOSS_LIMIT_PER_STRATEGY:
                logger.warning(
                    f"[{strategy_name}] 日次損失上限: "
                    f"{strategy_loss:,.0f}円 >= {DAILY_LOSS_LIMIT_PER_STRATEGY:,}円 → スキップ"
                )
                results[strategy_name] = []
                continue

            filter_type = strategy_config.get('filter_type', 'none')

            # --- フィルタ判定 ---
            if filter_type == 'divergence':
                # 乖離度フィルタは_strategy_kelly内でcombo単位で適用
                pass

            elif filter_type == 'entropy':
                max_entropy = strategy_config.get('max_entropy', 1.5)
                if entropy_1st >= max_entropy:
                    logger.info(
                        f"エントロピーフィルタ: {strategy_name} スキップ "
                        f"(H={entropy_1st:.3f} >= {max_entropy})"
                    )
                    results[strategy_name] = []
                    continue

            elif filter_type == 'ensemble':
                if not ensemble_agreed:
                    logger.info(
                        f"アンサンブル不一致: {strategy_name} スキップ"
                    )
                    results[strategy_name] = []
                    continue

            elif filter_type == 'div_confidence':
                max_entropy = strategy_config.get('max_entropy', 1.5)
                if entropy_1st >= max_entropy:
                    logger.info(
                        f"div_confidence エントロピーフィルタ: {strategy_name} スキップ "
                        f"(H={entropy_1st:.3f} >= {max_entropy})"
                    )
                    results[strategy_name] = []
                    continue

            elif filter_type == 'optuna':
                # Optuna最適化パラメータによるレースレベルフィルタ
                # (1) 1号艇確率上限: P(1着=1号艇) > threshold なら本命すぎてスキップ
                max_b1 = strategy_config.get('max_boat1_prob', 0.70)
                boat1_p = probs_1st[0]  # 0-indexed: 1号艇
                if boat1_p > max_b1:
                    logger.info(
                        f"optunaフィルタ: {strategy_name} スキップ "
                        f"(1号艇P={boat1_p:.3f} > {max_b1})"
                    )
                    results[strategy_name] = []
                    continue
                # (2) エントロピー下限: H < threshold なら確信しすぎ（低配当ゾーン）
                min_ent = strategy_config.get('min_entropy', 1.0)
                if entropy_1st < min_ent:
                    logger.info(
                        f"optunaフィルタ: {strategy_name} スキップ "
                        f"(H={entropy_1st:.3f} < {min_ent})"
                    )
                    results[strategy_name] = []
                    continue

            # --- bet_mode + 使用する確率の決定 ---
            bet_mode = strategy_config.get('bet_mode', 'sanrentan')
            use_monte_carlo = strategy_config.get('use_monte_carlo', False)
            model_type = strategy_config.get('model_type', 'standard')

            if bet_mode == 'nirentan':
                use_probs = nirentan_probs
                use_odds = odds_2t if odds_2t else {}
            elif model_type == 'are':
                # Model B (荒れ専門): are_predictionが必須
                if are_prediction is None:
                    logger.warning(
                        f"[{strategy_name}] Model B未ロード → スキップ"
                    )
                    results[strategy_name] = []
                    continue
                # 5クラス確率から三連単計算
                are_sanrentan = self._calculate_are_sanrentan(
                    are_prediction['probs_1st'],
                    are_prediction['probs_2nd'],
                    are_prediction['probs_3rd'],
                )
                if use_monte_carlo:
                    # MC-B: Model Bの確率でモンテカルロ/QMC
                    # 5クラス確率を6クラスに展開（1号艇=0）してMC/QMCに渡す
                    probs_6 = [0.0] + list(are_prediction['probs_1st'])
                    mc_ver = strategy_config.get('mc_version', 1)
                    if mc_ver >= 2:
                        from src.monte_carlo import qmc_sanrentan
                        mc_are = qmc_sanrentan(
                            probs_6, boats_data=boats_data, n_simulations=8192,
                        )
                    else:
                        from src.monte_carlo import monte_carlo_sanrentan
                        mc_are = monte_carlo_sanrentan(
                            probs_6, boats_data=boats_data, n_simulations=20000,
                        )
                    # 1号艇軸を除外
                    use_probs = {k: v for k, v in mc_are.items()
                                 if not k.startswith('1-')}
                    mc_label = "QMC" if mc_ver >= 2 else f"MC v{mc_ver}"
                    logger.info(f"MC-B {mc_label}確率生成: {len(use_probs)}通り")
                else:
                    use_probs = are_sanrentan
                    logger.info(f"NN-B確率生成: {len(use_probs)}通り")
                use_odds = odds_data
            elif use_monte_carlo:
                # モンテカルロ確率（mc_version で v1/QMC/v3 分岐）
                mc_ver = strategy_config.get('mc_version', 1)
                if mc_ver >= 3:
                    # QMC v3: 序盤特化ノイズモデル + Sobol列（全変数）
                    cache_key = f'qmc_v3_{strategy_name}'
                    if cache_key not in _mc_cache:
                        from src.monte_carlo import qmc_sanrentan_v3
                        _mc_cache[cache_key] = qmc_sanrentan_v3(
                            probs_1st, boats_data=boats_data,
                            n_simulations=8192,
                            race_data=race_data,
                            race_number=race_number,
                        )
                        logger.info(
                            f"QMC v3確率生成: {len(_mc_cache[cache_key])}通り "
                            f"(top={max(_mc_cache[cache_key].values()):.4f})"
                        )
                    use_probs = _mc_cache[cache_key]
                elif mc_ver >= 2:
                    # QMC: Sobol列 8192回 — v1と同じ3変数、サンプリングのみ改良
                    cache_key = f'qmc_{strategy_name}'
                    if cache_key not in _mc_cache:
                        from src.monte_carlo import qmc_sanrentan
                        _mc_cache[cache_key] = qmc_sanrentan(
                            probs_1st, boats_data=boats_data,
                            n_simulations=8192,
                        )
                        logger.info(
                            f"QMC確率生成: {len(_mc_cache[cache_key])}通り "
                            f"(top={max(_mc_cache[cache_key].values()):.4f})"
                        )
                    use_probs = _mc_cache[cache_key]
                else:
                    # MC v1: 20K/3変数 — 全v1戦略で共有
                    if mc_sanrentan is None:
                        from src.monte_carlo import monte_carlo_sanrentan
                        mc_sanrentan = monte_carlo_sanrentan(
                            probs_1st, boats_data=boats_data,
                            n_simulations=20000,
                        )
                        logger.info(
                            f"MC v1確率生成: {len(mc_sanrentan)}通り "
                            f"(top={max(mc_sanrentan.values()):.4f})"
                        )
                    use_probs = mc_sanrentan
                use_odds = odds_data
            elif filter_type == 'ensemble' and ensemble_sanrentan is not None:
                use_probs = ensemble_sanrentan
                use_odds = odds_data
            else:
                use_probs = sanrentan_probs
                use_odds = odds_data

            # 各戦略で独立bankroll
            if bankroll is not None:
                br = bankroll
            else:
                profit = get_current_bankroll(strategy_type=strategy_name)
                br = float(self.initial_bankroll + profit)

            # ドローダウン防止1: bankroll破産ガード
            if br <= 0:
                logger.warning(f"[{strategy_name}] bankroll≤0 ({br:.0f}円) → スキップ")
                results[strategy_name] = []
                continue

            # ドローダウン防止2: bankroll半減時にKelly縮小
            drawdown_ratio = br / self.initial_bankroll
            dd_multiplier = 1.0
            if drawdown_ratio < 0.5:
                dd_multiplier = 0.5  # 半減以下: Kelly半額
                logger.info(f"[{strategy_name}] DD防止: bankroll={br:.0f}円 "
                            f"({drawdown_ratio:.0%}) → Kelly×0.5")
            elif drawdown_ratio < 0.75:
                dd_multiplier = 0.75  # 25%減: Kelly75%
                logger.info(f"[{strategy_name}] DD防止: bankroll={br:.0f}円 "
                            f"({drawdown_ratio:.0%}) → Kelly×0.75")

            # ドローダウン防止3: 当日連敗ブレーキ
            consecutive_losses = _get_today_consecutive_losses(strategy_name)
            if consecutive_losses >= 5:
                logger.warning(f"[{strategy_name}] 当日{consecutive_losses}連敗 → スキップ")
                results[strategy_name] = []
                continue

            bets = self._strategy_kelly(
                strategy_config, use_probs, use_odds, br,
                strategy_name, venue_id, race_number,
                divergence_map=divergence_map,
                dd_multiplier=dd_multiplier,
                calibration_bands=self.calibration_bands,
                bet_mode=bet_mode,
            )
            results[strategy_name] = bets

        return results

    def _strategy_kelly(self, config, sanrentan_probs, odds_data,
                         bankroll, strategy_name,
                         venue_id=None, race_number=None,
                         divergence_map=None, dd_multiplier=1.0,
                         calibration_bands=None,
                         bet_mode='sanrentan'):
        """共通ケリー戦略: EVゾーンフィルタ → Kelly計算

        v9: kelly_prob_gain パラメータでKelly計算時の確率をブースト。
        gain=1.0 → Kelly負 → ¥100固定 (Standard)
        gain=1.5 → Kelly正 → 複利運用 (High_confidence覚醒モード)
        """
        candidates = []
        kelly_frac = config['kelly_fraction']
        min_ev = config['min_expected_value']
        max_bets = config['max_recommended_bets']
        max_ticket = bankroll * config['max_ticket_bet_ratio']
        max_total = bankroll * config['max_total_bet_ratio']
        min_bet = config['min_bet_amount']
        odds_discount_static = config.get('odds_discount_factor', 0.92)  # fallback
        use_dynamic_discount = config.get('use_dynamic_discount', True)
        base_max_odds = config.get('max_odds', 9999)
        filter_type = config.get('filter_type', 'none')
        min_divergence = config.get('min_divergence_ratio', 2.0)
        min_prob = config.get('min_probability', 0.0)
        min_odds = config.get('min_odds', 0.0)  # 最低オッズ (ガチガチ本命除外)
        max_ev = config.get('max_expected_value', 9999.0)  # EV上限 (過大推定フィルタ)
        kelly_prob_gain = config.get('kelly_prob_gain', 1.0)  # ハッパ係数 (1.0=補正なし, 1.5=覚醒)

        # 発見1+2: 場・R番号でオッズ上限を動的調整
        if venue_id is not None and race_number is not None:
            max_odds = _adjust_max_odds(base_max_odds, venue_id, race_number)
        else:
            max_odds = base_max_odds

        # デバッグ: フィルター状態ログ
        odds_available = sum(1 for c in sanrentan_probs if odds_data.get(c, 0.0) > 1.0)
        discount_mode = "dynamic" if use_dynamic_discount else f"static={odds_discount_static}"
        logger.info(
            f"[{strategy_name}] フィルター設定: "
            f"max_odds={max_odds}(base={base_max_odds}), "
            f"min_odds={min_odds}, min_ev={min_ev}, min_prob={min_prob}, "
            f"discount={discount_mode}, "
            f"odds有効={odds_available}/{len(sanrentan_probs)}件"
        )

        skip_counts = {'no_odds': 0, 'low_prob': 0, 'low_odds': 0,
                       'high_odds': 0, 'low_ev': 0, 'high_ev': 0,
                       'divergence': 0, 'kelly_neg': 0}

        for combo, prob in sanrentan_probs.items():
            raw_odds = odds_data.get(combo, 0.0)
            if raw_odds <= 1.0 or prob <= 0:
                skip_counts['no_odds'] += 1
                continue

            # 最低確率フィルター: 宝くじ買い目を排除
            if prob < min_prob:
                skip_counts['low_prob'] += 1
                continue

            # オッズ下限フィルター: ガチガチ本命を除外
            if min_odds > 0 and raw_odds < min_odds:
                skip_counts['low_odds'] += 1
                continue

            # オッズ上限フィルター（動的調整済み）
            if raw_odds > max_odds:
                skip_counts['high_odds'] += 1
                continue

            # --- 乖離度フィルター (C, F) ---
            if filter_type in ('divergence', 'div_confidence'):
                if divergence_map:
                    div_ratio = divergence_map.get(combo, 0.0)
                    if div_ratio < min_divergence:
                        skip_counts['divergence'] += 1
                        continue

            # オッズ割引（動的 or 静的）
            if use_dynamic_discount:
                odds_discount = _get_dynamic_discount(raw_odds)
            else:
                odds_discount = odds_discount_static
            discounted_odds = raw_odds * odds_discount

            # キャリブレーション
            flat_cal = config.get('calibration_factor', 0)
            if flat_cal > 0:
                # 定石戦略: フラット倍率キャリブレーション (×2.3等)
                cal_prob = min(prob * flat_cal, 0.95)
            else:
                # 既存戦略: 確率帯別キャリブレーション
                cal_prob = _apply_calibration(prob, calibration_bands)

            # EV計算（補正確率ベース）: EV = cal_prob × discounted_odds
            ev = cal_prob * discounted_odds
            if ev < min_ev:
                skip_counts['low_ev'] += 1
                continue
            if ev > max_ev:
                skip_counts['high_ev'] += 1
                continue
            b = discounted_odds - 1.0
            if b < 0.01:
                skip_counts['kelly_neg'] += 1
                continue

            # ケリー基準
            real_kelly = config.get('real_kelly', False)
            boosted_prob = min(cal_prob * kelly_prob_gain, 0.99)
            q_boosted = 1.0 - boosted_prob
            kelly = (b * boosted_prob - q_boosted) / b
            if kelly > 0:
                kelly_amount = bankroll * kelly * kelly_frac * dd_multiplier
                kelly_cap = config.get('kelly_cap', 1000)
                if real_kelly:
                    # リアルKelly: キャップなし（max_ticket制限のみ）
                    kelly_amount = max(min_bet, min(max_ticket, kelly_amount))
                else:
                    # 設定可能キャップ (デフォルト¥1,000)
                    kelly_amount = max(min_bet, min(kelly_cap, max_ticket, kelly_amount))
                bet_amount = int(round(kelly_amount / 100) * 100)
            else:
                # Kelly≤0 → 全戦略でスキップ（EV<1.0の出血を止める）
                skip_counts['kelly_neg'] += 1
                continue

            if bet_amount >= min_bet:
                candidates.append({
                    'bet_type': bet_mode,
                    'combination': combo,
                    'amount': bet_amount,
                    'odds': raw_odds,
                    'discounted_odds': discounted_odds,
                    'expected_value': ev,
                    'kelly_fraction': kelly * kelly_frac,
                    'probability': prob,  # 生確率を保存（DB記録用）
                    'calibrated_prob': cal_prob,  # 補正確率（参考用）
                    'strategy_type': strategy_name,
                })

        # デバッグ: フィルターサマリ
        skip_summary = (
            f"no={skip_counts['no_odds']},prob={skip_counts['low_prob']},"
            f"lo={skip_counts['low_odds']},hi={skip_counts['high_odds']},"
            f"ev_lo={skip_counts['low_ev']},ev_hi={skip_counts['high_ev']},"
            f"div={skip_counts['divergence']},"
            f"kelly={skip_counts['kelly_neg']},ok={len(candidates)}"
        )
        logger.info(f"[{strategy_name}] skip: {skip_summary}")

        # DB診断: conservativeのスキップ内訳をscheduler_healthに記録
        if strategy_name == 'conservative':
            try:
                from src.database import get_db_connection
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO scheduler_health (status, detail) VALUES (%s, %s)",
                        ('skip_detail', f'v={venue_id}R={race_number} {skip_summary} min_ev={min_ev} min_odds={min_odds} min_prob={min_prob}'),
                    )
            except Exception:
                pass

        # 割引EV上位N点に絞る
        candidates.sort(key=lambda x: x['expected_value'], reverse=True)
        candidates = candidates[:max_bets]

        # レース合計上限カット
        total = sum(c['amount'] for c in candidates)
        if total > max_total and candidates:
            ratio = max_total / total
            for c in candidates:
                c['amount'] = max(
                    min_bet,
                    int(round(c['amount'] * ratio / 100) * 100)
                )

        return candidates

    def _calculate_sanrentan_bets_conditional(self, probs_1st, probs_2nd,
                                               probs_3rd):
        """条件付き確率による3連単確率計算

        P(i,j,k) = P(1st=i) × P(2nd=j|1st=i) × P(3rd=k|1st=i,2nd=j)
        同一艇の組み合わせは除外。ゼロ除算ガード付き。
        """
        sanrentan = {}

        for combo in permutations(range(6), 3):
            i, j, k = combo

            if len(set(combo)) != 3:
                continue

            p1 = probs_1st[i]
            if p1 <= 0:
                continue

            # P(2nd=j | 1st=i): i以外の中でのjの確率
            remaining_2nd = [probs_2nd[x] for x in range(6) if x != i]
            sum_remaining_2nd = sum(remaining_2nd)
            if sum_remaining_2nd <= 0:
                continue
            p2_given_1 = probs_2nd[j] / sum_remaining_2nd

            # P(3rd=k | 1st=i, 2nd=j): i,j以外の中でのkの確率
            remaining_3rd = [
                probs_3rd[x] for x in range(6) if x != i and x != j
            ]
            sum_remaining_3rd = sum(remaining_3rd)
            if sum_remaining_3rd <= 0:
                continue
            p3_given_12 = probs_3rd[k] / sum_remaining_3rd

            prob = p1 * p2_given_1 * p3_given_12
            if prob > 0:
                # 艇番号は1始まり
                key = f"{i+1}-{j+1}-{k+1}"
                sanrentan[key] = prob

        return sanrentan

    def _calculate_are_sanrentan(self, probs_1st_5, probs_2nd, probs_3rd):
        """Model B (荒れ専門) の三連単確率計算

        probs_1st_5: [5] — 2号艇〜6号艇の1着確率 (idx 0=2号艇, ..., 4=6号艇)
        probs_2nd:   [6] — 2着確率（通常の6クラス）
        probs_3rd:   [6] — 3着確率（通常の6クラス）

        1号艇が1着の組み合わせは生成しない（荒れ専門のため）
        """
        sanrentan = {}

        for i_5 in range(5):  # 1着: 2-6号艇 (5クラス)
            i_6 = i_5 + 1     # 6クラスインデックス (1=2号艇, ..., 5=6号艇)
            p1 = probs_1st_5[i_5]
            if p1 <= 0:
                continue

            for j in range(6):  # 2着: 全6艇
                if j == i_6:
                    continue
                remaining_2nd = [probs_2nd[x] for x in range(6) if x != i_6]
                sum_r2 = sum(remaining_2nd)
                if sum_r2 <= 0:
                    continue
                p2 = probs_2nd[j] / sum_r2

                for k in range(6):  # 3着: 全6艇
                    if k == i_6 or k == j:
                        continue
                    remaining_3rd = [probs_3rd[x] for x in range(6)
                                     if x != i_6 and x != j]
                    sum_r3 = sum(remaining_3rd)
                    if sum_r3 <= 0:
                        continue
                    p3 = probs_3rd[k] / sum_r3

                    prob = p1 * p2 * p3
                    if prob > 0:
                        key = f"{i_6+1}-{j+1}-{k+1}"
                        sanrentan[key] = prob

        return sanrentan

    def _calculate_nirentan_probs(self, probs_1st, probs_2nd):
        """条件付き確率による2連単確率計算

        P(i,j) = P(1st=i) × P(2nd=j|1st=i)
        30通り (6×5) の確率辞書を返す。
        """
        nirentan = {}

        for i in range(6):
            for j in range(6):
                if i == j:
                    continue

                p1 = probs_1st[i]
                if p1 <= 0:
                    continue

                # P(2nd=j | 1st=i): i以外の中でのjの確率
                remaining_2nd = [probs_2nd[x] for x in range(6) if x != i]
                sum_remaining_2nd = sum(remaining_2nd)
                if sum_remaining_2nd <= 0:
                    continue
                p2_given_1 = probs_2nd[j] / sum_remaining_2nd

                prob = p1 * p2_given_1
                if prob > 0:
                    key = f"{i+1}-{j+1}"
                    nirentan[key] = prob

        return nirentan

    def save_bets(self, bets, prediction_id, race_id):
        """ベットをDBに保存（重複時はスキップ）"""
        with get_db_connection() as conn:
            cur = conn.cursor()
            saved = 0
            for bet in bets:
                cur.execute("""
                    INSERT INTO bets
                    (prediction_id, race_id, bet_type, combination,
                     amount, odds, expected_value, kelly_fraction,
                     strategy_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (race_id, strategy_type, combination)
                    DO NOTHING
                """, (
                    prediction_id, race_id,
                    bet['bet_type'], bet['combination'],
                    bet['amount'], bet['odds'],
                    bet['expected_value'], bet['kelly_fraction'],
                    bet['strategy_type'],
                ))
                saved += cur.rowcount
            logger.info(f"ベット保存: {saved}/{len(bets)}件 (race_id={race_id})")
