"""ケリー基準ベッティング戦略（最重要モジュール）

A/Bテスト → 6戦略並列テスト:
- 戦略A (conservative): 1/8ケリー、EV≥1.15(割引後)、最大3点、filter=none
- 戦略B (standard):     1/4ケリー、EV≥1.10(割引後)、最大5点、filter=none
- 戦略C (divergence):   市場乖離度フィルター、model_prob/market_prob≥2.0
- 戦略D (high_confidence): エントロピーフィルター、H<2.3の確信レースのみ
- 戦略E (ensemble):     4モデル合議、3/4多数決一致時のみ平均確率でKelly
- 戦略F (div_confidence): C+D合わせ技、乖離度≥1.5+エントロピーH<2.3両方パス

条件別最適化:
- 場の荒れ度でオッズ上限を動的調整
- レース番号で調整（R11-12は堅い、R2-4は荒れる）
- 5-6号艇軸の予測はゾーン外としてスキップ
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

# --- 発見1: 場の荒れ度分類 (1号艇勝率ベース) ---
# 堅い場 (1kaku > 60%): オッズ上限を締める
VENUE_STABLE = {18, 19, 24}  # 徳山, 下関, 大村
# 荒れる場 (1kaku < 48%): オッズ上限を広げる
VENUE_CHAOTIC = {2, 3, 4, 14}  # 戸田, 江戸川, 平和島, 鳴門

# --- 発見2: レース番号の荒れ度 ---
RACE_STABLE = {11, 12}     # R11-R12: 1kaku 67-69%
RACE_CHAOTIC = {2, 3, 4}   # R2-R4: 1kaku 45-48%


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

    return base_max_odds * factor


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


class KellyBettingStrategy:
    """ケリー基準 + 6戦略並列テスト ベッティング戦略"""

    def __init__(self, initial_bankroll=None):
        self.config = _load_config()
        self.initial_bankroll = (
            initial_bankroll or self.config['initial_bankroll']
        )

    def calculate_all_strategies(self, probs_1st, probs_2nd, probs_3rd,
                                  odds_data, bankroll=None,
                                  venue_id=None, race_number=None,
                                  ensemble_predictions=None):
        """全戦略を計算して返す

        既存A/B（filter_type=none）は従来パスを完全に通る。
        C-F戦略はフィルタ判定後にKelly計算。
        ensemble_predictions: EnsemblePredictor.predict_all()の結果（戦略E用）
        """
        # 発見3: 5-6号艇軸なら保守的戦略(A/B)のみスキップ、C-Fは継続
        skip_56 = _should_skip_by_top_boat(probs_1st)
        if skip_56:
            top_boat = max(range(6), key=lambda i: probs_1st[i]) + 1
            logger.info(
                f"5-6号艇軸: A/Bスキップ、C-F継続 "
                f"(モデル1着予測={top_boat}号艇, 場{venue_id} R{race_number})"
            )

        # 通常の3連単確率（A/B/C/D/F用）
        sanrentan_probs = self._calculate_sanrentan_bets_conditional(
            probs_1st, probs_2nd, probs_3rd
        )

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
            filter_type = strategy_config.get('filter_type', 'none')

            # 5-6号艇軸: A(conservative)/B(standard)のみスキップ
            if skip_56 and filter_type == 'none':
                results[strategy_name] = []
                continue

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

            # --- 使用する確率の決定 ---
            if filter_type == 'ensemble' and ensemble_sanrentan is not None:
                use_sanrentan = ensemble_sanrentan
            else:
                use_sanrentan = sanrentan_probs

            # 各戦略で独立bankroll
            if bankroll is not None:
                br = bankroll
            else:
                profit = get_current_bankroll(strategy_type=strategy_name)
                br = float(self.initial_bankroll + profit)

            bets = self._strategy_kelly(
                strategy_config, use_sanrentan, odds_data, br,
                strategy_name, venue_id, race_number,
                divergence_map=divergence_map,
            )
            results[strategy_name] = bets

        return results

    def _strategy_kelly(self, config, sanrentan_probs, odds_data,
                         bankroll, strategy_name,
                         venue_id=None, race_number=None,
                         divergence_map=None):
        """共通ケリー戦略: オッズ割引 → 割引EV判定 → Kelly計算 → 上限制限"""
        candidates = []
        kelly_frac = config['kelly_fraction']
        min_ev = config['min_expected_value']
        max_bets = config['max_recommended_bets']
        max_ticket = bankroll * config['max_ticket_bet_ratio']
        max_total = bankroll * config['max_total_bet_ratio']
        min_bet = config['min_bet_amount']
        odds_discount = config['odds_discount_factor']
        base_max_odds = config.get('max_odds', 9999)
        filter_type = config.get('filter_type', 'none')
        min_divergence = config.get('min_divergence_ratio', 2.0)

        # 発見1+2: 場・R番号でオッズ上限を動的調整
        if venue_id is not None and race_number is not None:
            max_odds = _adjust_max_odds(base_max_odds, venue_id, race_number)
        else:
            max_odds = base_max_odds

        for combo, prob in sanrentan_probs.items():
            raw_odds = odds_data.get(combo, 0.0)
            if raw_odds <= 1.0 or prob <= 0:
                continue

            # オッズ上限フィルター（動的調整済み）
            if raw_odds > max_odds:
                continue

            # --- 乖離度フィルター (C, F) ---
            if filter_type in ('divergence', 'div_confidence'):
                if divergence_map:
                    div_ratio = divergence_map.get(combo, 0.0)
                    if div_ratio < min_divergence:
                        continue

            # オッズ割引
            discounted_odds = raw_odds * odds_discount

            # 割引後EVで判定
            ev = prob * discounted_odds
            if ev < min_ev:
                continue

            # ケリー基準（割引オッズで計算）: f = (b*p - q) / b
            b = discounted_odds - 1.0
            if b <= 0:
                continue
            q = 1.0 - prob
            kelly = (b * prob - q) / b

            if kelly <= 0:
                continue

            # フラクショナル・ケリー
            bet_amount = bankroll * kelly * kelly_frac
            # 1点上限
            bet_amount = max(min_bet, min(max_ticket, bet_amount))
            bet_amount = int(round(bet_amount / 100) * 100)  # 100円単位

            if bet_amount >= min_bet:
                candidates.append({
                    'bet_type': 'sanrentan',
                    'combination': combo,
                    'amount': bet_amount,
                    'odds': raw_odds,
                    'discounted_odds': discounted_odds,
                    'expected_value': ev,
                    'kelly_fraction': kelly * kelly_frac,
                    'probability': prob,
                    'strategy_type': strategy_name,
                })

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

    def save_bets(self, bets, prediction_id, race_id):
        """ベットをDBに保存"""
        with get_db_connection() as conn:
            cur = conn.cursor()
            for bet in bets:
                cur.execute("""
                    INSERT INTO bets
                    (prediction_id, race_id, bet_type, combination,
                     amount, odds, expected_value, kelly_fraction,
                     strategy_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    prediction_id, race_id,
                    bet['bet_type'], bet['combination'],
                    bet['amount'], bet['odds'],
                    bet['expected_value'], bet['kelly_fraction'],
                    bet['strategy_type'],
                ))
            logger.info(f"ベット保存: {len(bets)}件 (race_id={race_id})")
