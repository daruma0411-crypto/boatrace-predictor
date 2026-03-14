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


DAILY_LOSS_LIMIT = 50000  # 全戦略合計の1日最大損失額


def _get_today_total_loss():
    """本日の全戦略合計損失額をDBから取得"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COALESCE(SUM(amount), 0) as total_wagered,
                       COALESCE(SUM(payout), 0) as total_payout
                FROM bets
                WHERE created_at >= CURRENT_DATE
            """)
            row = cur.fetchone()
            wagered = float(row['total_wagered'])
            payout = float(row['total_payout'])
            return wagered - payout  # 正=損失, 負=利益
    except Exception as e:
        logger.warning(f"日次損失チェックDB障害: {e}")
        return 0  # DB障害時は継続（bankroll制限が別途効く）


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
        # 日次損失制限チェック
        today_loss = _get_today_total_loss()
        if today_loss >= DAILY_LOSS_LIMIT:
            logger.warning(
                f"日次損失上限到達: {today_loss:,.0f}円 >= {DAILY_LOSS_LIMIT:,}円 → 全戦略スキップ"
            )
            return {name: [] for name in self.config['strategies']}

        # 5-6号艇軸: ログのみ（max_oddsフィルタに委ねる）
        skip_56 = _should_skip_by_top_boat(probs_1st)
        if skip_56:
            top_boat = max(range(6), key=lambda i: probs_1st[i]) + 1
            logger.info(
                f"5-6号艇軸: 注意 "
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
        odds_discount_static = config.get('odds_discount_factor', 0.92)  # fallback
        use_dynamic_discount = config.get('use_dynamic_discount', True)
        base_max_odds = config.get('max_odds', 9999)
        filter_type = config.get('filter_type', 'none')
        min_divergence = config.get('min_divergence_ratio', 2.0)
        min_prob = config.get('min_probability', 0.0)
        min_odds = config.get('min_odds', 0.0)  # 最低オッズ (ガチガチ本命除外)

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
                       'high_odds': 0, 'low_ev': 0, 'divergence': 0,
                       'kelly_neg': 0}

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

            # TODO: [TEST MODE] 戦略比較のため一律100円ベット
            # テスト中は生オッズEVで判定、Kelly計算をスキップ
            # 戦略間ダービー終了後、このブロックを削除して下のKellyブロックを復活
            ev = prob * raw_odds  # 生オッズでEV算出（割引なし）
            if ev < min_ev:
                skip_counts['low_ev'] += 1
                continue
            bet_amount = 100
            kelly = 0.0  # テストモード: Kelly不使用
            # --- [TEST MODE END] ---

            # # ケリー基準（割引オッズで計算）: f = (b*p - q) / b
            # ev = prob * discounted_odds
            # if ev < min_ev:
            #     skip_counts['low_ev'] += 1
            #     continue
            # b = discounted_odds - 1.0
            # if b < 0.01:
            #     skip_counts['kelly_neg'] += 1
            #     continue
            # q = 1.0 - prob
            # kelly = (b * prob - q) / b
            # if kelly <= 0:
            #     skip_counts['kelly_neg'] += 1
            #     continue
            # kelly_amount = bankroll * kelly * kelly_frac
            # kelly_amount = max(min_bet, min(max_ticket, kelly_amount))
            # kelly_amount = int(round(kelly_amount / 100) * 100)
            # bet_amount = kelly_amount

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

        # デバッグ: フィルターサマリ
        logger.info(
            f"[{strategy_name}] スキップ内訳: "
            f"odds無し={skip_counts['no_odds']}, "
            f"低確率={skip_counts['low_prob']}, "
            f"低オッズ={skip_counts['low_odds']}, "
            f"高オッズ={skip_counts['high_odds']}, "
            f"低EV={skip_counts['low_ev']}, "
            f"乖離={skip_counts['divergence']}, "
            f"Kelly負={skip_counts['kelly_neg']}, "
            f"→ 候補={len(candidates)}件"
        )

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
