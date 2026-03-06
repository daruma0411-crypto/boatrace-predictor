"""ケリー基準ベッティング戦略（最重要モジュール）

A/Bテスト対応:
- 戦略A (kelly_strict): 期待値1.20以上、ハーフ・ケリー、上位5点
- 戦略B (top_prob_fixed): 確率上位10点、期待値1.0以上、固定100円
"""
import json
import logging
import os
import numpy as np
from itertools import permutations
from src.database import get_db_connection, get_current_bankroll

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'config', 'betting_config.json'
)


def _load_config():
    """ベッティング設定を読み込み"""
    defaults = {
        'kelly_fraction': 0.5,
        'max_bet_ratio': 0.05,
        'min_expected_value': 1.20,
        'min_bet_amount': 100,
        'max_bet_amount': 10000,
        'max_kelly_bets': 5,
    }
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        defaults.update(config)
    except FileNotFoundError:
        pass
    return defaults


class KellyBettingStrategy:
    """ケリー基準 + A/Bテスト ベッティング戦略"""

    def __init__(self, initial_bankroll=100000):
        self.config = _load_config()
        self.initial_bankroll = initial_bankroll

    def calculate_all_strategies(self, probs_1st, probs_2nd, probs_3rd,
                                  odds_data, bankroll=None):
        """戦略A + 戦略Bの両方を計算し統合して返す"""
        if bankroll is None:
            profit = get_current_bankroll()
            bankroll = float(self.initial_bankroll + profit)

        sanrentan_probs = self._calculate_sanrentan_bets_conditional(
            probs_1st, probs_2nd, probs_3rd
        )

        bets_a = self._strategy_kelly_strict(
            sanrentan_probs, odds_data, bankroll
        )
        bets_b = self._strategy_top_prob_fixed(
            sanrentan_probs, odds_data
        )

        return {
            'kelly_strict': bets_a,
            'top_prob_fixed': bets_b,
        }

    def _strategy_kelly_strict(self, sanrentan_probs, odds_data, bankroll):
        """戦略A: 期待値1.20以上 × ハーフ・ケリー × 上位N点"""
        candidates = []
        kelly_frac = self.config['kelly_fraction']
        min_ev = self.config['min_expected_value']
        max_kelly_bets = self.config.get('max_kelly_bets', 5)
        max_bet = min(
            self.config['max_bet_amount'],
            bankroll * self.config['max_bet_ratio']
        )
        min_bet = self.config['min_bet_amount']

        for combo, prob in sanrentan_probs.items():
            odds = odds_data.get(combo, 0.0)
            if odds <= 1.0 or prob <= 0:
                continue

            ev = prob * odds
            if ev < min_ev:
                continue

            # ケリー基準: f = (b*p - q) / b  (b = odds-1)
            b = odds - 1.0
            if b <= 0:
                continue
            q = 1.0 - prob
            kelly = (b * prob - q) / b

            if kelly <= 0:
                continue

            # ハーフ・ケリー
            bet_amount = bankroll * kelly * kelly_frac
            bet_amount = max(min_bet, min(max_bet, bet_amount))
            bet_amount = int(round(bet_amount / 100) * 100)  # 100円単位

            if bet_amount >= min_bet:
                candidates.append({
                    'bet_type': 'sanrentan',
                    'combination': combo,
                    'amount': bet_amount,
                    'odds': odds,
                    'expected_value': ev,
                    'kelly_fraction': kelly * kelly_frac,
                    'probability': prob,
                    'strategy_type': 'kelly_strict',
                })

        # 期待値上位N点に絞る
        candidates.sort(key=lambda x: x['expected_value'], reverse=True)
        return candidates[:max_kelly_bets]

    def _strategy_top_prob_fixed(self, sanrentan_probs, odds_data):
        """戦略B: 確率上位10点、期待値1.0以上、固定100円"""
        bets = []
        sorted_combos = sorted(
            sanrentan_probs.items(), key=lambda x: x[1], reverse=True
        )

        for combo, prob in sorted_combos:
            if len(bets) >= 10:
                break

            odds = odds_data.get(combo, 0.0)
            if odds <= 1.0 or prob <= 0:
                continue

            ev = prob * odds
            if ev < 1.0:
                continue

            bets.append({
                'bet_type': 'sanrentan',
                'combination': combo,
                'amount': 100,
                'odds': odds,
                'expected_value': ev,
                'kelly_fraction': 0.0,
                'probability': prob,
                'strategy_type': 'top_prob_fixed',
            })

        return bets

    def _calculate_sanrentan_bets_conditional(self, probs_1st, probs_2nd,
                                               probs_3rd):
        """条件付き確率による3連単確率計算

        P(i,j,k) = P(1st=i) × P(2nd=j|1st=i) × P(3rd=k|1st=i,2nd=j)
        同一艇の組み合わせは除外。ゼロ除算ガード付き。
        """
        sanrentan = {}

        for combo in permutations(range(6), 3):
            i, j, k = combo

            # 同一艇チェック（冗長だがpermutationsなので不要、明示的にガード）
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
