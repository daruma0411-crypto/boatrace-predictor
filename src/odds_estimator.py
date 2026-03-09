"""統計的オッズ推定モジュール

Phase 1: 過去データの的中払戻金から、モデル確率→推定オッズの変換関数を構築
Phase 2: リアルタイムオッズ取得に差し替え可能なインターフェース

設計方針:
    ボートレースはパリミュチュエル方式（控除率25%）なので、
    理論オッズ = 0.75 / P(市場確率)
    だが市場確率とモデル確率には乖離がある（favorite-longshot bias等）。
    過去データから log(実オッズ) = f(log(1/P_model)) を回帰フィットして、
    モデル確率→推定市場オッズへの変換を学習する。
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

TAKEOUT_RATE = 0.25  # 3連単控除率


class OddsEstimatorBase:
    """オッズ推定の抽象基底クラス（Phase 2 差し替え用インターフェース）"""

    def estimate_odds(self, probability):
        """モデルの予測確率からオッズを推定

        Args:
            probability: float, 3連単の予測確率 (0 < p < 1)

        Returns:
            float: 推定オッズ（100円あたりの払戻）
        """
        raise NotImplementedError

    def estimate_odds_batch(self, sanrentan_probs):
        """全組み合わせの推定オッズを一括計算

        Args:
            sanrentan_probs: dict, {"1-2-3": 0.05, "1-3-2": 0.03, ...}

        Returns:
            dict: {"1-2-3": 15.0, "1-3-2": 25.0, ...}
        """
        return {
            combo: self.estimate_odds(prob)
            for combo, prob in sanrentan_probs.items()
            if prob > 0
        }


class TheoreticalOddsEstimator(OddsEstimatorBase):
    """理論オッズ推定（控除率のみ考慮、バイアス補正なし）

    odds = (1 - takeout) / P = 0.75 / P
    """

    def estimate_odds(self, probability):
        if probability <= 0:
            return 0.0
        return (1.0 - TAKEOUT_RATE) / probability


class CalibratedOddsEstimator(OddsEstimatorBase):
    """キャリブレーション済みオッズ推定

    過去データから log(実オッズ) vs log(1/P_model) の関係を
    線形回帰でフィットし、バイアスを補正する。

    favorite-longshot bias:
        - 本命（高確率）: 市場オッズが理論値より高い → 過小評価される
        - 穴（低確率）: 市場オッズが理論値より低い → 過大評価される
    """

    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0
        self.fitted = False

    def fit(self, model_probs, actual_payouts):
        """過去データからオッズ推定モデルをフィット

        Args:
            model_probs: array, モデルが予測した的中組み合わせの確率
            actual_payouts: array, 実際の払戻金（100円単位のオッズ）
        """
        # フィルタリング: 有効データのみ
        mask = (model_probs > 1e-8) & (actual_payouts > 0)
        probs = model_probs[mask]
        payouts = actual_payouts[mask]

        if len(probs) < 100:
            logger.warning(f"フィットデータ不足: {len(probs)}件")
            self.slope = 1.0
            self.intercept = np.log(1.0 - TAKEOUT_RATE)
            self.fitted = True
            return

        # log-log 回帰: log(payout/100) = a * log(1/P) + b
        x = np.log(1.0 / probs)
        y = np.log(payouts / 100.0)  # 100円単位 → 倍率

        # numpy polyfit (1次)
        coeffs = np.polyfit(x, y, 1)
        self.slope = coeffs[0]
        self.intercept = coeffs[1]
        self.fitted = True

        # フィット品質レポート
        y_pred = self.slope * x + self.intercept
        residuals = y - y_pred
        r_squared = 1 - np.var(residuals) / np.var(y)

        logger.info(f"オッズ推定フィット完了:")
        logger.info(f"  log(odds) = {self.slope:.3f} * log(1/P) + {self.intercept:.3f}")
        logger.info(f"  R² = {r_squared:.4f}")
        logger.info(f"  データ: {len(probs):,}件")

        # バイアス分析
        # 理論値: slope=1.0, intercept=log(0.75)=-0.288
        logger.info(f"  理論値: slope=1.000, intercept=-0.288")
        if self.slope < 0.95:
            logger.info(f"  → 穴目のオッズが理論値より低い（longshot bias）")
        elif self.slope > 1.05:
            logger.info(f"  → 穴目のオッズが理論値より高い（reverse bias）")

    def estimate_odds(self, probability):
        if probability <= 0:
            return 0.0

        log_inv_p = np.log(1.0 / probability)
        log_odds = self.slope * log_inv_p + self.intercept
        odds_multiplier = np.exp(log_odds)

        # 倍率 → 100円単位の払戻金
        return odds_multiplier * 100.0


class RealtimeOddsProvider(OddsEstimatorBase):
    """リアルタイムオッズ取得: boatrace.jp から3連単オッズをスクレイピング

    - 5分TTLキャッシュ（同一レースの重複リクエスト防止）
    - 空dictもキャッシュ（未発売の無限リトライ防止）
    - フォールバック: 理論オッズ 0.75/P
    """

    def __init__(self):
        self._cache = {}  # key: (date_str, venue, race) → (timestamp, odds_dict)
        self._cache_ttl = 300  # 5分
        self._session = None
        self._last_fetched_odds = {}

    def _get_session(self):
        if self._session is None:
            from src.scraper import _get_session
            self._session = _get_session()
        return self._session

    def fetch_odds(self, race_date, venue_id, race_number):
        """レースの3連単オッズを取得（5分TTLキャッシュ付き）

        Args:
            race_date: datetime.date
            venue_id: int (1-24)
            race_number: int (1-12)

        Returns:
            dict: {"1-2-3": 12.7, ...} 倍率形式。取得失敗時は空dict
        """
        import time
        from src.scraper import scrape_odds_3t

        cache_key = (str(race_date), venue_id, race_number)

        # キャッシュチェック
        if cache_key in self._cache:
            cached_time, cached_odds = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.debug(
                    f"オッズキャッシュヒット: 場{venue_id} R{race_number} "
                    f"({len(cached_odds)}通り)"
                )
                return cached_odds

        # スクレイピング
        odds = scrape_odds_3t(
            self._get_session(), race_date, venue_id, race_number
        )

        # 空dictもキャッシュ（未発売の無限リトライ防止）
        result = odds if odds else {}
        self._cache[cache_key] = (time.time(), result)
        self._last_fetched_odds = result

        return result

    def estimate_odds(self, probability):
        """フォールバック: 理論オッズ 0.75/P"""
        if probability <= 0:
            return 0.0
        return (1.0 - TAKEOUT_RATE) / probability

    def estimate_odds_batch(self, sanrentan_probs):
        """取得済みリアルオッズを返す。未取得の組み合わせは理論値フォールバック"""
        result = {}
        for combo, prob in sanrentan_probs.items():
            if prob <= 0:
                continue
            if combo in self._last_fetched_odds:
                result[combo] = self._last_fetched_odds[combo]
            else:
                result[combo] = self.estimate_odds(prob)
        return result
