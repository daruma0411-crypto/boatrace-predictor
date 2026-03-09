"""リアルタイムデータ収集（scraper.py ベース）

pyjpboatrace を廃止し、src/scraper.py の直接HTMLパースを使用。
"""
import time
import logging
from datetime import timedelta
from src.scraper import _get_session, scrape_racelist, scrape_beforeinfo
from utils.timezone import now_jst

logger = logging.getLogger(__name__)


class RealtimeDataCollector:
    """レース直前データの収集"""

    def __init__(self):
        self.session = _get_session()
        self._odds_provider = None

    def _get_odds_provider(self):
        """RealtimeOddsProvider を遅延初期化"""
        if self._odds_provider is None:
            from src.odds_estimator import RealtimeOddsProvider
            self._odds_provider = RealtimeOddsProvider()
        return self._odds_provider

    def get_exhibition_data(self, race_date, venue_id, race_number, deadline_time):
        """展示データを取得（最大3回リトライ）

        scraper.py の scrape_beforeinfo で展示タイム・進入コースを取得。
        取得失敗時は枠なりダミーを返す。
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = scrape_beforeinfo(self.session, race_date, venue_id, race_number)
                if data:
                    logger.info(
                        f"展示データ取得成功: 場{venue_id} R{race_number}"
                    )
                    return data
            except Exception as e:
                logger.warning(
                    f"展示データ取得失敗(試行{attempt+1}/{max_retries}): {e}"
                )
            time.sleep(5)

        # フォールバック: 枠なりダミーデータ
        logger.info(
            f"展示データフォールバック(枠なり): 場{venue_id} R{race_number}"
        )
        return self._generate_fallback_exhibition()

    def get_racelist_data(self, race_date, venue_id, race_number):
        """出走表データを取得"""
        try:
            boats = scrape_racelist(self.session, race_date, venue_id, race_number)
            if boats:
                logger.info(f"出走表取得成功: 場{venue_id} R{race_number}")
            return boats
        except Exception as e:
            logger.warning(f"出走表取得失敗: 場{venue_id} R{race_number}: {e}")
            return None

    def get_realtime_odds(self, race_date, venue_id, race_number, deadline_time):
        """リアルタイム3連単オッズを取得

        RealtimeOddsProvider 経由で boatrace.jp からスクレイピング。
        例外時は空dictを返す（ベッティング計算はオッズなしでスキップされる）。
        """
        try:
            provider = self._get_odds_provider()
            odds = provider.fetch_odds(race_date, venue_id, race_number)
            if odds:
                logger.info(
                    f"オッズ取得成功: 場{venue_id} R{race_number} "
                    f"({len(odds)}通り)"
                )
            else:
                logger.info(
                    f"オッズ未取得: 場{venue_id} R{race_number} "
                    f"(未発売 or 取得失敗)"
                )
            return odds
        except Exception as e:
            logger.warning(
                f"オッズ取得例外: 場{venue_id} R{race_number}: {e}"
            )
            return {}

    def _generate_fallback_exhibition(self):
        """展示データ取得失敗時のフォールバック（枠なり仮定）"""
        return [
            {
                'boat_number': i,
                'exhibition_time': None,
                'tilt': None,
                'approach_course': i,
                'fallback_flag': True,
                'weight': None,
            }
            for i in range(1, 7)
        ]
