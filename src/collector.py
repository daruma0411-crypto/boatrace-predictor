"""リアルタイムデータ収集"""
import time
import logging
from datetime import timedelta
from pyjpboatrace import PyJPBoatrace
from utils.timezone import now_jst

logger = logging.getLogger(__name__)


class RealtimeDataCollector:
    """レース直前データの収集"""

    def __init__(self):
        self.client = PyJPBoatrace()

    def get_exhibition_data(self, race_date, venue_id, race_number, deadline_time):
        """展示データを取得（締切2分前まで20秒間隔リトライ）

        進入コース欠損時は枠なり仮定。
        """
        cutoff = deadline_time - timedelta(minutes=2)
        data = None

        while now_jst() < cutoff:
            try:
                data = self.client.get_just_before_info(
                    d=race_date, stadium=venue_id, race=race_number
                )
                if data:
                    logger.info(
                        f"展示データ取得成功: 場{venue_id} R{race_number}"
                    )
                    break
            except Exception as e:
                logger.warning(f"展示データ取得リトライ: {e}")

            time.sleep(20)

        if not data:
            logger.warning(
                f"展示データ取得失敗: 場{venue_id} R{race_number}"
            )
            return None

        boats = self._parse_exhibition_data(data)
        return boats

    def get_realtime_odds(self, race_date, venue_id, race_number, deadline_time):
        """リアルタイムオッズを取得（締切30秒前まで15秒間隔リトライ）"""
        cutoff = deadline_time - timedelta(seconds=30)
        odds_data = None

        while now_jst() < cutoff:
            try:
                odds_data = self.client.get_odds(
                    d=race_date, stadium=venue_id, race=race_number
                )
                if odds_data:
                    logger.info(
                        f"オッズ取得成功: 場{venue_id} R{race_number}"
                    )
                    break
            except Exception as e:
                logger.warning(f"オッズ取得リトライ: {e}")

            time.sleep(15)

        if not odds_data:
            logger.warning(
                f"オッズ取得失敗: 場{venue_id} R{race_number}"
            )

        return odds_data

    def _parse_exhibition_data(self, data):
        """展示データをパースし、進入コース欠損時は枠なり仮定"""
        boats = []
        for i, entry in enumerate(data if isinstance(data, list) else [data]):
            boat_number = i + 1
            boat_info = entry if isinstance(entry, dict) else {}

            approach_course = boat_info.get('approach_course')
            fallback_flag = False
            if approach_course is None:
                approach_course = boat_number
                fallback_flag = True

            boats.append({
                'boat_number': boat_number,
                'exhibition_time': boat_info.get('exhibition_time'),
                'tilt': boat_info.get('tilt'),
                'approach_course': approach_course,
                'fallback_flag': fallback_flag,
                'weight': boat_info.get('weight'),
            })

        return boats
