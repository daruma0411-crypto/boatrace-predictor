"""動的レーススケジューラ"""
import time
import logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from pyjpboatrace import PyJPBoatrace
from src.database import get_db_connection
from src.collector import RealtimeDataCollector
from src.predictor import RealtimePredictor
from src.betting import KellyBettingStrategy
from utils.timezone import now_jst, format_jst

logger = logging.getLogger(__name__)


class DynamicRaceScheduler:
    """動的レーススケジューラ: 1分間隔ポーリング"""

    def __init__(self, model_path='models/boatrace_model.pth'):
        self.client = PyJPBoatrace()
        self.collector = RealtimeDataCollector()
        self.predictor = RealtimePredictor(model_path)
        self.betting = KellyBettingStrategy()
        self.processed_races = set()

    def fetch_daily_schedule(self):
        """当日のレーススケジュールを取得しDBに保存"""
        today = now_jst().date()
        try:
            schedule = self.client.get_race_schedule(d=today)
        except Exception as e:
            logger.error(f"スケジュール取得失敗: {e}")
            return []

        races = []
        if not schedule:
            return races

        for entry in schedule if isinstance(schedule, list) else [schedule]:
            if not isinstance(entry, dict):
                continue
            venue_id = entry.get('stadium') or entry.get('venue_id')
            race_number = entry.get('race') or entry.get('race_number')
            if not venue_id or not race_number:
                continue

            race_id = self._upsert_race(today, venue_id, race_number, entry)
            if race_id:
                races.append({
                    'race_id': race_id,
                    'venue_id': venue_id,
                    'race_number': race_number,
                    'deadline_time': entry.get('deadline_time'),
                })

        logger.info(f"本日のレース: {len(races)}件")
        return races

    def run_polling(self):
        """1分間隔ポーリング、締切10-2分前に予測実行"""
        logger.info("ポーリング開始")
        schedule = self.fetch_daily_schedule()

        while True:
            current = now_jst()
            logger.debug(f"ポーリング: {format_jst(current)}")

            if current.hour >= 21:
                logger.info("21時以降: 翌日待機")
                schedule = []

            if current.hour == 7 and current.minute == 0:
                schedule = self.fetch_daily_schedule()

            for race in schedule:
                race_key = (race['venue_id'], race['race_number'])
                if race_key in self.processed_races:
                    continue

                deadline = race.get('deadline_time')
                if not deadline:
                    continue

                if hasattr(deadline, 'tzinfo') and deadline.tzinfo is None:
                    from utils.timezone import to_jst
                    deadline = to_jst(deadline)

                minutes_before = (deadline - current).total_seconds() / 60

                if 2 <= minutes_before <= 10:
                    logger.info(
                        f"予測開始: 場{race['venue_id']} "
                        f"R{race['race_number']} "
                        f"(締切{minutes_before:.1f}分前)"
                    )
                    self.predict_and_bet_safe(race)
                    self.processed_races.add(race_key)

            time.sleep(60)

    def predict_and_bet_safe(self, race):
        """ThreadPoolExecutorで並列実行"""
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ex = executor.submit(
                    self.collector.get_exhibition_data,
                    now_jst().date(),
                    race['venue_id'],
                    race['race_number'],
                    race.get('deadline_time', now_jst() + timedelta(minutes=5)),
                )
                future_odds = executor.submit(
                    self.collector.get_realtime_odds,
                    now_jst().date(),
                    race['venue_id'],
                    race['race_number'],
                    race.get('deadline_time', now_jst() + timedelta(minutes=5)),
                )

                exhibition_data = future_ex.result(timeout=300)
                odds_data = future_odds.result(timeout=300)

            if not exhibition_data:
                logger.warning(
                    f"展示データなし: 場{race['venue_id']} R{race['race_number']}"
                )
                return

            race_data, boats_data = self.predictor._get_pre_race_data(
                race['race_id']
            )
            if not race_data:
                return

            if exhibition_data:
                for ex_boat in exhibition_data:
                    for db_boat in boats_data:
                        if db_boat['boat_number'] == ex_boat['boat_number']:
                            db_boat['exhibition_time'] = ex_boat.get(
                                'exhibition_time'
                            )
                            db_boat['approach_course'] = ex_boat.get(
                                'approach_course'
                            )
                            db_boat['fallback_flag'] = ex_boat.get(
                                'fallback_flag', False
                            )
                            break

            prediction = self.predictor.predict(race_data, boats_data)

            odds_dict = self._parse_odds(odds_data) if odds_data else {}

            all_bets = self.betting.calculate_all_strategies(
                prediction['probs_1st'],
                prediction['probs_2nd'],
                prediction['probs_3rd'],
                odds_dict,
            )

            for strategy_type, bets in all_bets.items():
                if bets:
                    pred_id = self.predictor.save_prediction(
                        race['race_id'], prediction,
                        recommended_bets=bets,
                        strategy_type=strategy_type,
                    )
                    self.betting.save_bets(bets, pred_id, race['race_id'])

            total_bets = sum(len(b) for b in all_bets.values())
            logger.info(
                f"予測完了: 場{race['venue_id']} R{race['race_number']} "
                f"({total_bets}件)"
            )

        except Exception as e:
            logger.error(
                f"予測エラー: 場{race['venue_id']} R{race['race_number']}: {e}",
                exc_info=True,
            )

    def _upsert_race(self, race_date, venue_id, race_number, entry):
        """レースをDB登録/取得"""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO races (race_date, venue_id, race_number,
                                       deadline_time)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (race_date, venue_id, race_number)
                    DO UPDATE SET deadline_time = EXCLUDED.deadline_time
                    RETURNING id
                """, (
                    race_date, venue_id, race_number,
                    entry.get('deadline_time'),
                ))
                return cur.fetchone()['id']
        except Exception as e:
            logger.error(f"レースDB登録失敗: {e}")
            return None

    def _parse_odds(self, odds_data):
        """オッズデータを {組み合わせ: 倍率} 辞書に変換"""
        if isinstance(odds_data, dict):
            return odds_data
        result = {}
        if isinstance(odds_data, list):
            for entry in odds_data:
                if isinstance(entry, dict):
                    combo = entry.get('combination', '')
                    odds_val = entry.get('odds', 0.0)
                    if combo and odds_val:
                        result[combo] = float(odds_val)
        return result
