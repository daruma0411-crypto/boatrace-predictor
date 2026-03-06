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
from src.result_collector import ResultCollector
from utils.timezone import now_jst, format_jst

logger = logging.getLogger(__name__)


class DynamicRaceScheduler:
    """動的レーススケジューラ: 1分間隔ポーリング"""

    def __init__(self, model_path='models/boatrace_model.pth'):
        self.client = PyJPBoatrace()
        self.collector = RealtimeDataCollector()
        self.predictor = RealtimePredictor(model_path)
        self.betting = KellyBettingStrategy()
        self.result_collector = ResultCollector()
        self.processed_races = set()
        self.settled_today = False

    def fetch_daily_schedule(self):
        """当日のレーススケジュールを取得しDBに保存

        pyjpboatrace API:
          get_stadiums(d) → 開催場一覧（HTMLパースが不安定なため非使用）
          get_12races(d, stadium) → {race_key: {vote_limit, status, racers}, ...}

        フォールバック方式: 全24場を get_12races で試行し、
        レスポンスがあった場を当日開催と判定。
        """
        today = now_jst().date()
        races = []
        from pyjpboatrace.const import NUM_STADIUMS

        for venue_id in range(1, NUM_STADIUMS + 1):
            try:
                race_data = self.client.get_12races(d=today, stadium=venue_id)
            except Exception:
                continue

            if not race_data or not isinstance(race_data, dict):
                continue

            found_races = False
            for race_key, race_info in race_data.items():
                if race_key in ('date', 'stadium'):
                    continue
                if not isinstance(race_info, dict):
                    continue

                # レース番号をパース (例: "1R" → 1)
                try:
                    race_number = int(str(race_key).replace('R', ''))
                except (ValueError, TypeError):
                    continue

                # 締切時刻をパース
                deadline_time = None
                vote_limit = race_info.get('vote_limit')
                if vote_limit:
                    try:
                        from datetime import datetime as dt
                        import pytz
                        jst = pytz.timezone('Asia/Tokyo')
                        deadline_time = dt.strptime(
                            vote_limit, '%Y-%m-%d %H:%M:%S'
                        )
                        deadline_time = jst.localize(deadline_time)
                    except (ValueError, TypeError):
                        pass

                entry = {'deadline_time': deadline_time}
                race_id = self._upsert_race(
                    today, venue_id, race_number, entry
                )
                if race_id:
                    races.append({
                        'race_id': race_id,
                        'venue_id': venue_id,
                        'race_number': race_number,
                        'deadline_time': deadline_time,
                    })
                    found_races = True

            if found_races:
                logger.info(f"場{venue_id}: レース取得成功")

        logger.info(f"本日のレース: {len(races)}件")
        return races

    def run_polling(self):
        """1分間隔ポーリング、締切10-2分前に予測実行"""
        logger.info("ポーリング開始")
        schedule = self.fetch_daily_schedule()

        while True:
            current = now_jst()
            logger.info(f"ポーリング: {format_jst(current)} (未処理: {len(schedule) - len(self.processed_races)}件)")

            if current.hour >= 23:
                # 23時以降: 結果収集 → 翌日待機
                if not self.settled_today:
                    logger.info("23時以降: レース結果収集開始")
                    try:
                        count = self.result_collector.settle_today()
                        logger.info(f"結果収集完了: {count}件精算")
                    except Exception as e:
                        logger.error(f"結果収集エラー: {e}", exc_info=True)
                    self.settled_today = True
                schedule = []

            if current.hour == 7 and current.minute == 0:
                self.settled_today = False
                self.processed_races = set()
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
