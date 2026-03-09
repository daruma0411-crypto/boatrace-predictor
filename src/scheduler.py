"""動的レーススケジューラ（scraper.py ベース）

pyjpboatrace を廃止し、公式サイト直接パースで当日スケジュール取得。
"""
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from src.scraper import _get_session, scrape_racelist, scrape_race_deadlines
from src.database import get_db_connection
from src.collector import RealtimeDataCollector
from src.predictor import RealtimePredictor
from src.betting import KellyBettingStrategy
from src.result_collector import ResultCollector
from utils.timezone import now_jst, format_jst

logger = logging.getLogger(__name__)

NUM_VENUES = 24

# 締切前リードタイム（分）
# 締切直前の確定オッズに近い値で計算するため2〜3分前に処理。
# 処理時間 ~20秒 + 購入バッファ ~60秒 → 最低1.5分必要。
LEAD_TIME_MIN = 2  # 最小リードタイム（これ未満はスキップ）
LEAD_TIME_MAX = 3  # 最大リードタイム（この範囲内で処理開始）


class DynamicRaceScheduler:
    """動的レーススケジューラ: 1分間隔ポーリング"""

    def __init__(self, model_path='models/boatrace_model.pth'):
        self.session = _get_session()
        self.collector = RealtimeDataCollector()
        self.predictor = RealtimePredictor(model_path)
        self.betting = KellyBettingStrategy()
        self.result_collector = ResultCollector()
        self.processed_races = set()
        self.settled_today = False

    def fetch_daily_schedule(self):
        """当日のレーススケジュールを取得しDBに保存

        全24場のR1出走表を試行し、レスポンスがあった場を当日開催と判定。
        開催場のR1〜R12を全てDB登録する。
        """
        today = now_jst().date()
        races = []

        for venue_id in range(1, NUM_VENUES + 1):
            try:
                test_boats = scrape_racelist(self.session, today, venue_id, 1)
                if not test_boats:
                    continue
            except Exception:
                continue

            time.sleep(0.5)

            # 締切時刻を一括取得
            deadlines = scrape_race_deadlines(self.session, today, venue_id)
            time.sleep(0.5)

            # この会場は開催中 → R1〜R12を登録
            for race_number in range(1, 13):
                # 締切時刻を datetime に変換
                deadline_dt = None
                dl_str = deadlines.get(race_number)
                if dl_str:
                    try:
                        h, m = dl_str.split(':')
                        deadline_dt = now_jst().replace(
                            hour=int(h), minute=int(m), second=0, microsecond=0,
                        )
                    except (ValueError, TypeError):
                        pass

                race_id = self._upsert_race(
                    today, venue_id, race_number,
                    {'deadline_time': deadline_dt},
                )
                if race_id:
                    races.append({
                        'race_id': race_id,
                        'venue_id': venue_id,
                        'race_number': race_number,
                        'deadline_time': deadline_dt,
                    })

            dl_summary = f"{deadlines.get(1, '?')}〜{deadlines.get(12, '?')}"
            logger.info(f"場{venue_id}: 12R登録完了 (締切: {dl_summary})")

        logger.info(f"本日のレース: {len(races)}件")
        return races

    def _catch_up_missed_races(self, schedule):
        """起動時キャッチアップ: 締切超過だが未処理のレースを遡って予測

        レース結果がまだ確定していないレースのみ処理する。
        """
        current = now_jst()
        missed = []
        for race in schedule:
            race_key = (race['venue_id'], race['race_number'])
            if race_key in self.processed_races:
                continue
            deadline = race.get('deadline_time')
            if deadline is None:
                continue
            minutes_left = (deadline - current).total_seconds() / 60
            if minutes_left < LEAD_TIME_MIN:
                missed.append(race)

        if not missed:
            logger.info("キャッチアップ: 見逃しレースなし")
            return

        # 結果確定済みのレースを除外
        settled_keys = set()
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT venue_id, race_number FROM races
                    WHERE race_date = %s AND status IN ('finished', 'settled')
                """, (current.date(),))
                for row in cur.fetchall():
                    settled_keys.add((row['venue_id'], row['race_number']))
        except Exception as e:
            logger.warning(f"キャッチアップ: 確定済みレース取得失敗: {e}")

        targets = [
            r for r in missed
            if (r['venue_id'], r['race_number']) not in settled_keys
        ]

        if not targets:
            logger.info("キャッチアップ: 全て結果確定済み、スキップ")
            for r in missed:
                self.processed_races.add((r['venue_id'], r['race_number']))
            return

        logger.info(f"キャッチアップ開始: {len(targets)}レースを遡って予測")
        for race in targets:
            race_key = (race['venue_id'], race['race_number'])
            logger.info(
                f"キャッチアップ予測: 場{race['venue_id']} "
                f"R{race['race_number']}"
            )
            self.predict_and_bet_safe(race)
            self.processed_races.add(race_key)

        # 結果確定済みも処理済みマーク
        for r in missed:
            self.processed_races.add((r['venue_id'], r['race_number']))

        logger.info(f"キャッチアップ完了: {len(targets)}レース予測保存")

    def run_polling(self):
        """1分間隔ポーリング、各レースを順番に予測実行

        起動時に見逃しレースをキャッチアップしてから通常ポーリングに移行。
        """
        logger.info("ポーリング開始")
        schedule = self.fetch_daily_schedule()

        # 起動時キャッチアップ: 締切超過の未処理レースを遡って予測
        self._catch_up_missed_races(schedule)

        while True:
            current = now_jst()
            remaining = len(schedule) - len(self.processed_races)
            logger.info(
                f"ポーリング: {format_jst(current)} "
                f"(未処理: {remaining}件)"
            )

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
                self._catch_up_missed_races(schedule)

            # 締切時間フィルター: 締切2〜3分前のレースを処理
            for race in schedule:
                race_key = (race['venue_id'], race['race_number'])
                if race_key in self.processed_races:
                    continue

                deadline = race.get('deadline_time')
                if deadline is None:
                    # 締切時刻不明 → 旧方式フォールバック
                    if 8 <= current.hour <= 17:
                        logger.info(
                            f"予測開始(締切不明): 場{race['venue_id']} "
                            f"R{race['race_number']}"
                        )
                        self.predict_and_bet_safe(race)
                        self.processed_races.add(race_key)
                        break
                    continue

                minutes_left = (deadline - current).total_seconds() / 60

                if minutes_left < LEAD_TIME_MIN:
                    # キャッチアップ済みなのでスキップのみ
                    self.processed_races.add(race_key)
                    continue

                if LEAD_TIME_MIN <= minutes_left <= LEAD_TIME_MAX:
                    logger.info(
                        f"予測開始: 場{race['venue_id']} "
                        f"R{race['race_number']} "
                        f"(締切まで{minutes_left:.0f}分)"
                    )
                    self.predict_and_bet_safe(race)
                    self.processed_races.add(race_key)
                    break  # 1レースずつ処理して次のポーリングへ

            time.sleep(60)

    def predict_and_bet_safe(self, race):
        """展示データ取得 → 予測 → ベット計算"""
        try:
            today = now_jst().date()
            deadline = race.get(
                'deadline_time',
                now_jst() + timedelta(minutes=5),
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ex = executor.submit(
                    self.collector.get_exhibition_data,
                    today,
                    race['venue_id'],
                    race['race_number'],
                    deadline,
                )
                future_odds = executor.submit(
                    self.collector.get_realtime_odds,
                    today,
                    race['venue_id'],
                    race['race_number'],
                    deadline,
                )

                exhibition_data = future_ex.result(timeout=300)
                odds_data = future_odds.result(timeout=300)

            if not exhibition_data or not exhibition_data.get('boats'):
                logger.warning(
                    f"直前情報なし: 場{race['venue_id']} R{race['race_number']}"
                )
                return

            # 直前情報から天候データを抽出
            beforeinfo_weather = exhibition_data.get('weather', {})
            beforeinfo_boats = exhibition_data.get('boats', [])

            race_data, boats_data = self.predictor._get_pre_race_data(
                race['race_id']
            )
            if not race_data:
                # DBにboatsがない場合、出走表をスクレイピングして登録
                boats_data = self._fetch_and_store_boats(
                    today, race['venue_id'], race['race_number'], race['race_id']
                )
                if not boats_data:
                    logger.warning(
                        f"出走表取得失敗: 場{race['venue_id']} R{race['race_number']}"
                    )
                    return
                race_data = {
                    'venue_id': race['venue_id'],
                    'month': today.month,
                    'distance': 1800,
                    'wind_speed': 0,
                    'wind_direction': 'calm',
                    'temperature': 20,
                }

            # 天候データをrace_dataにマージ
            if beforeinfo_weather:
                if beforeinfo_weather.get('wind_speed') is not None:
                    race_data['wind_speed'] = beforeinfo_weather['wind_speed']
                if beforeinfo_weather.get('wind_direction'):
                    race_data['wind_direction'] = beforeinfo_weather['wind_direction']
                if beforeinfo_weather.get('temperature') is not None:
                    race_data['temperature'] = beforeinfo_weather['temperature']
                race_data['wave_height'] = beforeinfo_weather.get('wave_height', 0)
                race_data['water_temperature'] = beforeinfo_weather.get(
                    'water_temperature', 20
                )

                # 天候データをDBに永続化（再学習用）
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE races
                            SET wind_speed = %s, wind_direction = %s,
                                temperature = %s, wave_height = %s,
                                water_temperature = %s
                            WHERE id = %s
                        """, (
                            race_data.get('wind_speed'),
                            race_data.get('wind_direction'),
                            race_data.get('temperature'),
                            race_data.get('wave_height'),
                            race_data.get('water_temperature'),
                            race['race_id'],
                        ))
                except Exception as e:
                    logger.warning(f"天候データDB保存失敗: {e}")

            # 直前情報(展示タイム・チルト・部品交換・進入コース)をマージ
            if beforeinfo_boats and boats_data:
                for ex_boat in beforeinfo_boats:
                    for db_boat in boats_data:
                        if db_boat['boat_number'] == ex_boat['boat_number']:
                            db_boat['exhibition_time'] = ex_boat.get(
                                'exhibition_time'
                            )
                            db_boat['approach_course'] = ex_boat.get(
                                'approach_course', db_boat['boat_number']
                            )
                            db_boat['tilt'] = ex_boat.get('tilt')
                            db_boat['parts_changed'] = ex_boat.get(
                                'parts_changed', False
                            )
                            db_boat['fallback_flag'] = False
                            if ex_boat.get('weight') is not None:
                                db_boat['weight'] = ex_boat['weight']
                            break

            # チルト・部品交換・展示タイムをDBに永続化（再学習用）
            if beforeinfo_boats and boats_data:
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        for db_boat in boats_data:
                            cur.execute("""
                                UPDATE boats
                                SET tilt = %s, parts_changed = %s,
                                    exhibition_time = %s, approach_course = %s
                                WHERE race_id = %s AND boat_number = %s
                            """, (
                                db_boat.get('tilt'),
                                db_boat.get('parts_changed', False),
                                db_boat.get('exhibition_time'),
                                db_boat.get('approach_course'),
                                race['race_id'],
                                db_boat['boat_number'],
                            ))
                except Exception as e:
                    logger.warning(f"艇データDB保存失敗: {e}")

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

    def _fetch_and_store_boats(self, race_date, venue_id, race_number, race_id):
        """出走表をスクレイピングしてDB格納、boats_dataとして返す"""
        boats = scrape_racelist(self.session, race_date, venue_id, race_number)
        if not boats or len(boats) != 6:
            return None

        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM boats WHERE race_id = %s", (race_id,))
                for b in boats:
                    cur.execute("""
                        INSERT INTO boats
                        (race_id, boat_number, player_id, player_name,
                         player_class, win_rate, win_rate_2, win_rate_3,
                         local_win_rate, local_win_rate_2,
                         motor_win_rate_2, motor_win_rate_3,
                         boat_win_rate_2, weight, avg_st, approach_course,
                         tilt, parts_changed)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        race_id, b['boat_number'],
                        b.get('player_id'), b.get('player_name'),
                        b.get('player_class'),
                        b.get('win_rate'), b.get('win_rate_2'), b.get('win_rate_3'),
                        b.get('local_win_rate'), b.get('local_win_rate_2'),
                        b.get('motor_win_rate_2'), b.get('motor_win_rate_3'),
                        b.get('boat_win_rate_2'),
                        b.get('weight'), b.get('avg_st'),
                        b['boat_number'],
                        b.get('tilt'), b.get('parts_changed', False),
                    ))
        except Exception as e:
            logger.error(f"出走表DB格納失敗: {e}")
            return None

        # predictor が使える形式に変換
        boats_data = []
        for b in boats:
            boats_data.append({
                'boat_number': b['boat_number'],
                'player_class': b.get('player_class'),
                'win_rate': b.get('win_rate'),
                'win_rate_2': b.get('win_rate_2'),
                'win_rate_3': b.get('win_rate_3'),
                'local_win_rate': b.get('local_win_rate'),
                'local_win_rate_2': b.get('local_win_rate_2'),
                'avg_st': b.get('avg_st'),
                'motor_win_rate_2': b.get('motor_win_rate_2'),
                'motor_win_rate_3': b.get('motor_win_rate_3'),
                'boat_win_rate_2': b.get('boat_win_rate_2'),
                'weight': b.get('weight'),
                'exhibition_time': None,
                'approach_course': b['boat_number'],
                'is_new_motor': False,
                'fallback_flag': False,
            })

        return boats_data

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
