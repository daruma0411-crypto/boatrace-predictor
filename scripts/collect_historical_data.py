"""過去データ収集スクリプト: PyJPBoatraceで過去レースデータを取得"""
import sys
import os
import time
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pyjpboatrace import PyJPBoatrace
from src.database import get_db_connection, init_database
from utils.timezone import now_jst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VENUES = list(range(1, 25))


def collect_date_range(start_date, end_date):
    """指定期間のレースデータを収集"""
    client = PyJPBoatrace()
    current = start_date
    total_races = 0
    total_boats = 0

    while current <= end_date:
        logger.info(f"収集中: {current}")

        for venue_id in VENUES:
            for race_number in range(1, 13):
                try:
                    race_info = client.get_race_info(
                        d=current, stadium=venue_id, race=race_number
                    )
                    if not race_info:
                        continue

                    race_id = _save_race(current, venue_id, race_number)
                    if not race_id:
                        continue

                    _save_boats(race_id, race_info, venue_id, race_number,
                                client, current)

                    total_races += 1
                    total_boats += 6

                except Exception as e:
                    logger.debug(
                        f"場{venue_id} R{race_number}: {e}"
                    )
                    continue

                time.sleep(1)

        logger.info(f"{current}: {total_races}レース, {total_boats}艇")
        current += timedelta(days=1)

    logger.info(f"収集完了: {total_races}レース, {total_boats}艇")


def _save_race(race_date, venue_id, race_number):
    """レースをDB登録"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO races (race_date, venue_id, race_number, status)
                VALUES (%s, %s, %s, 'finished')
                ON CONFLICT (race_date, venue_id, race_number)
                DO UPDATE SET status = 'finished'
                RETURNING id
            """, (race_date, venue_id, race_number))
            return cur.fetchone()['id']
    except Exception as e:
        logger.error(f"レース保存エラー: {e}")
        return None


def _save_boats(race_id, race_info, venue_id, race_number, client, race_date):
    """艇データをDB登録"""
    entries = race_info if isinstance(race_info, list) else [race_info]

    for i, entry in enumerate(entries[:6]):
        if not isinstance(entry, dict):
            continue
        boat_number = i + 1

        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO boats
                    (race_id, boat_number, player_id, player_name,
                     player_class, win_rate, win_rate_2, win_rate_3,
                     local_win_rate, local_win_rate_2,
                     motor_win_rate_2, motor_win_rate_3,
                     boat_win_rate_2, weight, avg_st)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s)
                """, (
                    race_id, boat_number,
                    entry.get('player_id'),
                    entry.get('player_name'),
                    entry.get('player_class'),
                    entry.get('win_rate'),
                    entry.get('win_rate_2'),
                    entry.get('win_rate_3'),
                    entry.get('local_win_rate'),
                    entry.get('local_win_rate_2'),
                    entry.get('motor_win_rate_2'),
                    entry.get('motor_win_rate_3'),
                    entry.get('boat_win_rate_2'),
                    entry.get('weight'),
                    entry.get('avg_st'),
                ))
        except Exception as e:
            logger.error(f"艇データ保存エラー: {e}")


def main():
    """過去3年分のデータを収集"""
    init_database()

    end_date = now_jst().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 3)

    logger.info(f"=== 過去データ収集: {start_date} 〜 {end_date} ===")
    collect_date_range(start_date, end_date)


if __name__ == '__main__':
    main()
