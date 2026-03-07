"""既存レースのboatsテーブルを公式サイトからバックフィル

DBに結果が入っているがboatsが空のレースに対して、
公式サイトから選手データをスクレイピングして埋める。
"""
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_db_connection
from src.scraper import _get_session, scrape_racelist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def backfill():
    session = _get_session()

    # boatsが空のレースを取得
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.race_date, r.venue_id, r.race_number
            FROM races r
            WHERE r.result_1st IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM boats b WHERE b.race_id = r.id)
            ORDER BY r.race_date, r.venue_id, r.race_number
        """)
        races = cur.fetchall()

    total = len(races)
    logger.info(f"バックフィル対象: {total}レース")

    success = 0
    fail = 0
    last_date = None

    for i, race in enumerate(races):
        race_id = race['id']
        race_date = race['race_date']
        venue_id = race['venue_id']
        race_number = race['race_number']

        if race_date != last_date:
            if last_date is not None:
                logger.info(
                    f"[{i}/{total}] {last_date} 完了 "
                    f"(累計: {success}成功, {fail}失敗)"
                )
            last_date = race_date

        boats = scrape_racelist(session, race_date, venue_id, race_number)
        if not boats or len(boats) != 6:
            fail += 1
            continue

        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                # 既存データ削除
                cur.execute("DELETE FROM boats WHERE race_id = %s", (race_id,))

                for b in boats:
                    cur.execute("""
                        INSERT INTO boats
                        (race_id, boat_number, player_id, player_name,
                         player_class, win_rate, win_rate_2, win_rate_3,
                         local_win_rate, local_win_rate_2,
                         motor_win_rate_2, motor_win_rate_3,
                         boat_win_rate_2, weight, avg_st,
                         approach_course)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        race_id, b['boat_number'],
                        b.get('player_id'), b.get('player_name'),
                        b.get('player_class'),
                        b.get('win_rate'), b.get('win_rate_2'), b.get('win_rate_3'),
                        b.get('local_win_rate'), b.get('local_win_rate_2'),
                        b.get('motor_win_rate_2'), b.get('motor_win_rate_3'),
                        b.get('boat_win_rate_2'),
                        b.get('weight'), b.get('avg_st'),
                        b['boat_number'],  # default: 枠なり
                    ))
                success += 1
        except Exception as e:
            logger.warning(f"DB書込エラー race_id={race_id}: {e}")
            fail += 1

        # 1秒ウェイト（サーバー負荷軽減）
        time.sleep(1)

    logger.info(
        f"=== バックフィル完了: {success}/{total}成功, {fail}失敗 ==="
    )


if __name__ == '__main__':
    backfill()
