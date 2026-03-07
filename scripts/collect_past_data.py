"""過去データ一括収集パイプライン (scraper.py ベース)

公式サイトHTMLを直接パースして選手情報・結果を収集しDBに格納。
pyjpboatrace は使わない（HTML構造変更で壊れているため）。

使い方:
    python scripts/collect_past_data.py --start 2025-01-01 --end 2025-12-31
    python scripts/collect_past_data.py --start 2026-01-01 --end 2026-03-05 --venue 4

高速版（並列）は collect_parallel.py を使用。
"""
import sys
import os
import time
import argparse
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_db_connection, init_database
from src.scraper import _get_session, scrape_racelist, scrape_result

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

VENUES = list(range(1, 25))


def _is_already_collected(cur, race_date, venue_id, race_number):
    """既に結果+boats が入っているかチェック（中断再開用）"""
    cur.execute("""
        SELECT r.id FROM races r
        WHERE r.race_date = %s AND r.venue_id = %s AND r.race_number = %s
          AND r.result_1st IS NOT NULL
          AND EXISTS (SELECT 1 FROM boats b WHERE b.race_id = r.id)
    """, (race_date, venue_id, race_number))
    return cur.fetchone() is not None


def collect_date_range(start_date, end_date, venue_filter=None, delay=1.0):
    """指定期間のレースデータ + 結果を一括収集"""
    session = _get_session()
    current = start_date
    total_races = 0
    total_skipped = 0
    total_errors = 0

    venues = [venue_filter] if venue_filter else VENUES

    while current <= end_date:
        day_venues = []

        for venue_id in venues:
            venue_races = 0

            # 開催判定: R1の出走表が取れるか
            test = scrape_racelist(session, current, venue_id, 1)
            if not test:
                continue
            time.sleep(delay)

            for race_number in range(1, 13):
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()

                        if _is_already_collected(cur, current, venue_id, race_number):
                            total_skipped += 1
                            continue

                    # 出走表
                    boats = scrape_racelist(session, current, venue_id, race_number)
                    if not boats or len(boats) != 6:
                        total_errors += 1
                        time.sleep(delay)
                        continue
                    time.sleep(delay)

                    # 結果
                    result = scrape_result(session, current, venue_id, race_number)
                    if not result:
                        total_errors += 1
                        time.sleep(delay)
                        continue

                    # DB格納
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            INSERT INTO races (race_date, venue_id, race_number, status,
                                               result_1st, result_2nd, result_3rd, payout_sanrentan)
                            VALUES (%s, %s, %s, 'finished', %s, %s, %s, %s)
                            ON CONFLICT (race_date, venue_id, race_number)
                            DO UPDATE SET status = 'finished',
                                result_1st = EXCLUDED.result_1st,
                                result_2nd = EXCLUDED.result_2nd,
                                result_3rd = EXCLUDED.result_3rd,
                                payout_sanrentan = EXCLUDED.payout_sanrentan
                            RETURNING id
                        """, (current, venue_id, race_number,
                              result['result_1st'], result['result_2nd'],
                              result['result_3rd'], result['payout_sanrentan']))
                        row = cur.fetchone()
                        if not row:
                            continue
                        race_id = row['id']

                        cur.execute("DELETE FROM boats WHERE race_id = %s", (race_id,))
                        for b in boats:
                            cur.execute("""
                                INSERT INTO boats
                                (race_id, boat_number, player_id, player_name,
                                 player_class, win_rate, win_rate_2, win_rate_3,
                                 local_win_rate, local_win_rate_2,
                                 motor_win_rate_2, motor_win_rate_3,
                                 boat_win_rate_2, weight, avg_st, approach_course)
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
                                b['boat_number'],
                            ))

                        venue_races += 1
                        total_races += 1

                except Exception as e:
                    logger.warning(f"error: {current} v{venue_id} R{race_number}: {e}")
                    total_errors += 1

                time.sleep(delay)

            if venue_races > 0:
                day_venues.append(f"v{venue_id}({venue_races}R)")

        if day_venues:
            logger.info(f"{current}: {', '.join(day_venues)} ({total_races}累計)")
        else:
            logger.info(f"{current}: no races or all skipped")

        current += timedelta(days=1)

    logger.info(
        f"=== done: {total_races} collected, "
        f"{total_skipped} skipped, {total_errors} errors ==="
    )


def main():
    parser = argparse.ArgumentParser(description='ボートレース過去データ一括収集')
    parser.add_argument('--start', required=True, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--venue', type=int, help='場ID (1-24)')
    parser.add_argument('--delay', type=float, default=1.0, help='リクエスト間隔(秒)')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    if args.venue and not (1 <= args.venue <= 24):
        logger.error("場IDは1〜24で指定してください")
        sys.exit(1)

    logger.info(f"=== collect: {start_date} ~ {end_date} ===")
    if args.venue:
        logger.info(f"venue filter: {args.venue}")

    init_database()
    collect_date_range(start_date, end_date, venue_filter=args.venue, delay=args.delay)


if __name__ == '__main__':
    main()
