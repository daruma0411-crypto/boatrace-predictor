"""並列データ収集スクリプト

公式サイトから出走表+結果を並列スクレイピングでDB格納。
日付単位でスレッド分散し、高速に大量データを収集する。

使い方:
    python scripts/collect_parallel.py --months 3
    python scripts/collect_parallel.py --start 2025-12-01 --end 2026-03-06
    python scripts/collect_parallel.py --months 3 --workers 8
"""
import sys
import os
import time
import argparse
import logging
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_db_connection, init_database
from src.scraper import _get_session, scrape_racelist, scrape_result

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

VENUES = list(range(1, 25))


def _collect_one_race(session, race_date, venue_id, race_number):
    """1レース分の出走表+結果を取得してDB格納

    Returns:
        (success: bool, race_date, venue_id, race_number)
    """
    try:
        # 出走表（選手データ）
        boats = scrape_racelist(session, race_date, venue_id, race_number)
        if not boats or len(boats) != 6:
            return (False, race_date, venue_id, race_number)

        # 結果（着順+払戻金）
        result = scrape_result(session, race_date, venue_id, race_number)
        if not result:
            return (False, race_date, venue_id, race_number)

        # DB格納
        with get_db_connection() as conn:
            cur = conn.cursor()

            # races upsert
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
            """, (race_date, venue_id, race_number,
                  result['result_1st'], result['result_2nd'],
                  result['result_3rd'], result['payout_sanrentan']))
            row = cur.fetchone()
            if not row:
                return (False, race_date, venue_id, race_number)
            race_id = row['id']

            # boats 入れ替え
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

        return (True, race_date, venue_id, race_number)

    except Exception as e:
        logger.debug(f"error {race_date} v{venue_id} R{race_number}: {e}")
        return (False, race_date, venue_id, race_number)


def _collect_one_date(race_date, delay=0.3):
    """1日分の全会場・全レースを収集

    Returns:
        (date, success_count, fail_count)
    """
    session = _get_session()  # スレッドごとにセッション生成
    success = 0
    fail = 0

    for venue_id in VENUES:
        # R1が取れなければその会場は開催なし → スキップ
        test_boats = scrape_racelist(session, race_date, venue_id, 1)
        if not test_boats:
            continue
        time.sleep(delay)

        for race_number in range(1, 13):
            if race_number == 1:
                # R1は既にtest_boatsで取得済み → 結果だけ取得
                result = scrape_result(session, race_date, venue_id, 1)
                if result:
                    try:
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
                            """, (race_date, venue_id, 1,
                                  result['result_1st'], result['result_2nd'],
                                  result['result_3rd'], result['payout_sanrentan']))
                            row = cur.fetchone()
                            if row:
                                race_id = row['id']
                                cur.execute("DELETE FROM boats WHERE race_id = %s", (race_id,))
                                for b in test_boats:
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
                                success += 1
                    except Exception:
                        fail += 1
                else:
                    fail += 1
                time.sleep(delay)
                continue

            ok, _, _, _ = _collect_one_race(session, race_date, venue_id, race_number)
            if ok:
                success += 1
            else:
                fail += 1
            time.sleep(delay)

    return (race_date, success, fail)


def collect_range(start_date, end_date, workers=5, delay=0.3):
    """指定期間を並列収集"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    logger.info(f"収集期間: {start_date} ~ {end_date} ({len(dates)}日間)")
    logger.info(f"並列数: {workers}, ウェイト: {delay}s")

    total_success = 0
    total_fail = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_collect_one_date, d, delay): d
            for d in dates
        }

        for future in as_completed(futures):
            d = futures[future]
            try:
                race_date, success, fail = future.result()
                total_success += success
                total_fail += fail
                completed += 1

                if success > 0:
                    logger.info(
                        f"[{completed}/{len(dates)}] {race_date}: "
                        f"{success}成功 {fail}失敗 "
                        f"(累計: {total_success})"
                    )
            except Exception as e:
                completed += 1
                logger.warning(f"[{completed}/{len(dates)}] {d}: exception {e}")

    logger.info(
        f"=== 収集完了: {total_success}レース取得, "
        f"{total_fail}失敗 ({len(dates)}日間) ==="
    )
    return total_success


def main():
    parser = argparse.ArgumentParser(description='ボートレース並列データ収集')
    parser.add_argument('--start', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=3, help='直近N ヶ月 (--start/end未指定時)')
    parser.add_argument('--workers', type=int, default=5, help='並列スレッド数')
    parser.add_argument('--delay', type=float, default=0.3, help='リクエスト間隔(秒)')
    args = parser.parse_args()

    if args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=30 * args.months)

    init_database()
    collect_range(start_date, end_date, workers=args.workers, delay=args.delay)


if __name__ == '__main__':
    main()
