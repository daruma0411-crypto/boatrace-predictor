"""過去データ一括収集パイプライン

選手情報・展示データ・着順・払戻金を一括収集し DB に格納する。
既に結果が入っているレースはスキップ（中断再開対応）。

使い方:
    python scripts/collect_past_data.py --start 2025-01-01 --end 2025-12-31
    python scripts/collect_past_data.py --start 2026-01-01 --end 2026-03-05 --venue 4
"""
import sys
import os
import time
import argparse
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
from pyjpboatrace import PyJPBoatrace
from pyjpboatrace.drivers import create_httpget_driver
from src.database import get_db_connection, init_database


def _create_ssl_tolerant_client():
    """SSL検証問題を回避する PyJPBoatrace クライアントを生成"""
    session = requests.Session()
    session.verify = False
    # SSL警告を抑制
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    driver = create_httpget_driver(http_get=session.get)
    return PyJPBoatrace(driver=driver)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

VENUES = list(range(1, 25))  # 1〜24場


def _save_race(cur, race_date, venue_id, race_number):
    """レースを INSERT (既存は UPDATE) → race_id を返す"""
    cur.execute("""
        INSERT INTO races (race_date, venue_id, race_number, status)
        VALUES (%s, %s, %s, 'finished')
        ON CONFLICT (race_date, venue_id, race_number)
        DO UPDATE SET status = 'finished'
        RETURNING id
    """, (race_date, venue_id, race_number))
    row = cur.fetchone()
    return row['id'] if row else None


def _save_boats(cur, race_id, race_info):
    """艇データを INSERT（既存は削除→再INSERT）"""
    # 同一 race_id の古いデータがあれば削除
    cur.execute("DELETE FROM boats WHERE race_id = %s", (race_id,))

    entries = race_info if isinstance(race_info, list) else [race_info]
    for i, entry in enumerate(entries[:6]):
        if not isinstance(entry, dict):
            continue
        boat_number = i + 1
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


def _update_exhibition(cur, race_id, just_before_info):
    """展示タイムデータで boats テーブルを UPDATE"""
    entries = just_before_info if isinstance(just_before_info, list) else [just_before_info]
    for i, entry in enumerate(entries[:6]):
        if not isinstance(entry, dict):
            continue
        boat_number = i + 1
        cur.execute("""
            UPDATE boats
            SET exhibition_time = %s,
                approach_course = %s
            WHERE race_id = %s AND boat_number = %s
        """, (
            entry.get('exhibition_time'),
            entry.get('approach_course', boat_number),
            race_id,
            boat_number,
        ))


def _parse_result(result_data):
    """レース結果から着順と3連単払戻金をパース

    Returns:
        (result_1st, result_2nd, result_3rd, payout_sanrentan) or None
    """
    if not result_data:
        return None

    # 着順パース
    ranking = result_data.get('result', [])
    if not ranking:
        return None

    # ranking は [{'boat': 1, 'rank': 1}, ...] or 着順リスト
    if isinstance(ranking, list):
        if isinstance(ranking[0], dict):
            sorted_ranking = sorted(ranking, key=lambda x: x.get('rank', 99))
            boats = [r.get('boat') for r in sorted_ranking]
        else:
            # 着順そのまま [1着艇, 2着艇, 3着艇, ...]
            boats = list(ranking)
    else:
        return None

    if len(boats) < 3:
        return None

    result_1st = boats[0]
    result_2nd = boats[1]
    result_3rd = boats[2]

    # 3連単払戻金
    payoff = result_data.get('payoff', {})
    trifecta = payoff.get('trifecta', {})
    payout_sanrentan = trifecta.get('payoff', 0)
    if isinstance(payout_sanrentan, str):
        payout_sanrentan = int(payout_sanrentan.replace(',', '').replace('円', ''))

    return (result_1st, result_2nd, result_3rd, payout_sanrentan)


def _save_result(cur, race_date, venue_id, race_number, result_1st, result_2nd, result_3rd, payout_sanrentan):
    """レース結果を races テーブルに UPDATE"""
    cur.execute("""
        UPDATE races
        SET result_1st = %s, result_2nd = %s, result_3rd = %s, payout_sanrentan = %s
        WHERE race_date = %s AND venue_id = %s AND race_number = %s
    """, (result_1st, result_2nd, result_3rd, payout_sanrentan,
          race_date, venue_id, race_number))


def _is_already_collected(cur, race_date, venue_id, race_number):
    """既に結果が入っているかチェック（中断再開用）"""
    cur.execute("""
        SELECT result_1st FROM races
        WHERE race_date = %s AND venue_id = %s AND race_number = %s
          AND result_1st IS NOT NULL
    """, (race_date, venue_id, race_number))
    return cur.fetchone() is not None


def collect_date_range(start_date, end_date, venue_filter=None):
    """指定期間のレースデータ + 結果を一括収集"""
    client = _create_ssl_tolerant_client()
    current = start_date
    total_races = 0
    total_skipped = 0
    total_errors = 0

    venues = [venue_filter] if venue_filter else VENUES

    while current <= end_date:
        day_races = 0
        day_venues = []

        for venue_id in venues:
            venue_races = 0

            # まず開催判定: get_12races で開催有無を確認
            try:
                schedule = client.get_12races(d=current, stadium=venue_id)
                if not schedule:
                    continue
            except Exception:
                continue

            for race_number in range(1, 13):
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()

                        # 中断再開: 既に結果があればスキップ
                        if _is_already_collected(cur, current, venue_id, race_number):
                            total_skipped += 1
                            continue

                        # races テーブルに INSERT/UPDATE
                        race_id = _save_race(cur, current, venue_id, race_number)
                        if not race_id:
                            continue

                        # 1. 選手情報取得（パーサーがHTML変更に追従できない場合はスキップ）
                        try:
                            race_info = client.get_race_info(
                                d=current, stadium=venue_id, race=race_number
                            )
                            if race_info:
                                _save_boats(cur, race_id, race_info)
                        except Exception as e:
                            logger.debug(f"選手情報取得スキップ: 場{venue_id} R{race_number}: {e}")

                        time.sleep(1)

                        # 2. 展示タイム取得（同上、パーサー互換性問題あり）
                        try:
                            just_before = client.get_just_before_info(
                                d=current, stadium=venue_id, race=race_number
                            )
                            if just_before:
                                _update_exhibition(cur, race_id, just_before)
                        except Exception as e:
                            logger.debug(f"展示データ取得スキップ: 場{venue_id} R{race_number}: {e}")

                        time.sleep(1)

                        # 3. レース結果取得（これが本命）
                        try:
                            result_data = client.get_race_result(
                                d=current, stadium=venue_id, race=race_number
                            )
                            parsed = _parse_result(result_data)
                            if parsed:
                                _save_result(cur, current, venue_id, race_number, *parsed)
                                venue_races += 1
                                total_races += 1
                            else:
                                logger.debug(f"結果パース失敗: 場{venue_id} R{race_number}")
                        except Exception as e:
                            logger.debug(f"結果取得失敗: 場{venue_id} R{race_number}: {e}")

                except Exception as e:
                    logger.warning(f"エラー: {current} 場{venue_id} R{race_number}: {e}")
                    total_errors += 1
                    continue

                time.sleep(1)

            if venue_races > 0:
                day_venues.append(f"場{venue_id}({venue_races}R)")

        day_races = len(day_venues)
        if day_venues:
            logger.info(f"{current}: {', '.join(day_venues)} 完了")
        else:
            logger.info(f"{current}: 開催なし or 全スキップ")

        current += timedelta(days=1)

    logger.info(
        f"=== 収集完了: {total_races}レース取得, "
        f"{total_skipped}件スキップ, {total_errors}件エラー ==="
    )


def main():
    parser = argparse.ArgumentParser(description='ボートレース過去データ一括収集')
    parser.add_argument('--start', required=True, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--venue', type=int, help='場ID (1-24) 指定時はその場のみ')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    if args.venue and not (1 <= args.venue <= 24):
        logger.error("場IDは1〜24で指定してください")
        sys.exit(1)

    logger.info(f"=== 過去データ収集: {start_date} 〜 {end_date} ===")
    if args.venue:
        logger.info(f"対象場: {args.venue}")

    init_database()
    collect_date_range(start_date, end_date, venue_filter=args.venue)


if __name__ == '__main__':
    main()
