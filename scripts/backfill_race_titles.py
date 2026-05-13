"""race_titles バックフィル (A3 of Phase A roadmap, Issue #4)

期間内の races に対して race title を順次 scrape し、race_titles に UPSERT する。

実行例:
  python scripts/backfill_race_titles.py --from 2026-02-01 --to 2026-04-30
  python scripts/backfill_race_titles.py --from 2026-04-30 --to 2026-04-30  # smoke
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.scraper import scrape_race_title, _get_session
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

SLEEP_SEC = 0.5


def upsert_title(conn, race_id, title):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO race_titles (race_id, title, scraped_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (race_id) DO UPDATE
        SET title = EXCLUDED.title, scraped_at = NOW()
    """, (race_id, title))


def process_date(session, conn, target_date):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, venue_id, race_number
        FROM races
        WHERE race_date = %s
        ORDER BY venue_id, race_number
    """, (target_date,))
    races = cur.fetchall()
    if not races:
        logger.info(f"  {target_date}: 0 races")
        return 0, 0
    success = 0
    for r in races:
        title = scrape_race_title(session, target_date, r['venue_id'], r['race_number'])
        if title is not None:
            upsert_title(conn, r['id'], title)
            success += 1
        else:
            upsert_title(conn, r['id'], None)
        time.sleep(SLEEP_SEC)
    conn.commit()
    logger.info(f"  {target_date}: {success}/{len(races)} 件取得成功")
    return success, len(races)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='from_', required=True, help='YYYY-MM-DD')
    parser.add_argument('--to', dest='to', required=True, help='YYYY-MM-DD')
    args = parser.parse_args()

    from_date = datetime.strptime(args.from_, '%Y-%m-%d').date()
    to_date = datetime.strptime(args.to, '%Y-%m-%d').date()
    if from_date > to_date:
        raise SystemExit("--from は --to より前である必要")

    session = _get_session()
    total_success, total = 0, 0
    d = from_date
    while d <= to_date:
        # 日ごとに接続を開閉して長時間 idle によるタイムアウトを回避
        with get_db_connection() as conn:
            s, t = process_date(session, conn, d)
        total_success += s
        total += t
        d += timedelta(days=1)
    logger.info(f"=== 完了: {total_success}/{total} 件取得成功 ===")


if __name__ == '__main__':
    main()
