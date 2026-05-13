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

from src.scraper import scrape_race_meta, _get_session
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

SLEEP_SEC = 0.2


def upsert_meta(conn, race_id, meta):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO race_titles (race_id, title, subtitle, day_label, scraped_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (race_id) DO UPDATE
        SET title = EXCLUDED.title,
            subtitle = EXCLUDED.subtitle,
            day_label = EXCLUDED.day_label,
            scraped_at = NOW()
    """, (race_id, meta.get('title'), meta.get('subtitle'), meta.get('day_label')))


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
        meta = scrape_race_meta(session, target_date, r['venue_id'], r['race_number'])
        if meta.get('title') is not None:
            success += 1
        upsert_meta(conn, r['id'], meta)
        time.sleep(SLEEP_SEC)
    conn.commit()
    logger.info(f"  {target_date}: {success}/{len(races)} 件 title 取得成功")
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
