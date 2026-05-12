"""race_titles 増分 scrape (A3 of Phase A roadmap, Issue #4)

当日の races に対して race title を scrape し、race_titles に UPSERT する。
GCP daily cron で 09:00 JST (00:00 UTC) に実行する想定。

実行:
  python scripts/scrape_race_titles.py        # 当日 (CURRENT_DATE)
  python scripts/scrape_race_titles.py --date 2026-05-12  # 任意日付
"""
import os
import sys
import logging
import argparse
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.scraper import _get_session
from src.database import get_db_connection
from scripts.backfill_race_titles import process_date

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYY-MM-DD (省略時は当日)')
    args = parser.parse_args()
    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else date.today()
    logger.info(f"対象日付: {target}")
    session = _get_session()
    with get_db_connection() as conn:
        success, total = process_date(session, conn, target)
    if total == 0:
        logger.info(f"races 0 件 ({target})、スクレイプ対象なし")
    else:
        logger.info(f"=== {target}: {success}/{total} 件取得成功 ===")


if __name__ == '__main__':
    main()
