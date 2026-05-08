"""既存 racelist.pkl を元に直前情報 (beforeinfo) を追加スクレイピング

READ-ONLY: production code には影響しない。

入力: analysis/historical_data/{year}_{month:02d}_v{XX-YY}/racelist.pkl
出力: analysis/historical_data/{year}_{month:02d}_v{XX-YY}/beforeinfo.pkl
  list[dict] {race_date, venue_id, race_number, weather, boats}

使い方:
  python analysis/scrape_beforeinfo_historical.py --year 2024 --month 4 --venue-start 1 --venue-end 3
"""
import os
import sys
import time
import pickle
import logging
import argparse
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scraper import _get_session, scrape_beforeinfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).parent / 'historical_data'
SLEEP_SEC = float(os.environ.get('SCRAPE_SLEEP_SEC', '0.3'))


def main(year: int, month: int, venue_start: int = 1, venue_end: int = 24):
    suffix = f'_v{venue_start:02d}-{venue_end:02d}' if (venue_start, venue_end) != (1, 24) else ''
    out_dir = OUT_ROOT / f'{year}_{month:02d}{suffix}'
    if not out_dir.exists():
        logger.error(f'shard ディレクトリが無い: {out_dir}')
        return

    racelist_path = out_dir / 'racelist.pkl'
    if not racelist_path.exists():
        logger.error(f'racelist.pkl が無い: {racelist_path}')
        return

    with open(racelist_path, 'rb') as f:
        racelist_records = pickle.load(f)

    logger.info(f'=== beforeinfo 追加スクレイピング開始 ===')
    logger.info(f'対象 shard: {out_dir.name} (racelist {len(racelist_records)} races)')
    logger.info(f'SLEEP_SEC={SLEEP_SEC}')

    beforeinfo_path = out_dir / 'beforeinfo.pkl'
    beforeinfo_records = _load(beforeinfo_path)
    logger.info(f'既存 beforeinfo: {len(beforeinfo_records)}')

    done = {(r['race_date'], r['venue_id'], r['race_number']) for r in beforeinfo_records}

    session = _get_session()
    total = len(racelist_records)
    n_new = 0
    n_fail = 0
    t_start = time.time()
    save_every = 60

    for i, r in enumerate(racelist_records):
        date_str = r['race_date']
        venue_id = r['venue_id']
        race_num = r['race_number']
        key = (date_str, venue_id, race_num)
        if key in done:
            continue

        race_date = date.fromisoformat(date_str)
        try:
            bi = scrape_beforeinfo(session, race_date, venue_id, race_num)
        except Exception as e:
            logger.warning(f'beforeinfo error {key}: {str(e)[:120]}')
            bi = None
        if bi:
            beforeinfo_records.append({
                'race_date': date_str,
                'venue_id': venue_id,
                'race_number': race_num,
                'weather': bi['weather'],
                'boats': bi['boats'],
            })
            n_new += 1
        else:
            n_fail += 1
        time.sleep(SLEEP_SEC)

        if (i + 1) % save_every == 0:
            _save(beforeinfo_path, beforeinfo_records)
            elapsed = time.time() - t_start
            eta = elapsed / (n_new + n_fail) * (total - i - 1) if (n_new + n_fail) else 0
            logger.info(
                f'進捗 {i+1}/{total} ({(i+1)/total*100:.1f}%) '
                f'new={n_new} fail={n_fail} 経過={int(elapsed/60)}分 残り={int(eta/60)}分'
            )

    _save(beforeinfo_path, beforeinfo_records)
    logger.info(f'=== 完了 === total={len(beforeinfo_records)} new={n_new} fail={n_fail}')
    logger.info(f'総時間: {int((time.time()-t_start)/60)}分')


def _load(path: Path):
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []


def _save(path: Path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--venue-start', type=int, default=1)
    parser.add_argument('--venue-end', type=int, default=24)
    args = parser.parse_args()
    main(args.year, args.month, args.venue_start, args.venue_end)
