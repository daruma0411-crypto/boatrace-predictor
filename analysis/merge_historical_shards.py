"""8 shard の pkl を統合して 1 ディレクトリに

入力: analysis/historical_data/{year}_{month:02d}_v{XX-YY}/ 8 ディレクトリ
出力: analysis/historical_data/{year}_{month:02d}/ 1 ディレクトリ

使い方:
  python analysis/merge_historical_shards.py --year 2024 --month 4
  python analysis/merge_historical_shards.py --year 2025 --month 4
"""
import argparse
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).parent / 'historical_data'


def main(year: int, month: int):
    base = OUT_ROOT
    pattern = f'{year}_{month:02d}_v*'
    shards = sorted(base.glob(pattern))
    if not shards:
        logger.error(f'shard が見つからない: {pattern}')
        return

    out_dir = base / f'{year}_{month:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'統合先: {out_dir}')
    logger.info(f'shard 数: {len(shards)}')

    files = ['racelist.pkl', 'result.pkl', 'odds_3t.pkl', 'beforeinfo.pkl', 'skipped.pkl']
    summary = {}

    for fname in files:
        merged = []
        seen = set()
        dup_count = 0
        for s in shards:
            p = s / fname
            if not p.exists():
                logger.debug(f'欠落: {p}')
                continue
            with open(p, 'rb') as f:
                records = pickle.load(f)
            for r in records:
                if fname == 'skipped.pkl':
                    merged.append(r)
                    continue
                key = (r['race_date'], r['venue_id'], r['race_number'])
                if key in seen:
                    dup_count += 1
                    continue
                seen.add(key)
                merged.append(r)

        with open(out_dir / fname, 'wb') as f:
            pickle.dump(merged, f)
        summary[fname] = (len(merged), dup_count)
        logger.info(f'  {fname}: {len(merged)} records (重複除外 {dup_count})')

    logger.info(f'=== 統合完了 ===')
    for f, (n, d) in summary.items():
        logger.info(f'{f}: {n} ({d} dup)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()
    main(args.year, args.month)
