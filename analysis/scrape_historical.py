"""過去年データを公式サイトからスクレイピングして pkl 保存

READ-ONLY: production code (src/, streamlit_app/) には影響しない。
DB にも書き込まない。

出力:
  analysis/historical_data/{year}_{month:02d}/
    racelist.pkl    : list[dict] 6艇×n_races (出走表)
    result.pkl      : list[dict] n_races (着順+払戻)
    odds_3t.pkl     : list[dict] n_races (3連単オッズ)
    skipped.pkl     : list[tuple] 取得失敗ログ

使い方:
  python analysis/scrape_historical.py --year 2024 --month 4
  python analysis/scrape_historical.py --year 2024 --month 4 --max-days 1  # テスト
  python analysis/scrape_historical.py --year 2024 --month 4 --start-day 15  # 途中再開

実行環境: ローカル PC (Windows でも動く)、フォアグラウンド実行
所要時間: 1日分 約15分、1ヶ月分 約7-8時間
"""
import os
import sys
import time
import pickle
import logging
import argparse
import calendar
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scraper import _get_session, scrape_racelist, scrape_result, scrape_odds_3t

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).parent / 'historical_data'
SLEEP_SEC = float(os.environ.get('SCRAPE_SLEEP_SEC', '0.5'))  # 加速A: 2リクエスト/秒、8並列で 16 RPS 想定


def main(year: int, month: int, max_days: int | None = None,
         start_day: int = 1,
         venue_start: int = 1, venue_end: int = 24):
    suffix = f'_v{venue_start:02d}-{venue_end:02d}' if (venue_start, venue_end) != (1, 24) else ''
    out_dir = OUT_ROOT / f'{year}_{month:02d}{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 月の日数（うるう年対応）
    days_in_month = calendar.monthrange(year, month)[1]
    end_day = min(days_in_month, start_day + max_days - 1) if max_days else days_in_month

    logger.info(f'=== 過去データスクレイピング開始 ===')
    logger.info(f'対象: {year}年{month}月 day {start_day}-{end_day} venue {venue_start}-{venue_end}')
    logger.info(f'出力先: {out_dir}')

    # 既存ファイルがあれば読み込んで append（途中再開対応）
    racelist_records = _load(out_dir / 'racelist.pkl')
    result_records = _load(out_dir / 'result.pkl')
    odds_records = _load(out_dir / 'odds_3t.pkl')
    skipped = _load(out_dir / 'skipped.pkl')

    logger.info(
        f'既存レコード: racelist={len(racelist_records)} '
        f'result={len(result_records)} odds={len(odds_records)} '
        f'skipped={len(skipped)}'
    )

    session = _get_session()

    # 既に取得済みのキーセット (再実行時の重複回避)
    done_racelist = {
        (r['race_date'], r['venue_id'], r['race_number'])
        for r in racelist_records
    }
    done_result = {
        (r['race_date'], r['venue_id'], r['race_number'])
        for r in result_records
    }
    done_odds = {
        (r['race_date'], r['venue_id'], r['race_number'])
        for r in odds_records
    }

    n_venues = venue_end - venue_start + 1
    total_steps = (end_day - start_day + 1) * n_venues * 12 * 3
    step = 0
    t_start = time.time()

    for d in range(start_day, end_day + 1):
        race_date = date(year, month, d)
        date_str = race_date.isoformat()
        # 休場会場のキャッシュ: そのレース日に R1 で racelist が None なら全 R で skip
        closed_venues = set()

        for venue_id in range(venue_start, venue_end + 1):
            for race_num in range(1, 13):
                key = (date_str, venue_id, race_num)

                # その日その会場が休場確定なら R2 以降は丸ごとスキップ
                if venue_id in closed_venues:
                    step += 3
                    continue

                # racelist
                step += 1
                rl_ok = key in done_racelist
                if not rl_ok:
                    try:
                        rl = scrape_racelist(session, race_date, venue_id, race_num)
                        if rl:
                            racelist_records.append({
                                'race_date': date_str,
                                'venue_id': venue_id,
                                'race_number': race_num,
                                'boats': rl,
                            })
                            rl_ok = True
                    except Exception as e:
                        skipped.append((date_str, venue_id, race_num, 'racelist', str(e)[:100]))
                    time.sleep(SLEEP_SEC)

                # racelist が取れない = 休場の可能性。R1 なら以降全 R をスキップ
                if not rl_ok:
                    if race_num == 1:
                        closed_venues.add(venue_id)
                    step += 2
                    continue

                # result
                step += 1
                if key not in done_result:
                    try:
                        res = scrape_result(session, race_date, venue_id, race_num)
                        if res:
                            result_records.append({
                                'race_date': date_str,
                                'venue_id': venue_id,
                                'race_number': race_num,
                                **res,
                            })
                    except Exception as e:
                        skipped.append((date_str, venue_id, race_num, 'result', str(e)[:100]))
                    time.sleep(SLEEP_SEC)

                # odds_3t
                step += 1
                if key not in done_odds:
                    try:
                        odds = scrape_odds_3t(session, race_date, venue_id, race_num)
                        if odds:
                            odds_records.append({
                                'race_date': date_str,
                                'venue_id': venue_id,
                                'race_number': race_num,
                                'odds': odds,
                            })
                    except Exception as e:
                        skipped.append((date_str, venue_id, race_num, 'odds_3t', str(e)[:100]))
                    time.sleep(SLEEP_SEC)

            # venue 単位で進捗表示
            elapsed = time.time() - t_start
            pct = step / total_steps * 100
            eta_sec = elapsed / step * (total_steps - step) if step else 0
            logger.info(
                f'進捗 {step}/{total_steps} ({pct:.1f}%) '
                f'date={date_str} venue={venue_id:02d} '
                f'racelist={len(racelist_records)} result={len(result_records)} odds={len(odds_records)} '
                f'skipped={len(skipped)} '
                f'経過={int(elapsed/60)}分 残り={int(eta_sec/60)}分'
            )

        # 1日終わるごとに中間保存
        _save(out_dir / 'racelist.pkl', racelist_records)
        _save(out_dir / 'result.pkl', result_records)
        _save(out_dir / 'odds_3t.pkl', odds_records)
        _save(out_dir / 'skipped.pkl', skipped)
        logger.info(f'★ {date_str} 中間保存完了')

    elapsed_total = time.time() - t_start
    logger.info(f'=== スクレイピング完了 ===')
    logger.info(f'racelist: {len(racelist_records)} records')
    logger.info(f'result: {len(result_records)} records')
    logger.info(f'odds_3t: {len(odds_records)} records')
    logger.info(f'skipped: {len(skipped)} entries')
    logger.info(f'総時間: {int(elapsed_total/60)}分 ({elapsed_total/3600:.1f}時間)')


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
    parser.add_argument('--max-days', type=int, default=None,
                        help='最大日数（テスト用、例: 1 で 1日分のみ）')
    parser.add_argument('--start-day', type=int, default=1,
                        help='開始日（再開用）')
    parser.add_argument('--venue-start', type=int, default=1,
                        help='開始 venue_id（並列実行用、1-24）')
    parser.add_argument('--venue-end', type=int, default=24,
                        help='終了 venue_id（並列実行用、1-24）')
    args = parser.parse_args()
    main(args.year, args.month, args.max_days, args.start_day,
         args.venue_start, args.venue_end)
