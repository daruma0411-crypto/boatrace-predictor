"""6プロセス並列 気象・展示データ補完収集

既存レースの気象データ(wind_speed等)が欠損しているものを補完するため、
6つの日付チャンクを subprocess.Popen で並列起動し collect_parallel.py を呼ぶ。

ON CONFLICT DO UPDATE SET ... COALESCE(...) により既存データは保持しつつ欠損列のみ補完。

使い方:
    python scripts/collect_weather_chunks.py
    python scripts/collect_weather_chunks.py --workers 2 --delay 0.8
"""
import sys
import os
import subprocess
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# 6チャンク定義
CHUNKS = [
    ('2025-06-01', '2025-07-18'),   # Chunk 1: 48日
    ('2025-07-19', '2025-09-04'),   # Chunk 2: 48日
    ('2025-09-05', '2025-10-22'),   # Chunk 3: 48日
    ('2025-10-23', '2025-12-08'),   # Chunk 4: 47日
    ('2025-12-09', '2026-01-24'),   # Chunk 5: 47日
    ('2026-01-25', '2026-03-14'),   # Chunk 6: 49日
]


def show_weather_stats():
    """DB内の気象データ充足率を表示"""
    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM races WHERE status = 'finished'")
        total = cur.fetchone()['total']

        cur.execute("""
            SELECT COUNT(*) as cnt FROM races
            WHERE status = 'finished' AND wind_speed IS NOT NULL
        """)
        with_weather = cur.fetchone()['cnt']

        cur.execute("""
            SELECT COUNT(*) as cnt FROM boats
            WHERE exhibition_time IS NOT NULL AND exhibition_time > 0
        """)
        with_exhibition = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM boats")
        total_boats = cur.fetchone()['cnt']

        cur.execute("""
            SELECT COUNT(*) as cnt FROM boats
            WHERE tilt IS NOT NULL
        """)
        with_tilt = cur.fetchone()['cnt']

    weather_pct = (with_weather / total * 100) if total > 0 else 0
    exhibition_pct = (with_exhibition / total_boats * 100) if total_boats > 0 else 0
    tilt_pct = (with_tilt / total_boats * 100) if total_boats > 0 else 0

    logger.info("=" * 60)
    logger.info("DB 気象・展示データ充足率")
    logger.info("=" * 60)
    logger.info(f"  総レース数      : {total:,}")
    logger.info(f"  気象データあり   : {with_weather:,} ({weather_pct:.1f}%)")
    logger.info(f"  総ボート数      : {total_boats:,}")
    logger.info(f"  展示タイムあり  : {with_exhibition:,} ({exhibition_pct:.1f}%)")
    logger.info(f"  チルトあり      : {with_tilt:,} ({tilt_pct:.1f}%)")
    logger.info("=" * 60)

    return weather_pct


def main():
    import argparse
    parser = argparse.ArgumentParser(description='6プロセス並列 気象データ補完収集')
    parser.add_argument('--workers', type=int, default=1,
                        help='各チャンクの並列スレッド数 (default: 1)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='リクエスト間隔(秒) (default: 0.5)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("6プロセス並列 気象・展示データ補完収集")
    logger.info("=" * 60)

    # 収集前の統計
    logger.info("--- 収集前 ---")
    show_weather_stats()

    # 各チャンクを subprocess で並列起動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    collect_script = os.path.join(script_dir, 'collect_parallel.py')

    processes = []
    for i, (start, end) in enumerate(CHUNKS):
        cmd = [
            sys.executable, collect_script,
            '--start', start,
            '--end', end,
            '--workers', str(args.workers),
            '--delay', str(args.delay),
        ]
        logger.info(f"Chunk {i+1}: {start} ~ {end} 起動中...")
        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((i + 1, start, end, proc))
        # 起動間隔を少し空ける（DB接続の輻輳回避）
        time.sleep(2)

    logger.info(f"\n全{len(processes)}プロセス起動完了。完了を待機中...")

    # 完了待ち
    for chunk_num, start, end, proc in processes:
        returncode = proc.wait()
        # 最後の数行だけログ出力
        output = proc.stdout.read() if proc.stdout else ''
        last_lines = output.strip().split('\n')[-5:]
        status = "完了" if returncode == 0 else f"エラー(code={returncode})"
        logger.info(f"\nChunk {chunk_num} ({start} ~ {end}): {status}")
        for line in last_lines:
            if line.strip():
                logger.info(f"  {line.strip()}")

    # 収集後の統計
    logger.info("\n--- 収集後 ---")
    weather_pct = show_weather_stats()

    if weather_pct >= 80:
        logger.info("気象データ充足率 80% 以上達成！リトレインに進めます。")
    else:
        logger.info(f"気象データ充足率 {weather_pct:.1f}% — 追加収集が必要かもしれません。")

    logger.info("全チャンク処理完了。")


if __name__ == '__main__':
    main()
