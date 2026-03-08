"""収集完了監視 → 自動再学習 → 精度評価

4ターミナルの並列収集が完了したら自動でモデル再学習＆評価を実行する。
5分間隔でDB件数を監視し、増加が止まったら収集完了と判定。

使い方:
    python scripts/auto_retrain.py
    python scripts/auto_retrain.py --target 50000 --interval 300
"""
import sys
import os
import time
import logging
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://boatrace:boatrace123@localhost:5432/boatrace_db',
)


def get_race_count():
    """結果ありレース数を取得"""
    os.environ['DATABASE_URL'] = DB_URL
    from src.database import get_db_connection
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) as cnt FROM races WHERE result_1st IS NOT NULL')
        total = cur.fetchone()['cnt']
        cur.execute('SELECT COUNT(*) as cnt FROM boats')
        boats = cur.fetchone()['cnt']
    return total, boats


def get_monthly_breakdown():
    """月別レース数を取得"""
    os.environ['DATABASE_URL'] = DB_URL
    from src.database import get_db_connection
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DATE_TRUNC('month', race_date)::date as m, COUNT(*) as c
            FROM races WHERE result_1st IS NOT NULL
            GROUP BY m ORDER BY m
        """)
        return cur.fetchall()


def run_training():
    """学習スクリプトを実行"""
    logger.info("=== 再学習開始 ===")
    env = os.environ.copy()
    env['DATABASE_URL'] = DB_URL
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(
        [sys.executable, 'scripts/train_model.py'],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,  # 最大1時間
    )

    logger.info("--- train stdout ---")
    for line in result.stdout.strip().split('\n')[-20:]:
        logger.info(f"  {line}")

    if result.returncode != 0:
        logger.error(f"学習失敗 (exit={result.returncode})")
        logger.error(result.stderr[-500:] if result.stderr else "no stderr")
        return False

    logger.info("=== 再学習完了 ===")
    return True


def run_evaluation():
    """精度評価スクリプトを実行"""
    logger.info("=== 精度評価開始 ===")
    env = os.environ.copy()
    env['DATABASE_URL'] = DB_URL
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(
        [sys.executable, 'scripts/evaluate_model.py'],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    logger.info("--- evaluate stdout ---")
    for line in result.stdout.strip().split('\n'):
        logger.info(f"  {line}")

    if result.returncode != 0:
        logger.error(f"評価失敗 (exit={result.returncode})")
        return False

    logger.info("=== 精度評価完了 ===")
    return True


def monitor_and_retrain(target=45000, interval=300, stable_count=3):
    """収集監視 → 再学習 → 評価

    Args:
        target: 目標レース数（これ以上なら即実行）
        interval: チェック間隔（秒）
        stable_count: 件数変動なしが何回続いたら完了判定
    """
    logger.info(f"=== 自動再学習モニター開始 ===")
    logger.info(f"目標: {target:,}件 | チェック間隔: {interval}秒 | 安定判定: {stable_count}回")

    prev_count = 0
    stable_checks = 0
    check_num = 0

    while True:
        check_num += 1
        try:
            race_count, boat_count = get_race_count()
        except Exception as e:
            logger.warning(f"DB接続エラー: {e}")
            time.sleep(interval)
            continue

        diff = race_count - prev_count
        logger.info(
            f"[check #{check_num}] レース: {race_count:,} (+{diff:,}) | "
            f"boats: {boat_count:,} | 目標: {target:,}"
        )

        # 目標到達チェック
        if race_count >= target:
            logger.info(f"目標{target:,}件到達！再学習開始します。")
            break

        # 安定判定（増加が止まった = 収集完了）
        if diff == 0 and race_count > 20000:
            stable_checks += 1
            logger.info(f"  安定チェック: {stable_checks}/{stable_count}")
            if stable_checks >= stable_count:
                logger.info(f"収集完了判定（{stable_count}回連続変動なし）。再学習開始します。")
                break
        else:
            stable_checks = 0

        prev_count = race_count
        time.sleep(interval)

    # 月別サマリー表示
    logger.info("--- 収集データ月別サマリー ---")
    try:
        monthly = get_monthly_breakdown()
        for m in monthly:
            logger.info(f"  {m['m']}: {m['c']:,}R")
    except Exception:
        pass

    # 再学習実行
    train_ok = run_training()
    if not train_ok:
        logger.error("再学習失敗。手動で確認してください。")
        return

    # 精度評価実行
    eval_ok = run_evaluation()
    if not eval_ok:
        logger.error("精度評価失敗。手動で確認してください。")
        return

    logger.info("=== 全工程完了 ===")
    logger.info(f"最終レース数: {get_race_count()[0]:,}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='収集完了監視 → 自動再学習')
    parser.add_argument('--target', type=int, default=45000, help='目標レース数')
    parser.add_argument('--interval', type=int, default=300, help='チェック間隔(秒)')
    parser.add_argument('--stable', type=int, default=3, help='安定判定回数')
    args = parser.parse_args()

    monitor_and_retrain(
        target=args.target,
        interval=args.interval,
        stable_count=args.stable,
    )


if __name__ == '__main__':
    main()
