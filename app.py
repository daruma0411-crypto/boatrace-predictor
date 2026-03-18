"""
ボートレース予想AIシステム メインエントリーポイント (worker)
"""
import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

_WORKER_VERSION = "v9.2-joseki-kelly"


def _write_health(status, detail):
    """scheduler_healthに書き込み（workerの状態追跡用）"""
    try:
        import psycopg2
        db_url = os.environ.get('DATABASE_URL', '')
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        if not db_url:
            return
        c = psycopg2.connect(db_url)
        cur = c.cursor()
        cur.execute(
            "INSERT INTO scheduler_health (status, detail) VALUES (%s, %s)",
            (status, f'[worker] {detail}'),
        )
        c.commit()
        c.close()
    except Exception as e:
        logger.warning(f"health書き込み失敗: {e}")


def main():
    logger.info(f"=== ボートレース予想AIシステム起動 (worker {_WORKER_VERSION}) ===")
    _write_health('worker_start', f'version={_WORKER_VERSION}')

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL環境変数が設定されていません")
        _write_health('error', 'DATABASE_URL未設定')
        sys.exit(1)

    try:
        from src.database import init_database
        init_database()
        logger.info("データベース初期化完了")
        _write_health('db_ready', 'DB初期化完了')
    except Exception as e:
        logger.error(f"データベース初期化失敗: {e}")
        _write_health('error', f'DB初期化失敗: {e}')
        sys.exit(1)

    # クラッシュ時自動復帰ループ（指数バックオフ）
    attempt = 0
    while True:
        attempt += 1
        try:
            from src.scheduler import DynamicRaceScheduler
            logger.info(f"スケジューラ起動中... (attempt={attempt})")
            _write_health('scheduler_start', f'attempt={attempt}')
            scheduler = DynamicRaceScheduler()
            _write_health('polling_start', f'attempt={attempt}, モデル読込完了')
            scheduler.run_polling()
            # 正常returnしても再起動
            logger.warning(f"run_polling()が正常終了、再起動 (attempt={attempt})")
            _write_health('polling_exit', f'正常終了 attempt={attempt}')
        except KeyboardInterrupt:
            logger.info("スケジューラ停止")
            break
        except Exception as e:
            logger.error(f"スケジューラエラー (attempt={attempt}): {e}", exc_info=True)
            _write_health('error', f'attempt={attempt}: {str(e)[:200]}')
        wait = min(60 * (2 ** min(attempt - 1, 3)), 600)
        logger.info(f"再起動待機: {wait}秒")
        time.sleep(wait)


if __name__ == "__main__":
    main()
