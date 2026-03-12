"""
ボートレース予想AIシステム メインエントリーポイント (worker)
"""
import os
import sys
import time
import logging
from src.database import init_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=== ボートレース予想AIシステム起動 ===")

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL環境変数が設定されていません")
        sys.exit(1)

    try:
        init_database()
        logger.info("データベース初期化完了")
    except Exception as e:
        logger.error(f"データベース初期化失敗: {e}")
        sys.exit(1)

    # クラッシュ時自動復帰ループ
    attempt = 0
    while True:
        attempt += 1
        try:
            from src.scheduler import DynamicRaceScheduler
            logger.info(f"スケジューラ起動中... (attempt={attempt})")
            scheduler = DynamicRaceScheduler()
            scheduler.run_polling()
            # 正常returnしても再起動
            logger.warning(f"run_polling()が正常終了、再起動 (attempt={attempt})")
        except KeyboardInterrupt:
            logger.info("スケジューラ停止")
            break
        except Exception as e:
            logger.error(f"スケジューラエラー (attempt={attempt}): {e}", exc_info=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
