"""
ボートレース予想AIシステム メインエントリーポイント
"""
import os
import sys
import logging
from src.scheduler import DynamicRaceScheduler
from src.database import init_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('boatrace_predictor.log'),
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
    
    logger.info("スケジューラ起動中...")
    scheduler = DynamicRaceScheduler()
    
    try:
        scheduler.run_polling()
    except KeyboardInterrupt:
        logger.info("スケジューラ停止")
    except Exception as e:
        logger.error(f"スケジューラエラー: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
