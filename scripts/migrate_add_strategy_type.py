"""マイグレーション: bets テーブルに strategy_type カラムを追加"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """bets テーブルに strategy_type VARCHAR(20) カラムを追加"""
    with get_db_connection() as conn:
        cur = conn.cursor()

        # カラムが存在するか確認
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'bets' AND column_name = 'strategy_type'
        """)
        exists = cur.fetchone()

        if exists:
            logger.info("strategy_type カラムは既に存在します")
            return

        # カラム追加
        cur.execute("""
            ALTER TABLE bets
            ADD COLUMN strategy_type VARCHAR(20)
        """)

        # 既存データにデフォルト値を設定
        cur.execute("""
            UPDATE bets
            SET strategy_type = 'kelly_strict'
            WHERE strategy_type IS NULL
        """)

        # NOT NULL 制約を追加
        cur.execute("""
            ALTER TABLE bets
            ALTER COLUMN strategy_type SET NOT NULL
        """)

        logger.info("マイグレーション完了: strategy_type カラムを追加しました")


if __name__ == '__main__':
    migrate()
