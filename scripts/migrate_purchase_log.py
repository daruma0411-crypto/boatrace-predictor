"""purchase_log テーブル マイグレーション

テレボート自動購入の結果を記録するテーブルを作成する。

使い方:
    python scripts/migrate_purchase_log.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.database import get_db_connection


def migrate():
    """purchase_log テーブルを作成"""
    with get_db_connection() as conn:
        cur = conn.cursor()

        # テーブル存在チェック
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'purchase_log'
        """)
        if cur.fetchone():
            print("purchase_log テーブルは既に存在します")
            return

        cur.execute("""
            CREATE TABLE IF NOT EXISTS purchase_log (
                id SERIAL PRIMARY KEY,
                bet_id INTEGER REFERENCES bets(id),
                race_id INTEGER NOT NULL,
                strategy_type VARCHAR(30) NOT NULL,
                combination VARCHAR(20) NOT NULL,
                amount INTEGER NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                error_message TEXT,
                screenshot_path TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                purchased_at TIMESTAMP WITH TIME ZONE
            )
        """)

        cur.execute("""
            CREATE INDEX idx_purchase_log_status
            ON purchase_log(status)
        """)

        cur.execute("""
            CREATE INDEX idx_purchase_log_bet_id
            ON purchase_log(bet_id)
        """)

        print("purchase_log テーブル作成完了")


if __name__ == '__main__':
    migrate()
