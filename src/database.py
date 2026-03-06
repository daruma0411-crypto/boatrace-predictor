"""PostgreSQLデータベース管理"""
import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


def _get_database_url():
    """DATABASE_URLを取得し、postgres:// → postgresql:// 変換"""
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise ValueError("DATABASE_URL環境変数が設定されていません")
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


@contextmanager
def get_db_connection():
    """PostgreSQL接続のコンテキストマネージャー"""
    conn = None
    try:
        conn = psycopg2.connect(_get_database_url(), cursor_factory=RealDictCursor)
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def init_database():
    """5テーブルを作成"""
    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS races (
                id SERIAL PRIMARY KEY,
                race_date DATE NOT NULL,
                venue_id INTEGER NOT NULL,
                race_number INTEGER NOT NULL,
                deadline_time TIMESTAMP WITH TIME ZONE,
                status VARCHAR(20) DEFAULT 'scheduled',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(race_date, venue_id, race_number)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS boats (
                id SERIAL PRIMARY KEY,
                race_id INTEGER REFERENCES races(id),
                boat_number INTEGER NOT NULL,
                player_id VARCHAR(10),
                player_name VARCHAR(50),
                player_class VARCHAR(5),
                win_rate REAL,
                win_rate_2 REAL,
                win_rate_3 REAL,
                local_win_rate REAL,
                local_win_rate_2 REAL,
                motor_win_rate_2 REAL,
                motor_win_rate_3 REAL,
                boat_win_rate_2 REAL,
                weight REAL,
                avg_st REAL,
                exhibition_time REAL,
                approach_course INTEGER,
                is_new_motor BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                race_id INTEGER REFERENCES races(id),
                prediction_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                probabilities_1st JSONB,
                probabilities_2nd JSONB,
                probabilities_3rd JSONB,
                recommended_bets JSONB,
                model_version VARCHAR(50),
                strategy_type VARCHAR(20) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id SERIAL PRIMARY KEY,
                prediction_id INTEGER REFERENCES predictions(id),
                race_id INTEGER REFERENCES races(id),
                bet_type VARCHAR(20) NOT NULL,
                combination VARCHAR(20) NOT NULL,
                amount INTEGER NOT NULL,
                odds REAL,
                expected_value REAL,
                kelly_fraction REAL,
                result VARCHAR(20),
                payout INTEGER DEFAULT 0,
                strategy_type VARCHAR(20) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                model_version VARCHAR(50),
                evaluation_date DATE NOT NULL,
                venue_id INTEGER,
                accuracy_1st REAL,
                accuracy_2nd REAL,
                accuracy_3rd REAL,
                total_bets INTEGER DEFAULT 0,
                total_amount INTEGER DEFAULT 0,
                total_payout INTEGER DEFAULT 0,
                roi REAL,
                strategy_type VARCHAR(20),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        logger.info("データベース初期化完了（5テーブル作成）")


def get_current_bankroll():
    """現在の収支を取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(SUM(payout - amount), 0) as profit
            FROM bets WHERE result IS NOT NULL
        """)
        row = cur.fetchone()
        return row['profit'] if row else 0
