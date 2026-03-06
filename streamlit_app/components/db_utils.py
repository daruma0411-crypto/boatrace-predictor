"""Streamlit用データベースユーティリティ"""
import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

INIT_SQL = """
CREATE TABLE IF NOT EXISTS races (
    id SERIAL PRIMARY KEY,
    venue_id INTEGER NOT NULL,
    race_number INTEGER NOT NULL,
    race_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    race_id INTEGER REFERENCES races(id),
    strategy_type VARCHAR(50),
    probabilities_1st JSONB,
    probabilities_2nd JSONB,
    probabilities_3rd JSONB,
    recommended_bets JSONB,
    prediction_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bets (
    id SERIAL PRIMARY KEY,
    race_id INTEGER REFERENCES races(id),
    strategy_type VARCHAR(50),
    combination VARCHAR(50),
    amount NUMERIC,
    odds NUMERIC,
    expected_value NUMERIC,
    result VARCHAR(20),
    payout NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db():
    """テーブルが存在しない場合に作成"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(INIT_SQL)
        logger.info("DB初期化完了")
    except Exception as e:
        logger.warning(f"DB初期化スキップ: {e}")


def _get_database_url():
    url = os.environ.get('DATABASE_URL', '')
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


@contextmanager
def get_db_connection():
    """PostgreSQL接続（Streamlit用）"""
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


def get_recent_predictions(limit=50, strategy_type=None):
    """最近の予測を取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        query = """
            SELECT p.*, r.race_date, r.venue_id, r.race_number
            FROM predictions p
            JOIN races r ON p.race_id = r.id
        """
        params = []
        if strategy_type:
            query += " WHERE p.strategy_type = %s"
            params.append(strategy_type)
        query += " ORDER BY p.created_at DESC LIMIT %s"
        params.append(limit)
        cur.execute(query, params)
        return cur.fetchall()


def get_performance_stats(days=30, strategy_type=None):
    """パフォーマンス統計を取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        query = """
            SELECT
                b.strategy_type,
                COUNT(*) as total_bets,
                SUM(b.amount) as total_amount,
                SUM(b.payout) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(b.payout)::float / SUM(b.amount) * 100
                     ELSE 0 END as roi,
                COUNT(CASE WHEN b.payout > 0 THEN 1 END) as wins
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE b.result IS NOT NULL
              AND r.race_date >= CURRENT_DATE - INTERVAL '%s days'
        """
        params = [days]
        if strategy_type:
            query += " AND b.strategy_type = %s"
            params.append(strategy_type)
        query += " GROUP BY b.strategy_type"
        cur.execute(query, params)
        return cur.fetchall()


def get_venue_stats():
    """場別パフォーマンスを取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                r.venue_id,
                b.strategy_type,
                COUNT(*) as total_bets,
                SUM(b.amount) as total_amount,
                SUM(b.payout) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(b.payout)::float / SUM(b.amount) * 100
                     ELSE 0 END as roi
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE b.result IS NOT NULL
            GROUP BY r.venue_id, b.strategy_type
            ORDER BY r.venue_id
        """)
        return cur.fetchall()


def get_daily_stats(days=30):
    """日別パフォーマンスを取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                r.race_date,
                b.strategy_type,
                COUNT(*) as total_bets,
                SUM(b.amount) as total_amount,
                SUM(b.payout) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(b.payout)::float / SUM(b.amount) * 100
                     ELSE 0 END as roi
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE b.result IS NOT NULL
              AND r.race_date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY r.race_date, b.strategy_type
            ORDER BY r.race_date
        """, (days,))
        return cur.fetchall()
