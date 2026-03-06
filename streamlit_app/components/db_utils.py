"""Streamlit用データベースユーティリティ"""
import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


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
