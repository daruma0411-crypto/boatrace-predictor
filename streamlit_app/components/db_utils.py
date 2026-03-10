"""Streamlit用データベースユーティリティ（接続プーリング対応）"""
import os
import logging
import threading
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()


def _get_database_url():
    url = os.environ.get('DATABASE_URL', '')
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


def _get_pool():
    """接続プールを取得（遅延初期化・スレッドセーフ）"""
    global _pool
    if _pool is None or _pool.closed:
        with _pool_lock:
            if _pool is None or _pool.closed:
                _pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=5,
                    dsn=_get_database_url(),
                    cursor_factory=RealDictCursor,
                )
    return _pool


@contextmanager
def get_db_connection():
    """PostgreSQL接続（プーリング版）"""
    pool = _get_pool()
    conn = None
    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            pool.putconn(conn)


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


def get_today_bets():
    """本日の買い目詳細を取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                r.venue_id, r.race_number, r.deadline_time,
                b.combination, b.amount, b.odds,
                b.expected_value, b.strategy_type, b.result, b.payout
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE r.race_date = CURRENT_DATE
            ORDER BY r.venue_id, r.race_number, b.strategy_type,
                     b.expected_value DESC
        """)
        return cur.fetchall()


def get_today_venues():
    """本日の開催場一覧を取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r.venue_id
            FROM races r
            WHERE r.race_date = CURRENT_DATE
            ORDER BY r.venue_id
        """)
        return [row['venue_id'] for row in cur.fetchall()]


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


def get_strategy_summary(start_date, end_date):
    """全戦略サマリー（期間指定）"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                b.strategy_type,
                COUNT(*) as total_bets,
                SUM(b.amount) as total_amount,
                SUM(b.payout) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(b.payout)::float / SUM(b.amount) * 100
                     ELSE 0 END as roi,
                COUNT(CASE WHEN b.payout > 0 THEN 1 END) as wins,
                COUNT(DISTINCT b.race_id) as total_races
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE b.result IS NOT NULL
              AND r.race_date >= %s
              AND r.race_date <= %s
            GROUP BY b.strategy_type
            ORDER BY b.strategy_type
        """, (start_date, end_date))
        return cur.fetchall()


def get_daily_stats_by_period(start_date, end_date):
    """期間指定の日別統計"""
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
              AND r.race_date >= %s
              AND r.race_date <= %s
            GROUP BY r.race_date, b.strategy_type
            ORDER BY r.race_date
        """, (start_date, end_date))
        return cur.fetchall()


def get_today_predictions():
    """本日の予想データ（レース選択用）"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r.venue_id, r.race_number, r.race_date,
                   p.strategy_type, p.id as prediction_id
            FROM predictions p
            JOIN races r ON p.race_id = r.id
            WHERE r.race_date = CURRENT_DATE
            ORDER BY r.venue_id, r.race_number
        """)
        return cur.fetchall()


def get_prediction_by_id(prediction_id):
    """予測データを取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM predictions WHERE id = %s",
            (prediction_id,)
        )
        return cur.fetchone()
