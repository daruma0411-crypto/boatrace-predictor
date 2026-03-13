"""Streamlit用データベースユーティリティ（接続プーリング対応）"""
import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import streamlit as st

logger = logging.getLogger(__name__)


def _get_database_url():
    url = os.environ.get('DATABASE_URL', '')
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


@st.cache_resource
def _get_pool():
    """接続プールを取得（Streamlitリソースキャッシュで1回だけ初期化）"""
    pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=_get_database_url(),
        cursor_factory=RealDictCursor,
        connect_timeout=10,
        options='-c statement_timeout=8000',
    )
    # インデックス作成（初回のみ）
    try:
        conn = pool.getconn()
        cur = conn.cursor()
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result)",
            "CREATE INDEX IF NOT EXISTS idx_bets_strategy ON bets(strategy_type)",
            "CREATE INDEX IF NOT EXISTS idx_bets_race_id ON bets(race_id)",
            "CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_races_date_venue ON races(race_date, venue_id)",
        ]:
            try:
                cur.execute(idx_sql)
            except Exception:
                conn.rollback()
                cur = conn.cursor()
        conn.commit()
        pool.putconn(conn)
    except Exception:
        pass
    return pool


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
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn:
            try:
                pool.putconn(conn)
            except Exception:
                pass


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
                SUM(COALESCE(b.return_amount, b.payout, 0)) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(COALESCE(b.return_amount, b.payout, 0))::float / SUM(b.amount) * 100
                     ELSE 0 END as roi,
                COUNT(CASE WHEN COALESCE(b.is_hit, b.payout > 0, FALSE) THEN 1 END) as wins
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE (b.is_hit IS NOT NULL OR b.result IS NOT NULL)
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
                SUM(COALESCE(b.return_amount, b.payout, 0)) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(COALESCE(b.return_amount, b.payout, 0))::float / SUM(b.amount) * 100
                     ELSE 0 END as roi
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE (b.is_hit IS NOT NULL OR b.result IS NOT NULL)
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
                b.expected_value, b.strategy_type,
                b.is_hit, COALESCE(b.return_amount, b.payout, 0) as return_amount,
                r.actual_result_trifecta, r.is_finished
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
                SUM(COALESCE(b.return_amount, b.payout, 0)) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(COALESCE(b.return_amount, b.payout, 0))::float / SUM(b.amount) * 100
                     ELSE 0 END as roi
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE (b.is_hit IS NOT NULL OR b.result IS NOT NULL)
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
                SUM(COALESCE(b.return_amount, b.payout, 0)) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(COALESCE(b.return_amount, b.payout, 0))::float / SUM(b.amount) * 100
                     ELSE 0 END as roi,
                COUNT(CASE WHEN COALESCE(b.is_hit, b.payout > 0, FALSE) THEN 1 END) as wins,
                COUNT(DISTINCT b.race_id) as total_races
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE (b.is_hit IS NOT NULL OR b.result IS NOT NULL)
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
                SUM(COALESCE(b.return_amount, b.payout, 0)) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(COALESCE(b.return_amount, b.payout, 0))::float / SUM(b.amount) * 100
                     ELSE 0 END as roi
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE (b.is_hit IS NOT NULL OR b.result IS NOT NULL)
              AND r.race_date >= %s
              AND r.race_date <= %s
            GROUP BY r.race_date, b.strategy_type
            ORDER BY r.race_date
        """, (start_date, end_date))
        return cur.fetchall()


def get_all_bankrolls():
    """全戦略のbankrollを1クエリで取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT strategy_type,
                   COALESCE(SUM(COALESCE(return_amount, payout, 0) - amount), 0) as profit
            FROM bets WHERE (is_hit IS NOT NULL OR result IS NOT NULL)
            GROUP BY strategy_type
        """)
        rows = cur.fetchall()
    result = {}
    for row in rows:
        result[row['strategy_type']] = 200000 + row['profit']
    return result


def get_dashboard_data(start_date, end_date):
    """ダッシュボードに必要な全データを1回のDB接続で取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()

        # 1. 本日のレース数・予測数・ベット数
        cur.execute("SELECT COUNT(*) as cnt FROM races WHERE race_date = CURRENT_DATE")
        today_races = cur.fetchone()['cnt']
        cur.execute(
            "SELECT COUNT(*) as cnt FROM predictions "
            "WHERE created_at::date = CURRENT_DATE"
        )
        today_preds = cur.fetchone()['cnt']
        cur.execute(
            "SELECT COUNT(*) as cnt FROM bets b "
            "JOIN races r ON b.race_id = r.id "
            "WHERE r.race_date = CURRENT_DATE"
        )
        today_bets = cur.fetchone()['cnt']

        # 2. 戦略サマリー（結果確定分のみ）
        cur.execute("""
            SELECT
                b.strategy_type,
                COUNT(*) as total_bets,
                SUM(b.amount) as total_amount,
                SUM(COALESCE(b.return_amount, b.payout, 0)) as total_payout,
                CASE WHEN SUM(b.amount) > 0
                     THEN SUM(COALESCE(b.return_amount, b.payout, 0))::float / SUM(b.amount) * 100
                     ELSE 0 END as roi,
                COUNT(CASE WHEN COALESCE(b.is_hit, b.payout > 0, FALSE) THEN 1 END) as wins,
                COUNT(DISTINCT b.race_id) as total_races
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE (b.is_hit IS NOT NULL OR b.result IS NOT NULL)
              AND r.race_date >= %s AND r.race_date <= %s
            GROUP BY b.strategy_type
            ORDER BY b.strategy_type
        """, (start_date, end_date))
        strategy_summary = cur.fetchall()

        # 3. 全戦略bankroll（全期間累計）
        cur.execute("""
            SELECT strategy_type,
                   COALESCE(SUM(COALESCE(return_amount, payout, 0) - amount), 0) as profit
            FROM bets WHERE (is_hit IS NOT NULL OR result IS NOT NULL)
            GROUP BY strategy_type
        """)
        bankroll_rows = cur.fetchall()
        bankrolls = {}
        for row in bankroll_rows:
            bankrolls[row['strategy_type']] = 200000 + row['profit']

        # 4. 本日の的中数
        cur.execute("""
            SELECT COUNT(*) as cnt FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE r.race_date = CURRENT_DATE AND b.is_hit = TRUE
        """)
        today_hits = cur.fetchone()['cnt']

    return {
        'today_races': today_races,
        'today_preds': today_preds,
        'today_bets': today_bets,
        'today_hits': today_hits,
        'strategy_summary': strategy_summary,
        'bankrolls': bankrolls,
        'db_ok': True,
    }


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
