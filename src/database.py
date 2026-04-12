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


def _migrate_tables(conn):
    """既存テーブルのスキーマをマイグレーション"""
    cur = conn.cursor()

    # races テーブルに不足カラムを追加
    for col, col_def in [
        ('deadline_time', 'TIMESTAMP WITH TIME ZONE'),
        ('status', "VARCHAR(20) DEFAULT 'scheduled'"),
        ('result_1st', 'INTEGER'),
        ('result_2nd', 'INTEGER'),
        ('result_3rd', 'INTEGER'),
        ('payout_sanrentan', 'INTEGER'),
    ]:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'races' AND column_name = %s
        """, (col,))
        if not cur.fetchone():
            cur.execute(f'ALTER TABLE races ADD COLUMN {col} {col_def}')
            logger.info(f"マイグレーション: races.{col} 追加")

    # races テーブルに UNIQUE 制約を追加
    cur.execute("""
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'races'::regclass AND contype = 'u'
    """)
    if not cur.fetchone():
        try:
            cur.execute("""
                ALTER TABLE races
                ADD CONSTRAINT races_date_venue_race_unique
                UNIQUE (race_date, venue_id, race_number)
            """)
            logger.info("マイグレーション: races UNIQUE制約追加")
        except Exception as e:
            logger.warning(f"UNIQUE制約追加スキップ (重複データ?): {e}")
            conn.rollback()
            conn.cursor()  # reset cursor after rollback

    # boats テーブルに不足カラムを追加 (208次元対応)
    for col, col_def in [
        ('tilt', 'REAL'),
        ('parts_changed', 'BOOLEAN DEFAULT FALSE'),
    ]:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'boats' AND column_name = %s
        """, (col,))
        if not cur.fetchone():
            cur.execute(f'ALTER TABLE boats ADD COLUMN {col} {col_def}')
            logger.info(f"マイグレーション: boats.{col} 追加")

    # races テーブルに天候カラムを追加 (208次元対応)
    for col, col_def in [
        ('wind_speed', 'REAL'),
        ('wind_direction', 'VARCHAR(10)'),
        ('temperature', 'REAL'),
        ('wave_height', 'REAL'),
        ('water_temperature', 'REAL'),
    ]:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'races' AND column_name = %s
        """, (col,))
        if not cur.fetchone():
            cur.execute(f'ALTER TABLE races ADD COLUMN {col} {col_def}')
            logger.info(f"マイグレーション: races.{col} 追加")

    # predictions テーブルに不足カラムを追加
    for col, col_def in [
        ('model_version', 'VARCHAR(50)'),
    ]:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'predictions' AND column_name = %s
        """, (col,))
        if not cur.fetchone():
            cur.execute(f'ALTER TABLE predictions ADD COLUMN {col} {col_def}')
            logger.info(f"マイグレーション: predictions.{col} 追加")

    # strategy_type VARCHAR(20) → VARCHAR(30) マイグレーション
    for table in ['predictions', 'bets', 'model_performance']:
        try:
            cur.execute("""
                SELECT character_maximum_length
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'strategy_type'
            """, (table,))
            row = cur.fetchone()
            if row and row['character_maximum_length'] and row['character_maximum_length'] < 30:
                cur.execute(
                    f'ALTER TABLE {table} ALTER COLUMN strategy_type TYPE VARCHAR(30)'
                )
                logger.info(f"マイグレーション: {table}.strategy_type VARCHAR(30)に拡張")
        except Exception as e:
            logger.warning(f"strategy_type拡張スキップ ({table}): {e}")
            conn.rollback()
            cur = conn.cursor()

    # bets テーブルに不足カラムを追加
    for col, col_def in [
        ('prediction_id', 'INTEGER'),
        ('bet_type', "VARCHAR(20) DEFAULT 'trifecta'"),
        ('kelly_fraction', 'REAL'),
        ('closing_odds', 'REAL'),
        ('clv', 'REAL'),
    ]:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'bets' AND column_name = %s
        """, (col,))
        if not cur.fetchone():
            cur.execute(f'ALTER TABLE bets ADD COLUMN {col} {col_def}')
            logger.info(f"マイグレーション: bets.{col} 追加")

    conn.commit()


def init_database():
    """5テーブルを作成（既存テーブルはマイグレーション）"""
    with get_db_connection() as conn:
        cur = conn.cursor()

        # 既存テーブルがあればマイグレーション
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'races'
        """)
        if cur.fetchone():
            _migrate_tables(conn)
            cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS races (
                id SERIAL PRIMARY KEY,
                race_date DATE NOT NULL,
                venue_id INTEGER NOT NULL,
                race_number INTEGER NOT NULL,
                deadline_time TIMESTAMP WITH TIME ZONE,
                status VARCHAR(20) DEFAULT 'scheduled',
                result_1st INTEGER,
                result_2nd INTEGER,
                result_3rd INTEGER,
                payout_sanrentan INTEGER,
                wind_speed REAL,
                wind_direction VARCHAR(10),
                temperature REAL,
                wave_height REAL,
                water_temperature REAL,
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
                tilt REAL,
                parts_changed BOOLEAN DEFAULT FALSE,
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
                strategy_type VARCHAR(30) NOT NULL,
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
                strategy_type VARCHAR(30) NOT NULL,
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
                strategy_type VARCHAR(30),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # レース処理ロック（複数プロセス間の重複防止用）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS race_processing (
                race_id INTEGER PRIMARY KEY REFERENCES races(id),
                locked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # bets重複防止用ユニーク制約（安全ネット）
        try:
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_bets_race_strategy_combo
                ON bets (race_id, strategy_type, combination)
            """)
        except Exception:
            conn.rollback()
            cur = conn.cursor()

        logger.info("データベース初期化完了（5テーブル + race_processing作成）")


# V4リセット日: この日以降のベットのみでbankrollを計算
BANKROLL_RESET_DATE = '2026-04-01'


def get_current_bankroll(strategy_type=None):
    """現在の収支を取得（戦略別フィルター対応、V4リセット日以降のみ）"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        if strategy_type:
            cur.execute("""
                SELECT COALESCE(SUM(payout - amount), 0) as profit
                FROM bets WHERE result IS NOT NULL AND strategy_type = %s
                  AND created_at >= %s
            """, (strategy_type, BANKROLL_RESET_DATE))
        else:
            cur.execute("""
                SELECT COALESCE(SUM(payout - amount), 0) as profit
                FROM bets WHERE result IS NOT NULL
                  AND created_at >= %s
            """, (BANKROLL_RESET_DATE,))
        row = cur.fetchone()
        return float(row['profit']) if row else 0.0
