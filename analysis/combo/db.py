import os
import psycopg2
from psycopg2.extras import RealDictCursor

DSN = os.environ.get(
    "BR_DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable",
)


def connect():
    conn = psycopg2.connect(DSN, connect_timeout=25, cursor_factory=RealDictCursor)
    conn.set_session(readonly=True, autocommit=True)
    return conn
