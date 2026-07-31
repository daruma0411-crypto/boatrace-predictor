"""race_softness の保存(自己隔離・冪等)と市場温度の取得。"""
import logging
from src.softness_params import TEMP_WINDOW

logger = logging.getLogger(__name__)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS race_softness (
    race_id   integer PRIMARY KEY,
    race_date date NOT NULL,
    score     real NOT NULL
)
"""
INSERT_SQL = """
INSERT INTO race_softness (race_id, race_date, score)
VALUES (%s, %s, %s) ON CONFLICT (race_id) DO NOTHING
"""
TEMP_SQL = """
SELECT race_date::text AS d, avg(score) AS m FROM race_softness
WHERE race_date >= (%s::date - %s * interval '1 day') AND race_date < %s::date
GROUP BY race_date
"""


def save_race_softness(race_id, race_date, score, conn_factory=None):
    """1レース1回 upsert。例外は投げない（本処理を止めない）。score None は保存しない。"""
    if score is None:
        return False
    try:
        if conn_factory is None:
            from src.database import get_db_connection
            conn_factory = get_db_connection
        with conn_factory() as conn:
            cur = conn.cursor()
            cur.execute(CREATE_SQL)
            cur.execute(INSERT_SQL, (int(race_id), str(race_date), float(score)))
        return True
    except Exception as e:
        logger.warning(f"race_softness 保存失敗(無視) race_id={race_id}: {e}")
        return False


def get_temperature(target_date, conn_factory=None):
    """前日までの直近 TEMP_WINDOW 日の日次平均スコアの平均。<3日 or 失敗は None。"""
    try:
        if conn_factory is None:
            from src.database import get_db_connection
            conn_factory = get_db_connection
        with conn_factory() as conn:
            cur = conn.cursor()
            cur.execute(TEMP_SQL, (str(target_date), TEMP_WINDOW, str(target_date)))
            rows = cur.fetchall()
        means = [float(r["m"]) for r in rows]
        if len(means) < 3:
            return None
        return sum(means) / len(means)
    except Exception as e:
        logger.warning(f"温度取得失敗(無視) date={target_date}: {e}")
        return None
