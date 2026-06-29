"""オッズ盤台帳: 判断時の全3連単オッズ盤を1レース1行で保存する（追記オンリー・自己隔離）。"""
import json
import logging

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS race_odds_board (
    race_id     integer PRIMARY KEY,
    captured_at timestamptz NOT NULL DEFAULT now(),
    odds_3t     jsonb NOT NULL,
    n_combos    integer NOT NULL
)
"""

INSERT_SQL = """
INSERT INTO race_odds_board (race_id, odds_3t, n_combos)
VALUES (%s, %s, %s)
ON CONFLICT (race_id) DO NOTHING
"""

HEALTH_SQL = "INSERT INTO scheduler_health (status, detail) VALUES (%s, %s)"


def build_board_row(race_id, odds_dict):
    """(race_id:int, odds_json:str, n_combos:int) を返す純粋関数。"""
    return (int(race_id), json.dumps(odds_dict), len(odds_dict))


def save_odds_board(race_id, odds_dict, conn_factory=None):
    """判断時オッズ盤を1レース1行で保存。例外は投げず bool を返す（本処理を止めない）。"""
    if not odds_dict:
        return False
    try:
        if conn_factory is None:
            from src.database import get_db_connection
            conn_factory = get_db_connection
        row = build_board_row(race_id, odds_dict)
        with conn_factory() as conn:
            cur = conn.cursor()
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(INSERT_SQL, row)
            cur.execute(HEALTH_SQL, ("odds_board_saved", f"race_id={race_id} n={row[2]}"))
        return True
    except Exception as e:
        logger.warning(f"odds_board 保存失敗(本処理に影響なし) race_id={race_id}: {e}")
        try:
            with conn_factory() as conn:
                conn.cursor().execute(
                    HEALTH_SQL, ("odds_board_failed", f"race_id={race_id}: {str(e)[:200]}")
                )
        except Exception:
            pass
        return False
