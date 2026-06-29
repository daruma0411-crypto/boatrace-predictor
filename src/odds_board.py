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
