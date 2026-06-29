ROW_KEYS = [
    "race_id", "race_date", "venue_id", "race_number",
    "odds_3t", "n_combos",
    "probabilities_1st", "probabilities_2nd", "probabilities_3rd",
    "result_1st", "result_2nd", "result_3rd", "payout_sanrentan",
]

SQL = """
SELECT DISTINCT ON (ob.race_id)
       ob.race_id, r.race_date, r.venue_id, r.race_number,
       ob.odds_3t, ob.n_combos,
       p.probabilities_1st, p.probabilities_2nd, p.probabilities_3rd,
       r.result_1st, r.result_2nd, r.result_3rd, r.payout_sanrentan
FROM race_odds_board ob
JOIN races r ON r.id = ob.race_id
JOIN predictions p ON p.race_id = ob.race_id
WHERE r.race_date BETWEEN %s AND %s AND r.result_1st IS NOT NULL
ORDER BY ob.race_id
"""


def load_candidate_board(conn, date_from, date_to):
    """盤⨝結果⨝確率(marginals) を 1レース1行で返す（read-only）。"""
    cur = conn.cursor()
    # race_odds_board が未作成の本番初期でも壊れないよう存在確認
    cur.execute("SELECT to_regclass('public.race_odds_board')")
    if cur.fetchone()["to_regclass"] is None:
        return []
    cur.execute(SQL, (date_from, date_to))
    return [dict(r) for r in cur.fetchall()]
