import json
from db import connect

_RACE_SQL = """
SELECT DISTINCT ON (ob.race_id)
       ob.race_id, ob.odds_3t, r.race_number, r.venue_id, r.race_date,
       r.wind_speed, r.wind_direction, r.temperature, r.wave_height, r.water_temperature,
       r.result_1st, r.result_2nd, r.result_3rd,
       p.probabilities_1st, p.probabilities_2nd, p.probabilities_3rd
FROM race_odds_board ob
JOIN races r ON r.id = ob.race_id
JOIN predictions p ON p.race_id = ob.race_id
WHERE r.race_date BETWEEN %s AND %s
      AND r.result_1st IS NOT NULL AND r.result_2nd IS NOT NULL AND r.result_3rd IS NOT NULL
ORDER BY ob.race_id
"""

_BOATS_SQL = "SELECT * FROM boats WHERE race_id = %s ORDER BY boat_number"


def _vec(v):
    if isinstance(v, str):
        v = json.loads(v)
    return [float(x) for x in v]


def _boats_data(cur, race_id):
    cur.execute(_BOATS_SQL, (race_id,))
    out = []
    for b in cur.fetchall():
        out.append({
            "boat_number": b["boat_number"], "player_class": b["player_class"],
            "win_rate": b["win_rate"], "win_rate_2": b["win_rate_2"], "win_rate_3": b["win_rate_3"],
            "local_win_rate": b["local_win_rate"], "local_win_rate_2": b["local_win_rate_2"],
            "avg_st": b["avg_st"], "motor_win_rate_2": b["motor_win_rate_2"],
            "motor_win_rate_3": b["motor_win_rate_3"], "boat_win_rate_2": b["boat_win_rate_2"],
            "weight": b["weight"], "exhibition_time": b["exhibition_time"],
            "approach_course": b["approach_course"], "is_new_motor": b["is_new_motor"],
            "tilt": b.get("tilt"), "parts_changed": b.get("parts_changed", False),
            "fallback_flag": False,
        })
    return out


def load_races(date_from, date_to, limit=None):
    conn = connect(); cur = conn.cursor()
    cur.execute(_RACE_SQL, (date_from, date_to))
    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        odds = r["odds_3t"] if isinstance(r["odds_3t"], dict) else json.loads(r["odds_3t"])
        if len(odds) < 100:
            continue
        out.append({
            "race_id": r["race_id"], "race_number": r["race_number"],
            "odds": {k: float(v) for k, v in odds.items()},
            "p1": _vec(r["probabilities_1st"]), "p2": _vec(r["probabilities_2nd"]),
            "p3": _vec(r["probabilities_3rd"]),
            "win_combo": f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}",
            "race_data": {
                "venue_id": r["venue_id"], "month": r["race_date"].month, "distance": 1800,
                "wind_speed": r["wind_speed"] or 0, "wind_direction": r["wind_direction"] or "calm",
                "temperature": r["temperature"] or 20, "wave_height": r["wave_height"] or 0,
                "water_temperature": r["water_temperature"] or 20,
            },
            "boats_data": _boats_data(cur, r["race_id"]),
        })
    return out
