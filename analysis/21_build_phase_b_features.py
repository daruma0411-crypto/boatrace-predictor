"""Phase B 特徴量 pkl 生成 (B1-B5 of Phase B roadmap, Issue #4)

入力: races + boats + race_titles テーブル (2026-02-01〜2026-04-30、READ-ONLY)
出力: analysis/features_phase_b.pkl (DataFrame、race_id index × 9 列)

特徴量:
- race_category (str): B1
- is_planned (bool): B1
- boat1_skill_gap (float): B2
- a_class_consumed (float [0,1]): B3 (SQL window で計算)
- day_in_meeting (int|NaN): B4
- day_label_raw (str|None): 保険、元値
"""
import os
import sys
import logging
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.database import get_db_connection
from src.phase_b_features import (
    classify_race_category,
    detect_planned_race,
    parse_day_label,
    compute_skill_gap,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "analysis" / "features_phase_b.pkl"

DATE_FROM = '2026-02-01'
DATE_TO = '2026-04-30'


def fetch_rows():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
                   rt.title, rt.subtitle, rt.day_label
            FROM races r
            LEFT JOIN race_titles rt ON r.id = rt.race_id
            WHERE r.race_date BETWEEN %s AND %s
            ORDER BY r.race_date, r.venue_id, r.race_number
        """, (DATE_FROM, DATE_TO))
        return cur.fetchall()


def fetch_boats(race_ids):
    if not race_ids:
        return {}
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT race_id, boat_number, player_class, win_rate_2
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        rows = cur.fetchall()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r['race_id']].append(dict(r))
    return dict(by_race)


def fetch_a_class_consumed():
    """B3: 同日同会場の累積 A 級出走比率を SQL window で計算"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            WITH per_race AS (
                SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
                       COUNT(b.*) AS n_boats,
                       COUNT(b.*) FILTER (WHERE b.player_class IN ('A1', 'A2')) AS n_a
                FROM races r
                JOIN boats b ON r.id = b.race_id
                WHERE r.race_date BETWEEN %s AND %s
                GROUP BY r.id, r.race_date, r.venue_id, r.race_number
            ),
            cumulative AS (
                SELECT race_id, race_date, venue_id, race_number,
                       SUM(n_a) OVER (
                           PARTITION BY race_date, venue_id
                           ORDER BY race_number
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                       ) AS prior_a,
                       SUM(n_boats) OVER (
                           PARTITION BY race_date, venue_id
                           ORDER BY race_number
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                       ) AS prior_total
                FROM per_race
            )
            SELECT race_id,
                   CASE
                       WHEN prior_total IS NULL OR prior_total = 0 THEN NULL
                       ELSE prior_a::float / prior_total
                   END AS a_class_consumed
            FROM cumulative
        """, (DATE_FROM, DATE_TO))
        return {r['race_id']: r['a_class_consumed'] for r in cur.fetchall()}


def main():
    logger.info(f"Phase B 特徴量生成 開始: {DATE_FROM} 〜 {DATE_TO}")

    rows = fetch_rows()
    logger.info(f"races 取得: {len(rows)} 行")

    race_ids = [r['race_id'] for r in rows]
    boats_by_race = fetch_boats(race_ids)
    logger.info(f"boats 取得: {sum(len(v) for v in boats_by_race.values())} 行")

    a_class_map = fetch_a_class_consumed()
    logger.info(f"a_class_consumed 計算: {len(a_class_map)} 行")

    records = []
    for r in rows:
        race_id = r['race_id']
        records.append({
            'race_id': race_id,
            'race_date': r['race_date'],
            'venue_id': r['venue_id'],
            'race_number': r['race_number'],
            'race_category': classify_race_category(r.get('subtitle')),
            'is_planned': detect_planned_race(r.get('title')),
            'boat1_skill_gap': compute_skill_gap(boats_by_race.get(race_id, [])),
            'a_class_consumed': a_class_map.get(race_id),
            'day_in_meeting': parse_day_label(r.get('day_label')),
            'day_label_raw': r.get('day_label'),
        })

    df = pd.DataFrame(records).set_index('race_id')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)
    logger.info(f"保存: {OUT_PATH} ({len(df)} 行 × {len(df.columns)} 列)")


if __name__ == '__main__':
    main()
