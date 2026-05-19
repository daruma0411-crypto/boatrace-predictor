"""全 24 会場 V10 NN 予測 backfill (Phase A)

各会場の V10 NN 予測を生成、pkl 保存。
Phase B (24 specialists 訓練) の前提。

出力: analysis/venue_v10_predictions.pkl (dict[venue_id -> {race_id -> pred}])
"""
import os
import sys
import pickle
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import psycopg2
from psycopg2.extras import RealDictCursor

from src.predictor import RealtimePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
TODA_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
KIRYU_PATH = ROOT / 'analysis' / 'kiryu_v10_predictions.pkl'


def fetch_races(venue_id):
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id AS race_id, race_date, race_number,
               wind_speed, wind_direction, temperature, wave_height, water_temperature,
               result_1st, result_2nd, result_3rd, payout_sanrentan, actual_result_trifecta
        FROM races
        WHERE venue_id = %s AND result_1st IS NOT NULL
          AND race_date >= '2025-06-01'
        ORDER BY race_date, race_number
    """, (venue_id,))
    races = [dict(r) for r in cur.fetchall()]
    race_ids = [r['race_id'] for r in races]
    if not race_ids:
        conn.close()
        return []
    boats_map = {}
    CHUNK = 1000
    for i in range(0, len(race_ids), CHUNK):
        chunk = race_ids[i:i+CHUNK]
        cur.execute("""
            SELECT race_id, boat_number, player_class, player_name, player_id,
                   win_rate, win_rate_2, win_rate_3, local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3, boat_win_rate_2,
                   exhibition_time, approach_course, is_new_motor, tilt,
                   parts_changed, weight
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (chunk,))
        for r in cur.fetchall():
            boats_map.setdefault(r['race_id'], []).append(dict(r))
    conn.close()
    valid = []
    for r in races:
        boats = sorted(boats_map.get(r['race_id'], []), key=lambda x: x['boat_number'])
        if len(boats) != 6:
            continue
        r['boats'] = boats
        valid.append(r)
    return valid


def predict_venue(venue_id, predictor):
    races = fetch_races(venue_id)
    predictions = {}
    for r in races:
        race_data = {
            'wind_speed': r['wind_speed'], 'wind_direction': r['wind_direction'],
            'temperature': r['temperature'], 'wave_height': r['wave_height'],
            'water_temperature': r['water_temperature'],
            'venue_id': venue_id, 'race_number': r['race_number'],
        }
        try:
            pred = predictor.predict(race_data, r['boats'])
            predictions[r['race_id']] = {
                'race_id': r['race_id'],
                'race_date': r['race_date'].isoformat(),
                'race_number': r['race_number'],
                'venue_id': venue_id,
                'probs_1st': pred['probs_1st'],
                'probs_2nd': pred['probs_2nd'],
                'probs_3rd': pred['probs_3rd'],
                'result_1st': r['result_1st'],
                'result_2nd': r['result_2nd'],
                'result_3rd': r['result_3rd'],
                'actual': r['actual_result_trifecta'] or f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}",
                'payout': r['payout_sanrentan'],
                'boats': r['boats'],
                'race_data': race_data,
            }
        except Exception:
            pass
    return predictions


def main():
    logger.info("全 24 会場 V10 backfill")
    venue_preds = {}

    # 既存 pkl 流用
    if TODA_PATH.exists():
        venue_preds[2] = pickle.load(open(TODA_PATH, 'rb'))
        logger.info(f"venue 2 (戸田): {len(venue_preds[2])} (既存 pkl)")
    if KIRYU_PATH.exists():
        venue_preds[1] = pickle.load(open(KIRYU_PATH, 'rb'))
        logger.info(f"venue 1 (桐生): {len(venue_preds[1])} (既存 pkl)")

    predictor = RealtimePredictor(model_path='models/boatrace_model.pth')

    for vid in range(1, 25):
        if vid in venue_preds:
            continue
        logger.info(f"venue {vid} 開始")
        preds = predict_venue(vid, predictor)
        venue_preds[vid] = preds
        logger.info(f"venue {vid}: {len(preds)} races")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(venue_preds, f)
    total = sum(len(v) for v in venue_preds.values())
    logger.info(f"保存: {OUT_PATH} (合計 {total} races)")


if __name__ == '__main__':
    main()
