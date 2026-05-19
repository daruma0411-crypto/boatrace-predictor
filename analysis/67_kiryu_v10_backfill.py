"""桐生 V10 NN 予測の backfill (Step 67-1)

戸田類似度 No.1 の桐生 (venue 1) の V10 NN 予測を生成、pkl 保存。
次の 68 で 戸田+桐生 LightGBM 再訓練 → 戸田 hold-out 評価。

出力: analysis/kiryu_v10_predictions.pkl
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
OUT_PATH = ROOT / 'analysis' / 'kiryu_v10_predictions.pkl'
VENUE_ID = 1  # 桐生


def fetch_races(venue_id):
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id AS race_id, race_date, race_number,
               wind_speed, wind_direction, temperature, wave_height, water_temperature,
               result_1st, result_2nd, result_3rd, payout_sanrentan, actual_result_trifecta
        FROM races
        WHERE venue_id = %s AND result_1st IS NOT NULL
        ORDER BY race_date, race_number
    """, (venue_id,))
    races = [dict(r) for r in cur.fetchall()]
    race_ids = [r['race_id'] for r in races]
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


def main():
    logger.info(f"venue={VENUE_ID} V10 NN backfill")
    races = fetch_races(VENUE_ID)
    logger.info(f"races: {len(races)}")

    predictor = RealtimePredictor(model_path='models/boatrace_model.pth')
    predictions = {}
    ok = 0
    for i, r in enumerate(races):
        if i % 500 == 0:
            logger.info(f"progress {i}/{len(races)}")
        race_data = {
            'wind_speed': r['wind_speed'], 'wind_direction': r['wind_direction'],
            'temperature': r['temperature'], 'wave_height': r['wave_height'],
            'water_temperature': r['water_temperature'],
            'venue_id': VENUE_ID, 'race_number': r['race_number'],
        }
        try:
            pred = predictor.predict(race_data, r['boats'])
            predictions[r['race_id']] = {
                'race_id': r['race_id'],
                'race_date': r['race_date'].isoformat(),
                'race_number': r['race_number'],
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
            ok += 1
        except Exception:
            pass
    logger.info(f"完了 ok={ok}")
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(predictions, f)
    logger.info(f"保存: {OUT_PATH}")


if __name__ == '__main__':
    main()
