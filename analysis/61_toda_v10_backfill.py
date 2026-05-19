"""戸田 V10 NN 予測の backfill (案 X Phase 1 Step 1)

戸田 全期間 2271 races のうち、DB にある V10 予測は 437 races のみ。
残り 1834 races (主に 2025-06〜2026-02) に V10 inference 実行、pkl 保存。

これにより 戸田 long-term backtest + QMC 係数 fit が可能になる。

入力: DB races + boats (戸田 2025-06〜2026-05)
出力: analysis/toda_v10_predictions.pkl
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
OUT_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
TODA_VENUE_ID = 2


def fetch_toda_races():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id AS race_id, race_date, race_number,
               wind_speed, wind_direction, temperature, wave_height, water_temperature,
               result_1st, result_2nd, result_3rd, payout_sanrentan,
               actual_result_trifecta
        FROM races
        WHERE venue_id = %s AND result_1st IS NOT NULL
        ORDER BY race_date, race_number
    """, (TODA_VENUE_ID,))
    races = [dict(r) for r in cur.fetchall()]
    race_ids = [r['race_id'] for r in races]

    # boats を chunk fetch
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

    # attach + filter
    valid = []
    for r in races:
        boats = sorted(boats_map.get(r['race_id'], []), key=lambda x: x['boat_number'])
        if len(boats) != 6:
            continue
        r['boats'] = boats
        valid.append(r)
    return valid


def main():
    logger.info("戸田 V10 NN 予測 backfill")
    races = fetch_toda_races()
    logger.info(f"戸田 races: {len(races)} (boats あり)")

    predictor = RealtimePredictor(model_path='models/boatrace_model.pth')
    logger.info("RealtimePredictor 初期化完了")

    predictions = {}
    n_ok = 0
    n_err = 0
    for i, r in enumerate(races):
        if i % 200 == 0:
            logger.info(f"progress {i}/{len(races)}, ok={n_ok}, err={n_err}")
        race_data = {
            'wind_speed': r['wind_speed'],
            'wind_direction': r['wind_direction'],
            'temperature': r['temperature'],
            'wave_height': r['wave_height'],
            'water_temperature': r['water_temperature'],
            'venue_id': TODA_VENUE_ID,
            'race_number': r['race_number'],
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
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err < 5:
                logger.warning(f"race {r['race_id']} 失敗: {e}")
    logger.info(f"完了 ok={n_ok}, err={n_err}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(predictions, f)
    logger.info(f"保存: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
