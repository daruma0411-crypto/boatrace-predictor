"""pkl 4本 (racelist/result/odds_3t/beforeinfo) から V11 fine-tune 用学習データを抽出

V10 と同じ FeatureEngineer.transform で 76次元特徴量を生成。
DB を一切触らない READ-ONLY スクリプト。

入力: analysis/historical_data/{year}_{month:02d}/  (merge_historical_shards.py 統合済)
出力: analysis/models_v11/v10_april_finetune/train_data_april.pkl

使い方:
  python analysis/extract_training_data_from_pkl.py --years 2024,2025 --month 4
"""
import os
import sys
import math
import pickle
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent / 'historical_data'
OUT_DIR = Path(__file__).parent / 'models_v11' / 'v10_april_finetune'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10_finetune_v10_2.py と同じ Miss Analysis mild preset
WEAK_VENUES = [1, 2, 3, 4, 5, 6]
STRONG_VENUES = [9, 10, 12, 20, 23]
WEIGHT_PRESET = {
    'weak_venue': 0.7, 'r1': 0.5, 'weak_wind': 1.2,
    'strong_venue': 2.0, 'strong_wind': 1.5,
}


def compute_sample_weight(venue_id, race_number, wind_speed, april_boost=2.0):
    """Miss Analysis weight + 4月データには april_boost を掛ける"""
    w = 1.0
    p = WEIGHT_PRESET
    if venue_id in WEAK_VENUES:
        w *= p['weak_venue']
    elif venue_id in STRONG_VENUES:
        w *= p['strong_venue']
    if race_number == 1:
        w *= p['r1']
    if wind_speed is not None:
        if 3 <= wind_speed <= 5:
            w *= p['strong_wind']
        elif 0 <= wind_speed <= 2:
            w *= p['weak_wind']
    return max(w, 0.1) * april_boost


def merge_boats(rl_boats, bi_boats):
    """racelist の boats と beforeinfo の boats をマージ"""
    bi_by_num = {b['boat_number']: b for b in bi_boats} if bi_boats else {}
    merged = []
    for b in rl_boats:
        bn = b['boat_number']
        bi_b = bi_by_num.get(bn, {})
        m = dict(b)
        if bi_b.get('weight'):
            m['weight'] = bi_b['weight']
        m['exhibition_time'] = bi_b.get('exhibition_time') or 6.8
        m['tilt'] = bi_b.get('tilt') if bi_b.get('tilt') is not None else 0.0
        m['approach_course'] = bi_b.get('approach_course', bn)
        m['parts_changed'] = bi_b.get('parts_changed', False)
        m['is_new_motor'] = False
        merged.append(m)
    return merged


def load_year_month(year: int, month: int):
    """指定 year_month の統合 pkl 4本を読み込む"""
    base = DATA_ROOT / f'{year}_{month:02d}'
    if not base.exists():
        raise FileNotFoundError(f'統合ディレクトリが無い: {base} (先に merge_historical_shards.py を実行)')

    racelist = pickle.load(open(base / 'racelist.pkl', 'rb'))
    result = pickle.load(open(base / 'result.pkl', 'rb'))
    bi_path = base / 'beforeinfo.pkl'
    beforeinfo = pickle.load(open(bi_path, 'rb')) if bi_path.exists() else []

    res_by = {(r['race_date'], r['venue_id'], r['race_number']): r for r in result}
    bi_by = {(r['race_date'], r['venue_id'], r['race_number']): r for r in beforeinfo}

    logger.info(
        f'{year}-{month:02d}: racelist={len(racelist)} result={len(res_by)} '
        f'beforeinfo={len(bi_by)}'
    )
    return racelist, res_by, bi_by


def main(years: list[int], month: int):
    fe = FeatureEngineer()
    X_rows, y1, y2, y3, weights, race_keys, race_dates = [], [], [], [], [], [], []
    n_skip_no_result, n_skip_no_bi, n_skip_fe_error, n_skip_invalid = 0, 0, 0, 0

    for year in years:
        racelist, res_by, bi_by = load_year_month(year, month)

        for rl in racelist:
            key = (rl['race_date'], rl['venue_id'], rl['race_number'])
            res = res_by.get(key)
            if not res:
                n_skip_no_result += 1
                continue
            bi = bi_by.get(key)
            if not bi:
                n_skip_no_bi += 1
                continue

            weather = bi.get('weather') or {}
            wind = weather.get('wind_speed')
            try:
                wind = float(wind) if wind is not None else 0
            except (TypeError, ValueError):
                wind = 0

            rd = {
                'venue_id': rl['venue_id'],
                'month': month,
                'distance': 1800,
                'wind_speed': wind,
                'wind_direction': weather.get('wind_direction') or 'calm',
                'temperature': weather.get('temperature') or 20,
                'wave_height': weather.get('wave_height') or 0,
                'water_temperature': weather.get('water_temperature') or 20,
            }
            boats = merge_boats(rl['boats'], bi.get('boats', []))
            if len(boats) != 6:
                n_skip_invalid += 1
                continue

            try:
                f = fe.transform(rd, boats)
            except Exception as e:
                n_skip_fe_error += 1
                if n_skip_fe_error <= 3:
                    logger.debug(f'transform error: {e}')
                continue

            try:
                b1 = int(res['result_1st']) - 1
                b2 = int(res['result_2nd']) - 1
                b3 = int(res['result_3rd']) - 1
            except (TypeError, ValueError):
                n_skip_invalid += 1
                continue
            if not (0 <= b1 <= 5 and 0 <= b2 <= 5 and 0 <= b3 <= 5):
                n_skip_invalid += 1
                continue

            X_rows.append(f)
            y1.append(b1)
            y2.append(b2)
            y3.append(b3)
            weights.append(compute_sample_weight(rl['venue_id'], rl['race_number'], wind))
            race_keys.append(f'{key[0]}_{key[1]:02d}_{key[2]:02d}')
            race_dates.append(key[0])

    X = np.array(X_rows, dtype=np.float32)
    y1 = np.array(y1, dtype=np.int64)
    y2 = np.array(y2, dtype=np.int64)
    y3 = np.array(y3, dtype=np.int64)
    wts = np.array(weights, dtype=np.float32)

    logger.info(f'=== 抽出完了 ===')
    logger.info(f'有効: {len(X)} races / {X.shape[1] if len(X) else 0}dim')
    logger.info(f'skip: no_result={n_skip_no_result} no_bi={n_skip_no_bi} '
                f'fe_error={n_skip_fe_error} invalid={n_skip_invalid}')
    logger.info(f'weight: min={wts.min():.3f} max={wts.max():.3f} mean={wts.mean():.3f}')
    if len(X):
        dist1 = np.bincount(y1, minlength=6) / len(y1)
        logger.info(f'1着艇分布: {[f"{x*100:.1f}%" for x in dist1]}')

    out = {
        'X': X,
        'y_1st': y1,
        'y_2nd': y2,
        'y_3rd': y3,
        'weights': wts,
        'race_keys': np.array(race_keys),
        'race_dates': np.array(race_dates),
        'years': years,
        'month': month,
        'extracted_at': datetime.now().isoformat(),
    }
    out_path = OUT_DIR / f'train_data_{"_".join(str(y) for y in years)}_{month:02d}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    logger.info(f'保存: {out_path} ({out_path.stat().st_size / 1024:.1f}KB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=str, required=True,
                        help='例: 2024,2025')
    parser.add_argument('--month', type=int, default=4)
    args = parser.parse_args()
    years = [int(y) for y in args.years.split(',')]
    main(years, args.month)
