"""V11 訓練データ抽出 + Miss Analysis ベースのサンプル重み設計

READ-ONLY: DBからSELECTのみ、production codeには一切影響しない。

出力:
  analysis/models_v11/train_data.pkl
    - X: (n_races, n_features) 特徴量行列
    - y_1st, y_2nd, y_3rd: (n_races,) 1着/2着/3着の艇番号 (0-5)
    - weights: (n_races,) Miss Analysisベースのサンプル重み
    - race_ids: (n_races,) races.id
    - feature_names: list[str]
"""
import os
import sys
import math
import pickle
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models_v11"
MODELS_DIR.mkdir(exist_ok=True)

# === Miss Analysis ベースの重み定義 ===
# 2026-04-22 分析結果より
WEAK_VENUES = [4, 3, 6, 5, 2, 1]    # 平和島/江戸川/浜名湖/多摩川/戸田/桐生 (ROI 0%)
STRONG_VENUES = [12, 23, 9]          # 住之江/唐津/津 (ROI 347-792%)
WEAK_R_NUMBER = 1                    # R1 (ROI 0%)
WEAK_WIND_RANGE = (0, 2)             # 弱風 (ROI 29%)
STRONG_WIND_RANGE = (3, 5)           # 中風 (ROI 432%)


def compute_sample_weight(race):
    """race dict から sample weight を計算

    基本重み 1.0。弱点セグメントで下げ、強みセグメントで上げる。
    ROI加重（配当ベース）も重ね、profit 最適化学習にする。
    """
    w = 1.0

    # 会場別重み
    venue_id = race.get('venue_id')
    if venue_id in WEAK_VENUES:
        w *= 0.3  # 弱点会場: 大幅に下げる（過学習抑制）
    elif venue_id in STRONG_VENUES:
        w *= 2.0  # 強み会場: 重視

    # R番号別重み
    if race.get('race_number') == WEAK_R_NUMBER:
        w *= 0.3  # R1 は弱点、学習に影響させない

    # 風速重み: 中風での予測を強化
    wind = race.get('wind_speed')
    if wind is not None:
        if STRONG_WIND_RANGE[0] <= wind <= STRONG_WIND_RANGE[1]:
            w *= 1.5  # 得意条件
        elif WEAK_WIND_RANGE[0] <= wind <= WEAK_WIND_RANGE[1]:
            w *= 1.8  # 弱風は不得意 → 学習を強化（より多く学ばせる）

    # ROI加重: 配当 log(1 + payout/1000) で荒れレース重視
    # train_custom_loss.py 流儀
    payout = race.get('payout_sanrentan') or 0
    if payout > 0:
        w *= math.log1p(float(payout) / 1000.0)

    return max(w, 0.01)  # 最小値ガード


def extract_features_per_boat(boats_row):
    """boat row から 14個の艇別特徴量を抽出"""
    return [
        boats_row.get('win_rate') or 0.0,
        boats_row.get('win_rate_2') or 0.0,
        boats_row.get('win_rate_3') or 0.0,
        boats_row.get('local_win_rate') or 0.0,
        boats_row.get('local_win_rate_2') or 0.0,
        boats_row.get('motor_win_rate_2') or 0.0,
        boats_row.get('motor_win_rate_3') or 0.0,
        boats_row.get('boat_win_rate_2') or 0.0,
        (boats_row.get('weight') or 52.0) - 52.0,
        boats_row.get('avg_st') or 0.16,
        boats_row.get('exhibition_time') or 6.8,
        {'A1': 3, 'A2': 2, 'B1': 1, 'B2': 0}.get(boats_row.get('player_class'), 1),
        float(boats_row.get('tilt') or 0.0),
        1.0 if boats_row.get('is_new_motor') else 0.0,
    ]


BOAT_FEATURE_NAMES = [
    'win_rate', 'win_rate_2', 'win_rate_3',
    'local_win_rate', 'local_win_rate_2',
    'motor_win_rate_2', 'motor_win_rate_3', 'boat_win_rate_2',
    'weight_diff', 'avg_st', 'exhibition_time',
    'class_ord', 'tilt', 'is_new_motor',
]


def build_feature_names():
    names = ['venue_id', 'race_number', 'wind_speed', 'wave_height',
             'temperature', 'water_temperature']
    for b in range(1, 7):
        for n in BOAT_FEATURE_NAMES:
            names.append(f'B{b}_{n}')
    return names


def main():
    logger.info("V11 訓練データ抽出 開始")

    conn = psycopg2.connect(os.environ['DATABASE_URL'],
                            cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # 確定レース取得 (is_finished=true + result有)
    logger.info("レース取得中...")
    cur.execute("""
        SELECT id, venue_id, race_number, race_date,
               actual_result_trifecta, payout_sanrentan,
               wind_speed, wave_height, temperature, water_temperature,
               result_1st, result_2nd, result_3rd
        FROM races
        WHERE is_finished = true
          AND actual_result_trifecta IS NOT NULL
          AND result_1st IS NOT NULL
          AND result_2nd IS NOT NULL
          AND result_3rd IS NOT NULL
        ORDER BY race_date ASC, id ASC
    """)
    races = cur.fetchall()
    logger.info(f"確定レース: {len(races)}件")

    # 各レースの boats 取得（6艇揃ってない行はスキップ）
    logger.info("艇情報取得中...")
    cur.execute("""
        SELECT race_id, boat_number,
               win_rate, win_rate_2, win_rate_3,
               local_win_rate, local_win_rate_2,
               motor_win_rate_2, motor_win_rate_3, boat_win_rate_2,
               weight, avg_st, exhibition_time, player_class,
               tilt, is_new_motor
        FROM boats
        ORDER BY race_id, boat_number
    """)
    boats_by_race = defaultdict(dict)
    for b in cur.fetchall():
        boats_by_race[b['race_id']][b['boat_number']] = b

    # 特徴量構築
    feature_names = build_feature_names()
    X_rows = []
    y_1st, y_2nd, y_3rd = [], [], []
    weights = []
    race_ids = []

    for race in races:
        rid = race['id']
        boats = boats_by_race.get(rid, {})
        if len(boats) != 6 or not all(i in boats for i in range(1, 7)):
            continue

        # レース特徴量
        row = [
            race['venue_id'] / 24.0,
            race['race_number'] / 12.0,
            (race.get('wind_speed') or 0) / 10.0,
            (race.get('wave_height') or 0) / 20.0,
            (race.get('temperature') or 20) / 40.0,
            (race.get('water_temperature') or 20) / 40.0,
        ]

        # 艇特徴量 (1-6号艇順)
        for i in range(1, 7):
            row.extend(extract_features_per_boat(boats[i]))

        # 1着/2着/3着艇番号 (0-5) 変換
        try:
            b1 = int(race['result_1st']) - 1
            b2 = int(race['result_2nd']) - 1
            b3 = int(race['result_3rd']) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= b1 <= 5 and 0 <= b2 <= 5 and 0 <= b3 <= 5):
            continue

        X_rows.append(row)
        y_1st.append(b1)
        y_2nd.append(b2)
        y_3rd.append(b3)
        weights.append(compute_sample_weight(race))
        race_ids.append(rid)

    X = np.array(X_rows, dtype=np.float32)
    y_1st = np.array(y_1st, dtype=np.int32)
    y_2nd = np.array(y_2nd, dtype=np.int32)
    y_3rd = np.array(y_3rd, dtype=np.int32)
    weights = np.array(weights, dtype=np.float32)
    race_ids = np.array(race_ids)

    logger.info(f"有効レース: {len(X)}件")
    logger.info(f"特徴量数: {X.shape[1]}")
    logger.info(f"重み統計: min={weights.min():.3f} "
                f"max={weights.max():.3f} mean={weights.mean():.3f}")

    # 1着艇番号分布 (バランス確認)
    dist_1st = np.bincount(y_1st, minlength=6) / len(y_1st)
    logger.info(f"1着艇分布: {[f'{x*100:.1f}%' for x in dist_1st]}")

    # 保存
    out = {
        'X': X,
        'y_1st': y_1st,
        'y_2nd': y_2nd,
        'y_3rd': y_3rd,
        'weights': weights,
        'race_ids': race_ids,
        'feature_names': feature_names,
        'extracted_at': datetime.now().isoformat(),
        'miss_analysis_weights': {
            'weak_venues': WEAK_VENUES,
            'strong_venues': STRONG_VENUES,
            'weak_r_number': WEAK_R_NUMBER,
            'weak_wind_range': WEAK_WIND_RANGE,
            'strong_wind_range': STRONG_WIND_RANGE,
        },
    }
    out_path = MODELS_DIR / "train_data.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    logger.info(f"保存: {out_path} ({out_path.stat().st_size / 1024:.1f}KB)")

    conn.close()


if __name__ == '__main__':
    main()
