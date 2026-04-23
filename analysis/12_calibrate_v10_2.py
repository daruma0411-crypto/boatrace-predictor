"""V10.2 用の calibrator を再fit

V10.2 の predict を訓練データで全レース分走らせ、
各ポジション (1st/2nd/3rd) 6クラスごとに Isotonic Regression で
予測確率 → 実確率 のマップ関数を fit する。

V10 の calibrators.pkl の構造を踏襲:
  {'1st': [iso_0, iso_1, ..., iso_5],
   '2nd': [...],
   '3rd': [...]}

保存先: analysis/models_v11/v10_2/calibrators_v10_2.pkl
"""
import os
import sys
import pickle
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_2_DIR = Path(__file__).parent / "models_v11" / "v10_2"
MODEL_PATH = V10_2_DIR / "boatrace_model_v10_2.pth"
SCALER_PATH = V10_2_DIR / "feature_scaler_v10_2.pkl"
CAL_PATH = V10_2_DIR / "calibrators_v10_2.pkl"


def fit_calibrators():
    logger.info("V10.2 calibrator 再fit 開始")

    # モデル
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    fe = FeatureEngineer()

    # 訓練データ取得（時系列 80% を calibrator fit 用に使用）
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date, r.race_number,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.is_finished = true
              AND r.actual_result_trifecta IS NOT NULL
              AND r.result_1st IS NOT NULL
              AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date ASC, r.id ASC
        """)
        races = cur.fetchall()
        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor, tilt, parts_changed
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()

    boats_by = defaultdict(list)
    for b in all_boats:
        boats_by[b['race_id']].append(dict(b))

    # 時系列 80% = calibrator fit 用
    n = len(races)
    split = int(n * 0.8)
    fit_races = races[:split]
    logger.info(f"fit 用レース: {len(fit_races)} 件")

    # 全レースで prob 予測 + ラベル収集
    probs_1 = np.zeros((len(fit_races), 6), dtype=np.float32)
    probs_2 = np.zeros((len(fit_races), 6), dtype=np.float32)
    probs_3 = np.zeros((len(fit_races), 6), dtype=np.float32)
    labels_1, labels_2, labels_3 = [], [], []

    valid_idx = []
    for i, race in enumerate(fit_races):
        boats = boats_by.get(race['id'], [])
        if len(boats) != 6:
            continue
        rd = {'venue_id': race['venue_id'], 'month': race['race_date'].month,
              'distance': 1800,
              'wind_speed': race.get('wind_speed') or 0,
              'wind_direction': race.get('wind_direction') or 'calm',
              'temperature': race.get('temperature') or 20,
              'wave_height': race.get('wave_height') or 0,
              'water_temperature': race.get('water_temperature') or 20}
        try:
            f = fe.transform(rd, boats)
        except Exception:
            continue
        f = scaler.transform(f.reshape(1, -1))
        X = torch.FloatTensor(f)
        with torch.no_grad():
            out = model(X)
        probs_1[i] = F.softmax(out[0], dim=1).numpy()[0]
        probs_2[i] = F.softmax(out[1], dim=1).numpy()[0]
        probs_3[i] = F.softmax(out[2], dim=1).numpy()[0]
        labels_1.append(race['result_1st'] - 1)
        labels_2.append(race['result_2nd'] - 1)
        labels_3.append(race['result_3rd'] - 1)
        valid_idx.append(i)
        if len(valid_idx) % 500 == 0:
            logger.info(f"  予測 {len(valid_idx)}件...")

    valid_idx = np.array(valid_idx)
    probs_1 = probs_1[valid_idx]
    probs_2 = probs_2[valid_idx]
    probs_3 = probs_3[valid_idx]
    labels_1 = np.array(labels_1)
    labels_2 = np.array(labels_2)
    labels_3 = np.array(labels_3)
    logger.info(f"有効 fit サンプル: {len(valid_idx)}")

    # 各クラス別に Isotonic Regression で fit
    def fit_position(probs, labels, pos_name):
        calibrators = []
        for cls in range(6):
            p = probs[:, cls]
            y = (labels == cls).astype(np.float32)
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            iso.fit(p, y)
            calibrators.append(iso)
            # 診断: 予測mean と 実際mean
            pred_mean = p.mean()
            actual_mean = y.mean()
            logger.info(f"  {pos_name}[{cls+1}号艇]: pred_mean={pred_mean:.4f} "
                        f"actual_mean={actual_mean:.4f}")
        return calibrators

    logger.info("1着 calibrator fit...")
    cal_1 = fit_position(probs_1, labels_1, '1st')
    logger.info("2着 calibrator fit...")
    cal_2 = fit_position(probs_2, labels_2, '2nd')
    logger.info("3着 calibrator fit...")
    cal_3 = fit_position(probs_3, labels_3, '3rd')

    # 保存
    out = {'1st': cal_1, '2nd': cal_2, '3rd': cal_3,
           'fitted_at': datetime.now().isoformat(),
           'n_samples': len(valid_idx),
           'source': 'V10.2 (fine-tuned from V10 + Miss Analysis mild weights)'}
    with open(CAL_PATH, 'wb') as f:
        pickle.dump(out, f)
    logger.info(f"保存: {CAL_PATH}")


if __name__ == '__main__':
    fit_calibrators()
