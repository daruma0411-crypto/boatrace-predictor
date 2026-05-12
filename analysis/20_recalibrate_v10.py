"""V10 キャリブレーター再 fit (A1 of Phase A roadmap, Issue #4)

- fit 区間: 2026-02-01 〜 2026-03-31 (2 ヶ月)
- hold-out 評価: 2026-04-01 〜 2026-04-30 (1 ヶ月)
- 出力 1: models/calibrators_v2.pkl (本番非投入、shadow 検証用)
- 出力 2: analysis/reports/calibration_v2_eval.md
- 本番 models/calibrators.pkl は touch しない (旧版として読み込み比較のみ)
"""
import os
import sys
import pickle
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "boatrace_model.pth"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"
OLD_CAL_PATH = ROOT / "models" / "calibrators.pkl"
NEW_CAL_PATH = ROOT / "models" / "calibrators_v2.pkl"
EVAL_PATH = ROOT / "analysis" / "reports" / "calibration_v2_eval.md"

FIT_START = '2026-02-01'
FIT_END = '2026-03-31'
EVAL_START = '2026-04-01'
EVAL_END = '2026-04-30'


def load_model_and_scaler():
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def fetch_races(date_from, date_to):
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
              AND r.race_date BETWEEN %s AND %s
            ORDER BY r.race_date ASC, r.id ASC
        """, (date_from, date_to))
        races = cur.fetchall()
        race_ids = [r['id'] for r in races]
        if not race_ids:
            return [], {}
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
    return races, boats_by


def predict_batch(races, boats_by, model, scaler, fe):
    """V10 推論を実行し raw probability + labels を返す"""
    probs_1 = np.zeros((len(races), 6), dtype=np.float32)
    probs_2 = np.zeros((len(races), 6), dtype=np.float32)
    probs_3 = np.zeros((len(races), 6), dtype=np.float32)
    labels_1, labels_2, labels_3 = [], [], []
    valid_idx = []
    for i, race in enumerate(races):
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
            logger.info(f"  予測 {len(valid_idx)} 件...")
    valid_idx = np.array(valid_idx)
    return (probs_1[valid_idx], probs_2[valid_idx], probs_3[valid_idx],
            np.array(labels_1), np.array(labels_2), np.array(labels_3))


def fit_position(probs, labels, pos_name):
    calibrators = []
    for cls in range(6):
        p = probs[:, cls]
        y = (labels == cls).astype(np.float32)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso.fit(p, y)
        calibrators.append(iso)
        logger.info(f"  {pos_name}[{cls+1}号艇]: pred_mean={p.mean():.4f} actual_mean={y.mean():.4f}")
    return calibrators


def apply_calibrators(probs, calibrators):
    out = np.zeros_like(probs)
    for cls in range(6):
        out[:, cls] = calibrators[cls].predict(probs[:, cls])
    return out


def reliability_bins(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for cls in range(6):
        p = probs[:, cls]
        y = (labels == cls).astype(np.float32)
        row = []
        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (p >= bins[b]) & (p < bins[b + 1])
            else:
                mask = (p >= bins[b]) & (p <= bins[b + 1])
            if mask.sum() == 0:
                row.append((bins[b], bins[b+1], 0, np.nan, np.nan))
            else:
                row.append((bins[b], bins[b+1], int(mask.sum()),
                            float(p[mask].mean()), float(y[mask].mean())))
        result.append(row)
    return result


def kpi_07_08_gap(reliability):
    """0.7-0.8 bin の |pred - hit| 平均 (pt) を返す。サンプル無しは除外"""
    gaps = []
    for cls_row in reliability:
        for lo, hi, n, pred, hit in cls_row:
            if abs(lo - 0.7) < 1e-6 and n > 0 and not np.isnan(hit):
                gaps.append(abs(pred - hit) * 100)
    return float(np.mean(gaps)) if gaps else float('nan')


def fit_calibrators():
    logger.info(f"=== A1 V10 calibrator 再 fit (fit: {FIT_START} 〜 {FIT_END}) ===")
    model, scaler = load_model_and_scaler()
    fe = FeatureEngineer()
    races, boats_by = fetch_races(FIT_START, FIT_END)
    logger.info(f"fit 区間レース取得: {len(races)} 件")
    p1, p2, p3, y1, y2, y3 = predict_batch(races, boats_by, model, scaler, fe)
    logger.info(f"有効 fit サンプル: {len(p1)}")
    cal_1 = fit_position(p1, y1, '1st')
    cal_2 = fit_position(p2, y2, '2nd')
    cal_3 = fit_position(p3, y3, '3rd')
    out = {
        '1st': cal_1, '2nd': cal_2, '3rd': cal_3,
        'fitted_at': datetime.now().isoformat(),
        'n_samples': int(len(p1)),
        'fit_period': f"{FIT_START} 〜 {FIT_END}",
        'source': 'V10 (boatrace_model.pth) Phase A A1 recalibration',
    }
    with open(NEW_CAL_PATH, 'wb') as f:
        pickle.dump(out, f)
    logger.info(f"保存: {NEW_CAL_PATH}")


def write_eval_report(results, n_samples):
    lines = []
    lines.append("# V10 キャリブレーター v2 評価レポート\n\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n\n")
    lines.append(f"- fit 期間: {FIT_START} 〜 {FIT_END}\n")
    lines.append(f"- hold-out 評価期間: {EVAL_START} 〜 {EVAL_END}\n")
    lines.append(f"- 評価有効サンプル数: {n_samples}\n\n")

    def fmt(x):
        if x is None:
            return "-"
        try:
            if np.isnan(x):
                return "-"
        except TypeError:
            return "-"
        return f"{x:.3f}"

    for pos in ['1st', '2nd', '3rd']:
        lines.append(f"## {pos} 着\n\n")
        for cls in range(6):
            lines.append(f"### {cls+1}号艇\n\n")
            lines.append("| bin | n | raw pred | raw hit | 旧 pred | 旧 hit | 新 pred | 新 hit |\n|---|---|---|---|---|---|---|---|\n")
            for b in range(10):
                raw_row = results[pos]['raw'][cls][b]
                old_row = results[pos]['old'][cls][b]
                new_row = results[pos]['new'][cls][b]
                lo, hi, n_b, raw_p, raw_h = raw_row
                _, _, _, old_p, old_h = old_row
                _, _, _, new_p, new_h = new_row
                bin_label = f"{lo:.1f}-{hi:.1f}"
                lines.append(f"| {bin_label} | {n_b} | {fmt(raw_p)} | {fmt(raw_h)} | {fmt(old_p)} | {fmt(old_h)} | {fmt(new_p)} | {fmt(new_h)} |\n")
            lines.append("\n")
    lines.append("## KPI: 0.7-0.8 帯ズレ (1着 全艇平均)\n\n")
    gap_raw = kpi_07_08_gap(results['1st']['raw'])
    gap_old = kpi_07_08_gap(results['1st']['old'])
    gap_new = kpi_07_08_gap(results['1st']['new'])
    lines.append("| 版 | |pred - hit| (pt) |\n|---|---|\n")
    lines.append(f"| raw (補正なし) | {gap_raw:.2f} |\n")
    lines.append(f"| 旧 calibrators.pkl | {gap_old:.2f} |\n")
    lines.append(f"| 新 calibrators_v2.pkl | {gap_new:.2f} |\n\n")
    if not np.isnan(gap_new) and gap_new <= 3.0:
        verdict = "KPI 達成 (±3pt 以内)"
    elif not np.isnan(gap_new) and gap_new < gap_old:
        verdict = "KPI 未達 (改善はしているが ±3pt 超え)"
    else:
        verdict = "KPI 未達 (旧版より悪化または同等)"
    lines.append(f"判定: **{verdict}** (新版ズレ {gap_new:.2f} pt)\n")
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {EVAL_PATH}")


def evaluate_calibrators():
    logger.info(f"=== A1 評価 (hold-out: {EVAL_START} 〜 {EVAL_END}) ===")
    model, scaler = load_model_and_scaler()
    fe = FeatureEngineer()
    with open(OLD_CAL_PATH, 'rb') as f:
        old_cal = pickle.load(f)
    with open(NEW_CAL_PATH, 'rb') as f:
        new_cal = pickle.load(f)
    races, boats_by = fetch_races(EVAL_START, EVAL_END)
    logger.info(f"評価区間レース取得: {len(races)} 件")
    p1, p2, p3, y1, y2, y3 = predict_batch(races, boats_by, model, scaler, fe)
    logger.info(f"有効評価サンプル: {len(p1)}")
    results = {}
    for pos_name, p, y in [('1st', p1, y1), ('2nd', p2, y2), ('3rd', p3, y3)]:
        p_old = apply_calibrators(p, old_cal[pos_name])
        p_new = apply_calibrators(p, new_cal[pos_name])
        results[pos_name] = {
            'raw': reliability_bins(p, y),
            'old': reliability_bins(p_old, y),
            'new': reliability_bins(p_new, y),
        }
    write_eval_report(results, int(len(p1)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['fit', 'eval', 'all'], default='all')
    args = parser.parse_args()
    if args.step in ('fit', 'all'):
        fit_calibrators()
    if args.step in ('eval', 'all'):
        evaluate_calibrators()
