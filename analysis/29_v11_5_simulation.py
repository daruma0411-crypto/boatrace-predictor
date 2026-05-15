"""V11.5 ミニシミュレーション (Phase C 本実装前の判定用)

目的: V10 prob + Phase B 6 特徴量を LightGBM (default) に渡し、
       hold-out 2026-04 で 1号艇 1着 binary 予測精度を V10 単独と比較。

入力:
  - models/boatrace_model.pth (V10)
  - models/feature_scaler.pkl
  - models/calibrators.pkl (V10 補正用)
  - analysis/features_phase_b.pkl

出力:
  - analysis/reports/v11_5_simulation.md (AUC/Brier/SHAP)

Train: 2026-02-01〜02-29 / Val: 2026-03 / Test: 2026-04
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
import pandas as pd
import torch
import torch.nn.functional as F
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "boatrace_model.pth"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"
CAL_PATH = ROOT / "models" / "calibrators.pkl"
PHASE_B_PATH = ROOT / "analysis" / "features_phase_b.pkl"
REPORT_PATH = ROOT / "analysis" / "reports" / "v11_5_simulation.md"


def load_v10():
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(CAL_PATH, 'rb') as f:
        cal = pickle.load(f)
    return model, scaler, cal


def fetch_races_and_boats(date_from, date_to):
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date, r.race_number,
                   r.result_1st, r.wind_speed, r.wind_direction, r.temperature,
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


def compute_v10_probs(races, boats_by, model, scaler, fe, cal):
    """V10 推論 + キャリブレーターで補正された prob_1st_boat1 を取得"""
    results = []
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
        probs_1 = F.softmax(out[0], dim=1).numpy()[0]
        # V10 raw P(boat1 1着)
        raw_p1 = float(probs_1[0])
        # 旧 calibrator で補正
        cal_p1 = float(cal['1st'][0].predict(np.array([raw_p1]))[0])
        results.append({
            'race_id': race['id'],
            'race_date': race['race_date'],
            'v10_prob_boat1_raw': raw_p1,
            'v10_prob_boat1_cal': cal_p1,
            'label': 1 if race['result_1st'] == 1 else 0,
        })
        if (i + 1) % 1000 == 0:
            logger.info(f"  V10 推論 {i+1}/{len(races)}")
    return pd.DataFrame(results)


def encode_features(df):
    """Phase B 特徴量を LightGBM に渡せる数値表現に変換"""
    df = df.copy()
    # race_category を label encoding (LightGBM categorical で扱う)
    cat_map = {'qualifier': 0, 'semifinal': 1, 'final': 2, 'general': 3, 'other': 4}
    df['race_category_enc'] = df['race_category'].map(cat_map).fillna(4).astype(int)
    df['is_planned_int'] = df['is_planned'].astype(int)
    # NaN は LightGBM のネイティブ NaN 対応で処理
    return df


def split_train_val_test(df):
    # 2026-02 は DB 未登録 (is_finished=false) のため使用不可
    # 2026-03 を train+val に分割、2026-04 を hold-out test
    rd = pd.to_datetime(df['race_date'])
    train = df[(rd >= '2026-03-01') & (rd <= '2026-03-22')]
    val = df[(rd >= '2026-03-23') & (rd <= '2026-03-31')]
    test = df[(rd >= '2026-04-01') & (rd <= '2026-04-30')]
    return train, val, test


FEATURE_COLS = [
    'v10_prob_boat1_raw',
    'race_category_enc',
    'is_planned_int',
    'boat1_skill_gap',
    'a_class_consumed',
    'day_in_meeting',
]


def train_lgb(train_df, val_df):
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['label'].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df['label'].values
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, categorical_feature=['race_category_enc'])
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, categorical_feature=['race_category_enc'], reference=train_ds)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
    }
    model = lgb.train(
        params, train_ds,
        num_boost_round=500,
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    return model


def evaluate(test_df, lgb_model):
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['label'].values
    # V10 baseline (calibrated)
    p_v10 = test_df['v10_prob_boat1_cal'].values
    # V11.5 (LightGBM)
    p_v11_5 = lgb_model.predict(X_test)
    return {
        'n_test': int(len(y_test)),
        'base_rate': float(y_test.mean()),
        'v10_auc': float(roc_auc_score(y_test, p_v10)),
        'v10_brier': float(brier_score_loss(y_test, p_v10)),
        'v11_5_auc': float(roc_auc_score(y_test, p_v11_5)),
        'v11_5_brier': float(brier_score_loss(y_test, p_v11_5)),
    }


def shap_top5(lgb_model, test_df):
    X_test = test_df[FEATURE_COLS].values
    # sample to speed up SHAP
    sample = np.random.RandomState(42).choice(len(X_test), size=min(500, len(X_test)), replace=False)
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test[sample])
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: -x[1])
    return ranked


def write_report(metrics, shap_ranked, train_n, val_n, test_n):
    lines = []
    lines.append("# V11.5 ミニシミュレーションレポート\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n")
    lines.append(f"目的: Phase C 本実装前の判定 (V10 + Phase B 特徴量 stacking)\n\n")
    lines.append(f"- Train: 2026-03-01 〜 2026-03-22 ({train_n} races) — 2026-02 は DB に結果未登録のため除外\n")
    lines.append(f"- Val:   2026-03-23 〜 2026-03-31 ({val_n} races)\n")
    lines.append(f"- Test:  2026-04 ({test_n} races)\n")
    lines.append(f"- 1号艇 1着 base rate (test): {metrics['base_rate']:.4f}\n\n")
    lines.append("## メトリクス (hold-out 2026-04)\n\n")
    lines.append("| 指標 | V10 (calibrated) | V11.5 (mini) | 差 |\n|---|---|---|---|\n")
    lines.append(f"| AUC | {metrics['v10_auc']:.4f} | {metrics['v11_5_auc']:.4f} | {metrics['v11_5_auc']-metrics['v10_auc']:+.4f} |\n")
    lines.append(f"| Brier Score (低いほど良) | {metrics['v10_brier']:.4f} | {metrics['v11_5_brier']:.4f} | {metrics['v11_5_brier']-metrics['v10_brier']:+.4f} |\n")
    lines.append("\n## SHAP TOP 5 特徴量寄与度 (平均 |shap|)\n\n")
    lines.append("| rank | feature | mean |shap| |\n|---|---|---|\n")
    for i, (feat, val) in enumerate(shap_ranked[:5], 1):
        lines.append(f"| {i} | `{feat}` | {val:.4f} |\n")
    lines.append("\n## 判定\n\n")
    auc_diff = metrics['v11_5_auc'] - metrics['v10_auc']
    if auc_diff > 0.005:
        verdict = "✅ Phase C 本実装に進める価値あり"
    elif auc_diff > 0:
        verdict = "🟡 微改善、Phase C 本実装は要検討 (パラメータ調整で伸びる可能性あり)"
    else:
        verdict = "❌ V10 を上回らず、Phase B 価値疑わしい。撤退検討"
    lines.append(f"AUC 差分 {auc_diff:+.4f} → **{verdict}**\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


def main():
    logger.info("=== V11.5 ミニシミュレーション ===")
    model, scaler, cal = load_v10()
    fe = FeatureEngineer()

    logger.info("races + boats 取得 (2026-02-01 〜 2026-04-30)")
    races, boats_by = fetch_races_and_boats('2026-02-01', '2026-04-30')
    logger.info(f"races: {len(races)} 件")

    v10_df = compute_v10_probs(races, boats_by, model, scaler, fe, cal)
    logger.info(f"V10 推論完了: {len(v10_df)} 件")

    phase_b = pd.read_pickle(PHASE_B_PATH)
    logger.info(f"Phase B pkl: {len(phase_b)} 件")

    # JOIN on race_id
    v10_df = v10_df.set_index('race_id')
    merged = v10_df.join(phase_b[['race_category', 'is_planned', 'boat1_skill_gap',
                                    'a_class_consumed', 'day_in_meeting']], how='inner')
    merged = merged.reset_index()
    logger.info(f"JOIN 後: {len(merged)} 件")

    merged = encode_features(merged)
    train_df, val_df, test_df = split_train_val_test(merged)
    logger.info(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    lgb_model = train_lgb(train_df, val_df)
    metrics = evaluate(test_df, lgb_model)
    shap_ranked = shap_top5(lgb_model, test_df)

    write_report(metrics, shap_ranked, len(train_df), len(val_df), len(test_df))
    logger.info("=== 完了 ===")
    logger.info(f"V10  AUC={metrics['v10_auc']:.4f} Brier={metrics['v10_brier']:.4f}")
    logger.info(f"V11.5 AUC={metrics['v11_5_auc']:.4f} Brier={metrics['v11_5_brier']:.4f}")
    logger.info(f"SHAP TOP3: {shap_ranked[:3]}")


if __name__ == '__main__':
    main()
