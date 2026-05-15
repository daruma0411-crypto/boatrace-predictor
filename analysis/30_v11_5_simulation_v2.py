"""V11.5 ミニシミュレーション v2 (Phase C 本実装前の再判定)

v1 (29_*.py) では V10 prob_1st_boat1 1個のみ + Phase B 6 で AUC -0.023 でした。
v2 では:
  - V10 全 18 出力 (prob_1st/2nd/3rd × 6 艇)
  - 簡易グリッドサーチ (num_leaves × learning_rate × min_data_in_leaf)
  - 同じ Train/Val/Test split

出力: analysis/reports/v11_5_simulation_v2.md
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
REPORT_PATH = ROOT / "analysis" / "reports" / "v11_5_simulation_v2.md"


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
    """V10 推論 + 補正の全 18 出力 (raw + calibrated 含む)"""
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
        probs_2 = F.softmax(out[1], dim=1).numpy()[0]
        probs_3 = F.softmax(out[2], dim=1).numpy()[0]
        rec = {
            'race_id': race['id'],
            'race_date': race['race_date'],
            'label': 1 if race['result_1st'] == 1 else 0,
        }
        for cls in range(6):
            rec[f'v10_p1_{cls+1}'] = float(probs_1[cls])
            rec[f'v10_p2_{cls+1}'] = float(probs_2[cls])
            rec[f'v10_p3_{cls+1}'] = float(probs_3[cls])
        # 旧 calibrator 補正 (1着 boat1 のみ参考用)
        rec['v10_p1_1_cal'] = float(cal['1st'][0].predict(np.array([probs_1[0]]))[0])
        results.append(rec)
        if (i + 1) % 1000 == 0:
            logger.info(f"  V10 推論 {i+1}/{len(races)}")
    return pd.DataFrame(results)


def encode_features(df):
    df = df.copy()
    cat_map = {'qualifier': 0, 'semifinal': 1, 'final': 2, 'general': 3, 'other': 4}
    df['race_category_enc'] = df['race_category'].map(cat_map).fillna(4).astype(int)
    df['is_planned_int'] = df['is_planned'].astype(int)
    return df


def split(df):
    rd = pd.to_datetime(df['race_date'])
    train = df[(rd >= '2026-03-01') & (rd <= '2026-03-22')]
    val = df[(rd >= '2026-03-23') & (rd <= '2026-03-31')]
    test = df[(rd >= '2026-04-01') & (rd <= '2026-04-30')]
    return train, val, test


V10_COLS = [f'v10_p{p}_{c}' for p in (1, 2, 3) for c in range(1, 7)]  # 18
PHASE_B_COLS = ['race_category_enc', 'is_planned_int', 'boat1_skill_gap', 'a_class_consumed', 'day_in_meeting']
FEATURE_COLS = V10_COLS + PHASE_B_COLS


GRID = [
    {'num_leaves': 15, 'learning_rate': 0.05, 'min_data_in_leaf': 50},
    {'num_leaves': 15, 'learning_rate': 0.02, 'min_data_in_leaf': 50},
    {'num_leaves': 31, 'learning_rate': 0.05, 'min_data_in_leaf': 50},
    {'num_leaves': 31, 'learning_rate': 0.02, 'min_data_in_leaf': 30},
    {'num_leaves': 63, 'learning_rate': 0.02, 'min_data_in_leaf': 50},
    {'num_leaves': 7,  'learning_rate': 0.05, 'min_data_in_leaf': 100},
]


def train_one(params, train_df, val_df):
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['label'].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df['label'].values
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, categorical_feature=['race_category_enc'])
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, categorical_feature=['race_category_enc'], reference=train_ds)
    p = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, 'feature_pre_filter': False, **params}
    model = lgb.train(p, train_ds, num_boost_round=500,
                      valid_sets=[val_ds], valid_names=['val'],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    return model


def evaluate(test_df, lgb_model):
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['label'].values
    p_v10 = test_df['v10_p1_1_cal'].values
    p_v11_5 = lgb_model.predict(X_test)
    return {
        'n_test': int(len(y_test)),
        'base_rate': float(y_test.mean()),
        'v10_auc': float(roc_auc_score(y_test, p_v10)),
        'v10_brier': float(brier_score_loss(y_test, p_v10)),
        'v11_5_auc': float(roc_auc_score(y_test, p_v11_5)),
        'v11_5_brier': float(brier_score_loss(y_test, p_v11_5)),
    }


def grid_search(train_df, val_df, test_df):
    rng = np.random.RandomState(42)
    results = []
    for i, params in enumerate(GRID):
        logger.info(f"  grid {i+1}/{len(GRID)}: {params}")
        model = train_one(params, train_df, val_df)
        # val AUC
        X_val = val_df[FEATURE_COLS].values
        y_val = val_df['label'].values
        p_val = model.predict(X_val)
        val_auc = roc_auc_score(y_val, p_val)
        # test AUC
        metrics = evaluate(test_df, model)
        results.append({
            'params': params,
            'val_auc': val_auc,
            'test_auc': metrics['v11_5_auc'],
            'test_brier': metrics['v11_5_brier'],
            'model': model,
        })
        logger.info(f"    val_auc={val_auc:.4f} test_auc={metrics['v11_5_auc']:.4f}")
    # val AUC 最大のもの
    best = max(results, key=lambda x: x['val_auc'])
    return best, results


def shap_top5(model, test_df):
    X_test = test_df[FEATURE_COLS].values
    sample = np.random.RandomState(42).choice(len(X_test), size=min(500, len(X_test)), replace=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[sample])
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: -x[1])
    return ranked


def write_report(best, all_results, shap_ranked, train_n, val_n, test_n, v10_auc, v10_brier):
    lines = []
    lines.append("# V11.5 ミニシミュレーション v2 レポート\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n")
    lines.append("v1 で AUC -0.023 だったので、V10 全 18 prob + 小グリッドサーチで再判定\n\n")
    lines.append(f"- Train: 2026-03-01〜03-22 ({train_n}) — 2026-02 は DB 未登録\n")
    lines.append(f"- Val: 2026-03-23〜03-31 ({val_n})\n")
    lines.append(f"- Test: 2026-04 ({test_n})\n\n")

    lines.append("## グリッドサーチ結果 (test AUC、val AUC で best 選定)\n\n")
    lines.append("| params | val AUC | test AUC | test Brier |\n|---|---|---|---|\n")
    for r in all_results:
        p = r['params']
        ptxt = f"leaves={p['num_leaves']} lr={p['learning_rate']} min_data={p['min_data_in_leaf']}"
        marker = " ✅" if r is best else ""
        lines.append(f"| {ptxt}{marker} | {r['val_auc']:.4f} | {r['test_auc']:.4f} | {r['test_brier']:.4f} |\n")

    lines.append("\n## hold-out 比較 (val-best モデル)\n\n")
    lines.append("| 指標 | V10 (calibrated) | V11.5 v2 (best) | 差 |\n|---|---|---|---|\n")
    lines.append(f"| AUC | {v10_auc:.4f} | {best['test_auc']:.4f} | {best['test_auc']-v10_auc:+.4f} |\n")
    lines.append(f"| Brier (低いほど良) | {v10_brier:.4f} | {best['test_brier']:.4f} | {best['test_brier']-v10_brier:+.4f} |\n")

    lines.append("\n## SHAP TOP 5 (best モデル)\n\n")
    lines.append("| rank | feature | mean |shap| |\n|---|---|---|\n")
    for i, (f, v) in enumerate(shap_ranked[:5], 1):
        lines.append(f"| {i} | `{f}` | {v:.4f} |\n")

    lines.append("\n### Phase B 特徴量だけの順位\n\n")
    pb_only = [(f, v) for f, v in shap_ranked if f in PHASE_B_COLS]
    lines.append("| rank in PhB | feature | mean |shap| |\n|---|---|---|\n")
    for i, (f, v) in enumerate(pb_only, 1):
        lines.append(f"| {i} | `{f}` | {v:.4f} |\n")

    auc_diff = best['test_auc'] - v10_auc
    lines.append("\n## 判定\n\n")
    if auc_diff > 0.005:
        verdict = "✅ Phase C 本実装に進める価値あり"
    elif auc_diff > 0:
        verdict = "🟡 微改善、Phase C 本実装は要検討"
    else:
        verdict = "❌ V10 を上回らず、撤退検討"
    lines.append(f"AUC 差分 {auc_diff:+.4f} → **{verdict}**\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


def main():
    logger.info("=== V11.5 ミニシミュレーション v2 (V10 18 prob + grid search) ===")
    model, scaler, cal = load_v10()
    fe = FeatureEngineer()
    races, boats_by = fetch_races_and_boats('2026-03-01', '2026-04-30')
    logger.info(f"races: {len(races)} 件")
    v10_df = compute_v10_probs(races, boats_by, model, scaler, fe, cal)
    logger.info(f"V10 推論完了: {len(v10_df)} 件")
    phase_b = pd.read_pickle(PHASE_B_PATH)
    v10_df = v10_df.set_index('race_id')
    merged = v10_df.join(phase_b[['race_category', 'is_planned', 'boat1_skill_gap',
                                    'a_class_consumed', 'day_in_meeting']], how='inner').reset_index()
    merged = encode_features(merged)
    train_df, val_df, test_df = split(merged)
    logger.info(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    if len(train_df) == 0 or len(test_df) == 0:
        raise SystemExit("train or test empty")
    best, all_results = grid_search(train_df, val_df, test_df)
    logger.info(f"best params: {best['params']}, val_auc={best['val_auc']:.4f}, test_auc={best['test_auc']:.4f}")
    shap_ranked = shap_top5(best['model'], test_df)
    # V10 ベースライン (test)
    metrics_v10 = evaluate(test_df, best['model'])
    write_report(best, all_results, shap_ranked,
                 len(train_df), len(val_df), len(test_df),
                 metrics_v10['v10_auc'], metrics_v10['v10_brier'])
    logger.info("=== 完了 ===")


if __name__ == '__main__':
    main()
