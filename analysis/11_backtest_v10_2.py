"""V10.1 vs V10 公平 retrospective backtest

V10.1 の NN 予測確率を使って、V10 と完全同じフィルタ・Kelly ロジックで
評価する。V10 の実bets を真の universe として、V10.1 が各bet に対して
同等以上の判断を下すかを検証。

手法:
  1. V10.1 モデルロード (analysis/models_v11/v10_1/)
  2. V10 が val 期間にbetした 80件を取得
  3. 各bet について V10.1 の prob を predict
     - FeatureEngineer (V4) で特徴量構築
     - scaler で正規化
     - NN forward で logits → softmax で probs_1st/2nd/3rd
  4. V10と同じ trifecta prob (独立) で EV 計算
     EV = prob × bets.odds  (V10は discounted_odds=odds*0.92だがここでは簡略)
  5. V10 filter 適用:
     - EV ∈ [0.5, 0.8]
     - odds ∈ [5, 40]
     - Kelly > 0
     - Kelly stake computation
  6. V10 と Σ profit 比較

V10 実績 ROI 124.5% と直接比較。
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn.functional as F
import psycopg2
from psycopg2.extras import RealDictCursor

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection
from src.predictor import _load_calibrators, _apply_calibrators

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_2_DIR = Path(__file__).parent / "models_v11" / "v10_2"
MODEL_PATH = V10_2_DIR / "boatrace_model_v10_2.pth"
SCALER_PATH = V10_2_DIR / "feature_scaler_v10_2.pkl"
REPORT_DIR = Path(__file__).parent / "reports"

# V10 フィルタ設定（betting_config.json mc_early_race と同一）
EV_MIN = 0.5
EV_MAX = 0.8
ODDS_MIN = 5.0
ODDS_MAX = 40.0
ODDS_DISCOUNT = 0.92  # V10 の discount
KELLY_FRAC = 0.0625
MAX_TICKET_RATIO = 0.008
BANKROLL = 200000


def load_v10_1():
    """V10.1 モデルをロード + V10 の calibrators.pkl を流用"""
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'],
        hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'],
        dropout=state['dropout'],
    )
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # V10.2 専用 calibrator（12_calibrate_v10_2.py で fit 済み）
    cal_path = V10_2_DIR / "calibrators_v10_2.pkl"
    if cal_path.exists():
        with open(cal_path, 'rb') as f:
            calibrators = pickle.load(f)
        cal_status = f"V10.2専用 (n={calibrators.get('n_samples', '?')})"
    else:
        calibrators = _load_calibrators()
        cal_status = "V10流用 (V10.2用が未fit)"
    logger.info(f"V10.2 ロード: epoch={state['metadata']['epoch']} "
                f"val_acc_1st={state['metadata']['val_acc_1st']:.1f}% "
                f"calibrator={cal_status}")
    return model, scaler, calibrators


def predict_race_probs(model, scaler, calibrators, race, boats):
    """1レースの全トリフェクタ prob を予測（calibrator適用後）"""
    fe = FeatureEngineer()
    race_data = {
        'venue_id': race['venue_id'],
        'month': race['race_date'].month,
        'distance': 1800,
        'wind_speed': race.get('wind_speed') or 0,
        'wind_direction': race.get('wind_direction') or 'calm',
        'temperature': race.get('temperature') or 20,
        'wave_height': race.get('wave_height') or 0,
        'water_temperature': race.get('water_temperature') or 20,
    }
    try:
        features = fe.transform(race_data, boats)
    except Exception:
        return None
    features = scaler.transform(features.reshape(1, -1))
    X = torch.FloatTensor(features)
    with torch.no_grad():
        out = model(X)
    p1 = F.softmax(out[0], dim=1).numpy()[0]
    p2 = F.softmax(out[1], dim=1).numpy()[0]
    p3 = F.softmax(out[2], dim=1).numpy()[0]
    if calibrators is not None:
        p1 = _apply_calibrators(p1, calibrators['1st'])
        p2 = _apply_calibrators(p2, calibrators['2nd'])
        p3 = _apply_calibrators(p3, calibrators['3rd'])
    return p1, p2, p3


def kelly_stake(prob, odds, bankroll, kelly_frac=KELLY_FRAC):
    """V10と同じ Kelly sizing"""
    if odds <= 1 or prob <= 0 or prob >= 1:
        return 0
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    if f_star <= 0:
        return 0
    raw = f_star * kelly_frac * bankroll
    max_ticket = MAX_TICKET_RATIO * bankroll
    raw = min(raw, max_ticket)
    stake = max(100, int(raw / 100) * 100)
    return stake


def evaluate_v10_1():
    """V10 の実bets に対し V10.1 が同等判断をしたらどうなるか"""
    model, scaler, calibrators = load_v10_1()

    conn = psycopg2.connect(os.environ['DATABASE_URL'],
                            cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # V10.1 が見てない val 期間を取得（学習 80%の外側）
    # 学習データは race_date ASC で 6253件、80%で split → 残り20%
    cur.execute("""
        SELECT r.id, r.venue_id, r.race_number, r.race_date,
               r.wind_speed, r.wind_direction, r.temperature,
               r.wave_height, r.water_temperature,
               r.actual_result_trifecta
        FROM races r
        WHERE r.is_finished = true
          AND r.actual_result_trifecta IS NOT NULL
          AND r.result_1st IS NOT NULL
          AND r.wind_speed IS NOT NULL
        ORDER BY r.race_date ASC, r.id ASC
    """)
    all_races = cur.fetchall()
    n = len(all_races)
    split = int(n * 0.8)
    val_race_ids = set(r['id'] for r in all_races[split:])
    logger.info(f"val 期間: {len(val_race_ids)}レース")

    # V10 の実bets
    cur.execute("""
        SELECT b.id, b.race_id, b.combination, b.amount,
               b.odds, b.expected_value, b.is_hit, b.payout
        FROM bets b
        WHERE b.strategy_type = 'mc_early_race'
          AND b.race_id = ANY(%s)
          AND b.result IS NOT NULL
    """, (list(val_race_ids),))
    v10_bets = [dict(b) for b in cur.fetchall()]
    logger.info(f"V10 bets (val期間): {len(v10_bets)}件")

    # 必要な races + boats を一括取得
    race_ids_for_bets = [b['race_id'] for b in v10_bets]
    cur.execute("""
        SELECT r.id, r.venue_id, r.race_number, r.race_date,
               r.wind_speed, r.wind_direction, r.temperature,
               r.wave_height, r.water_temperature
        FROM races r WHERE r.id = ANY(%s)
    """, (race_ids_for_bets,))
    races_map = {r['id']: dict(r) for r in cur.fetchall()}

    cur.execute("""
        SELECT race_id, boat_number, player_class,
               win_rate, win_rate_2, win_rate_3,
               local_win_rate, local_win_rate_2,
               avg_st, motor_win_rate_2, motor_win_rate_3,
               boat_win_rate_2, weight, exhibition_time,
               approach_course, is_new_motor,
               tilt, parts_changed
        FROM boats WHERE race_id = ANY(%s)
        ORDER BY race_id, boat_number
    """, (race_ids_for_bets,))
    boats_map = {}
    for b in cur.fetchall():
        boats_map.setdefault(b['race_id'], []).append(dict(b))
    conn.close()

    # V10.1 評価
    probs_cache = {}
    stats = {
        'considered': 0, 'accepted': 0,
        'reject_ev_low': 0, 'reject_ev_high': 0,
        'reject_odds': 0, 'reject_kelly': 0, 'reject_no_features': 0,
        'invest': 0, 'payout': 0, 'hits': 0,
    }

    for bet in v10_bets:
        rid = bet['race_id']
        stats['considered'] += 1

        # predict once per race
        if rid not in probs_cache:
            race = races_map.get(rid)
            boats = boats_map.get(rid, [])
            if not race or len(boats) != 6:
                probs_cache[rid] = None
            else:
                probs_cache[rid] = predict_race_probs(model, scaler, calibrators, race, boats)

        preds = probs_cache[rid]
        if preds is None:
            stats['reject_no_features'] += 1
            continue

        p1, p2, p3 = preds
        combo = bet['combination']
        try:
            a, b, c = [int(x) - 1 for x in combo.split('-')]
        except Exception:
            continue
        if not (0 <= a <= 5 and 0 <= b <= 5 and 0 <= c <= 5) or a in (b, c) or b == c:
            continue

        # V10.1 prob (独立確率の積、V10と同様)
        v11_prob = float(p1[a] * p2[b] * p3[c])
        raw_odds = float(bet['odds'] or 0)
        disc_odds = raw_odds * ODDS_DISCOUNT
        ev = v11_prob * disc_odds

        # Filters
        if ev < EV_MIN:
            stats['reject_ev_low'] += 1
            continue
        if ev > EV_MAX:
            stats['reject_ev_high'] += 1
            continue
        if not (ODDS_MIN <= raw_odds <= ODDS_MAX):
            stats['reject_odds'] += 1
            continue

        stake = kelly_stake(v11_prob, disc_odds, BANKROLL)
        if stake < 100:
            stats['reject_kelly'] += 1
            continue

        stats['accepted'] += 1
        stats['invest'] += stake
        if bet['is_hit']:
            # payout: raw_odds × stake (100円単位の配当なので rawで掛け算)
            stats['payout'] += int(raw_odds * stake)
            stats['hits'] += 1

    stats['roi'] = stats['payout'] / stats['invest'] if stats['invest'] else 0
    stats['hit_rate'] = stats['hits'] / stats['accepted'] if stats['accepted'] else 0
    stats['profit'] = stats['payout'] - stats['invest']

    # V10 実績
    v10_invest = sum(int(b['amount']) for b in v10_bets)
    v10_payout = sum(int(b['payout'] or 0) for b in v10_bets if b['is_hit'])
    v10_hits = sum(1 for b in v10_bets if b['is_hit'])
    v10_roi = v10_payout / v10_invest if v10_invest else 0

    logger.info("")
    logger.info("=== V10 実績 ===")
    logger.info(f"  bets={len(v10_bets)} hits={v10_hits} "
                f"投資={v10_invest:,} 回収={v10_payout:,} ROI={v10_roi*100:.1f}%")
    logger.info("")
    logger.info("=== V10.1 評価 ===")
    logger.info(f"  考慮bets: {stats['considered']}")
    logger.info(f"  採択:     {stats['accepted']}")
    logger.info(f"  EV<0.5:   {stats['reject_ev_low']}")
    logger.info(f"  EV>0.8:   {stats['reject_ev_high']}")
    logger.info(f"  odds範囲外: {stats['reject_odds']}")
    logger.info(f"  Kelly<=0: {stats['reject_kelly']}")
    logger.info(f"  特徴量失敗: {stats['reject_no_features']}")
    logger.info("")
    logger.info(f"  投資: ¥{stats['invest']:,}")
    logger.info(f"  回収: ¥{stats['payout']:,}")
    logger.info(f"  的中: {stats['hits']}/{stats['accepted']} "
                f"({stats['hit_rate']*100:.1f}%)")
    logger.info(f"  **ROI: {stats['roi']*100:.1f}%**")
    logger.info(f"  損益: {stats['profit']:+,}")
    logger.info("")
    if v10_roi > 0:
        diff = (stats['roi'] - v10_roi) * 100
        if diff > 0:
            logger.info(f"🏆 V10超え: +{diff:.1f}pt")
        else:
            logger.info(f"❌ V10未満: {diff:.1f}pt")

    # レポート保存
    report = {
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'v10_actual': {
            'bets': len(v10_bets), 'hits': v10_hits,
            'invest': v10_invest, 'payout': v10_payout, 'roi': v10_roi,
        },
        'v10_1_filtered': stats,
    }
    out_path = REPORT_DIR / "11_v10_2_backtest.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nレポート: {out_path}")


if __name__ == '__main__':
    evaluate_v10_1()
