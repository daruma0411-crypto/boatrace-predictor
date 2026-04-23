"""X3: 8変種を calibrator再fit + V10 universe で評価 + ランキング

各変種:
  1. variants/<name>/model.pth をロード
  2. train 80% で calibrator (Isotonic) 再fit
  3. V10 val期間bets (74件) に対して selection 評価
  4. ROI, 的中率, 採択率を記録

最終: ROI 降順でランキング出力
"""
import os
import sys
import json
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
import psycopg2
from psycopg2.extras import RealDictCursor

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.betting import _should_skip_by_top_boat, _get_dynamic_discount, VENUE_HONMEI, VENUE_ARE
from src.monte_carlo import monte_carlo_sanrentan

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).parent / "models_v11" / "v10_2_variants"
REPORT_DIR = Path(__file__).parent / "reports"

BANKROLL = 200000
KELLY_FRAC = 0.0625
MAX_TICKET_RATIO = 0.008
MIN_ODDS = 5.0
MAX_ODDS = 40.0
MIN_PROB = 0.005
MIN_BET = 100


def load_variant(name):
    vdir = OUT_ROOT / name
    state = torch.load(vdir / 'model.pth', map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(vdir / 'feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler, vdir


def predict_raw(model, scaler, race, boats):
    fe = FeatureEngineer()
    rd = {'venue_id': race['venue_id'], 'month': race['race_date'].month,
          'distance': 1800,
          'wind_speed': race.get('wind_speed') or 0,
          'wind_direction': race.get('wind_direction') or 'calm',
          'temperature': race.get('temperature') or 20,
          'wave_height': race.get('wave_height') or 0,
          'water_temperature': race.get('water_temperature') or 20}
    try:
        f = fe.transform(rd, boats)
    except Exception: return None
    f = scaler.transform(f.reshape(1, -1))
    X = torch.FloatTensor(f)
    with torch.no_grad():
        out = model(X)
    p1 = F.softmax(out[0], dim=1).numpy()[0]
    p2 = F.softmax(out[1], dim=1).numpy()[0]
    p3 = F.softmax(out[2], dim=1).numpy()[0]
    return p1, p2, p3


def apply_iso(probs, iso_list):
    out = np.array([float(iso_list[i].transform([probs[i]])[0])
                    for i in range(len(probs))])
    s = out.sum()
    return out / s if s > 0 else out


def fit_calibrator(model, scaler, train_races, boats_map):
    """訓練データ80% で Isotonic calibrator を fit"""
    logger.info(f"  calibrator fit 対象: {len(train_races)}レース")
    p1_all, p2_all, p3_all = [], [], []
    y1_all, y2_all, y3_all = [], [], []
    for race in train_races:
        boats = boats_map.get(race['id'], [])
        if len(boats) != 6: continue
        pred = predict_raw(model, scaler, race, boats)
        if pred is None: continue
        p1_all.append(pred[0])
        p2_all.append(pred[1])
        p3_all.append(pred[2])
        y1_all.append(race['result_1st'] - 1)
        y2_all.append(race['result_2nd'] - 1)
        y3_all.append(race['result_3rd'] - 1)
    p1_all = np.array(p1_all); p2_all = np.array(p2_all); p3_all = np.array(p3_all)
    y1_all = np.array(y1_all); y2_all = np.array(y2_all); y3_all = np.array(y3_all)

    def fit_pos(probs, labels):
        cals = []
        for cls in range(6):
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            iso.fit(probs[:, cls], (labels == cls).astype(np.float32))
            cals.append(iso)
        return cals

    return {'1st': fit_pos(p1_all, y1_all),
            '2nd': fit_pos(p2_all, y2_all),
            '3rd': fit_pos(p3_all, y3_all)}


def race_level_ok(probs_1st, venue_id, race_number):
    if race_number > 4: return False
    if _should_skip_by_top_boat(probs_1st): return False
    if venue_id in VENUE_HONMEI: return False
    if venue_id not in VENUE_ARE and race_number >= 7: return False
    return True


def kelly_stake(prob, odds, bankroll):
    disc = _get_dynamic_discount(odds)
    disc_odds = odds * disc
    b = disc_odds - 1.0
    if b < 0.01 or prob <= 0 or prob >= 1: return 0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0: return 0
    raw = bankroll * kelly * KELLY_FRAC
    raw = min(raw, MAX_TICKET_RATIO * bankroll)
    return max(MIN_BET, int(round(raw / 100) * 100))


def evaluate_variant(name, model, scaler, calibrators, v10_bets, val_race_map, boats_map):
    """V10 universe で 評価"""
    mc_cache = {}
    stats = {
        'considered': 0, 'accepted': 0,
        'reject_race_filter': 0, 'reject_low_prob': 0,
        'reject_odds': 0, 'reject_kelly': 0,
        'invest': 0, 'payout': 0, 'hits': 0,
    }
    for bet in v10_bets:
        rid = bet['race_id']
        stats['considered'] += 1
        race = val_race_map.get(rid)
        boats = boats_map.get(rid, [])
        if not race or len(boats) != 6:
            stats['reject_race_filter'] += 1; continue

        if rid not in mc_cache:
            pred = predict_raw(model, scaler, race, boats)
            if pred is None:
                mc_cache[rid] = None
            else:
                p1 = apply_iso(pred[0], calibrators['1st'])
                p2 = apply_iso(pred[1], calibrators['2nd'])
                p3 = apply_iso(pred[2], calibrators['3rd'])
                if not race_level_ok(p1.tolist(), race['venue_id'], race['race_number']):
                    mc_cache[rid] = None
                else:
                    try:
                        rdata = {'wind_speed': race.get('wind_speed') or 0,
                                 'wave_height': race.get('wave_height') or 0}
                        mc_cache[rid] = monte_carlo_sanrentan(
                            p1.tolist(), boats_data=boats, n_simulations=20000,
                            race_data=rdata, race_number=race['race_number'])
                    except Exception:
                        mc_cache[rid] = None

        mc_probs = mc_cache[rid]
        if mc_probs is None:
            stats['reject_race_filter'] += 1; continue

        combo = bet['combination']
        prob = mc_probs.get(combo, 0.0)
        odds = float(bet['odds'] or 0)
        if prob < MIN_PROB:
            stats['reject_low_prob'] += 1; continue
        if odds < MIN_ODDS or odds > MAX_ODDS:
            stats['reject_odds'] += 1; continue
        stake = kelly_stake(prob, odds, BANKROLL)
        if stake < MIN_BET:
            stats['reject_kelly'] += 1; continue

        stats['accepted'] += 1
        stats['invest'] += stake
        if bet['is_hit']:
            stats['hits'] += 1
            stats['payout'] += int(odds * stake)

    stats['roi'] = stats['payout'] / stats['invest'] if stats['invest'] else 0
    stats['hit_rate'] = stats['hits'] / stats['accepted'] if stats['accepted'] else 0
    stats['profit'] = stats['payout'] - stats['invest']
    return stats


def main():
    logger.info("X3: 8変種 evaluate 開始")

    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()

    cur.execute("""
        SELECT r.id, r.venue_id, r.race_number, r.race_date,
               r.result_1st, r.result_2nd, r.result_3rd,
               r.wind_speed, r.wind_direction, r.temperature,
               r.wave_height, r.water_temperature,
               r.actual_result_trifecta, r.payout_sanrentan
        FROM races r
        WHERE r.is_finished = true AND r.actual_result_trifecta IS NOT NULL
          AND r.result_1st IS NOT NULL AND r.wind_speed IS NOT NULL
        ORDER BY r.race_date ASC, r.id ASC
    """)
    all_races = [dict(r) for r in cur.fetchall()]
    n = len(all_races)
    split = int(n * 0.8)
    train_races = all_races[:split]
    val_races = all_races[split:]
    val_race_ids = [r['id'] for r in val_races]
    val_race_map = {r['id']: r for r in val_races}
    train_race_ids = [r['id'] for r in train_races]

    all_ids = train_race_ids + val_race_ids
    cur.execute("""
        SELECT race_id, boat_number, player_class,
               win_rate, win_rate_2, win_rate_3, local_win_rate, local_win_rate_2,
               avg_st, motor_win_rate_2, motor_win_rate_3, boat_win_rate_2,
               weight, exhibition_time, approach_course, is_new_motor,
               tilt, parts_changed
        FROM boats WHERE race_id = ANY(%s)
        ORDER BY race_id, boat_number
    """, (all_ids,))
    boats_map = defaultdict(list)
    for b in cur.fetchall():
        boats_map[b['race_id']].append(dict(b))

    cur.execute("""
        SELECT race_id, combination, amount, odds, is_hit, payout
        FROM bets
        WHERE strategy_type = 'mc_early_race' AND race_id = ANY(%s)
          AND result IS NOT NULL
    """, (val_race_ids,))
    v10_bets = [dict(b) for b in cur.fetchall()]
    conn.close()

    v10_invest = sum(int(b['amount']) for b in v10_bets)
    v10_payout = sum(int(b['payout'] or 0) for b in v10_bets if b['is_hit'])
    v10_hits = sum(1 for b in v10_bets if b['is_hit'])
    v10_roi = v10_payout / v10_invest if v10_invest else 0
    logger.info(f"V10 実績: bets={len(v10_bets)} ROI={v10_roi*100:.1f}% "
                f"hits={v10_hits} 投資={v10_invest:,} 回収={v10_payout:,}")

    # 変種名リスト
    variant_names = sorted([d.name for d in OUT_ROOT.iterdir() if d.is_dir()])
    logger.info(f"評価対象: {len(variant_names)}変種")

    results = []
    for name in variant_names:
        logger.info(f"\n--- {name} ---")
        try:
            model, scaler, vdir = load_variant(name)
            # calibrator fit (or load if exists)
            cal_path = vdir / 'calibrators.pkl'
            if cal_path.exists():
                with open(cal_path, 'rb') as f:
                    calibrators = pickle.load(f)
                logger.info(f"  calibrator loaded")
            else:
                calibrators = fit_calibrator(model, scaler, train_races, boats_map)
                with open(cal_path, 'wb') as f:
                    pickle.dump(calibrators, f)
                logger.info(f"  calibrator fitted")

            stats = evaluate_variant(name, model, scaler, calibrators,
                                      v10_bets, val_race_map, boats_map)
            stats['name'] = name
            results.append(stats)
            logger.info(f"  採択={stats['accepted']}/{stats['considered']} "
                        f"的中={stats['hits']}/{stats['accepted']} "
                        f"({stats['hit_rate']*100:.1f}%) "
                        f"ROI={stats['roi']*100:.1f}% 損益={stats['profit']:+,}")
        except Exception as e:
            logger.error(f"  失敗: {e}", exc_info=True)

    # ランキング
    results.sort(key=lambda x: -x['roi'])

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"V10 baseline: ROI {v10_roi*100:.1f}%")
    logger.info("=" * 80)
    logger.info(f"{'変種':<20} {'採択':>4} {'的中':>4} {'的中率':>7} {'ROI':>8} {'損益':>12}")
    logger.info("-" * 80)
    for r in results:
        v10_diff = (r['roi'] - v10_roi) * 100
        mark = '🏆' if v10_diff > 0 and r['accepted'] >= 10 else '  '
        logger.info(f"{mark}{r['name']:<18} {r['accepted']:>4} {r['hits']:>4} "
                    f"{r['hit_rate']*100:>6.1f}% "
                    f"{r['roi']*100:>6.1f}% "
                    f"{r['profit']:>+12,}")

    out = {
        'generated_at': datetime.now().isoformat(),
        'v10_actual': {'bets': len(v10_bets), 'hits': v10_hits,
                       'invest': v10_invest, 'payout': v10_payout, 'roi': v10_roi},
        'variants': results,
    }
    out_path = REPORT_DIR / "16_variants_eval.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nレポート: {out_path}")


if __name__ == '__main__':
    main()
