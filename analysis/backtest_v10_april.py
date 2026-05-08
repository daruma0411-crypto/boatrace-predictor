"""V10 April fine-tune 公平 backtest (2026年4月対象)

14_fair_backtest_v10_2.py の派生版:
- 対象: 2026-04-01 〜 2026-04-30 の V10 実bets (DB)
- モデル: analysis/models_v11/v10_april_finetune/
- calibrators: V10_april 専用が無ければ V10 流用

V10 が 2026年4月に実 bet した combo に対し、V10_april が:
  predict → MC sim → filter (race / prob / odds / Kelly) → 採択判定
真の payout (V10 実績) で ROI 計算。V10 実績 ROI と直接比較。
"""
import os
import sys
import json
import pickle
import logging
import argparse
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
import psycopg2
from psycopg2.extras import RealDictCursor

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.betting import (
    _should_skip_by_top_boat, _get_dynamic_discount,
    VENUE_HONMEI, VENUE_ARE,
)
from src.monte_carlo import monte_carlo_sanrentan
from src.predictor import _load_calibrators

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_APRIL_DIR = Path(__file__).parent / "models_v11" / "v10_april_finetune"
MODEL_PATH = V10_APRIL_DIR / "boatrace_model_v10_april.pth"
SCALER_PATH = V10_APRIL_DIR / "feature_scaler_v10_april.pkl"
CAL_PATH = V10_APRIL_DIR / "calibrators_v10_april.pkl"
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# mc_early_race 設定 (14_fair_backtest_v10_2.py と同一)
BANKROLL = 200000
KELLY_FRAC = 0.0625
MAX_TICKET_RATIO = 0.008
MIN_ODDS = 5.0
MAX_ODDS = 40.0
MIN_PROB = 0.005
MIN_BET = 100


def load_model():
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    if CAL_PATH.exists():
        with open(CAL_PATH, 'rb') as f:
            calibrators = pickle.load(f)
        cal_status = "V10_april 専用"
    else:
        calibrators = _load_calibrators()
        cal_status = "V10 流用 (専用未fit)"
    logger.info(
        f"モデルロード: epoch={state['metadata'].get('epoch')} "
        f"val_acc_1st={state['metadata'].get('val_acc_1st', 0):.1f}% "
        f"calibrator={cal_status}"
    )
    return model, scaler, calibrators


def apply_iso(probs, iso_list):
    out = np.array([float(iso_list[i].transform([probs[i]])[0]) for i in range(len(probs))])
    s = out.sum()
    return out / s if s > 0 else out


def predict_probs(model, scaler, calibrators, race, boats):
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
    except Exception:
        return None
    f = scaler.transform(f.reshape(1, -1))
    X = torch.FloatTensor(f)
    with torch.no_grad():
        out = model(X)
    p1 = apply_iso(F.softmax(out[0], dim=1).numpy()[0], calibrators['1st'])
    p2 = apply_iso(F.softmax(out[1], dim=1).numpy()[0], calibrators['2nd'])
    p3 = apply_iso(F.softmax(out[2], dim=1).numpy()[0], calibrators['3rd'])
    return p1, p2, p3


def race_level_ok(probs_1st, venue_id, race_number):
    if race_number > 4:
        return False, 'R>4'
    if _should_skip_by_top_boat(probs_1st):
        return False, 'skip_56'
    if venue_id in VENUE_HONMEI:
        return False, 'honmei_場'
    if venue_id not in VENUE_ARE and race_number >= 7:
        return False, 'joseki_gray_late'
    return True, ''


def kelly_stake(prob, odds, bankroll, kelly_frac=KELLY_FRAC):
    discount = _get_dynamic_discount(odds)
    disc_odds = odds * discount
    b = disc_odds - 1.0
    if b < 0.01 or prob <= 0 or prob >= 1:
        return 0, disc_odds
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0:
        return 0, disc_odds
    raw = bankroll * kelly * kelly_frac
    max_ticket = MAX_TICKET_RATIO * bankroll
    raw = min(raw, max_ticket)
    stake = max(MIN_BET, int(round(raw / 100) * 100))
    return stake, disc_odds


def main(date_from: str, date_to: str, strategy: str = 'mc_early_race'):
    logger.info(f"=== V10 April backtest 開始 期間 {date_from} 〜 {date_to} ===")
    model, scaler, calibrators = load_model()

    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()

    cur.execute("""
        SELECT r.id, r.venue_id, r.race_number, r.race_date,
               r.wind_speed, r.wind_direction, r.temperature,
               r.wave_height, r.water_temperature,
               r.actual_result_trifecta, r.payout_sanrentan
        FROM races r
        WHERE r.is_finished = true AND r.actual_result_trifecta IS NOT NULL
          AND r.result_1st IS NOT NULL AND r.wind_speed IS NOT NULL
          AND r.race_date BETWEEN %s AND %s
        ORDER BY r.race_date ASC, r.id ASC
    """, (date_from, date_to))
    races = cur.fetchall()
    race_ids = [r['id'] for r in races]
    race_map = {r['id']: dict(r) for r in races}
    logger.info(f"対象 races: {len(races)} ({date_from}〜{date_to}, finished)")

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
    boats_map = defaultdict(list)
    for b in cur.fetchall():
        boats_map[b['race_id']].append(dict(b))

    cur.execute("""
        SELECT race_id, combination, amount, odds, is_hit, payout
        FROM bets
        WHERE strategy_type = %s
          AND race_id = ANY(%s) AND result IS NOT NULL
        ORDER BY race_id, id
    """, (strategy, race_ids))
    v10_bets = [dict(b) for b in cur.fetchall()]
    conn.close()
    logger.info(f"V10 実bets ({strategy}): {len(v10_bets)}件")

    if not v10_bets:
        logger.warning(f"対象期間に {strategy} の bets が無い")
        return

    v10_invest = sum(int(b['amount']) for b in v10_bets)
    v10_payout = sum(int(b['payout'] or 0) for b in v10_bets if b['is_hit'])
    v10_hits = sum(1 for b in v10_bets if b['is_hit'])
    v10_roi = v10_payout / v10_invest if v10_invest else 0

    mc_cache = {}
    race_filter_cache = {}
    stats = {
        'considered': 0, 'accepted': 0,
        'reject_race_filter': 0, 'reject_low_prob': 0,
        'reject_odds': 0, 'reject_kelly': 0,
        'invest': 0, 'payout': 0, 'hits': 0,
        'filter_reasons': defaultdict(int),
    }
    detail = []

    for bet in v10_bets:
        rid = bet['race_id']
        stats['considered'] += 1
        race = race_map.get(rid)
        boats = boats_map.get(rid, [])
        if not race or len(boats) != 6:
            stats['reject_race_filter'] += 1
            continue

        if rid not in race_filter_cache:
            pred = predict_probs(model, scaler, calibrators, race, boats)
            if pred is None:
                race_filter_cache[rid] = (False, 'predict_fail', None, None, None)
            else:
                p1, p2, p3 = pred
                ok, reason = race_level_ok(p1.tolist(), race['venue_id'], race['race_number'])
                race_filter_cache[rid] = (ok, reason, p1, p2, p3)

        ok, reason, p1, p2, p3 = race_filter_cache[rid]
        if not ok:
            stats['reject_race_filter'] += 1
            stats['filter_reasons'][reason] += 1
            continue

        if rid not in mc_cache:
            try:
                rdata = {'wind_speed': race.get('wind_speed') or 0,
                         'wave_height': race.get('wave_height') or 0}
                mc_cache[rid] = monte_carlo_sanrentan(
                    p1.tolist(), boats_data=boats,
                    n_simulations=20000, race_data=rdata,
                    race_number=race['race_number'],
                )
            except Exception:
                mc_cache[rid] = None
        mc_probs = mc_cache[rid]
        if mc_probs is None:
            stats['reject_race_filter'] += 1
            continue

        combo = bet['combination']
        prob = mc_probs.get(combo, 0.0)
        raw_odds = float(bet['odds'] or 0)

        if prob < MIN_PROB:
            stats['reject_low_prob'] += 1
            continue
        if raw_odds < MIN_ODDS or raw_odds > MAX_ODDS:
            stats['reject_odds'] += 1
            continue

        stake, disc_odds = kelly_stake(prob, raw_odds, BANKROLL)
        if stake < MIN_BET:
            stats['reject_kelly'] += 1
            continue

        stats['accepted'] += 1
        stats['invest'] += stake
        if bet['is_hit']:
            stats['hits'] += 1
            stats['payout'] += int(raw_odds * stake)

        detail.append({
            'race_id': rid, 'race_date': str(race['race_date']),
            'venue_id': race['venue_id'], 'race_number': race['race_number'],
            'combo': combo, 'odds': raw_odds, 'prob': prob,
            'ev': prob * disc_odds, 'stake': stake, 'hit': bool(bet['is_hit']),
        })

    new_roi = stats['payout'] / stats['invest'] if stats['invest'] else 0
    new_hit_rate = stats['hits'] / stats['accepted'] if stats['accepted'] else 0
    new_profit = stats['payout'] - stats['invest']

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"V10 実績 ({strategy}, {date_from}〜{date_to})")
    logger.info(f"  bets={len(v10_bets)} hits={v10_hits} "
                f"ROI={v10_roi*100:.1f}% 投資=¥{v10_invest:,} 回収=¥{v10_payout:,}")
    logger.info("")
    logger.info(f"V10 April fine-tune 判断 (V10 universe上)")
    logger.info(f"  考慮:    {stats['considered']}")
    logger.info(f"  採択:    {stats['accepted']} ({stats['accepted']/max(1,stats['considered'])*100:.1f}%)")
    logger.info(f"  reject_race_filter: {stats['reject_race_filter']}")
    logger.info(f"  reject_low_prob:    {stats['reject_low_prob']}")
    logger.info(f"  reject_odds:        {stats['reject_odds']}")
    logger.info(f"  reject_kelly:       {stats['reject_kelly']}")
    logger.info(f"  的中: {stats['hits']}/{stats['accepted']} ({new_hit_rate*100:.1f}%)")
    logger.info(f"  投資: ¥{stats['invest']:,} 回収: ¥{stats['payout']:,}")
    logger.info(f"  損益: ¥{new_profit:+,}")
    logger.info(f"  **ROI: {new_roi*100:.1f}%**")
    logger.info("")
    logger.info("filter_reasons:")
    for k, v in sorted(stats['filter_reasons'].items(), key=lambda x: -x[1]):
        logger.info(f"  {k}: {v}")
    if v10_roi > 0:
        diff = (new_roi - v10_roi) * 100
        sym = '🏆 V10超え' if diff > 0 else '❌ V10未満'
        logger.info(f"\n{sym}: {diff:+.1f}pt")

    out_path = REPORT_DIR / f"backtest_v10_april_{date_from}_{date_to}_{strategy}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'date_from': date_from, 'date_to': date_to, 'strategy': strategy,
            'v10_actual': {'bets': len(v10_bets), 'hits': v10_hits,
                           'invest': v10_invest, 'payout': v10_payout, 'roi': v10_roi},
            'v10_april_selection': {
                'considered': stats['considered'], 'accepted': stats['accepted'],
                'invest': stats['invest'], 'payout': stats['payout'],
                'hits': stats['hits'], 'roi': new_roi, 'hit_rate': new_hit_rate,
                'reject_race_filter': stats['reject_race_filter'],
                'reject_low_prob': stats['reject_low_prob'],
                'reject_odds': stats['reject_odds'],
                'reject_kelly': stats['reject_kelly'],
                'filter_reasons': dict(stats['filter_reasons']),
            },
            'bets': detail,
        }, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nレポート: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='date_from', default='2026-04-01')
    parser.add_argument('--to', dest='date_to', default='2026-04-30')
    parser.add_argument('--strategy', default='mc_early_race')
    args = parser.parse_args()
    main(args.date_from, args.date_to, args.strategy)
