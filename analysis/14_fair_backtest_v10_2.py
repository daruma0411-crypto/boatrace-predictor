"""V10.2 公平 backtest (v2): V10 実bets に対する V10.2 の selection 能力評価

前版(13_)の問題: actual_payout を oddsに feed して 後出し的中評価になった。

本版の戦略:
  V10 が実際に betした 各combo に対して、V10.2 は:
    1. 同じ race_data + boats から predict → NN + calibrator → MC prob
    2. EV = V10.2_prob × V10.odds × dynamic_discount
    3. min_prob, odds range, Kelly>0 フィルタ
    4. 採択判定 (V10.2 の判断が V10 の選定より良いか？)
    5. 採択したら stake は V10.2 の Kelly 計算値
    6. 結果は V10 の is_hit / payout (真実)

これなら実 odds を使って V10.2 の **selection能力**を公平評価できる。
V10.2 独自の combo 発掘能力は評価外 (oddsデータない)。

V10 本番 ROI 132% と直接比較可能。
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
import psycopg2
from psycopg2.extras import RealDictCursor

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.betting import (
    _should_skip_by_top_boat, _get_dynamic_discount,
    VENUE_HONMEI, VENUE_ARE,
)
from src.monte_carlo import monte_carlo_sanrentan

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_2_DIR = Path(__file__).parent / "models_v11" / "v10_2"
MODEL_PATH = V10_2_DIR / "boatrace_model_v10_2.pth"
SCALER_PATH = V10_2_DIR / "feature_scaler_v10_2.pkl"
CAL_PATH = V10_2_DIR / "calibrators_v10_2.pkl"
REPORT_DIR = Path(__file__).parent / "reports"

# mc_early_race 設定
BANKROLL = 200000
KELLY_FRAC = 0.0625
MAX_TICKET_RATIO = 0.008
MIN_ODDS = 5.0
MAX_ODDS = 40.0
MIN_PROB = 0.005
MAX_ENTROPY = 2.3
MIN_BET = 100


def load_v10_2():
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(CAL_PATH, 'rb') as f: calibrators = pickle.load(f)
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
    except Exception: return None
    f = scaler.transform(f.reshape(1, -1))
    X = torch.FloatTensor(f)
    with torch.no_grad():
        out = model(X)
    p1 = apply_iso(F.softmax(out[0], dim=1).numpy()[0], calibrators['1st'])
    p2 = apply_iso(F.softmax(out[1], dim=1).numpy()[0], calibrators['2nd'])
    p3 = apply_iso(F.softmax(out[2], dim=1).numpy()[0], calibrators['3rd'])
    return p1, p2, p3


def race_level_ok(probs_1st, venue_id, race_number):
    """mc_early_race のレース全体フィルタ"""
    if race_number > 4: return False, 'R>4'
    if _should_skip_by_top_boat(probs_1st):
        return False, 'skip_56'
    # joseki_mode
    if venue_id in VENUE_HONMEI:
        return False, 'honmei_場'
    if venue_id not in VENUE_ARE and race_number >= 7:
        return False, 'joseki_gray_late'
    return True, ''


def kelly_stake(prob, odds, bankroll, kelly_frac=KELLY_FRAC):
    """V10 の Kelly 計算を再現（discounted odds 使用）"""
    discount = _get_dynamic_discount(odds)
    disc_odds = odds * discount
    b = disc_odds - 1.0
    if b < 0.01 or prob <= 0 or prob >= 1: return 0, disc_odds
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0: return 0, disc_odds
    raw = bankroll * kelly * kelly_frac
    max_ticket = MAX_TICKET_RATIO * bankroll
    raw = min(raw, max_ticket)
    stake = max(MIN_BET, int(round(raw / 100) * 100))
    return stake, disc_odds


def main():
    logger.info("V10.2 公平 backtest v2 開始")
    model, scaler, calibrators = load_v10_2()

    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # val集合
    cur.execute("""
        SELECT r.id, r.venue_id, r.race_number, r.race_date,
               r.wind_speed, r.wind_direction, r.temperature,
               r.wave_height, r.water_temperature,
               r.actual_result_trifecta, r.payout_sanrentan
        FROM races r
        WHERE r.is_finished = true AND r.actual_result_trifecta IS NOT NULL
          AND r.result_1st IS NOT NULL AND r.wind_speed IS NOT NULL
        ORDER BY r.race_date ASC, r.id ASC
    """)
    all_races = cur.fetchall()
    n = len(all_races)
    split = int(n * 0.8)
    val_races = all_races[split:]
    val_race_ids = [r['id'] for r in val_races]
    val_race_map = {r['id']: dict(r) for r in val_races}

    # boats
    cur.execute("""
        SELECT race_id, boat_number, player_class,
               win_rate, win_rate_2, win_rate_3,
               local_win_rate, local_win_rate_2,
               avg_st, motor_win_rate_2, motor_win_rate_3,
               boat_win_rate_2, weight, exhibition_time,
               approach_course, is_new_motor, tilt, parts_changed
        FROM boats WHERE race_id = ANY(%s)
        ORDER BY race_id, boat_number
    """, (val_race_ids,))
    boats_map = defaultdict(list)
    for b in cur.fetchall():
        boats_map[b['race_id']].append(dict(b))

    # V10 の val期間 bets
    cur.execute("""
        SELECT race_id, combination, amount, odds, is_hit, payout
        FROM bets
        WHERE strategy_type = 'mc_early_race'
          AND race_id = ANY(%s) AND result IS NOT NULL
        ORDER BY race_id, id
    """, (val_race_ids,))
    v10_bets = [dict(b) for b in cur.fetchall()]
    conn.close()
    logger.info(f"V10 実bets: {len(v10_bets)}件")

    # V10実績
    v10_invest = sum(int(b['amount']) for b in v10_bets)
    v10_payout = sum(int(b['payout'] or 0) for b in v10_bets if b['is_hit'])
    v10_hits = sum(1 for b in v10_bets if b['is_hit'])
    v10_roi = v10_payout / v10_invest if v10_invest else 0

    # 各V10 bet に V10.2 がどう判断するか評価
    # まずレース毎に V10.2 prob をキャッシュ
    mc_cache = {}
    race_filter_cache = {}

    stats = {
        'considered': 0, 'accepted': 0,
        'reject_race_filter': 0,
        'reject_low_prob': 0,
        'reject_odds': 0,
        'reject_kelly': 0,
        'v10_2_invest': 0, 'v10_2_payout': 0, 'v10_2_hits': 0,
        'filter_reasons': defaultdict(int),
    }

    v10_2_bets_detail = []

    for bet in v10_bets:
        rid = bet['race_id']
        stats['considered'] += 1
        race = val_race_map.get(rid)
        boats = boats_map.get(rid, [])
        if not race or len(boats) != 6:
            stats['reject_race_filter'] += 1
            continue

        # レース全体フィルタ (V10.2 predict必要)
        if rid not in race_filter_cache:
            pred = predict_probs(model, scaler, calibrators, race, boats)
            if pred is None:
                race_filter_cache[rid] = (False, 'predict_fail', None, None, None)
            else:
                probs_1st, probs_2nd, probs_3rd = pred
                ok, reason = race_level_ok(probs_1st.tolist(),
                                            race['venue_id'], race['race_number'])
                race_filter_cache[rid] = (ok, reason, probs_1st, probs_2nd, probs_3rd)

        ok, reason, probs_1st, probs_2nd, probs_3rd = race_filter_cache[rid]
        if not ok:
            stats['reject_race_filter'] += 1
            stats['filter_reasons'][reason] += 1
            continue

        # MC simulation
        if rid not in mc_cache:
            try:
                rdata = {'wind_speed': race.get('wind_speed') or 0,
                         'wave_height': race.get('wave_height') or 0}
                mc_cache[rid] = monte_carlo_sanrentan(
                    probs_1st.tolist(), boats_data=boats,
                    n_simulations=20000, race_data=rdata,
                    race_number=race['race_number'],
                )
            except Exception as e:
                mc_cache[rid] = None
        mc_probs = mc_cache[rid]
        if mc_probs is None:
            stats['reject_race_filter'] += 1
            continue

        # 各フィルタ判定
        combo = bet['combination']
        v11_prob = mc_probs.get(combo, 0.0)
        raw_odds = float(bet['odds'] or 0)

        if v11_prob < MIN_PROB:
            stats['reject_low_prob'] += 1
            continue
        if raw_odds < MIN_ODDS or raw_odds > MAX_ODDS:
            stats['reject_odds'] += 1
            continue

        stake, disc_odds = kelly_stake(v11_prob, raw_odds, BANKROLL)
        if stake < MIN_BET:
            stats['reject_kelly'] += 1
            continue

        # 採択
        stats['accepted'] += 1
        stats['v10_2_invest'] += stake
        if bet['is_hit']:
            stats['v10_2_hits'] += 1
            stats['v10_2_payout'] += int(raw_odds * stake)

        v10_2_bets_detail.append({
            'race_id': rid, 'combo': combo, 'odds': raw_odds,
            'v10_2_prob': v11_prob, 'ev': v11_prob * disc_odds,
            'stake': stake, 'hit': bool(bet['is_hit']),
        })

    v10_2_roi = (stats['v10_2_payout'] / stats['v10_2_invest']
                 if stats['v10_2_invest'] else 0)
    v10_2_hit_rate = (stats['v10_2_hits'] / stats['accepted']
                      if stats['accepted'] else 0)
    v10_2_profit = stats['v10_2_payout'] - stats['v10_2_invest']

    logger.info("")
    logger.info("=" * 60)
    logger.info("V10 実績 (val期間)")
    logger.info(f"  bets={len(v10_bets)} hits={v10_hits} "
                f"ROI={v10_roi*100:.1f}% 投資={v10_invest:,} 回収={v10_payout:,}")
    logger.info("")
    logger.info("V10.2 判断 (V10 universe上)")
    logger.info(f"  考慮: {stats['considered']}")
    logger.info(f"  採択: {stats['accepted']} "
                f"({stats['accepted']/max(1,stats['considered'])*100:.1f}%)")
    logger.info(f"  reject_race_filter: {stats['reject_race_filter']}")
    logger.info(f"  reject_low_prob: {stats['reject_low_prob']}")
    logger.info(f"  reject_odds: {stats['reject_odds']}")
    logger.info(f"  reject_kelly: {stats['reject_kelly']}")
    logger.info(f"  的中: {stats['v10_2_hits']}/{stats['accepted']} "
                f"({v10_2_hit_rate*100:.1f}%)")
    logger.info(f"  投資: ¥{stats['v10_2_invest']:,}")
    logger.info(f"  回収: ¥{stats['v10_2_payout']:,}")
    logger.info(f"  損益: ¥{v10_2_profit:+,}")
    logger.info(f"  **ROI: {v10_2_roi*100:.1f}%**")
    logger.info("")
    logger.info("filter_reasons:")
    for k, v in sorted(stats['filter_reasons'].items(), key=lambda x: -x[1]):
        logger.info(f"  {k}: {v}")
    logger.info("")
    if v10_roi > 0:
        diff = (v10_2_roi - v10_roi) * 100
        sym = '🏆 V10超え' if diff > 0 else '❌ V10未満'
        logger.info(f"{sym}: {diff:+.1f}pt")

    out = {
        'generated_at': datetime.now().isoformat(),
        'v10_actual': {'bets': len(v10_bets), 'hits': v10_hits,
                       'invest': v10_invest, 'payout': v10_payout, 'roi': v10_roi},
        'v10_2_selection': {
            'considered': stats['considered'],
            'accepted': stats['accepted'],
            'reject_race_filter': stats['reject_race_filter'],
            'reject_low_prob': stats['reject_low_prob'],
            'reject_odds': stats['reject_odds'],
            'reject_kelly': stats['reject_kelly'],
            'hits': stats['v10_2_hits'],
            'hit_rate': v10_2_hit_rate,
            'invest': stats['v10_2_invest'],
            'payout': stats['v10_2_payout'],
            'profit': v10_2_profit,
            'roi': v10_2_roi,
            'filter_reasons': dict(stats['filter_reasons']),
        },
        'bet_detail_top20': v10_2_bets_detail[:20],
    }
    out_path = REPORT_DIR / "14_v10_2_fair_backtest.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nレポート: {out_path}")


if __name__ == '__main__':
    main()
