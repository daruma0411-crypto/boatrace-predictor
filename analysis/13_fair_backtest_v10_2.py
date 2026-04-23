"""V10.2 公平 backtest: src.betting._strategy_kelly を流用して V10 同等フィルタで評価

V10 本番の処理を完全再現:
  1. V10.2 NN predict → probs_1st/2nd/3rd
  2. V10.2 calibrator 適用
  3. monte_carlo_sanrentan() で全120 combo 確率計算 (n=20000)
  4. レース全体フィルタ (skip_56, joseki_mode, max_race_number)
  5. src.betting._strategy_kelly で 各combo フィルタ + Kelly + top3
  6. 選定bet の実結果 (races.payout_sanrentan) で ROI 計算

オッズ: 実運用ではテレボートから取得だが、retrospective では推定必要。
  戦略:
    - 勝ち combo (actual_result): payout_sanrentan / 100 を真値odds として使用
    - 負け combo: 理論オッズ 0.75 / MC prob を使用 (過大評価ぎみだが公平比較)

V10 本番の実績 (mc_early_race: ROI 132%) と比較。
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
from src.database import get_db_connection
from src.betting import (
    KellyBettingStrategy, _should_skip_by_top_boat,
    VENUE_HONMEI, VENUE_ARE,
)
from src.monte_carlo import monte_carlo_sanrentan

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

V10_2_DIR = Path(__file__).parent / "models_v11" / "v10_2"
MODEL_PATH = V10_2_DIR / "boatrace_model_v10_2.pth"
SCALER_PATH = V10_2_DIR / "feature_scaler_v10_2.pkl"
CAL_PATH = V10_2_DIR / "calibrators_v10_2.pkl"

REPORT_DIR = Path(__file__).parent / "reports"

BANKROLL = 200000


def load_v10_2():
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(CAL_PATH, 'rb') as f:
        calibrators = pickle.load(f)
    logger.info(f"V10.2 loaded: val_acc={state['metadata']['val_acc_1st']:.1f}%")
    return model, scaler, calibrators


def apply_isotonic(probs, iso_list):
    """Isotonic 適用 (sklearn IsotonicRegression.transform)"""
    out = np.zeros_like(probs)
    for i in range(len(probs)):
        out[i] = float(iso_list[i].transform([probs[i]])[0])
    # normalize
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def predict_probs(model, scaler, calibrators, race, boats):
    """V10.2 predict → calibrated probs_1st/2nd/3rd"""
    fe = FeatureEngineer()
    rd = {'venue_id': race['venue_id'], 'month': race['race_date'].month,
          'distance': 1800,
          'wind_speed': race.get('wind_speed') or 0,
          'wind_direction': race.get('wind_direction') or 'calm',
          'temperature': race.get('temperature') or 20,
          'wave_height': race.get('wave_height') or 0,
          'water_temperature': race.get('water_temperature') or 20}
    try:
        features = fe.transform(rd, boats)
    except Exception:
        return None
    features = scaler.transform(features.reshape(1, -1))
    X = torch.FloatTensor(features)
    with torch.no_grad():
        out = model(X)
    p1 = F.softmax(out[0], dim=1).numpy()[0]
    p2 = F.softmax(out[1], dim=1).numpy()[0]
    p3 = F.softmax(out[2], dim=1).numpy()[0]
    p1 = apply_isotonic(p1, calibrators['1st'])
    p2 = apply_isotonic(p2, calibrators['2nd'])
    p3 = apply_isotonic(p3, calibrators['3rd'])
    return p1, p2, p3


def estimate_odds(sanrentan_probs, actual_combo, actual_payout):
    """全combo odds を推定

    - actual_combo: 実際の payout (payout_sanrentan / 100)
    - 他: 理論 0.75 / prob
    """
    odds = {}
    for combo, prob in sanrentan_probs.items():
        if prob <= 0:
            odds[combo] = 0.0
            continue
        if combo == actual_combo and actual_payout > 0:
            odds[combo] = actual_payout / 100.0  # 真の市場オッズ
        else:
            odds[combo] = min(999.9, 0.75 / prob)  # 理論オッズ（控除率25%）
    return odds


def race_level_filter(probs_1st, venue_id, race_number, strategy_config):
    """レース全体フィルタ (calculate_all_strategiesの該当部分を抽出)"""
    # max_race_number
    max_race = strategy_config.get('max_race_number', 12)
    if race_number > max_race:
        return False, f'R{race_number}>max_race'
    # exclude_race_numbers
    if race_number in strategy_config.get('exclude_race_numbers', []):
        return False, f'R{race_number} in exclude'
    # skip_56
    if strategy_config.get('skip_56', False):
        if _should_skip_by_top_boat(probs_1st):
            top = max(range(6), key=lambda i: probs_1st[i]) + 1
            return False, f'skip_56 top={top}'
    # joseki_mode
    if strategy_config.get('joseki_mode', False) and venue_id is not None:
        if venue_id in VENUE_HONMEI:
            return False, f'joseki本命場 V{venue_id}'
        if strategy_config.get('joseki_skip_gray_late', True):
            is_gray = venue_id not in VENUE_ARE
            if is_gray and race_number >= 7:
                return False, f'joseki グレー場後半 V{venue_id}R{race_number}'
    # include_venues
    include = strategy_config.get('include_venues', [])
    if include and venue_id not in include:
        return False, f'V{venue_id} not in include_venues'
    return True, ''


def main():
    logger.info("V10.2 公平 backtest 開始")
    model, scaler, calibrators = load_v10_2()

    # betting strategy を初期化 (_strategy_kelly 呼ぶため)
    kb = KellyBettingStrategy(initial_bankroll=BANKROLL)
    strategy_config = kb.config['strategies']['mc_early_race']
    logger.info(f"strategy_config: kelly_frac={strategy_config['kelly_fraction']} "
                f"max_odds={strategy_config['max_odds']} "
                f"min_prob={strategy_config['min_probability']}")

    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # val 集合 (時系列80/20)
    cur.execute("""
        SELECT r.id, r.venue_id, r.race_number, r.race_date,
               r.wind_speed, r.wind_direction, r.temperature,
               r.wave_height, r.water_temperature,
               r.actual_result_trifecta, r.payout_sanrentan
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
    val_races = all_races[split:]
    logger.info(f"val 集合: {len(val_races)}")

    # boats 一括取得
    val_race_ids = [r['id'] for r in val_races]
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

    # V10 実績
    cur.execute("""
        SELECT COUNT(*) as bets, SUM(amount) as invest,
               SUM(CASE WHEN is_hit THEN 1 ELSE 0 END) as hits,
               SUM(COALESCE(payout, 0)) as payout
        FROM bets
        WHERE strategy_type = 'mc_early_race'
          AND race_id = ANY(%s) AND result IS NOT NULL
    """, (val_race_ids,))
    row = cur.fetchone()
    v10_bets, v10_invest, v10_hits, v10_payout = row['bets'], int(row['invest'] or 0), row['hits'] or 0, int(row['payout'] or 0)
    v10_roi = v10_payout / v10_invest if v10_invest else 0
    conn.close()

    # V10.2 evaluate
    stats = {
        'considered_races': 0, 'race_filter_skip': 0,
        'predict_fail': 0, 'bets_generated': 0,
        'invest': 0, 'hits': 0, 'payout': 0,
        'filter_breakdown': defaultdict(int),
    }

    for i, race in enumerate(val_races):
        boats = boats_map.get(race['id'], [])
        if len(boats) != 6:
            stats['predict_fail'] += 1
            continue

        stats['considered_races'] += 1

        # NN predict + calibrate
        pred = predict_probs(model, scaler, calibrators, race, boats)
        if pred is None:
            stats['predict_fail'] += 1
            continue
        probs_1st, probs_2nd, probs_3rd = pred

        # レース全体フィルタ
        ok, reason = race_level_filter(probs_1st, race['venue_id'],
                                        race['race_number'], strategy_config)
        if not ok:
            stats['race_filter_skip'] += 1
            stats['filter_breakdown'][reason.split()[0]] += 1
            continue

        # monte_carlo_sanrentan
        try:
            race_data_for_mc = {
                'wind_speed': race.get('wind_speed') or 0,
                'wave_height': race.get('wave_height') or 0,
            }
            mc_probs = monte_carlo_sanrentan(
                probs_1st.tolist(),
                boats_data=boats,
                n_simulations=20000,
                race_data=race_data_for_mc,
                race_number=race['race_number'],
            )
        except Exception as e:
            logger.warning(f"MC失敗: race={race['id']}: {e}")
            stats['predict_fail'] += 1
            continue

        # オッズ推定
        actual_combo = race['actual_result_trifecta']
        actual_payout = int(race['payout_sanrentan'] or 0)
        odds_data = estimate_odds(mc_probs, actual_combo, actual_payout)

        # _strategy_kelly 呼び出し
        try:
            bets = kb._strategy_kelly(
                config=strategy_config,
                sanrentan_probs=mc_probs,
                odds_data=odds_data,
                bankroll=BANKROLL,
                strategy_name='v10_2_eval',
                venue_id=race['venue_id'],
                race_number=race['race_number'],
            )
        except Exception as e:
            logger.warning(f"strategy_kelly失敗: race={race['id']}: {e}")
            continue

        # 結果集計
        for bet in bets:
            stats['bets_generated'] += 1
            stats['invest'] += bet['amount']
            if bet['combination'] == actual_combo and actual_payout > 0:
                stats['hits'] += 1
                # payout 計算: actual_payout × bet_amount / 100
                stats['payout'] += int(actual_payout * bet['amount'] / 100)

        if (i + 1) % 100 == 0:
            logger.info(f"  進捗 {i+1}/{len(val_races)}: "
                        f"bets={stats['bets_generated']} "
                        f"ROI={stats['payout']/max(1,stats['invest'])*100:.1f}%")

    roi = stats['payout'] / stats['invest'] if stats['invest'] else 0
    hit_rate = stats['hits'] / stats['bets_generated'] if stats['bets_generated'] else 0
    profit = stats['payout'] - stats['invest']

    # レポート
    logger.info("")
    logger.info("=" * 60)
    logger.info("V10 (mc_early_race) 実績")
    logger.info(f"  bets={v10_bets} hits={v10_hits} 投資={v10_invest:,} "
                f"回収={v10_payout:,} ROI={v10_roi*100:.1f}%")
    logger.info("")
    logger.info("V10.2 backtest 結果")
    logger.info(f"  考慮レース: {stats['considered_races']}")
    logger.info(f"  レース全体filter除外: {stats['race_filter_skip']}")
    logger.info(f"  bets生成: {stats['bets_generated']}件")
    logger.info(f"  的中: {stats['hits']}件 ({hit_rate*100:.1f}%)")
    logger.info(f"  投資: ¥{stats['invest']:,}")
    logger.info(f"  回収: ¥{stats['payout']:,}")
    logger.info(f"  損益: ¥{profit:+,}")
    logger.info(f"  **ROI: {roi*100:.1f}%**")
    logger.info("")
    logger.info("filter breakdown:")
    for k, v in sorted(stats['filter_breakdown'].items(), key=lambda x: -x[1]):
        logger.info(f"  {k}: {v}")
    logger.info("")
    if v10_roi > 0:
        diff_pt = (roi - v10_roi) * 100
        if diff_pt > 0:
            logger.info(f"🏆 V10超え: +{diff_pt:.1f}pt")
        else:
            logger.info(f"❌ V10未満: {diff_pt:.1f}pt")

    out = {
        'generated_at': datetime.now().isoformat(),
        'v10_actual': {'bets': v10_bets, 'hits': v10_hits,
                       'invest': v10_invest, 'payout': v10_payout, 'roi': v10_roi},
        'v10_2_backtest': {
            'considered_races': stats['considered_races'],
            'race_filter_skip': stats['race_filter_skip'],
            'bets_generated': stats['bets_generated'],
            'hits': stats['hits'], 'hit_rate': hit_rate,
            'invest': stats['invest'], 'payout': stats['payout'],
            'profit': profit, 'roi': roi,
            'filter_breakdown': dict(stats['filter_breakdown']),
        }
    }
    out_path = REPORT_DIR / "13_v10_2_fair_backtest.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nレポート: {out_path}")


if __name__ == '__main__':
    main()
