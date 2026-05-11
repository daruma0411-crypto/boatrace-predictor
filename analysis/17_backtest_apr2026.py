"""2026年4月 forward backtest (高速版)

13_fair_backtest_v10_2.py のパターンで P6 戦略のみ簡潔に評価する。
calculate_all_strategies の全戦略走査と DB 参照フィルタ (当日連敗) を回避。

フロー:
  1. RealtimePredictor.predict (V10 or fine-tune後モデル) → probs_1st/2nd/3rd
  2. race_level_filter (skip_56, joseki_mode, include_venues, max_race_number)
  3. monte_carlo_sanrentan (n_simulations=20000)
  4. _strategy_kelly(config=P6) → bets
  5. odds は pkl の実 odds を使う、実 result/payout で照合 → ROI

入力:
  --model-path: 評価対象モデル
  --strategy: 戦略名 (default: mc3_venue_focus_r4)
  --year/--month: 期間 (default: 2026/4)

データ:
  - races/boats: DB SELECT のみ (READ-ONLY)
  - odds_3t: analysis/historical_data/{year}_{month:02d}/odds_3t.pkl
"""
import os
import sys
import json
import pickle
import logging
import argparse
import calendar
from datetime import date
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import psycopg2
from psycopg2.extras import RealDictCursor

from src.predictor import RealtimePredictor
from src.monte_carlo import monte_carlo_sanrentan
from src.betting import (
    KellyBettingStrategy, _should_skip_by_top_boat,
    VENUE_HONMEI, VENUE_ARE,
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ANALYSIS_DIR = Path(__file__).parent
HIST_DIR = ANALYSIS_DIR / 'historical_data'
REPORT_DIR = ANALYSIS_DIR / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def race_level_filter(probs_1st, venue_id, race_number, strategy_config):
    """13_fair_backtest_v10_2.py の同名関数を流用 (DB 参照無し)"""
    max_race = strategy_config.get('max_race_number', 12)
    if race_number > max_race:
        return False, f'R{race_number}>max_race'
    if race_number in strategy_config.get('exclude_race_numbers', []):
        return False, f'R{race_number} in exclude'
    if strategy_config.get('skip_56', False):
        if _should_skip_by_top_boat(probs_1st):
            top = max(range(6), key=lambda i: probs_1st[i]) + 1
            return False, f'skip_56 top={top}'
    if strategy_config.get('joseki_mode', False) and venue_id is not None:
        if venue_id in VENUE_HONMEI:
            return False, f'joseki本命場 V{venue_id}'
        if strategy_config.get('joseki_skip_gray_late', True):
            is_gray = venue_id not in VENUE_ARE
            if is_gray and race_number >= 7:
                return False, f'joseki グレー後半 V{venue_id}R{race_number}'
    include = strategy_config.get('include_venues', [])
    if include and venue_id not in include:
        return False, f'V{venue_id} not in include_venues'
    # 1号艇軸スキップ (BettingCalculator.calculate_all_strategies と同等)
    top_boat = max(range(6), key=lambda i: probs_1st[i])
    if top_boat == 0:
        return False, '1号艇軸'
    return True, ''


def fetch_races(date_from, date_to):
    conn = psycopg2.connect(os.environ['DATABASE_URL'],
                            cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, venue_id, race_number, race_date,
               result_1st, result_2nd, result_3rd,
               actual_result_trifecta, payout_sanrentan,
               wind_speed, wind_direction, wave_height,
               temperature, water_temperature
        FROM races
        WHERE is_finished = true
          AND race_date BETWEEN %s AND %s
          AND actual_result_trifecta IS NOT NULL
          AND result_1st IS NOT NULL
        ORDER BY race_date, venue_id, race_number
    """, (date_from, date_to))
    races = [dict(r) for r in cur.fetchall()]

    if not races:
        conn.close()
        return [], {}

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
    boats_map = defaultdict(list)
    for b in cur.fetchall():
        boats_map[b['race_id']].append(dict(b))

    conn.close()
    return races, boats_map


def load_odds_pkl(year, month):
    p = HIST_DIR / f'{year}_{month:02d}' / 'odds_3t.pkl'
    if not p.exists():
        raise FileNotFoundError(f'odds_3t.pkl が無い: {p}')
    with open(p, 'rb') as f:
        records = pickle.load(f)
    odds_map = {}
    for r in records:
        key = (str(r['race_date']), r['venue_id'], r['race_number'])
        odds_map[key] = r['odds']
    logger.info(f'Odds pkl loaded: {len(odds_map)} レース')
    return odds_map


def main(model_path, year, month, strategy, bankroll):
    date_from = date(year, month, 1)
    date_to = date(year, month, calendar.monthrange(year, month)[1])

    logger.info(f'=== Backtest {date_from} 〜 {date_to} ===')
    logger.info(f'  model: {model_path}')
    logger.info(f'  strategy: {strategy}')
    logger.info(f'  bankroll: ¥{bankroll:,}')

    races, boats_map = fetch_races(date_from, date_to)
    logger.info(f'対象レース: {len(races)}')

    odds_map = load_odds_pkl(year, month)

    predictor = RealtimePredictor(model_path=model_path)
    kb = KellyBettingStrategy(initial_bankroll=bankroll)
    strategy_config = kb.config['strategies'].get(strategy)
    if strategy_config is None:
        raise ValueError(f'戦略 {strategy} が config に無い')
    logger.info(f'  戦略config: include_venues={strategy_config.get("include_venues")} '
                f'max_race={strategy_config.get("max_race_number")} '
                f'min_prob={strategy_config.get("min_probability")} '
                f'max_odds={strategy_config.get("max_odds")}')

    bets_log = []
    stats = defaultdict(int)
    filter_reasons = defaultdict(int)
    cur_bankroll = bankroll

    for idx, race in enumerate(races):
        if (idx + 1) % 500 == 0:
            logger.info(f'進捗 {idx+1}/{len(races)} bets={len(bets_log)} '
                        f'pnl={sum(b["pnl"] for b in bets_log):+,}')

        boats = boats_map.get(race['id'], [])
        if len(boats) != 6:
            stats['no_boats'] += 1
            continue
        boats = sorted(boats, key=lambda b: b['boat_number'])

        odds_key = (race['race_date'].isoformat(), race['venue_id'], race['race_number'])
        odds_data = odds_map.get(odds_key)
        if not odds_data:
            stats['no_odds'] += 1
            continue

        race_data = {
            'venue_id': race['venue_id'],
            'race_number': race['race_number'],
            'month': race['race_date'].month,
            'distance': 1800,
            'wind_speed': race.get('wind_speed') or 0,
            'wind_direction': race.get('wind_direction') or 'calm',
            'temperature': race.get('temperature') or 20,
            'wave_height': race.get('wave_height') or 0,
            'water_temperature': race.get('water_temperature') or 20,
        }

        try:
            pred = predictor.predict(race_data, boats)
        except Exception as e:
            stats['predict_fail'] += 1
            continue
        probs_1st = pred['probs_1st']
        probs_2nd = pred['probs_2nd']
        probs_3rd = pred['probs_3rd']

        # レース全体フィルタ
        ok, reason = race_level_filter(probs_1st, race['venue_id'],
                                       race['race_number'], strategy_config)
        if not ok:
            stats['race_filter_skip'] += 1
            filter_reasons[reason.split()[0]] += 1
            continue

        # MC sanrentan
        try:
            mc_probs = monte_carlo_sanrentan(
                probs_1st, boats_data=boats, n_simulations=20000,
                race_data={'wind_speed': race_data['wind_speed'],
                           'wave_height': race_data['wave_height']},
                race_number=race['race_number'],
            )
        except Exception as e:
            stats['mc_fail'] += 1
            continue

        # _strategy_kelly
        try:
            bets = kb._strategy_kelly(
                config=strategy_config,
                sanrentan_probs=mc_probs,
                odds_data=odds_data,
                bankroll=cur_bankroll,
                strategy_name=strategy,
                venue_id=race['venue_id'],
                race_number=race['race_number'],
            )
        except Exception as e:
            stats['kelly_fail'] += 1
            continue

        stats['considered'] += 1
        if not bets:
            stats['no_bets'] += 1
            continue

        actual = race['actual_result_trifecta']
        payout = int(race['payout_sanrentan'] or 0)

        for bet in bets:
            stake = bet.get('amount', 100)
            combo = bet.get('combination')
            bet_odds = bet.get('odds', odds_data.get(combo, 0.0))
            is_hit = (combo == actual)
            actual_return = (stake / 100 * payout) if is_hit else 0
            pnl = actual_return - stake
            bets_log.append({
                'race_id': race['id'],
                'race_date': race['race_date'].isoformat(),
                'venue_id': race['venue_id'],
                'race_number': race['race_number'],
                'combo': combo,
                'odds': bet_odds,
                'stake': stake,
                'is_hit': is_hit,
                'payout_per_100': payout if is_hit else 0,
                'actual_return': actual_return,
                'pnl': pnl,
            })
            cur_bankroll += pnl

    n = len(bets_log)
    logger.info('')
    logger.info('=== 集計 ===')
    logger.info(f'  対象 {len(races)} / no_boats={stats["no_boats"]} / no_odds={stats["no_odds"]} '
                f'/ predict_fail={stats["predict_fail"]} / race_filter_skip={stats["race_filter_skip"]}')
    logger.info(f'  filter内訳: {dict(filter_reasons)}')
    logger.info(f'  considered={stats["considered"]} no_bets={stats["no_bets"]}')

    if n == 0:
        logger.warning('bets ゼロ')
        result = {'n_bets': 0, 'stats': dict(stats), 'filter_reasons': dict(filter_reasons)}
    else:
        n_hit = sum(b['is_hit'] for b in bets_log)
        total_stake = sum(b['stake'] for b in bets_log)
        total_payout = sum(b['actual_return'] for b in bets_log)
        total_pnl = sum(b['pnl'] for b in bets_log)
        roi = total_payout / total_stake * 100 if total_stake else 0
        hit_rate = n_hit / n * 100
        logger.info(f'  bets: {n} (hit={n_hit}, hit_rate={hit_rate:.1f}%)')
        logger.info(f'  stake: ¥{total_stake:,}')
        logger.info(f'  payout: ¥{total_payout:,}')
        logger.info(f'  PnL: ¥{total_pnl:+,}')
        logger.info(f'  ROI: {roi:.1f}%')
        logger.info(f'  bankroll: ¥{bankroll:,} → ¥{cur_bankroll:,}')
        result = {
            'n_bets': n, 'n_hit': n_hit, 'hit_rate': hit_rate,
            'total_stake': total_stake, 'total_payout': total_payout,
            'total_pnl': total_pnl, 'roi': roi,
            'final_bankroll': cur_bankroll,
            'stats': dict(stats), 'filter_reasons': dict(filter_reasons),
        }

    model_tag = Path(model_path).stem
    out_path = REPORT_DIR / f'backtest_{year}_{month:02d}_{strategy}_{model_tag}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': str(model_path),
            'period': [date_from.isoformat(), date_to.isoformat()],
            'strategy': strategy,
            'initial_bankroll': bankroll,
            'summary': result,
            'bets': bets_log,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f'  保存: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='models/boatrace_model.pth')
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--month', type=int, default=4)
    parser.add_argument('--strategy', default='mc3_venue_focus_r4')
    parser.add_argument('--bankroll', type=int, default=200000)
    args = parser.parse_args()
    main(args.model_path, args.year, args.month, args.strategy, args.bankroll)
