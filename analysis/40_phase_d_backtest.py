"""Phase D D0+D1: 6 組合せ backtest (kelly_fraction × min_expected_value)

ベース: mc3_venue_focus_r4 (P6) を 2026-04 で評価 (2026-03 の odds 不在のため April のみ)。
in-memory で config を override し、本番 config/betting_config.json は触らない。

V10 推論 + MC sanrentan の結果を 1 回キャッシュし、
6 組合せに対しては kelly 計算と PnL 集計のみを切替て高速化する。
"""
import os
import sys
import copy
import json
import pickle
import logging
from datetime import date
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from src.predictor import RealtimePredictor
from src.monte_carlo import monte_carlo_sanrentan
from src.betting import (
    KellyBettingStrategy, _should_skip_by_top_boat,
    VENUE_HONMEI, VENUE_ARE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = ROOT / 'analysis'
HIST_DIR = ANALYSIS_DIR / 'historical_data'
REPORT_DIR = ANALYSIS_DIR / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = ANALYSIS_DIR / 'phase_d_cache.pkl'
REPORT_PATH = REPORT_DIR / 'phase_d_grid.md'

BASE_STRATEGY = 'mc3_venue_focus_r4'
DATE_FROM = date(2026, 4, 1)
DATE_TO = date(2026, 4, 30)
INITIAL_BANKROLL = 200000

GRID = [
    {'kelly_fraction': 0.0625, 'min_expected_value': 0.0, 'label': 'P6 default (1/16, EV0)'},
    {'kelly_fraction': 0.0625, 'min_expected_value': 1.0, 'label': '1/16 + EV>=1.0'},
    {'kelly_fraction': 0.10,    'min_expected_value': 0.0, 'label': '1/10 + EV0'},
    {'kelly_fraction': 0.10,    'min_expected_value': 1.0, 'label': '1/10 + EV>=1.0'},
    {'kelly_fraction': 0.10,    'min_expected_value': 1.1, 'label': '1/10 + EV>=1.1'},
    {'kelly_fraction': 0.20,    'min_expected_value': 1.0, 'label': '1/5 + EV>=1.0'},
]


def race_level_filter(probs_1st, venue_id, race_number, strategy_config):
    max_race = strategy_config.get('max_race_number', 12)
    if race_number > max_race:
        return False, f'R{race_number}>max_race'
    if race_number in strategy_config.get('exclude_race_numbers', []):
        return False, f'R{race_number} in exclude'
    if strategy_config.get('skip_56', False):
        if _should_skip_by_top_boat(probs_1st):
            return False, 'skip_56'
    if strategy_config.get('joseki_mode', False) and venue_id is not None:
        if venue_id in VENUE_HONMEI:
            return False, f'joseki本命 V{venue_id}'
        if strategy_config.get('joseki_skip_gray_late', True):
            is_gray = venue_id not in VENUE_ARE
            if is_gray and race_number >= 7:
                return False, f'joseki グレー後半 V{venue_id}R{race_number}'
    include = strategy_config.get('include_venues', [])
    if include and venue_id not in include:
        return False, f'V{venue_id} not in include'
    top_boat = max(range(6), key=lambda i: probs_1st[i])
    if top_boat == 0:
        return False, '1号艇軸'
    return True, ''


def fetch_races(date_from, date_to):
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, venue_id, race_number, race_date,
               result_1st, actual_result_trifecta, payout_sanrentan,
               wind_speed, wind_direction, wave_height,
               temperature, water_temperature
        FROM races
        WHERE is_finished = true AND race_date BETWEEN %s AND %s
          AND actual_result_trifecta IS NOT NULL AND result_1st IS NOT NULL
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


def load_odds_map(date_from, date_to):
    odds_map = {}
    months = set()
    d = date_from
    while d <= date_to:
        months.add((d.year, d.month))
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)
    for y, m in sorted(months):
        p = HIST_DIR / f'{y}_{m:02d}' / 'odds_3t.pkl'
        if not p.exists():
            logger.warning(f'odds 不在: {p}')
            continue
        with open(p, 'rb') as f:
            for r in pickle.load(f):
                key = (str(r['race_date']), r['venue_id'], r['race_number'])
                odds_map[key] = r['odds']
        logger.info(f'odds 読込: {y}_{m:02d} (累計 {len(odds_map)})')
    return odds_map


def build_cache(predictor, races, boats_map, odds_map):
    cache = []
    skipped = defaultdict(int)
    for idx, race in enumerate(races):
        if (idx + 1) % 500 == 0:
            logger.info(f'cache 構築 {idx+1}/{len(races)}')
        boats = boats_map.get(race['id'], [])
        if len(boats) != 6:
            skipped['no_boats'] += 1
            continue
        boats = sorted(boats, key=lambda b: b['boat_number'])
        odds_key = (race['race_date'].isoformat(), race['venue_id'], race['race_number'])
        odds_data = odds_map.get(odds_key)
        if not odds_data:
            skipped['no_odds'] += 1
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
        except Exception:
            skipped['predict_fail'] += 1
            continue
        try:
            mc_probs = monte_carlo_sanrentan(
                pred['probs_1st'], boats_data=boats, n_simulations=20000,
                race_data={'wind_speed': race_data['wind_speed'],
                           'wave_height': race_data['wave_height']},
                race_number=race['race_number'],
            )
        except Exception:
            skipped['mc_fail'] += 1
            continue
        cache.append({
            'race_id': race['id'],
            'race_date': race['race_date'],
            'venue_id': race['venue_id'],
            'race_number': race['race_number'],
            'probs_1st': pred['probs_1st'],
            'mc_probs': mc_probs,
            'odds_data': odds_data,
            'actual_trifecta': race['actual_result_trifecta'],
            'payout_sanrentan': int(race['payout_sanrentan'] or 0),
        })
    logger.info(f'cache: {len(cache)} 件、skip {dict(skipped)}')
    return cache


def evaluate_combo(cache, base_config, override, kb):
    config = copy.deepcopy(base_config)
    config['kelly_fraction'] = override['kelly_fraction']
    config['min_expected_value'] = override['min_expected_value']
    bankroll = INITIAL_BANKROLL
    bets_log = []
    for c in cache:
        ok, _ = race_level_filter(c['probs_1st'], c['venue_id'], c['race_number'], config)
        if not ok:
            continue
        try:
            bets = kb._strategy_kelly(
                config=config, sanrentan_probs=c['mc_probs'],
                odds_data=c['odds_data'], bankroll=bankroll,
                strategy_name=BASE_STRATEGY,
                venue_id=c['venue_id'], race_number=c['race_number'],
            )
        except Exception:
            continue
        if not bets:
            continue
        actual = c['actual_trifecta']
        payout = c['payout_sanrentan']
        for bet in bets:
            stake = bet.get('amount', 100)
            combo = bet.get('combination')
            is_hit = (combo == actual)
            actual_return = (stake / 100 * payout) if is_hit else 0
            pnl = actual_return - stake
            bets_log.append({
                'race_id': c['race_id'],
                'race_date': c['race_date'],
                'combo': combo,
                'stake': stake,
                'is_hit': is_hit,
                'actual_return': actual_return,
                'pnl': pnl,
            })
            bankroll += pnl
    return summarize(bets_log)


def summarize(bets_log):
    n = len(bets_log)
    if n == 0:
        return {'n_bets': 0}
    df = pd.DataFrame(bets_log)
    n_hit = int(df['is_hit'].sum())
    total_stake = int(df['stake'].sum())
    total_payout = int(df['actual_return'].sum())
    total_pnl = int(df['pnl'].sum())
    roi = total_payout / total_stake * 100 if total_stake else 0
    daily = df.groupby('race_date')['pnl'].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    cum = df['pnl'].cumsum()
    running_max = cum.cummax()
    mdd = int((running_max - cum).max())
    return {
        'n_bets': n,
        'n_hit': n_hit,
        'hit_rate': float(n_hit / n * 100),
        'total_stake': total_stake,
        'total_payout': total_payout,
        'total_pnl': total_pnl,
        'roi': float(roi),
        'sharpe_daily': float(sharpe),
        'mdd': mdd,
    }


def write_report(results):
    lines = []
    lines.append("# Phase D D0+D1 グリッドサーチ結果\n\n")
    lines.append(f"対象戦略: {BASE_STRATEGY}\n")
    lines.append(f"期間: {DATE_FROM} 〜 {DATE_TO}\n")
    lines.append(f"初期 bankroll: ¥{INITIAL_BANKROLL:,}\n\n")
    lines.append("## 6 組合せ比較\n\n")
    lines.append("| # | 設定 | bets | hit_rate | ROI | PnL | Sharpe | MDD |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(results, 1):
        m = r['metrics']
        if m['n_bets'] == 0:
            lines.append(f"| {i} | {r['label']} | 0 | - | - | - | - | - |\n")
        else:
            lines.append(f"| {i} | {r['label']} | {m['n_bets']} | {m['hit_rate']:.1f}% | "
                         f"{m['roi']:.1f}% | ¥{m['total_pnl']:+,} | {m['sharpe_daily']:.3f} | ¥{m['mdd']:,} |\n")
    valid = [r for r in results if r['metrics']['n_bets'] > 0]
    if valid:
        best = max(valid, key=lambda r: r['metrics']['roi'])
        baseline = results[0]
        lines.append("\n## ベスト組合せ\n\n")
        lines.append(f"**{best['label']}** — ROI {best['metrics']['roi']:.1f}% "
                     f"(baseline P6 default {baseline['metrics']['roi']:.1f}% / 差 "
                     f"{best['metrics']['roi'] - baseline['metrics']['roi']:+.1f}pt)\n\n")
        improvement = best['metrics']['roi'] - baseline['metrics']['roi']
        if improvement > 5:
            verdict = "✅ 本番反映候補、D2 (動的 kelly) に進む"
        elif improvement > 0:
            verdict = "🟡 微改善、D2 で更に伸びるか検証"
        else:
            verdict = "❌ 現状維持推奨、Phase D 撤退検討"
        lines.append(f"判定: **{verdict}**\n")
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


def main():
    logger.info(f"=== Phase D D0+D1 backtest {DATE_FROM} 〜 {DATE_TO} ===")
    if CACHE_PATH.exists():
        logger.info(f'cache 既存、読み込み: {CACHE_PATH}')
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
    else:
        races, boats_map = fetch_races(DATE_FROM, DATE_TO)
        logger.info(f'races: {len(races)}')
        odds_map = load_odds_map(DATE_FROM, DATE_TO)
        predictor = RealtimePredictor()
        cache = build_cache(predictor, races, boats_map, odds_map)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f'cache 保存: {CACHE_PATH}')

    kb = KellyBettingStrategy(initial_bankroll=INITIAL_BANKROLL)
    base_config = kb.config['strategies'][BASE_STRATEGY]

    results = []
    for override in GRID:
        logger.info(f"  combo: {override['label']}")
        metrics = evaluate_combo(cache, base_config, override, kb)
        results.append({'label': override['label'], 'override': override, 'metrics': metrics})
        logger.info(f"    bets={metrics.get('n_bets', 0)} ROI={metrics.get('roi', 0):.1f}%")

    write_report(results)
    out_json = REPORT_DIR / 'phase_d_grid.json'
    out_json.write_text(json.dumps([{
        'label': r['label'], 'override': r['override'], 'metrics': r['metrics']
    } for r in results], indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    logger.info(f"JSON 出力: {out_json}")
    logger.info("=== 完了 ===")


if __name__ == '__main__':
    main()
