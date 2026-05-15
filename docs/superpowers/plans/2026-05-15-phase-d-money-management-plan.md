# Phase D 資金管理パラメータ最適化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** P6 戦略 (mc3_venue_focus_r4) で 6 組合せの `(kelly_fraction, min_expected_value)` を 2026-03〜04 の 2 ヶ月 backtest し、ROI/Sharpe/MDD/bets 数で比較。動的 `kelly_prob_gain` の効果も検証。最終的に「本番反映候補のパラメータ」または「現状維持推奨」を提言。

**Architecture:** 既存 `analysis/17_backtest_apr2026.py` の backtest 構造を流用。races + boats + odds + V10 推論結果を 1 回キャッシュし、6 組合せに対して kelly 計算と PnL 集計のみを切替て高速比較。本番 `config/betting_config.json` は触らず、`copy.deepcopy(strategy_config)` で in-memory override。

**Tech Stack:** Python 3、`RealtimePredictor` (V10)、`KellyBettingStrategy`、`monte_carlo_sanrentan`、pandas、psycopg2。

**Spec:** `docs/superpowers/specs/2026-05-15-phase-d-money-management-design.md`

**Issue:** https://github.com/daruma0411-crypto/boatrace-predictor/issues/4

---

## File Structure

| Path | Action | 責務 |
|---|---|---|
| `analysis/40_phase_d_backtest.py` | Create | D0+D1 (6 組合せ) backtest スクリプト、V10 推論キャッシュ + 並列 kelly 評価 |
| `analysis/41_phase_d_dynamic_kelly.py` | Create | D2 動的 kelly_prob_gain backtest、Task 1 のベスト組合せベース |
| `analysis/reports/phase_d_grid.md` | Generated | 6 組合せ比較表 (ROI/bets/Sharpe/MDD) |
| `analysis/reports/phase_d_dynamic.md` | Generated | 動的 gain 結果 + 最終提言 |
| `analysis/phase_d_cache.pkl` | Generated (一時) | races + V10 probs + MC probs キャッシュ (6 組合せ高速化用) |

---

## Task 1: D0+D1 — 6 組合せ backtest

**Files:**
- Create: `analysis/40_phase_d_backtest.py`
- Reference (READ-ONLY): `analysis/17_backtest_apr2026.py`, `src/betting.py`, `config/betting_config.json`

### - [ ] Step 1: スクリプト全体を作成

`analysis/40_phase_d_backtest.py` を新規作成:

```python
"""Phase D D0+D1: 6 組合せ backtest (kelly_fraction × min_expected_value)

ベース: mc3_venue_focus_r4 (P6) を 2026-03-01 〜 2026-04-30 で評価。
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
import calendar
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
DATE_FROM = date(2026, 3, 1)
DATE_TO = date(2026, 4, 30)
INITIAL_BANKROLL = 200000

GRID = [
    {'kelly_fraction': 0.0625, 'min_expected_value': 0.0, 'label': 'P6 default (1/16, EV0)'},
    {'kelly_fraction': 0.0625, 'min_expected_value': 1.0, 'label': '1/16 + EV≥1.0'},
    {'kelly_fraction': 0.10,    'min_expected_value': 0.0, 'label': '1/10 + EV0'},
    {'kelly_fraction': 0.10,    'min_expected_value': 1.0, 'label': '1/10 + EV≥1.0'},
    {'kelly_fraction': 0.10,    'min_expected_value': 1.1, 'label': '1/10 + EV≥1.1'},
    {'kelly_fraction': 0.20,    'min_expected_value': 1.0, 'label': '1/5 + EV≥1.0'},
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
    """historical_data/{year}_{month:02d}/odds_3t.pkl を期間分マージ"""
    odds_map = {}
    # 月単位で読み込み
    months = set()
    d = date_from
    while d <= date_to:
        months.add((d.year, d.month))
        # next month
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
    """V10 推論 + MC sanrentan を全 race で 1 回だけ実行してキャッシュ"""
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
    """1 組合せの backtest 実行"""
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
    # best
    valid = [r for r in results if r['metrics']['n_bets'] > 0]
    if valid:
        best = max(valid, key=lambda r: r['metrics']['roi'])
        baseline = results[0]
        lines.append("\n## ベスト組合せ\n\n")
        lines.append(f"**{best['label']}** — ROI {best['metrics']['roi']:.1f}% "
                     f"(baseline P6 default {baseline['metrics']['roi']:.1f}% vs ベスト 差 "
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
```

### - [ ] Step 2: 実行

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python -u analysis/40_phase_d_backtest.py 2>&1 | tee /tmp/phase_d_run.log
```

期待:
- 初回: V10 推論 + MC × 8000 races で 20-40 分 (background 起動推奨)
- cache 構築後の 6 組合せ評価は数十秒〜数分
- 完了時に `analysis/reports/phase_d_grid.md` 出力

長時間想定のため `run_in_background: true` で起動して完了通知を待つこと。

### - [ ] Step 3: レポート確認

```bash
cat analysis/reports/phase_d_grid.md
```

検証ポイント:
- 6 組合せ全てに bets > 0 (極端な EV フィルタで全 skip にならないこと)
- baseline (P6 default) ROI が memory 記載 224% 近辺 (±50pt 程度の誤差は許容、期間が違うため完全一致は期待しない)
- ベスト組合せの判定が表示されている

異常時の対処:
- bets ゼロ多発 → kb の `_strategy_kelly` で `min_probability` や他のフィルタが効いている可能性。ログを確認
- ROI が極端に低い (10% 以下) → odds_map のキーマッチ問題、`load_odds_map` のキー形式を確認
- ROI が極端に高い (500%+) → bets 数極端に少ない (1-2 件)、サンプル偏り。bets 数を表で確認

### - [ ] Step 4: コミット

```bash
git add analysis/40_phase_d_backtest.py analysis/reports/phase_d_grid.md analysis/reports/phase_d_grid.json
git commit -m "feat(phase-d): D0+D1 6 組合せ backtest 実装と結果"
```

cache pkl (`analysis/phase_d_cache.pkl`) は gitignore 対象 (中間生成物)。コミット不要。実際に commit する際 `.gitignore` 既存設定を確認、ない場合は別途追加判断。

---

## Task 2: D2 — 動的 kelly_prob_gain backtest

**Files:**
- Create: `analysis/41_phase_d_dynamic_kelly.py`

Task 1 で生成された `analysis/phase_d_cache.pkl` を再利用。

### - [ ] Step 1: スクリプト全体を作成

`analysis/41_phase_d_dynamic_kelly.py` を新規作成:

```python
"""Phase D D2: 動的 kelly_prob_gain backtest

Task 1 のベスト (kelly_fraction, min_expected_value) を固定し、
kelly_prob_gain を entropy 別に変動させて改善幅を確認する。
"""
import os
import sys
import copy
import json
import pickle
import logging
import math
from datetime import date
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import pandas as pd

from src.betting import (
    KellyBettingStrategy, _should_skip_by_top_boat,
    VENUE_HONMEI, VENUE_ARE,
)

# Task 1 のヘルパーを再利用
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
m40 = import_module('40_phase_d_backtest')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'phase_d_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'phase_d_dynamic.md'
GRID_JSON = ROOT / 'analysis' / 'reports' / 'phase_d_grid.json'

INITIAL_BANKROLL = 200000


def entropy_of_probs(probs):
    """3 連単確率分布のエントロピー (高いほど低確信)"""
    total = sum(probs.values())
    if total <= 0:
        return float('inf')
    e = 0.0
    for p in probs.values():
        if p > 0:
            r = p / total
            e -= r * math.log(r)
    return e


def gain_for_entropy(entropy):
    """entropy → kelly_prob_gain map"""
    if entropy < 1.5:
        return 1.5
    if entropy < 2.0:
        return 1.2
    return 1.0


def evaluate_dynamic(cache, base_config, override, kb):
    config = copy.deepcopy(base_config)
    config['kelly_fraction'] = override['kelly_fraction']
    config['min_expected_value'] = override['min_expected_value']
    bankroll = INITIAL_BANKROLL
    bets_log = []
    for c in cache:
        ok, _ = m40.race_level_filter(c['probs_1st'], c['venue_id'], c['race_number'], config)
        if not ok:
            continue
        # 動的 gain
        config['kelly_prob_gain'] = gain_for_entropy(entropy_of_probs(c['mc_probs']))
        try:
            bets = kb._strategy_kelly(
                config=config, sanrentan_probs=c['mc_probs'],
                odds_data=c['odds_data'], bankroll=bankroll,
                strategy_name='mc3_venue_focus_r4',
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
                'race_id': c['race_id'], 'race_date': c['race_date'],
                'combo': combo, 'stake': stake, 'is_hit': is_hit,
                'actual_return': actual_return, 'pnl': pnl,
            })
            bankroll += pnl
    return m40.summarize(bets_log)


def main():
    if not CACHE_PATH.exists():
        raise SystemExit(f"cache 不在: {CACHE_PATH}。先に 40_phase_d_backtest.py を実行してください")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f'cache 読込: {len(cache)} レース')

    if not GRID_JSON.exists():
        raise SystemExit(f"Task 1 結果 JSON 不在: {GRID_JSON}")
    with open(GRID_JSON, 'r', encoding='utf-8') as f:
        grid_results = json.load(f)
    valid = [r for r in grid_results if r['metrics'].get('n_bets', 0) > 0]
    best = max(valid, key=lambda r: r['metrics']['roi'])
    logger.info(f"Task 1 ベスト: {best['label']} ROI {best['metrics']['roi']:.1f}%")

    kb = KellyBettingStrategy(initial_bankroll=INITIAL_BANKROLL)
    base_config = kb.config['strategies']['mc3_venue_focus_r4']

    # 静的 (Task 1 ベスト) と動的 gain を比較
    static = best['metrics']
    dynamic = evaluate_dynamic(cache, base_config, best['override'], kb)

    lines = []
    lines.append("# Phase D D2 動的 kelly_prob_gain 結果\n\n")
    lines.append(f"Task 1 ベース: {best['label']}\n")
    lines.append(f"gain rule: entropy<1.5→1.5, 1.5≤<2.0→1.2, ≥2.0→1.0\n\n")
    lines.append("## 比較\n\n")
    lines.append("| 指標 | 静的 gain=1.0 (Task 1 ベスト) | 動的 gain | 差 |\n|---|---|---|---|\n")
    lines.append(f"| n_bets | {static['n_bets']} | {dynamic['n_bets']} | {dynamic['n_bets']-static['n_bets']:+d} |\n")
    lines.append(f"| hit_rate | {static['hit_rate']:.1f}% | {dynamic['hit_rate']:.1f}% | {dynamic['hit_rate']-static['hit_rate']:+.1f}pt |\n")
    lines.append(f"| ROI | {static['roi']:.1f}% | {dynamic['roi']:.1f}% | {dynamic['roi']-static['roi']:+.1f}pt |\n")
    lines.append(f"| PnL | ¥{static['total_pnl']:+,} | ¥{dynamic['total_pnl']:+,} | ¥{dynamic['total_pnl']-static['total_pnl']:+,} |\n")
    lines.append(f"| Sharpe | {static['sharpe_daily']:.3f} | {dynamic['sharpe_daily']:.3f} | {dynamic['sharpe_daily']-static['sharpe_daily']:+.3f} |\n")
    lines.append(f"| MDD | ¥{static['mdd']:,} | ¥{dynamic['mdd']:,} | ¥{dynamic['mdd']-static['mdd']:+,} |\n")

    diff = dynamic['roi'] - static['roi']
    lines.append("\n## 最終提言\n\n")
    if diff > 5:
        verdict = f"✅ 動的 gain 採用、本番反映候補 (差 {diff:+.1f}pt)"
    elif diff > 0:
        verdict = f"🟡 微改善 ({diff:+.1f}pt)。静的 gain で十分、動的化は見送り"
    else:
        verdict = f"❌ 動的 gain は逆効果 ({diff:+.1f}pt)、静的 gain 採用"
    lines.append(f"{verdict}\n\n")
    lines.append(f"### 本番反映候補のパラメータ (mc3_venue_focus_r4)\n\n")
    lines.append(f"- `kelly_fraction`: {best['override']['kelly_fraction']}\n")
    lines.append(f"- `min_expected_value`: {best['override']['min_expected_value']}\n")
    lines.append(f"- `kelly_prob_gain`: {'動的 (entropy 別)' if diff > 5 else '1.0 (静的)'}\n\n")
    lines.append("**本番反映は別セッションで実施** (shadow 1-2 週並走後)。\n")
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
```

### - [ ] Step 2: 実行 (cache 流用、短時間)

```bash
python -u analysis/41_phase_d_dynamic_kelly.py 2>&1 | tail -20
```

期待: cache 流用なので 1-2 分以内に完了。

### - [ ] Step 3: レポート確認

```bash
cat analysis/reports/phase_d_dynamic.md
```

検証:
- 静的 vs 動的 の比較表 (ROI/PnL/Sharpe/MDD) が出ている
- 最終提言が「✅/🟡/❌」のいずれかで明示されている
- 本番反映候補のパラメータが具体的に記載されている

### - [ ] Step 4: コミット

```bash
git add analysis/41_phase_d_dynamic_kelly.py analysis/reports/phase_d_dynamic.md
git commit -m "feat(phase-d): D2 動的 kelly_prob_gain backtest と最終提言"
```

---

## Self-Review

- **Spec 網羅**:
  - D0 (期待値フィルタ) → Task 1 で `min_expected_value` を 0.0/1.0/1.1 で組合せ評価
  - D1 (kelly_fraction 増) → Task 1 で 0.0625/0.10/0.20
  - D2 (動的 gain) → Task 2 で entropy 別 gain
  - 評価指標 (ROI/bets/Sharpe/MDD) → `summarize()` 関数で全部出力
  - 撤退ライン → レポート出力時に判定文字列 (✅/🟡/❌) で明示
- **プレースホルダ**: なし。各スクリプト全文を Step 1 に貼付済み
- **型整合**: `summarize()` の戻り値 dict は Task 1/2 で同じキー (`n_bets`, `roi`, `total_pnl`, ...) を使用、`race_level_filter` シグネチャも一貫
- **粒度**: 各 Task 4 ステップ、長時間処理 (Task 1 Step 2) のみ background 起動指示
- **cache の扱い**: Task 1 で `analysis/phase_d_cache.pkl` を生成、Task 2 で再利用。中間生成物のため git 管理しない

## 完了後の状態

- `analysis/reports/phase_d_grid.md`: 6 組合せ比較表
- `analysis/reports/phase_d_dynamic.md`: 静的 vs 動的 + 最終提言
- 本番反映候補パラメータが明示される (or 「現状維持推奨」)
- 別セッションで shadow 並走 → 本番反映の判断材料完備
