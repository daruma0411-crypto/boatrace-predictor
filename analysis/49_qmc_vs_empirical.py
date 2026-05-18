"""QMC 過去 120 通り再計算 + 実データ突合分析 (CLAUDE.md 批判プロトコル準拠)

P7 失敗の根本原因「QMC が予測した 120 通り分布と実データの突合が一度も
されていなかった」を解消する。QMC 再計算は 1 ヶ月 1.3 分 (15.3 ms/race)
で可能なので、過去 1 ヶ月分の全 race × 120 combo を全部再計算してから
突合する。

入力: races + boats + predictions (NN probs_1st 取得用) (READ-ONLY)
出力:
  - analysis/qmc_predictions_cache.pkl: race_id → {qmc_probs (120 通り), actual}
  - analysis/reports/qmc_vs_empirical.md
"""
import os
import sys
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

from src.monte_carlo import qmc_sanrentan_v3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'qmc_predictions_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'qmc_vs_empirical.md'

PERIOD_START = date(2026, 4, 11)
PERIOD_END = date(2026, 5, 18)


def fetch_data():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    # races + 結果
    cur.execute("""
        SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
               r.result_1st, r.result_2nd, r.result_3rd,
               r.actual_result_trifecta, r.payout_sanrentan,
               r.wind_speed, r.wind_direction, r.wave_height,
               r.temperature, r.water_temperature
        FROM races r
        WHERE r.is_finished = true AND r.actual_result_trifecta IS NOT NULL
          AND r.race_date BETWEEN %s AND %s
        ORDER BY r.race_date, r.id
    """, (PERIOD_START, PERIOD_END))
    races = [dict(r) for r in cur.fetchall()]
    race_ids = [r['race_id'] for r in races]
    # boats
    cur.execute("""
        SELECT race_id, boat_number, player_class, win_rate_2, local_win_rate_2,
               avg_st, motor_win_rate_2, exhibition_time, approach_course,
               is_new_motor, tilt, parts_changed, weight
        FROM boats WHERE race_id = ANY(%s) ORDER BY race_id, boat_number
    """, (race_ids,))
    boats_by = defaultdict(list)
    for b in cur.fetchall():
        boats_by[b['race_id']].append(dict(b))
    # NN probs (predictions table、最新のものを 1 件だけ)
    cur.execute("""
        SELECT DISTINCT ON (race_id) race_id, probabilities_1st
        FROM predictions
        WHERE race_id = ANY(%s) AND probabilities_1st IS NOT NULL
        ORDER BY race_id, id DESC
    """, (race_ids,))
    probs_by = {r['race_id']: r['probabilities_1st'] for r in cur.fetchall()}
    conn.close()
    return races, boats_by, probs_by


def build_cache(races, boats_by, probs_by):
    cache = {}
    skipped = defaultdict(int)
    for i, race in enumerate(races):
        if (i + 1) % 500 == 0:
            logger.info(f'  QMC 再計算 {i+1}/{len(races)}')
        rid = race['race_id']
        boats = boats_by.get(rid)
        if not boats or len(boats) != 6:
            skipped['no_boats'] += 1
            continue
        probs_1st = probs_by.get(rid)
        if not probs_1st or not isinstance(probs_1st, list) or len(probs_1st) != 6:
            skipped['no_probs'] += 1
            continue
        race_data = {
            'wind_speed': race.get('wind_speed') or 0,
            'wind_direction': race.get('wind_direction') or 'calm',
            'wave_height': race.get('wave_height') or 0,
            'temperature': race.get('temperature') or 20,
            'water_temperature': race.get('water_temperature') or 20,
        }
        try:
            qmc_probs = qmc_sanrentan_v3(
                probs_1st, boats_data=boats,
                race_data=race_data, race_number=race['race_number']
            )
        except Exception as e:
            skipped['qmc_fail'] += 1
            continue
        cache[rid] = {
            'qmc_probs': qmc_probs,
            'actual': race['actual_result_trifecta'],
            'venue_id': race['venue_id'],
            'race_number': race['race_number'],
            'payout': int(race['payout_sanrentan'] or 0),
            'result_1st': race['result_1st'],
            'result_2nd': race['result_2nd'],
            'result_3rd': race['result_3rd'],
            'boats': boats,
            'race_data': race_data,
        }
    logger.info(f'  QMC 再計算完了: {len(cache)} 件 (skipped: {dict(skipped)})')
    return cache


def detect_filter_triggers(cache):
    """各 race で P7 の 4 つの B フィルタ trigger 状態を判定"""
    triggers = {}
    for rid, c in cache.items():
        boats = c['boats']
        race_data = c['race_data']
        b1 = next((b for b in boats if b['boat_number'] == 1), None)
        b4_exh_rank_top = False
        b1_exh_offset = None
        if boats and len(boats) == 6:
            exhs = [(b['boat_number'], b.get('exhibition_time')) for b in boats
                    if b.get('exhibition_time') is not None]
            if len(exhs) == 6:
                fastest_boat = min(exhs, key=lambda x: x[1])[0]
                if fastest_boat == 4:
                    b4_exh_rank_top = True
                if b1 and b1.get('exhibition_time') is not None:
                    others = [float(b.get('exhibition_time')) for b in boats
                              if b['boat_number'] != 1 and b.get('exhibition_time') is not None]
                    if len(others) == 5:
                        b1_exh_offset = float(b1['exhibition_time']) - sum(others) / 5
        triggers[rid] = {
            'H09': b1_exh_offset is not None and b1_exh_offset > 0.10,
            'H10': b4_exh_rank_top,
            'H16': (race_data.get('wave_height') or 0) > 4,
            'H27': b1 is not None and b1.get('weight') is not None and float(b1['weight']) > 55,
        }
    return triggers


def calibration_analysis(cache, triggers, filter_name):
    """指定 filter trigger 時の QMC vs 実データ突合

    Returns: dict with bias info
    """
    if filter_name == 'baseline':
        race_ids = list(cache.keys())
    else:
        race_ids = [rid for rid, t in triggers.items() if t.get(filter_name)]
    if not race_ids:
        return None
    n_races = len(race_ids)
    # 各 combo に対し: QMC 平均確率, 実頻度
    combo_qmc_sum = defaultdict(float)
    combo_count = defaultdict(int)
    combo_actual_count = defaultdict(int)
    for rid in race_ids:
        c = cache[rid]
        actual = c['actual']
        if not actual:
            continue
        for combo, prob in c['qmc_probs'].items():
            combo_qmc_sum[combo] += prob
            combo_count[combo] += 1
        combo_actual_count[actual] += 1
    # combo 別 平均 QMC 予測 と 実頻度
    rows = []
    for combo in set(list(combo_qmc_sum.keys()) + list(combo_actual_count.keys())):
        qmc_mean = combo_qmc_sum[combo] / n_races if n_races else 0
        actual_rate = combo_actual_count[combo] / n_races if n_races else 0
        bias = actual_rate - qmc_mean  # 正 = QMC 過小評価、負 = QMC 過大評価
        rows.append({
            'combo': combo,
            'qmc_mean': qmc_mean * 100,
            'actual_rate': actual_rate * 100,
            'bias': bias * 100,
            'actual_hits': combo_actual_count[combo],
        })
    df = pd.DataFrame(rows).sort_values('bias', ascending=False)
    return {
        'n_races': n_races,
        'df': df,
    }


def main():
    if CACHE_PATH.exists():
        logger.info(f'cache 既存、読み込み: {CACHE_PATH}')
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
    else:
        logger.info(f"data fetch {PERIOD_START}〜{PERIOD_END}")
        races, boats_by, probs_by = fetch_data()
        logger.info(f"races: {len(races)}, NN probs 取得: {len(probs_by)}")
        logger.info("QMC v3 再計算開始")
        cache = build_cache(races, boats_by, probs_by)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f'cache 保存: {CACHE_PATH}')

    triggers = detect_filter_triggers(cache)

    lines = []
    lines.append("# QMC vs 実データ 突合分析\n\n")
    lines.append(f"対象: {PERIOD_START}〜{PERIOD_END} (cache races {len(cache)})\n\n")
    lines.append("各 filter trigger 時に QMC が予測した 120 通り 3 連単分布の平均と、\n")
    lines.append("実データの経験的着順頻度を比較。**bias = 実頻度 - QMC 平均**\n")
    lines.append("正 → QMC 過小評価 (= 実データで頻出するのに予測値低い、買い目候補)\n")
    lines.append("負 → QMC 過大評価\n\n")

    # 全体ベースライン
    bl = calibration_analysis(cache, triggers, 'baseline')
    if bl:
        lines.append(f"## ベースライン (全 {bl['n_races']} races)\n\n")
        lines.append("### QMC 過小評価 (bias 正、買い目候補) TOP 10\n\n")
        lines.append("| combo | QMC 予測 | 実頻度 | bias (pt) | actual hits |\n|---|---|---|---|---|\n")
        for _, r in bl['df'].head(10).iterrows():
            lines.append(f"| {r['combo']} | {r['qmc_mean']:.3f}% | {r['actual_rate']:.3f}% | **{r['bias']:+.3f}** | {r['actual_hits']} |\n")
        lines.append("\n### QMC 過大評価 (bias 負、回避候補) TOP 10\n\n")
        lines.append("| combo | QMC 予測 | 実頻度 | bias (pt) | actual hits |\n|---|---|---|---|---|\n")
        for _, r in bl['df'].tail(10).iterrows():
            lines.append(f"| {r['combo']} | {r['qmc_mean']:.3f}% | {r['actual_rate']:.3f}% | **{r['bias']:+.3f}** | {r['actual_hits']} |\n")

    # 各 filter trigger
    for filt in ['H09', 'H10', 'H16', 'H27']:
        res = calibration_analysis(cache, triggers, filt)
        if not res or res['n_races'] < 30:
            lines.append(f"\n## {filt} (n_races={res['n_races'] if res else 0} < 30、解析省略)\n")
            continue
        n = res['n_races']
        df = res['df']
        lines.append(f"\n## {filt} trigger 時 (n={n} races)\n\n")
        lines.append("### QMC が「採用外」とした買い目で実頻出した着順 (bias TOP 10、買い目候補)\n\n")
        lines.append("| combo | QMC 予測 | 実頻度 | bias (pt) | actual hits |\n|---|---|---|---|---|\n")
        # 採用外 = QMC が低確率付与、ただし実頻出 → bias 大
        sub = df[df['actual_hits'] >= 2].sort_values('bias', ascending=False).head(10)
        for _, r in sub.iterrows():
            lines.append(f"| {r['combo']} | {r['qmc_mean']:.3f}% | {r['actual_rate']:.3f}% | **{r['bias']:+.3f}** | {r['actual_hits']} |\n")
        lines.append("\n### QMC が「重視」したが実データ低頻度 (bias 負 TOP 5、回避候補)\n\n")
        lines.append("| combo | QMC 予測 | 実頻度 | bias (pt) | actual hits |\n|---|---|---|---|---|\n")
        sub2 = df.sort_values('bias').head(5)
        for _, r in sub2.iterrows():
            lines.append(f"| {r['combo']} | {r['qmc_mean']:.3f}% | {r['actual_rate']:.3f}% | **{r['bias']:+.3f}** | {r['actual_hits']} |\n")

    # サマリ
    lines.append("\n## 重要な留意 (CLAUDE.md 批判プロトコル準拠)\n\n")
    lines.append("- bias が正の combo: 「QMC が低く見積もったが実は頻出」= **モデルの blind spot、市場で過小オッズ**\n")
    lines.append("- bias が負の combo: 「QMC が高く見積もったが実は外れ」= **モデルの過信、市場で過大オッズだが買うべきでない**\n")
    lines.append("- ただし actual_hits < 5 の combo は標本不足、bias の信頼性低い\n")
    lines.append("- QMC の compute_ratings_early ヒューリスティック (展示偏差 / クラス係数 / 風波 等) の\n")
    lines.append("  どの項目が原因かは別途特定必要 (本レポートは現象論まで)\n")
    lines.append("- 採用判断は岩下さんが下す。本レポートは論点提供まで\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
