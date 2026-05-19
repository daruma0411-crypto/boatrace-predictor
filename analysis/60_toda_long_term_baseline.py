"""戸田 vs 全会場 長期 baseline (2025-2026)

cache (5 週間) ではなく **2025-2026 通年 data** で戸田特性を確認。
QMC compute_ratings_early の 11 項目係数が「全国 heuristic」のままだが、
戸田特有の特性 (淡水・狭水面・捲り発生) が data でどう現れるかを集計。

検証項目:
  Q1: 戸田 1号艇 1着率 × class (A1/A2/B1/B2)
  Q2: 戸田 1号艇 1着率 × 風速 / 波高
  Q3: 戸田 1号艇 1着率 × 展示タイム偏差
  Q4: 全国平均と戸田の各項目 effect size 差
  Q5: 戸田の 1着艇分布 (1-6号艇)

これで「全国 heuristic 係数を戸田で適用する場合の不適合度」を data 化。
"""
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / 'analysis' / 'reports' / '60_toda_long_term_baseline.md'

TODA_VENUE_ID = 2


def fetch_data():
    """戸田 + 全会場の race + boats data (2025-2026)."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id AS race_id, r.venue_id, r.race_date, r.race_number,
               r.result_1st, r.result_2nd, r.result_3rd, r.payout_sanrentan,
               r.wind_speed, r.wave_height
        FROM races r
        WHERE r.race_date >= '2025-06-01' AND r.result_1st IS NOT NULL
          AND r.payout_sanrentan IS NOT NULL
    """)
    races = {r['race_id']: dict(r) for r in cur.fetchall()}
    race_ids = list(races.keys())
    # boats を chunk で取得 (n大 → 分割)
    boats_map = defaultdict(dict)
    CHUNK = 10000
    for i in range(0, len(race_ids), CHUNK):
        chunk = race_ids[i:i+CHUNK]
        cur.execute("""
            SELECT race_id, boat_number, player_class, exhibition_time, motor_win_rate_2,
                   avg_st, approach_course, win_rate_2, local_win_rate_2
            FROM boats WHERE race_id = ANY(%s)
        """, (chunk,))
        for r in cur.fetchall():
            boats_map[r['race_id']][r['boat_number']] = dict(r)
    conn.close()
    # races に boats attach
    result = []
    for rid, race in races.items():
        boats = boats_map.get(rid, {})
        if len(boats) < 6:
            continue
        race['boats'] = [boats.get(i) for i in range(1, 7)]
        race['boat1'] = boats.get(1)
        race['boat4'] = boats.get(4)
        result.append(race)
    return result


def main():
    logger.info("戸田 vs 全会場 長期 baseline")
    races = fetch_data()
    toda = [r for r in races if r['venue_id'] == TODA_VENUE_ID]
    other = [r for r in races if r['venue_id'] != TODA_VENUE_ID]
    logger.info(f"戸田: {len(toda)} races, 他会場: {len(other)} races")

    lines = []
    lines.append("# 戸田 vs 全会場 長期 baseline (2025-06 〜 2026-05)\n\n")
    lines.append(f"対象: 戸田 {len(toda)} races / 他 23 会場 {len(other)} races\n")
    lines.append("**V10 不使用、実 actual data のみで戸田特性を集計**\n\n")

    # Q1: 1号艇 1着率 × 1号艇クラス
    lines.append("## Q1: 1号艇 1着率 × 1号艇クラス\n\n")
    lines.append("compute_ratings_early の class 係数は全国 heuristic (A1:0.75, A2:0.90 等)。\n")
    lines.append("戸田で 1号艇 A1 が実際どれだけ強いか確認。\n\n")
    lines.append("| 1号艇クラス | 戸田 n | 戸田 1着率 | 他会場 n | 他会場 1着率 | 戸田 effect |\n|---|---|---|---|---|---|\n")
    for cls in ['A1', 'A2', 'B1', 'B2']:
        toda_filt = [r for r in toda if r['boat1'] and r['boat1']['player_class'] == cls]
        other_filt = [r for r in other if r['boat1'] and r['boat1']['player_class'] == cls]
        toda_rate = sum(1 for r in toda_filt if r['result_1st'] == 1) / len(toda_filt) * 100 if toda_filt else 0
        other_rate = sum(1 for r in other_filt if r['result_1st'] == 1) / len(other_filt) * 100 if other_filt else 0
        effect = toda_rate - other_rate
        lines.append(f"| {cls} | {len(toda_filt)} | {toda_rate:.2f}% | {len(other_filt)} | {other_rate:.2f}% | {effect:+.2f}pt |\n")

    # Q2: 1号艇 1着率 × 風速
    lines.append("\n## Q2: 1号艇 1着率 × 風速\n\n")
    lines.append("compute_ratings_early の weather_factor は全国 heuristic (wind>=5 → 1.15x std)。\n")
    lines.append("戸田 (横風強い水面) で風の効果が違うか確認。\n\n")
    lines.append("| 風速 m/s | 戸田 n | 戸田 1号艇 1着率 | 他会場 n | 他会場 1号艇 1着率 | 戸田 effect |\n|---|---|---|---|---|---|\n")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 99)]:
        toda_filt = [r for r in toda if (r['wind_speed'] or 0) >= lo and (r['wind_speed'] or 0) < hi]
        other_filt = [r for r in other if (r['wind_speed'] or 0) >= lo and (r['wind_speed'] or 0) < hi]
        toda_rate = sum(1 for r in toda_filt if r['result_1st'] == 1) / len(toda_filt) * 100 if toda_filt else 0
        other_rate = sum(1 for r in other_filt if r['result_1st'] == 1) / len(other_filt) * 100 if other_filt else 0
        effect = toda_rate - other_rate
        lines.append(f"| {lo}-{hi} | {len(toda_filt)} | {toda_rate:.2f}% | {len(other_filt)} | {other_rate:.2f}% | {effect:+.2f}pt |\n")

    # Q3: 1号艇 1着率 × 展示タイム偏差
    lines.append("\n## Q3: 1号艇 1着率 × 1号艇展示タイム偏差\n\n")
    lines.append("compute_ratings_early の展示タイム補正は diff>0.10 → 1.12x std (全国)。\n")
    lines.append("戸田特性 (淡水・硬い水面で展示が変わりやすい?) を確認。\n\n")

    def b1_ex_diff(r):
        if not r.get('boats') or not r['boat1']:
            return None
        ex_times = [b['exhibition_time'] for b in r['boats'] if b and b.get('exhibition_time')]
        if len(ex_times) < 4 or not r['boat1'].get('exhibition_time'):
            return None
        avg = sum(ex_times) / len(ex_times)
        return r['boat1']['exhibition_time'] - avg

    lines.append("| 1号艇展示偏差 | 戸田 n | 戸田 1号艇 1着率 | 他会場 n | 他会場 1号艇 1着率 | 戸田 effect |\n|---|---|---|---|---|---|\n")
    for lo, hi in [(-1, -0.10), (-0.10, -0.05), (-0.05, 0.05), (0.05, 0.10), (0.10, 1)]:
        toda_filt = [r for r in toda if (d := b1_ex_diff(r)) is not None and lo <= d < hi]
        other_filt = [r for r in other if (d := b1_ex_diff(r)) is not None and lo <= d < hi]
        toda_rate = sum(1 for r in toda_filt if r['result_1st'] == 1) / len(toda_filt) * 100 if toda_filt else 0
        other_rate = sum(1 for r in other_filt if r['result_1st'] == 1) / len(other_filt) * 100 if other_filt else 0
        effect = toda_rate - other_rate
        lines.append(f"| {lo:+.2f}〜{hi:+.2f} | {len(toda_filt)} | {toda_rate:.2f}% | {len(other_filt)} | {other_rate:.2f}% | {effect:+.2f}pt |\n")

    # Q4: 1着艇分布全体 (戸田 vs 他会場)
    lines.append("\n## Q4: 全 1着艇分布\n\n")
    lines.append("戸田と他会場の艇別 1着率 (全 race ベース)\n\n")
    lines.append("| 艇 | 戸田 1着率 | 他会場 1着率 | 差 |\n|---|---|---|---|\n")
    for boat in range(1, 7):
        toda_rate = sum(1 for r in toda if r['result_1st'] == boat) / len(toda) * 100
        other_rate = sum(1 for r in other if r['result_1st'] == boat) / len(other) * 100
        lines.append(f"| {boat} | {toda_rate:.2f}% | {other_rate:.2f}% | {toda_rate-other_rate:+.2f}pt |\n")

    # Q5: 月別 1号艇 1着率 (時系列で安定性確認)
    lines.append("\n## Q5: 月別 1号艇 1着率 (時系列、戸田)\n\n")
    by_month = defaultdict(list)
    for r in toda:
        ym = r['race_date'].strftime('%Y-%m')
        by_month[ym].append(r['result_1st'] == 1)
    lines.append("| 月 | n | 戸田 1号艇 1着率 |\n|---|---|---|\n")
    for ym in sorted(by_month.keys()):
        vals = by_month[ym]
        rate = sum(vals) / len(vals) * 100
        lines.append(f"| {ym} | {len(vals)} | {rate:.2f}% |\n")

    # Q6: 展示タイム偏差 × クラスの相互作用 (戸田)
    lines.append("\n## Q6: 戸田 — 1号艇クラス × 展示タイム偏差 クロス\n\n")
    lines.append("class 係数と展示係数が独立に動くか、相互作用あるか確認。\n\n")
    lines.append("| 1号艇クラス | 展示偏差 < 0 (好) | 展示偏差 0-0.10 | 展示偏差 > 0.10 (悪) |\n|---|---|---|---|\n")
    for cls in ['A1', 'A2', 'B1', 'B2']:
        cells = []
        for lo, hi in [(-1, 0), (0, 0.10), (0.10, 1)]:
            sub = [r for r in toda if r['boat1'] and r['boat1']['player_class'] == cls
                   and (d := b1_ex_diff(r)) is not None and lo <= d < hi]
            if sub:
                rate = sum(1 for r in sub if r['result_1st'] == 1) / len(sub) * 100
                cells.append(f"{rate:.1f}% (n={len(sub)})")
            else:
                cells.append('-')
        lines.append(f"| {cls} | {cells[0]} | {cells[1]} | {cells[2]} |\n")

    # 結論ヒント
    lines.append("\n## ヒント (data から読める示唆)\n\n")
    # 戸田 A1 1号艇 1着率 vs 他会場 A1
    toda_a1 = [r for r in toda if r['boat1'] and r['boat1']['player_class'] == 'A1']
    other_a1 = [r for r in other if r['boat1'] and r['boat1']['player_class'] == 'A1']
    if toda_a1 and other_a1:
        toda_a1_rate = sum(1 for r in toda_a1 if r['result_1st'] == 1) / len(toda_a1) * 100
        other_a1_rate = sum(1 for r in other_a1 if r['result_1st'] == 1) / len(other_a1) * 100
        diff = toda_a1_rate - other_a1_rate
        lines.append(f"- 1号艇 A1: 戸田 {toda_a1_rate:.1f}% vs 他会場 {other_a1_rate:.1f}% (差 {diff:+.1f}pt)\n")
        if diff < -10:
            lines.append("  → 戸田は A1 が他会場より弱い、class 係数 0.75 は戸田で過大\n")

    # 月別変動性
    monthly_rates = [sum(by_month[ym]) / len(by_month[ym]) * 100 for ym in sorted(by_month.keys())]
    if monthly_rates:
        std_rate = float(np.std(monthly_rates))
        mean_rate = float(np.mean(monthly_rates))
        lines.append(f"- 戸田月別 1号艇 1着率: mean {mean_rate:.1f}% / std {std_rate:.1f}pt\n")
        if std_rate > 5:
            lines.append("  → 月別変動 大、短期 calibrator が機能しない理由\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- V10 NN は不使用、実 actual のみ集計\n")
    lines.append("- 2025-06 以降 約 1 年データ、戸田 ~2272 races\n")
    lines.append("- QMC 11 項目係数の戸田 fit を行うか、岩下さんの判断\n")
    lines.append("- 結論は出さず、岩下さんに判断委ねる\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
