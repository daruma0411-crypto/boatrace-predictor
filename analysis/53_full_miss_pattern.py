"""全 5828 races 外し方傾向分析

52 (P6 production 110 bets) はサンプル不足。cache 全 5828 races で QMC top pick の
外し方を仮説駆動で分析。岩下さん観察 (4号艇 A 級) も n>=1000 規模で再検証。

CLAUDE.md 批判プロトコル準拠、結論は出さず岩下さんの判断に委ねる。

検証する 6 仮説:
  H1: QMC top pick hit 率の baseline (全 races)
  H2: QMC が 1-X-X 予測時の actual 1着分布 (本命崩れ structural rate)
  H3: 4号艇 A 級観察の n>=1000 再検証 (52 で n=22 反証だったが確認)
  H4: 会場別 hit 率の散らばり
  H5: R 番号別 hit 率 (R4 systematic 負けの一般性検証)
  H6: 風波 / 展示タイム差 / クラス分散と hit 率の関係

出力: analysis/reports/53_full_miss_pattern.md
"""
import os
import sys
import pickle
import logging
import statistics
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
CACHE_PATH = ROOT / 'analysis' / 'qmc_predictions_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '53_full_miss_pattern.md'


def fetch_venue_race_meta(race_ids):
    """venue_id, race_number は cache に既にある (qmc_predictions_cache 確認済み)."""
    pass


def main():
    logger.info("全 5828 races miss pattern 分析")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"cache: {len(cache)} races")

    # 各 race の top pick を抽出
    races = []
    for rid, c in cache.items():
        qmc_probs = c.get('qmc_probs', {})
        actual = c.get('actual')
        if not qmc_probs or not actual:
            continue
        top_combo = max(qmc_probs.items(), key=lambda x: x[1])
        top_pick = top_combo[0]
        top_prob = top_combo[1]
        # top-3 picks も保持
        top3 = sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]
        top3_combos = [c[0] for c in top3]
        # 1-X-X 系合計
        sum_1xx = sum(p for k, p in qmc_probs.items() if k.startswith('1-'))
        sum_4xx = sum(p for k, p in qmc_probs.items() if k.startswith('4-'))
        # 4号艇クラス
        boat4_class = None
        boats = c.get('boats', [])
        if boats and len(boats) >= 4:
            boat4_class = boats[3].get('player_class')
        boat1_class = boats[0].get('player_class') if boats else None
        # クラス分散 (全 6 艇)
        class_values = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        class_arr = [class_values.get(b.get('player_class', 'B1'), 2) for b in boats] if len(boats) == 6 else None
        class_std = float(np.std(class_arr)) if class_arr else None

        rd = c.get('race_data', {})
        races.append({
            'race_id': rid,
            'venue_id': c.get('venue_id'),
            'race_number': c.get('race_number'),
            'top_pick': top_pick,
            'top_prob': top_prob,
            'top3': top3_combos,
            'sum_1xx': sum_1xx,
            'sum_4xx': sum_4xx,
            'actual': actual,
            'result_1st': c.get('result_1st'),
            'result_2nd': c.get('result_2nd'),
            'result_3rd': c.get('result_3rd'),
            'payout': c.get('payout'),
            'boat4_class': boat4_class,
            'boat1_class': boat1_class,
            'class_std': class_std,
            'wind_speed': rd.get('wind_speed') or 0,
            'wave_height': rd.get('wave_height') or 0,
            'top1_hit': top_pick == actual,
            'top3_hit': actual in top3_combos,
            'b1_win': c.get('result_1st') == 1,
        })

    logger.info(f"valid races: {len(races)}")

    lines = []
    lines.append("# 全 5828 races miss pattern 分析\n\n")
    lines.append(f"対象: cache {len(races)} races (49_qmc_vs_empirical.py 生成)\n")
    lines.append("各 race の QMC top pick / top-3 と actual を突合。岩下さん観察を n>=1000 規模で再検証。\n\n")

    # ============== ベースライン ==============
    n_total = len(races)
    n_top1 = sum(1 for r in races if r['top1_hit'])
    n_top3 = sum(1 for r in races if r['top3_hit'])
    n_b1_win = sum(1 for r in races if r['b1_win'])
    lines.append("## H1: QMC top pick hit 率の baseline\n\n")
    lines.append(f"- 全 races: **{n_total}**\n")
    lines.append(f"- top-1 hit: **{n_top1} ({n_top1/n_total*100:.2f}%)**\n")
    lines.append(f"- top-3 hit: **{n_top3} ({n_top3/n_total*100:.2f}%)** (3 連単 top-3 picks のいずれかが当たる)\n")
    lines.append(f"- actual 1着 = 1号艇: **{n_b1_win} ({n_b1_win/n_total*100:.2f}%)**\n\n")

    # ============== H2: 1-X-X 予測時の actual 分布 ==============
    lines.append("## H2: QMC が 1-X-X を top pick とした時の actual 1着分布\n\n")
    lines.append("**仮説**: QMC top pick = 1-X-X (本命買い) が外れた時、actual 1着がどう散らばるかで本命崩れの structural rate を測る。\n\n")
    pred1xx_actual = defaultdict(int)
    n_pred1xx = 0
    for r in races:
        if r['top_pick'].startswith('1-'):
            n_pred1xx += 1
            pred1xx_actual[r['result_1st']] += 1
    lines.append(f"- top pick = 1-X-X の race 数: {n_pred1xx}\n")
    lines.append("| actual 1着 | n | rate |\n|---|---|---|\n")
    for boat in range(1, 7):
        n = pred1xx_actual.get(boat, 0)
        lines.append(f"| {boat} 号艇 | {n} | {n/n_pred1xx*100:.2f}% |\n")

    # ============== H3: 4号艇 A 級観察 (n>=1000 再検証) ==============
    lines.append("\n## H3: 4号艇クラス別の挙動 (岩下さん観察 n>=1000 再検証)\n\n")
    lines.append("**観察**: 4号艇に A 級が入ると戦略発火、actual は 1号艇内の BC 級。\n")
    lines.append("52 (P6 110 bets) では n=22 で観察反証だが、cache 5828 races で再検証。\n\n")
    groups = {
        'A1': lambda c: c == 'A1',
        'A2': lambda c: c == 'A2',
        'A 級 (A1+A2)': lambda c: c in ('A1', 'A2'),
        'B1': lambda c: c == 'B1',
        'B2': lambda c: c == 'B2',
        'B 級 (B1+B2)': lambda c: c in ('B1', 'B2'),
    }
    lines.append("| 4号艇クラス | n_races | 1号艇 1着 (rate) | 4号艇 1着 (rate) | top-1 hit (rate) | top-3 hit (rate) |\n|---|---|---|---|---|---|\n")
    for label, cond in groups.items():
        rs = [r for r in races if cond(r['boat4_class'])]
        if not rs:
            continue
        n = len(rs)
        n_b1 = sum(1 for r in rs if r['b1_win'])
        n_b4 = sum(1 for r in rs if r['result_1st'] == 4)
        n_t1 = sum(1 for r in rs if r['top1_hit'])
        n_t3 = sum(1 for r in rs if r['top3_hit'])
        lines.append(f"| {label} | {n} | {n_b1} ({n_b1/n*100:.2f}%) | {n_b4} ({n_b4/n*100:.2f}%) | "
                     f"{n_t1} ({n_t1/n*100:.2f}%) | {n_t3} ({n_t3/n*100:.2f}%) |\n")

    # 「4号艇 A 級時に 1号艇 BC 級」のサブ分析
    lines.append("\n### 4号艇 A 級 × 1号艇クラス クロス\n\n")
    lines.append("「4号艇 A 級時、1号艇 BC 級が来る」の specifically検証:\n\n")
    lines.append("| 4号艇クラス | 1号艇クラス | n_races | 1号艇 1着 (rate) | 4号艇 1着 (rate) |\n|---|---|---|---|---|\n")
    for b4_label, b4_cond in [('A 級', lambda c: c in ('A1', 'A2')), ('B 級', lambda c: c in ('B1', 'B2'))]:
        for b1_label, b1_cond in [('A 級', lambda c: c in ('A1', 'A2')), ('B 級', lambda c: c in ('B1', 'B2'))]:
            rs = [r for r in races if b4_cond(r['boat4_class']) and b1_cond(r['boat1_class'])]
            if not rs:
                continue
            n = len(rs)
            n_b1 = sum(1 for r in rs if r['b1_win'])
            n_b4 = sum(1 for r in rs if r['result_1st'] == 4)
            lines.append(f"| 4号艇 {b4_label} | 1号艇 {b1_label} | {n} | {n_b1} ({n_b1/n*100:.2f}%) | {n_b4} ({n_b4/n*100:.2f}%) |\n")

    # ============== H4: 会場別 ==============
    lines.append("\n## H4: 会場別 hit 率\n\n")
    venue_stats = defaultdict(lambda: {'n': 0, 'top1': 0, 'top3': 0, 'b1': 0})
    for r in races:
        v = r['venue_id']
        if v is None:
            continue
        venue_stats[v]['n'] += 1
        venue_stats[v]['top1'] += int(r['top1_hit'])
        venue_stats[v]['top3'] += int(r['top3_hit'])
        venue_stats[v]['b1'] += int(r['b1_win'])
    # 上位 / 下位
    venue_rows = []
    for v, s in venue_stats.items():
        venue_rows.append({
            'venue': v, 'n': s['n'],
            'top1_rate': s['top1']/s['n']*100,
            'top3_rate': s['top3']/s['n']*100,
            'b1_rate': s['b1']/s['n']*100,
        })
    venue_rows.sort(key=lambda x: -x['top3_rate'])
    lines.append("**top-3 hit 率の高い会場 / 低い会場** (n=5828 / 24 venues = 平均 ~243 races/venue):\n\n")
    lines.append("| venue | n | top-1 hit% | top-3 hit% | 1号艇 1着% |\n|---|---|---|---|---|\n")
    for r in venue_rows[:5] + [None] + venue_rows[-5:]:
        if r is None:
            lines.append("| ... | | | | |\n")
            continue
        lines.append(f"| {r['venue']} | {r['n']} | {r['top1_rate']:.2f}% | {r['top3_rate']:.2f}% | {r['b1_rate']:.2f}% |\n")

    # ============== H5: R 番号別 ==============
    lines.append("\n## H5: R 番号別 hit 率 (R4 systematic 負けの一般性検証)\n\n")
    lines.append("**仮説**: 52 で R4 が systematic に negative。cache 全 races で R 番号別 hit 率の散らばりを確認。\n\n")
    r_stats = defaultdict(lambda: {'n': 0, 'top1': 0, 'top3': 0, 'b1': 0})
    for r in races:
        rn = r['race_number']
        if rn is None:
            continue
        r_stats[rn]['n'] += 1
        r_stats[rn]['top1'] += int(r['top1_hit'])
        r_stats[rn]['top3'] += int(r['top3_hit'])
        r_stats[rn]['b1'] += int(r['b1_win'])
    lines.append("| R | n | top-1 hit% | top-3 hit% | 1号艇 1着% |\n|---|---|---|---|---|\n")
    for rn in sorted(r_stats.keys()):
        s = r_stats[rn]
        lines.append(f"| R{rn} | {s['n']} | {s['top1']/s['n']*100:.2f}% | {s['top3']/s['n']*100:.2f}% | {s['b1']/s['n']*100:.2f}% |\n")

    # ============== H6: 気象 / クラス分散 ==============
    lines.append("\n## H6: 風波 / クラス分散と hit 率の関係\n\n")
    # 風速 buckets
    wind_buckets = [(0, 1.5), (1.5, 3), (3, 5), (5, 99)]
    lines.append("### 風速別\n\n")
    lines.append("| 風速 (m/s) | n | top-1 hit% | top-3 hit% | 1号艇 1着% |\n|---|---|---|---|---|\n")
    for lo, hi in wind_buckets:
        rs = [r for r in races if lo <= r['wind_speed'] < hi]
        if not rs: continue
        n = len(rs)
        t1 = sum(1 for r in rs if r['top1_hit'])
        t3 = sum(1 for r in rs if r['top3_hit'])
        b1 = sum(1 for r in rs if r['b1_win'])
        lines.append(f"| {lo:.1f}-{hi:.1f} | {n} | {t1/n*100:.2f}% | {t3/n*100:.2f}% | {b1/n*100:.2f}% |\n")

    # 波高 buckets
    wave_buckets = [(0, 2), (2, 4), (4, 99)]
    lines.append("\n### 波高別\n\n")
    lines.append("| 波高 (cm) | n | top-1 hit% | top-3 hit% | 1号艇 1着% |\n|---|---|---|---|---|\n")
    for lo, hi in wave_buckets:
        rs = [r for r in races if lo <= r['wave_height'] < hi]
        if not rs: continue
        n = len(rs)
        t1 = sum(1 for r in rs if r['top1_hit'])
        t3 = sum(1 for r in rs if r['top3_hit'])
        b1 = sum(1 for r in rs if r['b1_win'])
        lines.append(f"| {lo:.1f}-{hi:.1f} | {n} | {t1/n*100:.2f}% | {t3/n*100:.2f}% | {b1/n*100:.2f}% |\n")

    # クラス分散 buckets
    lines.append("\n### クラス分散別 (6 艇のクラス標準偏差)\n\n")
    lines.append("| class_std | n | top-1 hit% | top-3 hit% | 1号艇 1着% |\n|---|---|---|---|---|\n")
    cs_buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 99)]
    for lo, hi in cs_buckets:
        rs = [r for r in races if r['class_std'] is not None and lo <= r['class_std'] < hi]
        if not rs: continue
        n = len(rs)
        t1 = sum(1 for r in rs if r['top1_hit'])
        t3 = sum(1 for r in rs if r['top3_hit'])
        b1 = sum(1 for r in rs if r['b1_win'])
        lines.append(f"| {lo:.2f}-{hi:.2f} | {n} | {t1/n*100:.2f}% | {t3/n*100:.2f}% | {b1/n*100:.2f}% |\n")

    # ============== 採用候補抽出 ==============
    lines.append("\n## 採用基準による自動振り分け (CLAUDE.md 準拠)\n\n")

    # H3 判定
    a4_a_rs = [r for r in races if r['boat4_class'] in ('A1', 'A2')]
    a4_b_rs = [r for r in races if r['boat4_class'] in ('B1', 'B2')]
    if a4_a_rs and a4_b_rs:
        b1_rate_a = sum(1 for r in a4_a_rs if r['b1_win']) / len(a4_a_rs) * 100
        b1_rate_b = sum(1 for r in a4_b_rs if r['b1_win']) / len(a4_b_rs) * 100
        b4_rate_a = sum(1 for r in a4_a_rs if r['result_1st'] == 4) / len(a4_a_rs) * 100
        b4_rate_b = sum(1 for r in a4_b_rs if r['result_1st'] == 4) / len(a4_b_rs) * 100
        diff_b1 = b1_rate_a - b1_rate_b
        diff_b4 = b4_rate_a - b4_rate_b
        lines.append(f"- **H3 (4号艇 A 級観察)**: n_A={len(a4_a_rs)}, n_B={len(a4_b_rs)}\n")
        lines.append(f"  - 1号艇 1着率: A 級時 {b1_rate_a:.2f}% vs B 級時 {b1_rate_b:.2f}% (差 {diff_b1:+.2f}pt)\n")
        lines.append(f"  - 4号艇 1着率: A 級時 {b4_rate_a:.2f}% vs B 級時 {b4_rate_b:.2f}% (差 {diff_b4:+.2f}pt)\n")
        if diff_b1 < -5:
            lines.append(f"  - 🟢 観察支持 (4号艇 A 級時に 1号艇 1着率が顕著に低下)\n")
        elif diff_b4 > 10:
            lines.append(f"  - 🔴 観察反証 (4号艇 A 級時は 4号艇自身が 1着しやすい)\n")
        else:
            lines.append(f"  - ⚪ 効果限定 (差 小)\n")

    # H5 判定 (R 番号別の top-3 hit 率 spread)
    r_top3_rates = [s['top3']/s['n']*100 for s in r_stats.values() if s['n'] > 100]
    if r_top3_rates:
        spread = max(r_top3_rates) - min(r_top3_rates)
        lines.append(f"\n- **H5 (R 番号別)**: top-3 hit 率の spread = {spread:.2f}pt\n")
        if spread > 10:
            lines.append(f"  - 🟢 R 番号 依存性あり (filter 候補)\n")
        elif spread > 5:
            lines.append(f"  - 🟡 弱い R 番号依存性\n")
        else:
            lines.append(f"  - 🔴 R 番号で大差なし\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- n>=1000 規模の検証だが、観察基準 (4号艇 A 級時の挙動) は岩下さんの体感 / 印象との突合\n")
    lines.append("- top-3 hit 率は購入戦略 (Kelly/EV filter) を含まない単純突合、ROI proxy ではない\n")
    lines.append("- 採用候補は **shadow 2 週間必須**、P7 失敗教訓\n")
    lines.append("- QMC が systematic に 1-X-X 過大評価 (49 報告) を含む状態での top-3 hit 率である点に注意\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
