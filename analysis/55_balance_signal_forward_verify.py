"""「内 vs 外 balance」 signal の forward 検証 (Gate 1)

54 で発見した signal (balance <= -1.0 で 1号艇 1着 -14.7pt, 5号艇 +7.1pt) が、
期間分割で再現するか確認。再現しなければ backtest overfit (P7/Phase D' と同型) として
P8 design を凍結。

期間:
  Period A (train): 2026-04 (n=3095) — GW 含まず
  Period B-GW (forward 1): 2026-05-01 〜 2026-05-06 (GW)
  Period B-postGW (forward 2): 2026-05-07 〜 2026-05-18

CLAUDE.md 批判プロトコル準拠、結論は出さず岩下さんの判断に委ねる。

期待される結果:
  ✅ A / B-GW / B-postGW 全期間で 1号艇 -10pt 以上、5号艇 +5pt 以上 → 採用 candidate
  🟡 一部期間のみ → 環境依存性、追加分析
  🔴 forward (B-GW or B-postGW) で signal 消失 → overfit、凍結

出力: analysis/reports/55_balance_signal_forward_verify.md
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date

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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '55_balance_signal_forward_verify.md'

CLASS_SCORE = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
GW_START = date(2026, 5, 1)
GW_END = date(2026, 5, 6)
APR_END = date(2026, 4, 30)


def fetch_race_dates(race_ids):
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT id, race_date FROM races WHERE id = ANY(%s)", (race_ids,))
    dates = {r['id']: r['race_date'] for r in cur.fetchall()}
    conn.close()
    return dates


def period_label(d):
    if d <= APR_END:
        return 'A (2026-04)'
    if d <= GW_END:
        return 'B-GW (2026-05-01〜06)'
    return 'B-postGW (2026-05-07〜18)'


def class_score(c):
    return CLASS_SCORE.get(c, 2)


def main():
    logger.info("balance signal forward 検証 (Gate 1)")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    race_dates = fetch_race_dates(list(cache.keys()))
    logger.info(f"race_dates: {len(race_dates)}")

    races = []
    for rid, c in cache.items():
        boats = c.get('boats', [])
        if len(boats) != 6 or not c.get('result_1st') or not c.get('actual'):
            continue
        scores = [class_score(b.get('player_class', 'B1')) for b in boats]
        balance = float(np.mean(scores[:3]) - np.mean(scores[3:]))
        d = race_dates.get(rid)
        if not d:
            continue
        qmc_probs = c.get('qmc_probs', {})
        top3 = [k for k, _ in sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]]
        races.append({
            'rid': rid,
            'date': d,
            'period': period_label(d),
            'balance': balance,
            'result_1st': c['result_1st'],
            'result_2nd': c['result_2nd'],
            'result_3rd': c['result_3rd'],
            'actual': c['actual'],
            'payout': c.get('payout', 0) or 0,
            'top3_qmc': top3,
        })
    logger.info(f"valid races: {len(races)}")

    # 期間別 × balance 帯別集計
    buckets = [
        ('内強強 (>= +1.0)', lambda b: b >= 1.0),
        ('内強 (+0.5〜+1.0)', lambda b: 0.5 <= b < 1.0),
        ('均衡 (-0.5〜+0.5)', lambda b: -0.5 < b < 0.5),
        ('外強 (-1.0〜-0.5)', lambda b: -1.0 < b <= -0.5),
        ('**外強強 (<= -1.0)**', lambda b: b <= -1.0),
    ]
    periods = ['A (2026-04)', 'B-GW (2026-05-01〜06)', 'B-postGW (2026-05-07〜18)']

    lines = []
    lines.append("# balance signal forward 検証 (Gate 1)\n\n")
    lines.append("54 で発見した外強強帯 signal が、期間分割でも再現するか確認。\n")
    lines.append("P7 / Phase D' で経験した「backtest overfit」を回避するための gate.\n\n")
    lines.append(f"対象: cache {len(races)} races (2026-04-11 〜 2026-05-18)\n\n")

    # 期間別 race 数
    lines.append("## 期間別 race 数\n\n")
    period_counts = defaultdict(int)
    for r in races:
        period_counts[r['period']] += 1
    lines.append("| 期間 | n_races |\n|---|---|\n")
    for p in periods:
        lines.append(f"| {p} | {period_counts[p]} |\n")

    # 各期間 × バランス帯 → 1号艇 1着率 / 5号艇 1着率 / 4号艇 1着率
    lines.append("\n## 各期間 × バランス帯別 1着艇率\n\n")
    lines.append("| 期間 | バランス帯 | n | 1号艇 1着率 | 4号艇 1着率 | 5号艇 1着率 | top-3 hit率 |\n|---|---|---|---|---|---|---|\n")
    period_bucket_data = {}
    for p in periods:
        for label, cond in buckets:
            rs = [r for r in races if r['period'] == p and cond(r['balance'])]
            n = len(rs)
            if n == 0:
                lines.append(f"| {p} | {label} | 0 | - | - | - | - |\n")
                continue
            b1 = sum(1 for r in rs if r['result_1st'] == 1) / n * 100
            b4 = sum(1 for r in rs if r['result_1st'] == 4) / n * 100
            b5 = sum(1 for r in rs if r['result_1st'] == 5) / n * 100
            t3 = sum(1 for r in rs if r['actual'] in r['top3_qmc']) / n * 100
            period_bucket_data[(p, label)] = {
                'n': n, 'b1': b1, 'b4': b4, 'b5': b5, 't3': t3,
            }
            lines.append(f"| {p} | {label} | {n} | {b1:.2f}% | {b4:.2f}% | {b5:.2f}% | {t3:.2f}% |\n")

    # 外強強帯のみフォーカス比較
    lines.append("\n## 外強強帯 (balance <= -1.0) — 期間別比較\n\n")
    lines.append("**signal 再現性チェック**: 各期間で 1号艇 1着率と 5号艇 1着率が baseline (54 で出した全期間値) と整合するか\n\n")
    lines.append("| 期間 | n | 1号艇 1着率 (vs 全期間 40.69%) | 5号艇 1着率 (vs 全期間 13.10%) | 4号艇 1着率 (vs 全期間 12.76%) | top-3 hit率 (vs 全期間 12.07%) |\n|---|---|---|---|---|---|\n")
    ge_data = []
    for p in periods:
        d = period_bucket_data.get((p, '**外強強 (<= -1.0)**'))
        if not d:
            lines.append(f"| {p} | 0 | - | - | - | - |\n")
            continue
        ge_data.append((p, d))
        lines.append(f"| {p} | {d['n']} | {d['b1']:.2f}% ({d['b1']-40.69:+.2f}) | "
                     f"{d['b5']:.2f}% ({d['b5']-13.10:+.2f}) | "
                     f"{d['b4']:.2f}% ({d['b4']-12.76:+.2f}) | "
                     f"{d['t3']:.2f}% ({d['t3']-12.07:+.2f}) |\n")

    # 全期間平均 (baseline、外強強帯のみ)
    all_extreme = [r for r in races if r['balance'] <= -1.0]
    n_e = len(all_extreme)
    if n_e:
        b1_all = sum(1 for r in all_extreme if r['result_1st'] == 1) / n_e * 100
        b5_all = sum(1 for r in all_extreme if r['result_1st'] == 5) / n_e * 100
        lines.append(f"\n*全期間 (baseline): n={n_e}, 1号艇 1着率 {b1_all:.2f}%, 5号艇 1着率 {b5_all:.2f}%*\n")

    # signal 再現性の自動判定
    lines.append("\n## 自動判定 (CLAUDE.md 採用基準準拠)\n\n")
    if len(ge_data) >= 2:
        # B-GW と B-postGW の挙動を確認
        b_gw = next((d for p, d in ge_data if 'B-GW' in p), None)
        b_pg = next((d for p, d in ge_data if 'B-postGW' in p), None)
        a = next((d for p, d in ge_data if p.startswith('A')), None)

        if a and b_pg:
            # 1号艇 1着率が forward (B-postGW) でも train (A) と同方向か
            a_b1_low = a['b1'] < 50.0  # 平均 55% より低い
            f_b1_low = b_pg['b1'] < 50.0
            a_b5_high = a['b5'] > 8.0  # 平均 6% より高い
            f_b5_high = b_pg['b5'] > 8.0

            lines.append(f"- **Period A (train、2026-04 外強強 n={a['n']})**: 1号艇 {a['b1']:.2f}%、5号艇 {a['b5']:.2f}%\n")
            lines.append(f"- **Period B-postGW (forward、2026-05-07〜18 外強強 n={b_pg['n']})**: 1号艇 {b_pg['b1']:.2f}%、5号艇 {b_pg['b5']:.2f}%\n")

            if a_b1_low and f_b1_low and a_b5_high and f_b5_high:
                lines.append("\n🟢 **forward でも signal 再現** (1号艇 < 50%, 5号艇 > 8%) → P8 design 着手 candidate\n")
            elif (a_b1_low and f_b1_low) or (a_b5_high and f_b5_high):
                lines.append("\n🟡 **部分再現** (片方の signal は forward でも残存、もう片方は弱化) → 追加検証必要\n")
            else:
                lines.append("\n🔴 **forward で signal 消失** → backtest overfit 疑い、P8 design 凍結\n")
        if b_pg and b_pg['n'] < 30:
            lines.append(f"\n⚠️ B-postGW n={b_pg['n']} は検出力不足、判定は仮ベース\n")

    # GW vs post-GW の追加観察
    if len(ge_data) >= 3:
        b_gw_d = next((d for p, d in ge_data if 'B-GW' in p), None)
        b_pg_d = next((d for p, d in ge_data if 'B-postGW' in p), None)
        if b_gw_d and b_pg_d:
            diff_b1 = b_gw_d['b1'] - b_pg_d['b1']
            lines.append(f"\n### GW vs post-GW (外強強帯のみ)\n\n")
            lines.append(f"- B-GW: 1号艇 1着率 {b_gw_d['b1']:.2f}% (n={b_gw_d['n']})\n")
            lines.append(f"- B-postGW: 1号艇 1着率 {b_pg_d['b1']:.2f}% (n={b_pg_d['n']})\n")
            lines.append(f"- 差: {diff_b1:+.2f}pt\n")
            if abs(diff_b1) > 15:
                lines.append("- ⚠️ GW vs post-GW で 1号艇 1着率が大きく異なる → GW 効果が外強強 signal に交絡\n")
            else:
                lines.append("- 🟢 GW vs post-GW で 1号艇 1着率は安定 → balance signal は GW 効果と独立\n")

    # 留意事項
    lines.append("\n## 留意事項\n\n")
    lines.append("- cache 期間は 2026-04-11 〜 2026-05-18 のみ、5 週間で長期 forward は未確認\n")
    lines.append("- 外強強帯は全 races の 5% 程度、期間分割で各 n が小さい (~50-150)\n")
    lines.append("- GW 効果 (52 で確認、hit 率 11.4% → 1.52%) との交絡可能性を別途確認\n")
    lines.append("- forward 再現 OK でも、shadow 2 週間で本番検証必須 (P7 教訓)\n")
    lines.append("- 結論は出さず、岩下さんの判断 (P8 design 着手 / 凍結) を待つ\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
