"""内 (1-3号) vs 外 (4-6号) のクラスバランス別 着順分布 / 買い目分析

岩下さん観察「4号艇 A 級 + 1号艇 B 級」を一般化:
内 (1-3 号艇) が弱く、外 (4-6 号艇) が強い時、どう賭けるべきか?

クラス強度 score:
  A1=4, A2=3, B1=2, B2=1
  inner_score = mean(boat 1-3 のクラス値)
  outer_score = mean(boat 4-6 のクラス値)
  balance = inner_score - outer_score
  balance > 0: 内強 (通常)
  balance < 0: 外強 (反転、岩下さん観察パターン)
  balance ≈ 0: 均衡

CLAUDE.md 批判プロトコル準拠、結論は出さず岩下さんの判断に委ねる。

検証する 5 仮説:
  H1: balance score 別の actual 1/2/3 着艇分布
  H2: 各バランス帯での top 10 actual 3連単 combo
  H3: 反転帯 (balance < -0.5) で QMC が当てているか (alpha 検証)
  H4: 反転帯での候補買い目戦略 (4-1-X, 4-2-X, 5-1-X 等) の hit 率 + 平均 payout
  H5: 物理ストーリーとの整合 (会場特性 / 風波との交絡)

出力: analysis/reports/54_inner_outer_balance.md
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'analysis' / 'qmc_predictions_cache.pkl'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '54_inner_outer_balance.md'

CLASS_SCORE = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}


def class_score(c):
    return CLASS_SCORE.get(c, 2)


def main():
    logger.info("内外クラスバランス別 着順 / 買い目分析")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"cache: {len(cache)} races")

    races = []
    for rid, c in cache.items():
        boats = c.get('boats', [])
        if len(boats) != 6 or not c.get('actual') or not c.get('result_1st'):
            continue
        scores = [class_score(b.get('player_class', 'B1')) for b in boats]
        inner = np.mean(scores[:3])
        outer = np.mean(scores[3:])
        balance = inner - outer

        # QMC top-3 picks
        qmc_probs = c.get('qmc_probs', {})
        top3 = [k for k, _ in sorted(qmc_probs.items(), key=lambda x: -x[1])[:3]]
        top10 = [k for k, _ in sorted(qmc_probs.items(), key=lambda x: -x[1])[:10]]

        races.append({
            'rid': rid,
            'inner': inner,
            'outer': outer,
            'balance': balance,
            'boat1_class': boats[0].get('player_class'),
            'boat2_class': boats[1].get('player_class'),
            'boat3_class': boats[2].get('player_class'),
            'boat4_class': boats[3].get('player_class'),
            'boat5_class': boats[4].get('player_class'),
            'boat6_class': boats[5].get('player_class'),
            'result_1st': c['result_1st'],
            'result_2nd': c['result_2nd'],
            'result_3rd': c['result_3rd'],
            'actual': c['actual'],
            'payout': c.get('payout', 0) or 0,
            'venue_id': c.get('venue_id'),
            'race_number': c.get('race_number'),
            'top3_qmc': top3,
            'top10_qmc': top10,
            'qmc_probs': qmc_probs,
            'wind_speed': c.get('race_data', {}).get('wind_speed') or 0,
            'wave_height': c.get('race_data', {}).get('wave_height') or 0,
        })

    logger.info(f"valid races: {len(races)}")

    lines = []
    lines.append("# 内 (1-3 号) vs 外 (4-6 号) クラスバランス別 着順 / 買い目分析\n\n")
    lines.append(f"対象: cache {len(races)} races\n")
    lines.append("**バランス score**: inner_avg - outer_avg (A1=4, A2=3, B1=2, B2=1)\n")
    lines.append("- balance > +0.5: 内強 (通常)\n")
    lines.append("- balance -0.5 〜 +0.5: 均衡\n")
    lines.append("- balance < -0.5: **外強 (反転、岩下さん観察パターン)**\n\n")

    # ============== H1: バランス帯別 1着艇分布 ==============
    lines.append("## H1: バランス帯別の actual 1着艇分布\n\n")
    buckets = [
        ('内強強 (balance >= 1.0)', lambda b: b >= 1.0),
        ('内強 (0.5 <= balance < 1.0)', lambda b: 0.5 <= b < 1.0),
        ('均衡 (-0.5 < balance < 0.5)', lambda b: -0.5 < b < 0.5),
        ('外強 (-1.0 < balance <= -0.5)', lambda b: -1.0 < b <= -0.5),
        ('**外強強 (balance <= -1.0)**', lambda b: b <= -1.0),
    ]
    lines.append("| バランス帯 | n | 1号艇 1着% | 2号艇 1着% | 3号艇 1着% | 4号艇 1着% | 5号艇 1着% | 6号艇 1着% |\n|---|---|---|---|---|---|---|---|\n")
    bucket_races = {}
    for label, cond in buckets:
        rs = [r for r in races if cond(r['balance'])]
        bucket_races[label] = rs
        n = len(rs)
        if n == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - | - |\n")
            continue
        row = f"| {label} | {n} |"
        for boat in range(1, 7):
            cnt = sum(1 for r in rs if r['result_1st'] == boat)
            row += f" {cnt/n*100:.2f}% |"
        lines.append(row + "\n")

    # ============== H1-b: 2着 / 3着分布 (外強帯のみ) ==============
    lines.append("\n### 外強帯のみ詳細: 2 着 / 3 着艇分布\n\n")
    for label in ['外強 (-1.0 < balance <= -0.5)', '**外強強 (balance <= -1.0)**']:
        rs = bucket_races.get(label, [])
        if not rs:
            continue
        n = len(rs)
        lines.append(f"\n#### {label} (n={n})\n\n")
        lines.append("| 順位 | 1号艇 | 2号艇 | 3号艇 | 4号艇 | 5号艇 | 6号艇 |\n|---|---|---|---|---|---|---|\n")
        for pos_label, pos_key in [('1着', 'result_1st'), ('2着', 'result_2nd'), ('3着', 'result_3rd')]:
            row = f"| {pos_label} |"
            for boat in range(1, 7):
                cnt = sum(1 for r in rs if r[pos_key] == boat)
                row += f" {cnt/n*100:.2f}% |"
            lines.append(row + "\n")

    # ============== H2: 各バランス帯 top 10 actual 3連単 ==============
    lines.append("\n## H2: バランス帯別 top 10 actual 3連単 combo\n\n")
    for label, _ in buckets:
        rs = bucket_races.get(label, [])
        if not rs:
            continue
        n = len(rs)
        cnt = defaultdict(int)
        for r in rs:
            cnt[r['actual']] += 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:10]
        lines.append(f"\n### {label} (n={n})\n\n")
        lines.append("| rank | combo | n | rate |\n|---|---|---|---|\n")
        for i, (combo, c_) in enumerate(top):
            lines.append(f"| {i+1} | {combo} | {c_} | {c_/n*100:.2f}% |\n")

    # ============== H3: 外強帯で QMC が当てているか (alpha 検証) ==============
    lines.append("\n## H3: 外強帯での QMC alpha 検証\n\n")
    lines.append("**問い**: 外強 (balance <= -0.5) の race で、QMC top-3 / top-10 picks に actual combo が含まれる割合。\n")
    lines.append("低ければ「QMC が外している領域」= 非重複 filter / 代替軸 candidate。\n\n")
    lines.append("| バランス帯 | n | top-1 hit% | top-3 hit% | top-10 hit% | QMC が 4-X-X / 5-X-X / 6-X-X を top-3 に含む率 |\n|---|---|---|---|---|---|\n")
    for label, _ in buckets:
        rs = bucket_races.get(label, [])
        if not rs: continue
        n = len(rs)
        t1 = sum(1 for r in rs if r['top3_qmc'] and r['top3_qmc'][0] == r['actual'])
        t3 = sum(1 for r in rs if r['actual'] in r['top3_qmc'])
        t10 = sum(1 for r in rs if r['actual'] in r['top10_qmc'])
        # top-3 が外艇始まり (4/5/6) の率
        outer_top3 = sum(1 for r in rs if r['top3_qmc'] and any(t[0] in '456' for t in r['top3_qmc']))
        lines.append(f"| {label} | {n} | {t1/n*100:.2f}% | {t3/n*100:.2f}% | {t10/n*100:.2f}% | {outer_top3/n*100:.2f}% |\n")

    # ============== H4: 外強帯の候補買い目戦略 ==============
    lines.append("\n## H4: 外強帯での候補買い目戦略 hit 率 + 平均 payout\n\n")
    lines.append("**仮説的戦略 candidate** (外強帯で発動、3 連単 1 点 / 多点買い):\n\n")

    # 戦略候補 (組合せ)
    # 注: hit 率 と平均 payout を集計、ROI 推定
    strategy_specs = [
        ('4-1-X (流し、4-1- 全 2/3着)',  lambda actual: actual.startswith('4-1-')),
        ('4-2-X (流し、4-2-)', lambda actual: actual.startswith('4-2-')),
        ('4-3-X (流し、4-3-)', lambda actual: actual.startswith('4-3-')),
        ('X-1-X (1号艇 2着固定、流し)', lambda actual: actual.split('-')[1] == '1'),
        ('4-X-1 (4 軸、1号艇 3着)', lambda actual: actual.split('-')[0] == '4' and actual.split('-')[2] == '1'),
        ('5-X-X (5 軸 1着、流し)', lambda actual: actual.startswith('5-')),
        ('6-X-X (6 軸 1着、流し)', lambda actual: actual.startswith('6-')),
        ('4-1-2 (1 点)', lambda actual: actual == '4-1-2'),
        ('4-1-3 (1 点)', lambda actual: actual == '4-1-3'),
        ('4-1-5 (1 点)', lambda actual: actual == '4-1-5'),
    ]

    for label in ['外強 (-1.0 < balance <= -0.5)', '**外強強 (balance <= -1.0)**']:
        rs = bucket_races.get(label, [])
        if not rs: continue
        n = len(rs)
        lines.append(f"\n### {label} (n={n})\n\n")
        lines.append("| 戦略 | hit n | hit率 | 平均 payout (hit時) | 期待回収 (¥100 × n_picks 仮定) |\n|---|---|---|---|---|\n")
        for sname, scond in strategy_specs:
            hits = [r for r in rs if scond(r['actual'])]
            n_hit = len(hits)
            if n_hit == 0:
                lines.append(f"| {sname} | 0 | 0.00% | - | - |\n")
                continue
            avg_payout = np.mean([r['payout'] for r in hits])
            # ROI proxy: hit率 × 平均payout / 100¥ — 1点買い前提 (流し系は厳密でない)
            # 注: 流し系は picks 数で割らないと不当な利益、ここは hit時の payout のみ表示
            lines.append(f"| {sname} | {n_hit} | {n_hit/n*100:.2f}% | ¥{avg_payout:,.0f} | hit率 × 平均 |\n")

    # 単点買い (3 連単 1 点) の ROI 推定
    lines.append("\n### 1 点買い ROI 試算 (外強強帯、payout × hit率 / 100 ¥)\n\n")
    rs = bucket_races.get('**外強強 (balance <= -1.0)**', [])
    n = len(rs)
    if rs:
        single_strats = ['4-1-2', '4-1-3', '4-1-5', '4-1-6', '4-2-1', '4-2-3', '4-2-5',
                          '4-3-1', '4-3-2', '4-3-5', '5-1-2', '5-1-3', '5-2-1', '6-1-2', '1-4-2', '1-4-3']
        lines.append("| 1 点買い目 | hit n | hit率 | 平均 payout (hit時) | ROI (1点 ¥100 単位) |\n|---|---|---|---|---|\n")
        for combo in single_strats:
            n_hit = sum(1 for r in rs if r['actual'] == combo)
            if n_hit == 0:
                lines.append(f"| {combo} | 0 | 0.00% | - | -100% |\n")
                continue
            avg_payout = np.mean([r['payout'] for r in rs if r['actual'] == combo])
            roi = (n_hit * avg_payout - n * 100) / (n * 100) * 100
            lines.append(f"| {combo} | {n_hit} | {n_hit/n*100:.2f}% | ¥{avg_payout:,.0f} | {roi:+.1f}% |\n")

    # ============== H5: 物理ストーリー (会場 / 風波交絡確認) ==============
    lines.append("\n## H5: 外強帯の物理ストーリー (会場 / 風波交絡)\n\n")
    lines.append("**問い**: 外強帯が特定会場 / 荒天に偏っていないか? もし偏っていれば交絡で「外強」signal が弱まる可能性。\n\n")
    rs = bucket_races.get('**外強強 (balance <= -1.0)**', [])
    if rs:
        # 会場分布
        venue_cnt = defaultdict(int)
        for r in rs:
            venue_cnt[r['venue_id']] += 1
        lines.append(f"\n### 外強強帯 (n={len(rs)}) の会場分布 (top 10)\n\n")
        lines.append("| venue | n | 全 race 数比 |\n|---|---|---|\n")
        all_venue_cnt = defaultdict(int)
        for r in races:
            all_venue_cnt[r['venue_id']] += 1
        for v, c in sorted(venue_cnt.items(), key=lambda x: -x[1])[:10]:
            all_n = all_venue_cnt[v]
            ratio = c / all_n * 100 if all_n else 0
            lines.append(f"| {v} | {c} | {ratio:.2f}% (全 {all_n}) |\n")
        # 風速分布
        wind_avg = np.mean([r['wind_speed'] for r in rs])
        all_wind_avg = np.mean([r['wind_speed'] for r in races])
        wave_avg = np.mean([r['wave_height'] for r in rs])
        all_wave_avg = np.mean([r['wave_height'] for r in races])
        lines.append(f"\n- 外強強帯 平均風速: {wind_avg:.2f} m/s (全平均 {all_wind_avg:.2f})\n")
        lines.append(f"- 外強強帯 平均波高: {wave_avg:.2f} cm (全平均 {all_wave_avg:.2f})\n")

    # ============== 採用判定 ==============
    lines.append("\n## 採用基準による自動振り分け (CLAUDE.md 準拠)\n\n")
    rs_outer = bucket_races.get('**外強強 (balance <= -1.0)**', [])
    n_outer = len(rs_outer)
    if rs_outer:
        b1_rate = sum(1 for r in rs_outer if r['result_1st'] == 1) / n_outer * 100
        b4_rate = sum(1 for r in rs_outer if r['result_1st'] == 4) / n_outer * 100
        b5_rate = sum(1 for r in rs_outer if r['result_1st'] == 5) / n_outer * 100
        avg_b1_rate = sum(1 for r in races if r['result_1st'] == 1) / len(races) * 100
        avg_b4_rate = sum(1 for r in races if r['result_1st'] == 4) / len(races) * 100
        avg_b5_rate = sum(1 for r in races if r['result_1st'] == 5) / len(races) * 100
        lines.append(f"- **外強強帯** (balance <= -1.0): n={n_outer}\n")
        lines.append(f"  - 1号艇 1着率 {b1_rate:.2f}% (全平均 {avg_b1_rate:.2f}%, 差 {b1_rate-avg_b1_rate:+.2f}pt)\n")
        lines.append(f"  - 4号艇 1着率 {b4_rate:.2f}% (全平均 {avg_b4_rate:.2f}%, 差 {b4_rate-avg_b4_rate:+.2f}pt)\n")
        lines.append(f"  - 5号艇 1着率 {b5_rate:.2f}% (全平均 {avg_b5_rate:.2f}%, 差 {b5_rate-avg_b5_rate:+.2f}pt)\n")
        if n_outer >= 100 and abs(b1_rate - avg_b1_rate) > 10:
            lines.append("  - 🟢 標本数 + effect size 共に大、採用候補\n")
        elif n_outer >= 30:
            lines.append("  - 🟡 標本数中、effect size 中、保留 (追加検証)\n")
        else:
            lines.append("  - 🔴 標本不足\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- バランス score は class のみ、勝率 / モーター / 展示タイム未反映\n")
    lines.append("- 戦略 hit 率 は cache 全期間 (主に 2025-12 〜 2026-03)、forward 確認必要\n")
    lines.append("- 流し系 (4-1-X 等) の ROI は picks 数で割らないと過大、本表は hit 率のみ参考\n")
    lines.append("- QMC が既に拾っているなら filter 二重対処になる、H3 で alpha 確認後に判断\n")
    lines.append("- 採用候補は **shadow 2 週間必須**\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
