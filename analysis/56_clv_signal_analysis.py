"""CLV (Closing Line Value) signal 分析

仮説: 締切に向けてオッズが下がった picks (CLV > 0) = sharp money が乗った = ROI 高
逆に CLV < 0 = 市場が我々の picks を逆評価 = ROI 低

CLV = (odds_at_bet - closing_odds) / closing_odds
- CLV > 0: 我々の買い時 odds より締切時が低い → 市場が picks に集まった
- CLV < 0: 我々の買い時 odds より締切時が高い → 市場は picks を見送った

CLAUDE.md 批判プロトコル準拠、結論は出さず岩下さんの判断に委ねる。

検証する 5 仮説:
  H1: CLV 帯別 hit 率 (CLV 正 = hit 率高い?)
  H2: CLV 帯別 ROI (CLV 正 = ROI 黒字?)
  H3: 戦略別 CLV 効果 (mc/mc2/mc3 で違いあるか)
  H4: バランス帯別 CLV 効果 (外強強で CLV が特に効くか)
  H5: 採用候補: CLV < -X で skip すると ROI 改善するか

入力: DB bets (4月以降、全 mc 系戦略)
出力: analysis/reports/56_clv_signal.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '56_clv_signal.md'

CLASS_SCORE = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}


def fetch_bets():
    """全 mc 系戦略の bets + CLV を引く."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            b.id, b.race_id, b.strategy_type, b.combination,
            b.amount, b.is_hit, b.return_amount, b.created_at::date AS bd,
            b.odds, b.closing_odds, b.clv, b.expected_value
        FROM bets b
        WHERE b.created_at >= '2026-04-01'
          AND b.clv IS NOT NULL
          AND b.strategy_type LIKE 'mc%'
    """)
    bets = cur.fetchall()
    conn.close()
    return bets


def fetch_balance_scores(race_ids):
    """各 race の class balance score."""
    if not race_ids:
        return {}
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT race_id, boat_number, player_class
        FROM boats WHERE race_id = ANY(%s)
    """, (list(race_ids),))
    boats = defaultdict(dict)
    for r in cur.fetchall():
        boats[r['race_id']][r['boat_number']] = r['player_class']
    conn.close()
    bal = {}
    for rid, bd in boats.items():
        if len(bd) < 6:
            continue
        scores = [CLASS_SCORE.get(bd.get(i, 'B1'), 2) for i in range(1, 7)]
        bal[rid] = np.mean(scores[:3]) - np.mean(scores[3:])
    return bal


def bucket_stats(bets, label):
    """1 バケットの統計."""
    n = len(bets)
    if n == 0:
        return None
    hits = sum(1 for b in bets if b['is_hit'])
    invested = sum(float(b['amount'] or 0) for b in bets)
    returned = sum(float(b['return_amount'] or 0) for b in bets)
    return {
        'label': label,
        'n': n,
        'hits': hits,
        'hit_rate': hits / n * 100,
        'invested': invested,
        'returned': returned,
        'pnl': returned - invested,
        'roi': (returned - invested) / invested * 100 if invested else 0,
    }


def main():
    logger.info("CLV signal 分析")
    bets = fetch_bets()
    logger.info(f"bets with CLV: {len(bets)}")

    # CLV 帯定義
    clv_buckets = [
        ('CLV < -0.20 (市場大逆方向)', lambda c: c < -0.20),
        ('CLV -0.20〜-0.10', lambda c: -0.20 <= c < -0.10),
        ('CLV -0.10〜-0.05', lambda c: -0.10 <= c < -0.05),
        ('CLV -0.05〜+0.00', lambda c: -0.05 <= c < 0),
        ('CLV +0.00〜+0.05', lambda c: 0 <= c < 0.05),
        ('CLV +0.05〜+0.10', lambda c: 0.05 <= c < 0.10),
        ('CLV +0.10〜+0.20', lambda c: 0.10 <= c < 0.20),
        ('CLV > +0.20 (市場大順方向)', lambda c: c >= 0.20),
    ]

    lines = []
    lines.append("# CLV (Closing Line Value) signal 分析\n\n")
    lines.append(f"対象: 全 mc 系戦略 4月以降の bets n={len(bets)} (CLV あり)\n")
    lines.append("CLV = (odds_at_bet - closing_odds) / closing_odds\n")
    lines.append("- CLV > 0: 締切時オッズ低下 = sharp money 流入 = picks が市場に評価された\n")
    lines.append("- CLV < 0: 締切時オッズ上昇 = 市場が picks を見送った\n\n")

    # H1/H2: CLV 帯別 hit 率 + ROI
    lines.append("## H1/H2: CLV 帯別 hit 率 / ROI\n\n")
    lines.append("| CLV 帯 | n | hit | hit率 | invested | returned | PnL | ROI |\n|---|---|---|---|---|---|---|---|\n")
    for label, cond in clv_buckets:
        bs = [b for b in bets if cond(b['clv'])]
        s = bucket_stats(bs, label)
        if not s:
            lines.append(f"| {label} | 0 | - | - | - | - | - | - |\n")
            continue
        lines.append(f"| {label} | {s['n']} | {s['hits']} | {s['hit_rate']:.2f}% | "
                     f"¥{s['invested']:,.0f} | ¥{s['returned']:,.0f} | "
                     f"¥{s['pnl']:+,.0f} | {s['roi']:+.2f}% |\n")

    # CLV > 0 vs CLV <= 0 サマリ
    pos = [b for b in bets if b['clv'] > 0]
    neg = [b for b in bets if b['clv'] <= 0]
    s_pos = bucket_stats(pos, 'CLV > 0')
    s_neg = bucket_stats(neg, 'CLV <= 0')
    if s_pos and s_neg:
        lines.append(f"\n**シンプル 2 分割**:\n\n")
        lines.append("| 帯 | n | hit率 | ROI |\n|---|---|---|---|\n")
        lines.append(f"| {s_pos['label']} | {s_pos['n']} | {s_pos['hit_rate']:.2f}% | {s_pos['roi']:+.2f}% |\n")
        lines.append(f"| {s_neg['label']} | {s_neg['n']} | {s_neg['hit_rate']:.2f}% | {s_neg['roi']:+.2f}% |\n")

    # H3: 戦略別 CLV 効果
    lines.append("\n## H3: 戦略別 CLV 効果\n\n")
    lines.append("各戦略で CLV>0 vs CLV<=0 の ROI 差を確認。差が大きい戦略は CLV filter 効果大。\n\n")
    strat_groups = defaultdict(list)
    for b in bets:
        strat_groups[b['strategy_type']].append(b)
    lines.append("| 戦略 | n total | CLV>0 n | CLV>0 hit% | CLV>0 ROI | CLV<=0 n | CLV<=0 hit% | CLV<=0 ROI | ROI差 |\n|---|---|---|---|---|---|---|---|---|\n")
    for strat in sorted(strat_groups.keys(), key=lambda s: -len(strat_groups[s])):
        bs = strat_groups[strat]
        if len(bs) < 50:
            continue  # 小さい戦略は除外
        pos_s = bucket_stats([b for b in bs if b['clv'] > 0], 'pos')
        neg_s = bucket_stats([b for b in bs if b['clv'] <= 0], 'neg')
        if pos_s and neg_s:
            diff = pos_s['roi'] - neg_s['roi']
            lines.append(f"| {strat} | {len(bs)} | {pos_s['n']} | {pos_s['hit_rate']:.2f}% | "
                         f"{pos_s['roi']:+.2f}% | {neg_s['n']} | {neg_s['hit_rate']:.2f}% | "
                         f"{neg_s['roi']:+.2f}% | {diff:+.2f}pt |\n")

    # H4: バランス帯別 CLV 効果
    logger.info("バランス score fetch")
    race_ids = [b['race_id'] for b in bets]
    balance = fetch_balance_scores(set(race_ids))
    logger.info(f"balance scores: {len(balance)}")
    lines.append("\n## H4: バランス帯 × CLV クロス\n\n")
    lines.append("外強強帯 (1号艇 -15pt の signal あり) で CLV filter がさらに効くか?\n\n")
    bal_buckets = [
        ('inner_strong (>= +0.5)', lambda b: b >= 0.5),
        ('balanced (-0.5〜+0.5)', lambda b: -0.5 < b < 0.5),
        ('outer_mid (-1.0〜-0.5)', lambda b: -1.0 < b <= -0.5),
        ('outer_strong (<= -1.0)', lambda b: b <= -1.0),
    ]
    lines.append("| バランス帯 | CLV>0 n | CLV>0 ROI | CLV<=0 n | CLV<=0 ROI | 差 |\n|---|---|---|---|---|---|\n")
    for blabel, bcond in bal_buckets:
        bs = [b for b in bets if b['race_id'] in balance and bcond(balance[b['race_id']])]
        pos_s = bucket_stats([b for b in bs if b['clv'] > 0], 'pos')
        neg_s = bucket_stats([b for b in bs if b['clv'] <= 0], 'neg')
        if pos_s and neg_s:
            diff = pos_s['roi'] - neg_s['roi']
            lines.append(f"| {blabel} | {pos_s['n']} | {pos_s['roi']:+.2f}% | "
                         f"{neg_s['n']} | {neg_s['roi']:+.2f}% | {diff:+.2f}pt |\n")
        else:
            lines.append(f"| {blabel} | - | - | - | - | - |\n")

    # H5: CLV cut-off 試算
    lines.append("\n## H5: CLV cut-off filter 試算\n\n")
    lines.append("「CLV < threshold」で skip した時、残り bets の ROI と削減数を試算:\n\n")
    lines.append("| threshold | 残 n | 削減 n | 削減率 | 残 hit% | 残 ROI |\n|---|---|---|---|---|---|\n")
    thresholds = [-0.30, -0.20, -0.15, -0.10, -0.05, 0, 0.05]
    n_total = len(bets)
    for th in thresholds:
        kept = [b for b in bets if b['clv'] >= th]
        s = bucket_stats(kept, f'th={th}')
        if not s:
            continue
        cut = n_total - s['n']
        cut_rate = cut / n_total * 100
        lines.append(f"| {th:+.2f} | {s['n']} | {cut} | {cut_rate:.1f}% | "
                     f"{s['hit_rate']:.2f}% | {s['roi']:+.2f}% |\n")

    # 全体 baseline
    s_all = bucket_stats(bets, 'all')
    if s_all:
        lines.append(f"\n*全体 baseline: n={s_all['n']}, hit率 {s_all['hit_rate']:.2f}%, ROI {s_all['roi']:+.2f}%*\n")

    # 自動判定
    lines.append("\n## 自動判定 (CLAUDE.md 採用基準準拠)\n\n")
    if s_pos and s_neg and s_all:
        diff = s_pos['roi'] - s_neg['roi']
        if abs(diff) > 20 and s_pos['n'] >= 100 and s_neg['n'] >= 100:
            lines.append(f"- 🟢 **CLV 効果 大** (CLV>0 ROI vs CLV<=0 ROI 差 {diff:+.2f}pt)\n")
            lines.append(f"  → 採用 candidate: CLV < 0 で skip フィルタ\n")
        elif abs(diff) > 10:
            lines.append(f"- 🟡 **CLV 効果 中** (差 {diff:+.2f}pt)\n")
            lines.append(f"  → 保留、追加検証 (バランス帯クロス見て採用判断)\n")
        else:
            lines.append(f"- 🔴 **CLV 効果 小** (差 {diff:+.2f}pt)\n")
            lines.append(f"  → CLV は単純 filter として機能しない\n")

    # 留意事項
    lines.append("\n## 留意事項\n\n")
    lines.append("- CLV は **事後 metric** (締切時オッズ必要)、リアルタイム filter には使えない\n")
    lines.append("  → ただし 「過去 N 日 CLV ROI 高い picks 条件」を学習して real-time 適用は可\n")
    lines.append("- 戦略別差は picks 数に偏りあり、絶対 ROI ではなく相対比較\n")
    lines.append("- 採用候補は shadow 並走で検証 (P7 教訓)\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
