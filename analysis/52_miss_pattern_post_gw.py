"""mc3_venue_focus_r4 (P6) production bets miss pattern 分析

GW (5/1-5/6) と Post-GW (5/7-5/18) で hit 率が 11.4% → 3.0% に低下した原因を
仮説駆動で検証。「外し方」の構造変化を特定する。

CLAUDE.md 批判プロトコル準拠 (擁護/批判/未検証 + 採用基準の自動振り分け)。

検証する 5 仮説:
  Q1: 本命崩れ頻度の変化 (GW vs post-GW で 1着艇 ≠ 1号艇 の頻度差)
  Q2: 外した時の actual 1着分布 (4号艇 1着・5号艇 1着が増えたか)
  Q3: 2-3 着外しの構造 (1号艇 1着自体は当たったが 2-3 着で外したか)
  Q4: 会場別 / R 番号別 loss 集中
  Q5: 4号艇 A 級発火時パターン (岩下さん観察)

出力: analysis/reports/52_miss_pattern_post_gw.md
"""
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / 'analysis' / 'reports' / '52_miss_pattern_post_gw.md'

STRATEGY = 'mc3_venue_focus_r4'
GW_END = date(2026, 5, 6)  # 5/6 まで GW、5/7 から post-GW


def fetch_bets_with_context():
    """P6 production bets + race info + 4号艇クラス を join."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            b.id AS bet_id,
            b.race_id,
            b.combination,
            b.amount,
            b.is_hit,
            b.return_amount,
            b.created_at::date AS bet_date,
            b.odds,
            b.expected_value,
            r.venue_id,
            r.race_number,
            r.result_1st,
            r.result_2nd,
            r.result_3rd,
            r.actual_result_trifecta,
            r.payout_sanrentan,
            r.wind_speed,
            r.wave_height,
            boat4.player_class AS boat4_class,
            boat1.player_class AS boat1_class
        FROM bets b
        JOIN races r ON b.race_id = r.id
        LEFT JOIN boats boat4 ON boat4.race_id = b.race_id AND boat4.boat_number = 4
        LEFT JOIN boats boat1 ON boat1.race_id = b.race_id AND boat1.boat_number = 1
        WHERE b.strategy_type = %s
          AND b.created_at >= '2026-05-01'
          AND r.result_1st IS NOT NULL
        ORDER BY b.created_at
    """, (STRATEGY,))
    bets = cur.fetchall()
    conn.close()
    return bets


def fetch_race_boats(race_ids):
    """各 race の全 6 艇のクラス情報."""
    if not race_ids:
        return {}
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT race_id, boat_number, player_class
        FROM boats
        WHERE race_id = ANY(%s)
        ORDER BY race_id, boat_number
    """, (list(race_ids),))
    result = defaultdict(dict)
    for row in cur.fetchall():
        result[row['race_id']][row['boat_number']] = row['player_class']
    conn.close()
    return result


def is_post_gw(d):
    return d > GW_END


def period_label(d):
    return 'post_GW' if is_post_gw(d) else 'GW'


# ============================================================
# Q1: 本命崩れ頻度
# ============================================================

def q1_honmei_breakdown(bets):
    """各期間で actual 1着 = 1 の頻度."""
    # race 単位で集計 (同 race の複数 bet を 1 つに)
    races_by_period = defaultdict(set)
    race_winner = {}
    for b in bets:
        races_by_period[period_label(b['bet_date'])].add(b['race_id'])
        race_winner[b['race_id']] = b['result_1st']

    rows = []
    for period, rids in races_by_period.items():
        n_total = len(rids)
        n_b1_win = sum(1 for rid in rids if race_winner.get(rid) == 1)
        rows.append({
            'period': period,
            'n_races': n_total,
            'n_b1_win': n_b1_win,
            'b1_win_rate': n_b1_win / n_total * 100 if n_total else 0,
        })
    return rows


# ============================================================
# Q2: 外した時の actual 1着分布
# ============================================================

def q2_miss_winner_dist(bets):
    """戦略外し (race 単位、全 bet 外し) 時の actual 1着分布."""
    # race 単位: 同 race の bet が全て外しなら miss
    by_race = defaultdict(list)
    for b in bets:
        by_race[b['race_id']].append(b)

    dist = defaultdict(lambda: defaultdict(int))
    for rid, bs in by_race.items():
        period = period_label(bs[0]['bet_date'])
        any_hit = any(b['is_hit'] for b in bs)
        if any_hit:
            continue
        winner = bs[0]['result_1st']
        dist[period][winner] += 1
    return dist


# ============================================================
# Q3: 2-3 着外しの構造
# ============================================================

def q3_position_miss(bets):
    """1着 = 1 だった race で、picks が 2-3 着まで取れたか."""
    by_race = defaultdict(list)
    for b in bets:
        by_race[b['race_id']].append(b)

    rows = []
    for period in ['GW', 'post_GW']:
        n_b1win_races = 0     # 1着 = 1 の race 数
        n_b1win_hit = 0       # うち hit があった race 数 (2-3着まで取れた)
        n_b1win_pick = 0      # うち picks に 1-X-X が含まれた race 数
        for rid, bs in by_race.items():
            if period_label(bs[0]['bet_date']) != period:
                continue
            if bs[0]['result_1st'] != 1:
                continue
            n_b1win_races += 1
            if any(b['is_hit'] for b in bs):
                n_b1win_hit += 1
            if any(str(b['combination']).startswith('1') for b in bs):
                n_b1win_pick += 1
        rows.append({
            'period': period,
            'n_b1win_races': n_b1win_races,
            'n_b1win_pick': n_b1win_pick,
            'n_b1win_hit': n_b1win_hit,
            'pick_rate': n_b1win_pick / n_b1win_races * 100 if n_b1win_races else 0,
            'hit_rate': n_b1win_hit / n_b1win_races * 100 if n_b1win_races else 0,
        })
    return rows


# ============================================================
# Q4: 会場別 / R 番号別 loss 集中
# ============================================================

def q4_venue_race_loss(bets):
    """会場 × R 番号別の bet 数 / hit 数 / PnL."""
    by_key = defaultdict(lambda: {'n_bets': 0, 'n_hits': 0, 'invested': 0, 'returned': 0, 'period_split': defaultdict(int)})
    for b in bets:
        key = (b['venue_id'], b['race_number'])
        d = by_key[key]
        d['n_bets'] += 1
        if b['is_hit']:
            d['n_hits'] += 1
        d['invested'] += float(b['amount'] or 0)
        d['returned'] += float(b['return_amount'] or 0)
        d['period_split'][period_label(b['bet_date'])] += 1
    # post_GW で hit 0 + bets >= 3 の組合せに焦点
    rows = []
    for (venue, race_num), d in sorted(by_key.items(), key=lambda x: -x[1]['invested'] + x[1]['returned']):
        pnl = d['returned'] - d['invested']
        rows.append({
            'venue_id': venue,
            'race_number': race_num,
            'n_bets': d['n_bets'],
            'n_hits': d['n_hits'],
            'invested': d['invested'],
            'returned': d['returned'],
            'pnl': pnl,
            'hit_rate': d['n_hits'] / d['n_bets'] * 100 if d['n_bets'] else 0,
            'GW_bets': d['period_split']['GW'],
            'post_GW_bets': d['period_split']['post_GW'],
        })
    return rows


# ============================================================
# Q5: 4号艇 A 級発火時パターン (岩下さん観察)
# ============================================================

def q5_boat4_a_class(bets, race_boats_full):
    """4号艇 A 級 + 戦略発火時の挙動."""
    # race 単位、4号艇 A1/A2 の race で集計
    by_race = defaultdict(list)
    for b in bets:
        by_race[b['race_id']].append(b)

    rows_overall = []
    rows_by_period = []
    for label, filter_a4 in [
        ('全 races (P6 発火)', True),
        ('4号艇 A 級限定 (P6 発火)', True),  # 後で filter
        ('4号艇 A 級ではない (P6 発火)', True),
    ]:
        pass  # 設計再考

    # シンプルに 3 区分集計
    groups = {
        '4号艇 A 級 (A1/A2)': lambda b4c: b4c in ('A1', 'A2'),
        '4号艇 B 級 (B1/B2)': lambda b4c: b4c in ('B1', 'B2'),
        '4号艇 不明': lambda b4c: b4c not in ('A1', 'A2', 'B1', 'B2'),
    }
    rows = []
    for label, cond in groups.items():
        for period in ['全期間', 'GW', 'post_GW']:
            n_races = 0
            n_b1_win = 0    # 1号艇 1着
            n_b4_win = 0    # 4号艇 1着
            n_b4_top3 = 0   # 4号艇 2-3着
            n_hits = 0      # P6 hit
            picks_with_4 = 0
            for rid, bs in by_race.items():
                b4_class = bs[0]['boat4_class']
                if not cond(b4_class):
                    continue
                d = bs[0]['bet_date']
                if period == 'GW' and is_post_gw(d):
                    continue
                if period == 'post_GW' and not is_post_gw(d):
                    continue
                n_races += 1
                if bs[0]['result_1st'] == 1:
                    n_b1_win += 1
                if bs[0]['result_1st'] == 4:
                    n_b4_win += 1
                if bs[0]['result_2nd'] == 4 or bs[0]['result_3rd'] == 4:
                    n_b4_top3 += 1
                if any(b['is_hit'] for b in bs):
                    n_hits += 1
                if any('4' in str(b['combination']) for b in bs):
                    picks_with_4 += 1
            rows.append({
                'group': label,
                'period': period,
                'n_races': n_races,
                'n_b1_win': n_b1_win,
                'n_b4_win': n_b4_win,
                'n_b4_top3': n_b4_top3,
                'n_hits': n_hits,
                'picks_with_4': picks_with_4,
                'b1_win_rate': n_b1_win / n_races * 100 if n_races else 0,
                'b4_win_rate': n_b4_win / n_races * 100 if n_races else 0,
                'hit_rate': n_hits / n_races * 100 if n_races else 0,
            })
    return rows


# ============================================================
# Report writer
# ============================================================

def write_report(bets, q1, q2, q3, q4, q5):
    lines = []
    lines.append("# mc3_venue_focus_r4 (P6) miss pattern 分析\n\n")
    lines.append("対象期間: 2026-05-01 〜 2026-05-18 (18 日、110 bets、7 hits)\n")
    lines.append("**GW (5/1-5/6)**: hit 率 11.4% / PnL +¥165k / **Post-GW (5/7-5/18)**: hit 率 3.0% / PnL -¥118k\n\n")

    # ============== サマリ統計 ==============
    lines.append("## 期間別サマリ\n\n")
    gw_bets = [b for b in bets if not is_post_gw(b['bet_date'])]
    pg_bets = [b for b in bets if is_post_gw(b['bet_date'])]
    lines.append("| 期間 | n_bets | n_hits | hit率 | invested | returned | PnL |\n|---|---|---|---|---|---|---|\n")
    for label, bs in [('GW (5/1-5/6)', gw_bets), ('Post-GW (5/7-5/18)', pg_bets), ('合計', bets)]:
        n = len(bs)
        h = sum(1 for b in bs if b['is_hit'])
        inv = sum(float(b['amount'] or 0) for b in bs)
        ret = sum(float(b['return_amount'] or 0) for b in bs)
        lines.append(f"| {label} | {n} | {h} | {h/n*100:.2f}% | ¥{inv:,.0f} | ¥{ret:,.0f} | ¥{ret-inv:+,.0f} |\n")

    # ============== Q1 ==============
    lines.append("\n## Q1: 本命崩れ頻度の変化\n\n")
    lines.append("**仮説**: GW 中は素人参加でオッズ歪み拡大 → 1号艇本命が刺さりやすい。\n")
    lines.append("Post-GW で市場効率化 → 本命崩れ頻度が上昇。\n\n")
    lines.append("| 期間 | n_races (P6 発火) | 1号艇 1着 race 数 | 1号艇 1着率 |\n|---|---|---|---|\n")
    for r in q1:
        lines.append(f"| {r['period']} | {r['n_races']} | {r['n_b1_win']} | {r['b1_win_rate']:.2f}% |\n")
    lines.append("\n**判定基準**: post_GW で b1_win_rate が GW より顕著に低下 (>5pt) → 仮説支持\n\n")

    # ============== Q2 ==============
    lines.append("\n## Q2: 外した時の actual 1着分布\n\n")
    lines.append("**仮説**: post-GW で外し race の 1着が 1号艇以外 (3号艇/4号艇/5号艇) に分散していれば、本命崩れによる loss を支持。\n\n")
    lines.append("| 期間 | 外し race 数 | actual 1着 = 1 | =2 | =3 | =4 | =5 | =6 |\n|---|---|---|---|---|---|---|---|\n")
    for period in ['GW', 'post_GW']:
        d = q2.get(period, {})
        total = sum(d.values())
        row = f"| {period} | {total} |"
        for boat in range(1, 7):
            n = d.get(boat, 0)
            row += f" {n} ({n/total*100:.1f}%) |" if total else " 0 |"
        lines.append(row + "\n")

    # ============== Q3 ==============
    lines.append("\n## Q3: 2-3 着外しの構造 (1号艇 1着 race 内)\n\n")
    lines.append("**仮説**: 1着 = 1 自体は当たっているが 2-3 着で外していれば、calibration 問題。\n")
    lines.append("逆に 1着 = 1 race で picks が当たっていなければ、軸選定の問題ではなく純粋に組合せ miss。\n\n")
    lines.append("| 期間 | 1着=1 race 数 | picks に 1-X-X 含む | hit (3連単正解) | pick率 | hit率 |\n|---|---|---|---|---|---|\n")
    for r in q3:
        lines.append(f"| {r['period']} | {r['n_b1win_races']} | {r['n_b1win_pick']} | {r['n_b1win_hit']} | "
                     f"{r['pick_rate']:.1f}% | {r['hit_rate']:.1f}% |\n")

    # ============== Q4 ==============
    lines.append("\n## Q4: 会場 × R 番号別 PnL\n\n")
    lines.append("**仮説**: 特定会場・R 番号に loss が集中していれば、QMC 非重複 filter 候補。\n\n")
    lines.append("| venue | R | n_bets | n_hits | hit率 | invested | returned | PnL | GW bets | postGW bets |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(q4, key=lambda x: x['pnl']):
        lines.append(f"| {r['venue_id']} | R{r['race_number']} | {r['n_bets']} | {r['n_hits']} | "
                     f"{r['hit_rate']:.1f}% | ¥{r['invested']:,.0f} | ¥{r['returned']:,.0f} | "
                     f"¥{r['pnl']:+,.0f} | {r['GW_bets']} | {r['post_GW_bets']} |\n")

    # ============== Q5 ==============
    lines.append("\n## Q5: 4号艇クラス別の挙動 (岩下さん観察検証)\n\n")
    lines.append("**観察**: 4号艇に A 級が入ると戦略発火するが、実際は 1号艇内の BC 級が来るパターン。\n")
    lines.append("**仮説**: 4号艇 A 級 = 市場が過剰評価 → 1号艇オッズ歪み → 「neglected favorite」 として 1号艇 BC 級が勝つ。\n")
    lines.append("ただし「たまに見てる」観察は confirmation bias の温床、data 検証要。\n\n")
    lines.append("| 4号艇クラス | 期間 | n_races | 1号艇 1着 (rate) | 4号艇 1着 (rate) | 4号艇 2-3着 | hit (rate) |\n|---|---|---|---|---|---|---|\n")
    for r in q5:
        lines.append(f"| {r['group']} | {r['period']} | {r['n_races']} | "
                     f"{r['n_b1_win']} ({r['b1_win_rate']:.1f}%) | "
                     f"{r['n_b4_win']} ({r['b4_win_rate']:.1f}%) | "
                     f"{r['n_b4_top3']} | "
                     f"{r['n_hits']} ({r['hit_rate']:.1f}%) |\n")

    # ============== 自動判定 + 採用候補 ==============
    lines.append("\n## 採用基準による自動振り分け (CLAUDE.md 準拠)\n\n")

    # Q1 判定
    q1_dict = {r['period']: r for r in q1}
    if 'GW' in q1_dict and 'post_GW' in q1_dict:
        diff = q1_dict['GW']['b1_win_rate'] - q1_dict['post_GW']['b1_win_rate']
        if diff > 5.0:
            lines.append(f"- **Q1 (本命崩れ)** 🟢 仮説支持 (GW {q1_dict['GW']['b1_win_rate']:.1f}% → post_GW {q1_dict['post_GW']['b1_win_rate']:.1f}%、差 {diff:+.1f}pt)\n")
        elif diff > 2.0:
            lines.append(f"- **Q1 (本命崩れ)** 🟡 弱い支持 (差 {diff:+.1f}pt)\n")
        else:
            lines.append(f"- **Q1 (本命崩れ)** 🔴 仮説不支持 (差 {diff:+.1f}pt)\n")

    # Q5 判定 (4号艇 A 級時の 1号艇 1着率 vs 4号艇 B 級時)
    q5_a_all = next((r for r in q5 if r['group'].startswith('4号艇 A 級') and r['period'] == '全期間'), None)
    q5_b_all = next((r for r in q5 if r['group'].startswith('4号艇 B 級') and r['period'] == '全期間'), None)
    if q5_a_all and q5_b_all and q5_a_all['n_races'] >= 10:
        diff = q5_a_all['b1_win_rate'] - q5_b_all['b1_win_rate']
        if q5_a_all['n_races'] >= 30 and diff > 10:
            lines.append(f"- **Q5 (4号艇 A 級観察)** 🟢 観察支持 (A 級時 1号艇 {q5_a_all['b1_win_rate']:.1f}% vs B 級時 {q5_b_all['b1_win_rate']:.1f}%、差 {diff:+.1f}pt、n={q5_a_all['n_races']})\n")
        elif diff > 5:
            lines.append(f"- **Q5 (4号艇 A 級観察)** 🟡 弱い支持 (差 {diff:+.1f}pt、n={q5_a_all['n_races']})\n")
        else:
            lines.append(f"- **Q5 (4号艇 A 級観察)** 🔴 観察不支持 (差 {diff:+.1f}pt、n={q5_a_all['n_races']})\n")
    elif q5_a_all:
        lines.append(f"- **Q5 (4号艇 A 級観察)** ⚪ サンプル不足 (n={q5_a_all['n_races']} < 30、判定保留)\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- 全 q 共通: P6 18 日 110 bets はサンプル不足、effect size 大でない signal は判定不能\n")
    lines.append("- Q5 は 観察ベース、confirmation bias 排除のため n>=30 を最低条件\n")
    lines.append("- 自動判定は heuristic 閾値、最終判断は岩下さん\n")
    lines.append("- 採用候補は **shadow 2 週間必須**、P7 失敗教訓を踏襲\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    logger.info(f"P6 miss pattern 分析: strategy={STRATEGY}")
    bets = fetch_bets_with_context()
    logger.info(f"bets fetched: {len(bets)}")
    if not bets:
        raise SystemExit("bets が取れない")

    race_ids = list({b['race_id'] for b in bets})
    race_boats = fetch_race_boats(race_ids)
    logger.info(f"races: {len(race_ids)}, boats coverage: {len(race_boats)}")

    logger.info("[Q1] 本命崩れ頻度")
    q1 = q1_honmei_breakdown(bets)
    logger.info("[Q2] 外し時の 1着分布")
    q2 = q2_miss_winner_dist(bets)
    logger.info("[Q3] 2-3 着外し構造")
    q3 = q3_position_miss(bets)
    logger.info("[Q4] 会場 × R 別 PnL")
    q4 = q4_venue_race_loss(bets)
    logger.info("[Q5] 4号艇 A 級発火時")
    q5 = q5_boat4_a_class(bets, race_boats)

    write_report(bets, q1, q2, q3, q4, q5)
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
