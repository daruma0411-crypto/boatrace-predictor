"""
純粋な予測精度分析スクリプト
==============================
predictions テーブルの確率 × races テーブルの実結果を突き合わせ、
ベット・オッズ・EV フィルターを一切除外した「AIモデルの実力」を評価する。

出力指標:
  1. Top-1 / Top-5 / Top-10 予測精度 (3連単120通り中の正解ランク)
  2. 1着単体の予測精度
  3. 本命 vs 穴の精度比較 (払戻金額で分類)
  4. 戦略別・日別の推移

Usage:
  python scripts/analyze_prediction_accuracy.py
"""

import os
import json
import psycopg2
import psycopg2.extras
from itertools import permutations
from collections import defaultdict
from datetime import date

# ─── DB接続 ───
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable"
)

# ─── 3連単120通りの順列 (0-indexed) ───
SANRENTAN_COMBOS = list(permutations(range(6), 3))
# コンボ文字列 → インデックス
COMBO_TO_IDX = {}
for idx, (i, j, k) in enumerate(SANRENTAN_COMBOS):
    key = f"{i+1}-{j+1}-{k+1}"
    COMBO_TO_IDX[key] = idx


def compute_sanrentan_probs(probs_1st, probs_2nd, probs_3rd):
    """条件付き確率で3連単120通りの確率を計算

    P(i,j,k) = P(1st=i) * P(2nd=j|1st≠i) * P(3rd=k|1st≠i,2nd≠j)
    """
    probs = [0.0] * 120
    for idx, (i, j, k) in enumerate(SANRENTAN_COMBOS):
        p1 = probs_1st[i]

        # 2着: i を除外して正規化
        mask_2nd = [probs_2nd[x] for x in range(6) if x != i]
        sum_2nd = sum(mask_2nd) or 1e-10
        p2_given_1 = probs_2nd[j] / sum_2nd

        # 3着: i, j を除外して正規化
        mask_3rd = [probs_3rd[x] for x in range(6) if x != i and x != j]
        sum_3rd = sum(mask_3rd) or 1e-10
        p3_given_12 = probs_3rd[k] / sum_3rd

        probs[idx] = p1 * p2_given_1 * p3_given_12

    return probs


def get_rank(probs, target_idx):
    """確率リストの中で target_idx が上位何番目か (1-indexed)"""
    target_prob = probs[target_idx]
    rank = 1
    for p in probs:
        if p > target_prob:
            rank += 1
    return rank


def classify_payout(payout_amount):
    """払戻金額でレースを分類
    ガチガチ: ~1,999円 (オッズ ~20倍未満)
    中穴:     2,000~9,999円
    万舟:     10,000円以上 (オッズ100倍以上)
    """
    if payout_amount is None or payout_amount <= 0:
        return "不明"
    if payout_amount < 2000:
        return "本命 (<2,000円)"
    if payout_amount < 10000:
        return "中穴 (2,000~9,999円)"
    return "万舟 (10,000円以上)"


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # ─── データ取得 ───
    # 各レース×戦略ごとに1件の予測を取得 (結果確定済みのみ)
    cur.execute("""
        SELECT
            p.id AS prediction_id,
            p.race_id,
            p.strategy_type,
            p.probabilities_1st,
            p.probabilities_2nd,
            p.probabilities_3rd,
            p.model_version,
            r.race_date,
            r.venue_id,
            r.race_number,
            r.actual_result_trifecta,
            r.payout_amount,
            r.result_1st,
            r.result_2nd,
            r.result_3rd
        FROM predictions p
        JOIN races r ON p.race_id = r.id
        WHERE r.actual_result_trifecta IS NOT NULL
          AND r.is_finished = TRUE
          AND r.race_date >= '2026-03-11'
        ORDER BY r.race_date, r.venue_id, r.race_number, p.strategy_type
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("結果確定済みの予測データが0件です。")
        return

    # ─── 戦略ごとに重複除去: 同一 race_id × strategy_type で最初の1件 ───
    seen = set()
    predictions = []
    for row in rows:
        key = (row['race_id'], row['strategy_type'])
        if key in seen:
            continue
        seen.add(key)
        predictions.append(dict(row))

    # ─── 戦略ごとの分析 ───
    # まず「全戦略共通」（同じモデルの確率を使うのでrace_idごとに1件だけ使用）
    race_seen = set()
    unique_predictions = []
    for p in predictions:
        if p['race_id'] not in race_seen:
            race_seen.add(p['race_id'])
            unique_predictions.append(p)

    print("=" * 70)
    print("  ボートレース AI 純粋予測精度レポート")
    print(f"  分析日: {date.today()}  |  対象: 2026-03-11 以降")
    print("=" * 70)

    # ─── 全体分析 (レースごとに1件) ───
    analyze_group("全レース (モデル共通)", unique_predictions)

    # ─── 戦略別分析 ───
    strategy_groups = defaultdict(list)
    for p in predictions:
        strategy_groups[p['strategy_type']].append(p)

    print("\n" + "=" * 70)
    print("  戦略別分析")
    print("=" * 70)

    for strategy in sorted(strategy_groups.keys()):
        group = strategy_groups[strategy]
        # 戦略内でもrace_idの重複除去
        s_seen = set()
        unique = []
        for p in group:
            if p['race_id'] not in s_seen:
                s_seen.add(p['race_id'])
                unique.append(p)
        analyze_group(f"戦略: {strategy}", unique, compact=True)

    # ─── 日別推移 ───
    print("\n" + "=" * 70)
    print("  日別 Top-10 的中率推移 (全レース)")
    print("=" * 70)

    date_groups = defaultdict(list)
    for p in unique_predictions:
        date_groups[str(p['race_date'])].append(p)

    print(f"{'日付':<14} {'レース数':>8} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'1着正解':>8}")
    print("-" * 62)

    for d in sorted(date_groups.keys()):
        group = date_groups[d]
        stats = compute_stats(group)
        n = stats['total']
        print(f"{d:<14} {n:>8} "
              f"{stats['top1']/n*100:>7.1f}% "
              f"{stats['top5']/n*100:>7.1f}% "
              f"{stats['top10']/n*100:>7.1f}% "
              f"{stats['first_correct']/n*100:>7.1f}%")


def compute_stats(predictions):
    """予測リストの統計を計算"""
    stats = {
        'total': 0,
        'top1': 0,
        'top3': 0,
        'top5': 0,
        'top10': 0,
        'top20': 0,
        'first_correct': 0,
        'second_correct': 0,
        'third_correct': 0,
        'ranks': [],
        'payout_groups': defaultdict(lambda: {'total': 0, 'top1': 0, 'top5': 0, 'top10': 0}),
    }

    for p in predictions:
        result = p['actual_result_trifecta']
        if result not in COMBO_TO_IDX:
            continue

        # 確率パース
        p1 = p['probabilities_1st']
        p2 = p['probabilities_2nd']
        p3 = p['probabilities_3rd']
        if isinstance(p1, str):
            p1 = json.loads(p1)
        if isinstance(p2, str):
            p2 = json.loads(p2)
        if isinstance(p3, str):
            p3 = json.loads(p3)

        # 3連単120通り確率
        sanrentan = compute_sanrentan_probs(p1, p2, p3)
        target_idx = COMBO_TO_IDX[result]
        rank = get_rank(sanrentan, target_idx)

        stats['total'] += 1
        stats['ranks'].append(rank)

        if rank <= 1:
            stats['top1'] += 1
        if rank <= 3:
            stats['top3'] += 1
        if rank <= 5:
            stats['top5'] += 1
        if rank <= 10:
            stats['top10'] += 1
        if rank <= 20:
            stats['top20'] += 1

        # 各着順の単体精度 (actual_result_trifecta からパース、result_1st等はNULLが多い)
        parts = result.split('-')
        actual_1st = int(parts[0])
        actual_2nd = int(parts[1])
        actual_3rd = int(parts[2])

        pred_1st = max(range(6), key=lambda x: p1[x]) + 1
        pred_2nd = max(range(6), key=lambda x: p2[x]) + 1
        pred_3rd = max(range(6), key=lambda x: p3[x]) + 1

        if pred_1st == actual_1st:
            stats['first_correct'] += 1
        if pred_2nd == actual_2nd:
            stats['second_correct'] += 1
        if pred_3rd == actual_3rd:
            stats['third_correct'] += 1

        # 払戻金額別
        payout_class = classify_payout(p['payout_amount'])
        pg = stats['payout_groups'][payout_class]
        pg['total'] += 1
        if rank <= 1:
            pg['top1'] += 1
        if rank <= 5:
            pg['top5'] += 1
        if rank <= 10:
            pg['top10'] += 1

    return stats


def analyze_group(title, predictions, compact=False):
    """グループの分析結果を出力"""
    stats = compute_stats(predictions)
    n = stats['total']

    if n == 0:
        print(f"\n[{title}] 有効データ0件")
        return

    ranks = stats['ranks']
    avg_rank = sum(ranks) / len(ranks)
    median_rank = sorted(ranks)[len(ranks) // 2]

    if compact:
        print(f"\n[{title}]  N={n:,}  "
              f"Top1={stats['top1']/n*100:.1f}%  "
              f"Top5={stats['top5']/n*100:.1f}%  "
              f"Top10={stats['top10']/n*100:.1f}%  "
              f"1着={stats['first_correct']/n*100:.1f}%  "
              f"平均rank={avg_rank:.1f}")
        return

    print(f"\n{'─' * 60}")
    print(f"  [{title}]  対象レース数: {n:,}")
    print(f"{'─' * 60}")

    # ランダム基準
    random_top1 = 1 / 120 * 100
    random_top5 = 5 / 120 * 100
    random_top10 = 10 / 120 * 100

    print(f"\n  ■ 3連単予測精度 (120通り中)")
    print(f"  {'指標':<20} {'的中数':>8} {'的中率':>10} {'ランダム基準':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Top-1 (完全一致)':<20} {stats['top1']:>8} {stats['top1']/n*100:>9.2f}% {random_top1:>11.2f}%")
    print(f"  {'Top-3':<20} {stats['top3']:>8} {stats['top3']/n*100:>9.2f}% {3/120*100:>11.2f}%")
    print(f"  {'Top-5':<20} {stats['top5']:>8} {stats['top5']/n*100:>9.2f}% {random_top5:>11.2f}%")
    print(f"  {'Top-10':<20} {stats['top10']:>8} {stats['top10']/n*100:>9.2f}% {random_top10:>11.2f}%")
    print(f"  {'Top-20':<20} {stats['top20']:>8} {stats['top20']/n*100:>9.2f}% {20/120*100:>11.2f}%")
    print(f"  {'平均ランク':<20} {'':>8} {avg_rank:>9.1f}{'位':>4}")
    print(f"  {'中央値ランク':<20} {'':>8} {median_rank:>9}{'位':>4}")

    # ランダム比
    if stats['top1'] > 0:
        lift_top1 = (stats['top1'] / n) / (1 / 120)
    else:
        lift_top1 = 0
    lift_top10 = (stats['top10'] / n) / (10 / 120)
    print(f"\n  → Top-1 リフト (vs ランダム): {lift_top1:.1f}倍")
    print(f"  → Top-10 リフト (vs ランダム): {lift_top10:.1f}倍")

    print(f"\n  ■ 各着順の単体予測精度 (最高確率の艇が正解だった率)")
    print(f"  {'1着予測精度':<20} {stats['first_correct']:>8} {stats['first_correct']/n*100:>9.2f}% {'(ランダム 16.7%)':>16}")
    print(f"  {'2着予測精度':<20} {stats['second_correct']:>8} {stats['second_correct']/n*100:>9.2f}% {'(ランダム 16.7%)':>16}")
    print(f"  {'3着予測精度':<20} {stats['third_correct']:>8} {stats['third_correct']/n*100:>9.2f}% {'(ランダム 16.7%)':>16}")

    # ─── 本命 vs 穴 ───
    print(f"\n  ■ 本命 vs 穴レースの精度比較")
    print(f"  {'分類':<22} {'レース数':>8} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8}")
    print(f"  {'-'*58}")

    for cls in ["本命 (<2,000円)", "中穴 (2,000~9,999円)", "万舟 (10,000円以上)", "不明"]:
        pg = stats['payout_groups'].get(cls)
        if pg and pg['total'] > 0:
            t = pg['total']
            print(f"  {cls:<22} {t:>8} "
                  f"{pg['top1']/t*100:>7.1f}% "
                  f"{pg['top5']/t*100:>7.1f}% "
                  f"{pg['top10']/t*100:>7.1f}%")

    # ─── ランク分布ヒストグラム ───
    print(f"\n  ■ 正解ランク分布 (3連単)")
    buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 40), (41, 60), (61, 120)]
    for lo, hi in buckets:
        count = sum(1 for r in ranks if lo <= r <= hi)
        bar = "#" * int(count / n * 50)
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"  {label:>7}位: {count:>5} ({count/n*100:>5.1f}%) {bar}")


if __name__ == "__main__":
    main()
