"""V11 グリッド backtest: 15変種を val集合で一括評価 + V10実績と比較

評価手法（全変種共通）:
  - val 1,240レースで top3 combo を予測
  - ¥1000/点 × 3点 固定で購入
  - races.actual_result_trifecta と一致で payout_sanrentan 獲得
  - Miss分析フィルタ適用版も同時評価

V10比較:
  - 同期間のV10実績 (mc_early_race) を bets テーブルから取得
  - V10のROIを真値として並べる

出力:
  analysis/reports/06_grid_backtest.md  (markdown比較表)
  analysis/reports/06_grid_backtest.json (生データ)
"""
import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import lightgbm as lgb
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models_v11"
GRID_DIR = MODELS_DIR / "grid"
REPORT_DIR = Path(__file__).parent / "reports"

STRONG_VENUES = [9, 10, 12, 20, 23]
MAX_R = 4
EXCLUDE_R = [1]
BET_AMOUNT = 1000
TOP_K = 3


def predict_topk(models, X_row, k=3):
    """topK組合せ予測"""
    p1 = models['1st'].predict([X_row])[0]
    p2 = models['2nd'].predict([X_row])[0]
    p3 = models['3rd'].predict([X_row])[0]
    combos = []
    for a in range(6):
        for b in range(6):
            if b == a: continue
            for c in range(6):
                if c in (a, b): continue
                score = p1[a] * p2[b] * p3[c]
                combos.append((f"{a+1}-{b+1}-{c+1}", score))
    combos.sort(key=lambda x: -x[1])
    return combos[:k]


def evaluate_variant(variant_name, X_val, race_info_val, apply_filter=False):
    """変種名を読み込んで backtest 実行"""
    vdir = GRID_DIR / variant_name
    models = {pos: lgb.Booster(model_file=str(vdir / f"{pos}.txt"))
              for pos in ['1st', '2nd', '3rd']}

    stats = {'bets': 0, 'hits': 0, 'invest': 0, 'payout': 0,
             'filtered_out': 0}

    for i, race in enumerate(race_info_val):
        if apply_filter:
            if race['venue_id'] not in STRONG_VENUES:
                stats['filtered_out'] += 1
                continue
            if race['race_number'] in EXCLUDE_R or race['race_number'] > MAX_R:
                stats['filtered_out'] += 1
                continue

        top = predict_topk(models, X_val[i], k=TOP_K)
        actual = race.get('actual_result_trifecta')
        payout = int(race.get('payout_sanrentan') or 0)

        for combo, _ in top:
            stats['bets'] += 1
            stats['invest'] += BET_AMOUNT
            if combo == actual and payout > 0:
                stats['hits'] += 1
                stats['payout'] += payout * (BET_AMOUNT // 100)

    stats['roi'] = stats['payout'] / stats['invest'] if stats['invest'] else 0
    stats['hit_rate'] = stats['hits'] / stats['bets'] if stats['bets'] else 0
    stats['profit'] = stats['payout'] - stats['invest']
    return stats


def fetch_v10_actual(cur, start_rid, end_rid):
    """V10 の実運用 bets を取得して ROI算出"""
    cur.execute("""
        SELECT COUNT(*) as bets,
               SUM(amount) as invest,
               SUM(CASE WHEN is_hit THEN 1 ELSE 0 END) as hits,
               SUM(COALESCE(payout, 0)) as payout
        FROM bets b
        WHERE strategy_type = 'mc_early_race'
          AND b.race_id BETWEEN %s AND %s
          AND b.result IS NOT NULL
    """, (start_rid, end_rid))
    row = cur.fetchone()
    if not row or not row['bets']:
        return None
    invest = int(row['invest'] or 0)
    payout = int(row['payout'] or 0)
    return {
        'bets': row['bets'],
        'hits': row['hits'] or 0,
        'invest': invest,
        'payout': payout,
        'profit': payout - invest,
        'roi': payout / invest if invest else 0,
        'hit_rate': (row['hits'] or 0) / row['bets'] if row['bets'] else 0,
    }


def main():
    logger.info("V11 グリッド backtest 開始")

    # val データ
    with open(MODELS_DIR / "train_data.pkl", 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    race_ids = data['race_ids']
    n = len(X)
    split = int(n * 0.8)
    X_val = X[split:]
    race_ids_val = [int(r) for r in race_ids[split:]]
    logger.info(f"val: {len(X_val)}レース")

    # race info
    conn = psycopg2.connect(os.environ['DATABASE_URL'],
                            cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, venue_id, race_number, actual_result_trifecta,
               payout_sanrentan, wind_speed, wave_height
        FROM races WHERE id = ANY(%s)
    """, (race_ids_val,))
    race_info_map = {r['id']: dict(r) for r in cur.fetchall()}
    race_info_val = [race_info_map[rid] for rid in race_ids_val]

    # V10実績（val期間のrace_id範囲内）
    v10_actual = fetch_v10_actual(cur, min(race_ids_val), max(race_ids_val))
    conn.close()

    if v10_actual:
        logger.info(f"V10実績: bets={v10_actual['bets']} "
                    f"ROI={v10_actual['roi']*100:.1f}% "
                    f"的中率={v10_actual['hit_rate']*100:.1f}%")
    else:
        logger.info("V10実績: データなし")

    # 全変種評価
    variants = sorted([d.name for d in GRID_DIR.iterdir() if d.is_dir()])
    results = {}
    logger.info(f"評価対象: {len(variants)}変種 × 2フィルタ")

    for v in variants:
        logger.info(f"  {v}...")
        r_raw = evaluate_variant(v, X_val, race_info_val, apply_filter=False)
        r_flt = evaluate_variant(v, X_val, race_info_val, apply_filter=True)
        results[v] = {'raw': r_raw, 'filtered': r_flt}

    # レポート作成
    lines = [
        "# V11 グリッド Backtest レポート",
        f"\n生成: {datetime.now():%Y-%m-%d %H:%M:%S} JST",
        f"\nval集合: {len(X_val)}レース (時系列80/20分割の直近20%)",
        f"評価方法: top-3 combo × ¥1000/点 固定購入",
        "",
    ]

    if v10_actual:
        lines.append("## V10 実績（同val期間）")
        lines.append("")
        lines.append(f"- bets: **{v10_actual['bets']}** | "
                     f"的中: {v10_actual['hits']} ({v10_actual['hit_rate']*100:.1f}%) | "
                     f"投資: ¥{v10_actual['invest']:,} | "
                     f"回収: ¥{v10_actual['payout']:,} | "
                     f"**ROI: {v10_actual['roi']*100:.1f}%**")
        lines.append("")
        lines.append("※ V10は EV/entropy/Kelly フィルタ通過分のみの実運用bets。公平ではないが実績値として参考。")
        lines.append("")

    # raw 比較表
    lines.append("## V11 全15変種（raw / top3購入）")
    lines.append("")
    lines.append("| 変種 | bets | 的中 | 的中率 | ROI | 損益 |")
    lines.append("|---|---|---|---|---|---|")
    rows_raw = []
    for name, r in results.items():
        rr = r['raw']
        rows_raw.append((name, rr))
    rows_raw.sort(key=lambda x: -x[1]['roi'])
    for name, rr in rows_raw:
        profit_sign = '+' if rr['profit'] >= 0 else ''
        lines.append(f"| {name} | {rr['bets']:,} | {rr['hits']} | "
                     f"{rr['hit_rate']*100:.2f}% | "
                     f"**{rr['roi']*100:.1f}%** | "
                     f"{profit_sign}¥{rr['profit']:,} |")
    lines.append("")

    # filtered 比較表
    lines.append("## V11 全15変種（Miss分析フィルタ適用: 勝ち会場R2-R4のみ）")
    lines.append("")
    lines.append("| 変種 | bets | 的中 | 的中率 | ROI | 損益 |")
    lines.append("|---|---|---|---|---|---|")
    rows_flt = []
    for name, r in results.items():
        rf = r['filtered']
        rows_flt.append((name, rf))
    rows_flt.sort(key=lambda x: -x[1]['roi'])
    for name, rf in rows_flt:
        profit_sign = '+' if rf['profit'] >= 0 else ''
        lines.append(f"| {name} | {rf['bets']:,} | {rf['hits']} | "
                     f"{rf['hit_rate']*100:.2f}% | "
                     f"**{rf['roi']*100:.1f}%** | "
                     f"{profit_sign}¥{rf['profit']:,} |")
    lines.append("")

    # V10超え判定
    if v10_actual:
        v10_roi = v10_actual['roi']
        lines.append(f"## 🎯 V10超え判定（V10 ROI={v10_roi*100:.1f}%）")
        lines.append("")
        winners_raw = [(n, r) for n, r in rows_raw if r['roi'] > v10_roi]
        winners_flt = [(n, r) for n, r in rows_flt if r['roi'] > v10_roi]
        if winners_raw:
            lines.append("### raw（top3購入）でV10超え:")
            for n, r in winners_raw:
                lines.append(f"- **{n}**: ROI {r['roi']*100:.1f}% (+{(r['roi']-v10_roi)*100:.1f}pt)")
        else:
            lines.append("### raw: V10超えなし")
        lines.append("")
        if winners_flt:
            lines.append("### filtered（Miss分析フィルタ）でV10超え:")
            for n, r in winners_flt:
                lines.append(f"- **{n}**: ROI {r['roi']*100:.1f}% (+{(r['roi']-v10_roi)*100:.1f}pt)")
        else:
            lines.append("### filtered: V10超えなし")
        lines.append("")
        lines.append("⚠️ 注: V11の評価は top3 naive 購入。V10はEV/Kellyフィルタ通過後。")
        lines.append("完全にフェアではないが、**V11変種間の相対比較**は有効。")

    # 保存
    md_path = REPORT_DIR / "06_grid_backtest.md"
    md_path.write_text('\n'.join(lines), encoding='utf-8')

    json_path = REPORT_DIR / "06_grid_backtest.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'v10_actual': v10_actual,
            'variants': results,
        }, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\nレポート: {md_path}")
    logger.info(f"JSON:    {json_path}")

    # コンソール概要
    logger.info("\n=== raw top5 ===")
    for name, rr in rows_raw[:5]:
        logger.info(f"  {name:28s} ROI {rr['roi']*100:5.1f}% "
                    f"hit {rr['hit_rate']*100:4.1f}%")
    logger.info("\n=== filtered top5 ===")
    for name, rf in rows_flt[:5]:
        logger.info(f"  {name:28s} ROI {rf['roi']*100:5.1f}% "
                    f"hit {rf['hit_rate']*100:4.1f}%")


if __name__ == '__main__':
    main()
