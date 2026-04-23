"""V11 vs V10 公平 Retrospective Backtest

公平比較ロジック:
  V10が実際にbetした同じ組合せに対して、V11が「同じ判断を下すか？」を評価。
  V10の記録した `bets.odds` を真の市場オッズとして使用（retrospective公平性）。

  各V10-betについて:
    V11_prob = V11が予測する P(combo=V10の選定組合せ)
    V11_EV = V11_prob × bets.odds  (市場オッズは V10記録を流用)

    V10同等フィルタ:
      - V11_EV in [0.5, 0.8]       (EVフィルタ)
      - bets.odds in [5, 40]       (odds帯フィルタ, V10設定継承)
      - Kelly sizing: stake = kelly_fraction × edge × bankroll

    V11が採択した bets のみで ROI集計:
      - stake 支出
      - bet が is_hit なら payout受取

  V10超え判定:
    同じbet universe で V11の ROI が V10実績 (124.5%) を超えるか。

V11の「selection能力」（どの組合せを選ぶかの判断力）を純粋に評価する。
"""
import os
import sys
import json
import pickle
import math
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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

# V10 フィルタ設定（config/betting_config.json の mc_early_race と同一）
EV_MIN = 0.5
EV_MAX = 0.8
ODDS_MIN = 5.0
ODDS_MAX = 40.0
KELLY_FRAC = 0.0625
MAX_TICKET_RATIO = 0.008
MAX_TOTAL_RATIO = 0.02
BANKROLL = 200000  # initial_bankroll


def predict_all_combo_probs(models, X_row):
    """全120組合せの raw prob をnormalize して真の確率分布化

    1st/2nd/3rd 独立モデルの積は合計が1にならない（3連単として不正）ので
    全120組合せで正規化して確率分布化する。
    """
    p1 = models['1st'].predict([X_row])[0]
    p2 = models['2nd'].predict([X_row])[0]
    p3 = models['3rd'].predict([X_row])[0]
    combos = {}
    total = 0.0
    for a in range(6):
        for b in range(6):
            if b == a: continue
            for c in range(6):
                if c == a or c == b: continue
                raw = p1[a] * p2[b] * p3[c]
                combos[f"{a+1}-{b+1}-{c+1}"] = raw
                total += raw
    if total > 0:
        for k in combos:
            combos[k] /= total
    return combos


def predict_combo_prob(models, X_row, combo):
    """特定組合せ combo の確率（normalized）"""
    try:
        a, b, c = [int(x) - 1 for x in combo.split('-')]
    except (ValueError, AttributeError):
        return 0.0
    if not (0 <= a <= 5 and 0 <= b <= 5 and 0 <= c <= 5):
        return 0.0
    if a == b or a == c or b == c:
        return 0.0
    probs = predict_all_combo_probs(models, X_row)
    return probs.get(combo, 0.0)


def kelly_stake(prob, odds, bankroll, kelly_frac=KELLY_FRAC):
    """Kelly 賭け金（実装は V10 に準拠）

    f* = (p*b - q) / b, b = odds - 1, q = 1-p
    stake = f* × kelly_frac × bankroll (100円単位丸め、最低100円)
    max_ticket で上限キャップ
    """
    if odds <= 1 or prob <= 0 or prob >= 1:
        return 0
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (prob * b - q) / b
    if f_star <= 0:
        return 0
    raw = f_star * kelly_frac * bankroll
    max_ticket = MAX_TICKET_RATIO * bankroll  # ¥1,600
    raw = min(raw, max_ticket)
    stake = max(100, int(raw / 100) * 100)
    return stake


def evaluate_variant_on_v10_universe(variant_name, v10_bets, X_by_race_id):
    """V10の betsユニバースで V11 を評価"""
    vdir = GRID_DIR / variant_name
    models = {pos: lgb.Booster(model_file=str(vdir / f"{pos}.txt"))
              for pos in ['1st', '2nd', '3rd']}

    stats = {
        'considered_bets': 0,     # V10がbetした全件数
        'v11_accepted': 0,        # V11が採択した件数
        'v11_rejected_ev': 0,
        'v11_rejected_odds': 0,
        'v11_invest': 0,
        'v11_payout': 0,
        'v11_hits': 0,
    }

    # race_id ごとに全combo probs をキャッシュ
    prob_cache = {}

    for bet in v10_bets:
        rid = bet['race_id']
        if rid not in X_by_race_id:
            continue
        X_row = X_by_race_id[rid]
        v10_odds = float(bet.get('odds') or 0)
        v10_combo = bet['combination']

        stats['considered_bets'] += 1

        # V11の prob 計算（race_id単位でキャッシュ）
        if rid not in prob_cache:
            prob_cache[rid] = predict_all_combo_probs(models, X_row)
        v11_prob = prob_cache[rid].get(v10_combo, 0.0)
        if v11_prob <= 0:
            stats['v11_rejected_ev'] += 1
            continue

        # V11_EV = v11_prob × v10_odds
        v11_ev = v11_prob * v10_odds

        # EV フィルタ
        if not (EV_MIN <= v11_ev <= EV_MAX):
            stats['v11_rejected_ev'] += 1
            continue

        # odds フィルタ
        if not (ODDS_MIN <= v10_odds <= ODDS_MAX):
            stats['v11_rejected_odds'] += 1
            continue

        # V11のKellyで賭け金決定
        stake = kelly_stake(v11_prob, v10_odds, BANKROLL)
        if stake < 100:
            continue

        stats['v11_accepted'] += 1
        stats['v11_invest'] += stake

        if bet.get('is_hit'):
            # payout = v10_odds × stake / 100 (100円単位の配当率なのでそのまま掛け算)
            payout = int(v10_odds * stake)
            stats['v11_payout'] += payout
            stats['v11_hits'] += 1

    stats['v11_roi'] = (stats['v11_payout'] / stats['v11_invest']
                       if stats['v11_invest'] else 0)
    stats['v11_hit_rate'] = (stats['v11_hits'] / stats['v11_accepted']
                            if stats['v11_accepted'] else 0)
    stats['v11_profit'] = stats['v11_payout'] - stats['v11_invest']
    return stats


def main():
    logger.info("V11 vs V10 公平 Retrospective Backtest 開始")

    # val データ
    with open(MODELS_DIR / "train_data.pkl", 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    race_ids = [int(r) for r in data['race_ids']]
    n = len(X)
    split = int(n * 0.8)
    val_race_ids = set(race_ids[split:])
    X_by_rid = {race_ids[i]: X[i] for i in range(split, n)}
    logger.info(f"val レース数: {len(val_race_ids)}")

    # V10 の bets (mc_early_race) を val期間で取得
    conn = psycopg2.connect(os.environ['DATABASE_URL'],
                            cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, race_id, combination, amount, odds, expected_value,
               is_hit, payout, result
        FROM bets
        WHERE strategy_type = 'mc_early_race'
          AND race_id = ANY(%s)
          AND result IS NOT NULL
        ORDER BY id
    """, (list(val_race_ids),))
    v10_bets = [dict(b) for b in cur.fetchall()]
    conn.close()
    logger.info(f"V10 bets (val期間): {len(v10_bets)}件")

    # V10 実績集計
    v10_invest = sum(int(b['amount']) for b in v10_bets)
    v10_payout = sum(int(b['payout'] or 0) for b in v10_bets if b['is_hit'])
    v10_hits = sum(1 for b in v10_bets if b['is_hit'])
    v10_roi = v10_payout / v10_invest if v10_invest else 0
    logger.info(f"V10 実績: bets={len(v10_bets)} 投資={v10_invest:,} "
                f"回収={v10_payout:,} ROI={v10_roi*100:.1f}%")

    # 全変種評価
    variants = sorted([d.name for d in GRID_DIR.iterdir() if d.is_dir()])
    results = {}
    logger.info(f"評価: {len(variants)}変種")

    for v in variants:
        logger.info(f"  {v}...")
        stats = evaluate_variant_on_v10_universe(v, v10_bets, X_by_rid)
        results[v] = stats

    # レポート生成
    lines = [
        "# V11 vs V10 公平 Retrospective Backtest",
        f"\n生成: {datetime.now():%Y-%m-%d %H:%M:%S} JST",
        f"\n評価方法: V10実bets {len(v10_bets)}件に対し、V11が各betを採択/却下",
        f"- 採択判定: V11_EV in [{EV_MIN}, {EV_MAX}] かつ odds in [{ODDS_MIN}, {ODDS_MAX}]",
        f"- 賭け金: V11のKelly計算 (kelly_frac={KELLY_FRAC}, bankroll=¥{BANKROLL:,})",
        f"- 市場オッズは V10記録 (bets.odds) を使用",
        "",
        "## V10 実績（val期間ベースライン）",
        f"- bets: **{len(v10_bets)}** | 的中: {v10_hits} ({v10_hits/len(v10_bets)*100:.1f}%)",
        f"- 投資: ¥{v10_invest:,} | 回収: ¥{v10_payout:,} | "
        f"**ROI: {v10_roi*100:.1f}%** | 損益: ¥{v10_payout-v10_invest:+,}",
        "",
        "## V11 全15変種（V10と同じユニバースで selection 能力を評価）",
        "",
        "| 変種 | 採択 | EV却下 | 的中 | 的中率 | V11投資 | V11回収 | **V11 ROI** | 損益 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    ranked = sorted(results.items(), key=lambda x: -x[1]['v11_roi'])
    for name, s in ranked:
        profit_sign = '+' if s['v11_profit'] >= 0 else ''
        hit_rate_pct = s['v11_hit_rate'] * 100
        roi_pct = s['v11_roi'] * 100
        lines.append(
            f"| {name} | {s['v11_accepted']} | {s['v11_rejected_ev']} | "
            f"{s['v11_hits']} | {hit_rate_pct:.1f}% | "
            f"¥{s['v11_invest']:,} | ¥{s['v11_payout']:,} | "
            f"**{roi_pct:.1f}%** | "
            f"{profit_sign}¥{s['v11_profit']:,} |"
        )
    lines.append("")

    # V10超え判定
    winners = [(n, s) for n, s in ranked if s['v11_roi'] > v10_roi
               and s['v11_accepted'] >= 10]  # 最低10 bets で信頼性確保
    lines.append(f"## 🎯 V10超え判定（V10 ROI {v10_roi*100:.1f}% / n>=10 bets）")
    lines.append("")
    if winners:
        lines.append(f"### ✅ V10を超えた変種: {len(winners)}個")
        lines.append("")
        for n, s in winners:
            lines.append(f"- **{n}**: ROI {s['v11_roi']*100:.1f}% "
                        f"(+{(s['v11_roi']-v10_roi)*100:.1f}pt) / "
                        f"採択{s['v11_accepted']}件")
    else:
        lines.append("### ❌ V10超えなし")
        lines.append("")
        lines.append("全変種が V10 の ROI を下回った。")
        lines.append("- V11の selection 能力は V10 より劣る（この評価方法では）")
        lines.append("- ただし V11が V10と異なる組合せを選ぶケースは評価外")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 💡 注意事項")
    lines.append("")
    lines.append("- この評価は「V10がbetした組合せの中で、V11も賭けるか？」のみを測定")
    lines.append("- V11独自の新規組合せ発掘能力は評価外")
    lines.append("- 完全な評価には live shadow 運用 or 全combo odds推定が必要")
    lines.append("- val=1,240レース、V10 bets=" + str(len(v10_bets)) + "件 なので"
                 "統計的に不十分（最低100-200件欲しい）")

    # 保存
    md_path = REPORT_DIR / "07_fair_backtest.md"
    md_path.write_text('\n'.join(lines), encoding='utf-8')

    json_path = REPORT_DIR / "07_fair_backtest.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'v10_stats': {
                'bets': len(v10_bets), 'hits': v10_hits,
                'invest': v10_invest, 'payout': v10_payout,
                'roi': v10_roi,
            },
            'variants': results,
        }, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\nレポート: {md_path}")

    # コンソール
    logger.info("\n=== Top 5 (V10=" + f"{v10_roi*100:.1f}%)" + " ===")
    for name, s in ranked[:5]:
        mark = "🏆" if s['v11_roi'] > v10_roi and s['v11_accepted'] >= 10 else "  "
        logger.info(f"  {mark} {name:28s} ROI {s['v11_roi']*100:5.1f}% "
                    f"採択 {s['v11_accepted']:3d}件")


if __name__ == '__main__':
    main()
