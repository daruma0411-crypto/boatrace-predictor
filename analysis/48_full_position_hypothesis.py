"""30 仮説 6 艇 × {1着率/2着率/3着率} 拡張 screening (CLAUDE.md 批判プロトコル準拠)

46_hypothesis_screening.py は「1号艇 1着率」だけ見ていたため、中穴狙い戦略の
2着・3着構造を見落とした。本スクリプトで各仮説 × 全 6 艇 × 3 着位置 = 18 metric
で再評価する。

期待: 1着率では signal なかった仮説が、2着率/3着率で +5pt 以上の lift 出る
ケースを発掘 (例: H10 trigger 時に 2 号艇の 2着率が上昇する 等)。

入力: races + boats + race_titles (READ-ONLY)
出力: analysis/reports/full_position_hypothesis.md
"""
import os
import sys
import logging
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy.stats import fisher_exact

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'full_position_hypothesis.md'

PERIOD_START = date(2026, 4, 11)
PERIOD_END = date(2026, 5, 18)


def fetch_data():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
               r.result_1st, r.result_2nd, r.result_3rd,
               r.wind_speed, r.wind_direction, r.wave_height,
               rt.title AS race_title, rt.subtitle AS race_subtitle, rt.day_label
        FROM races r LEFT JOIN race_titles rt ON r.id = rt.race_id
        WHERE r.is_finished = true
          AND r.result_1st IS NOT NULL AND r.result_2nd IS NOT NULL AND r.result_3rd IS NOT NULL
          AND r.race_date BETWEEN %s AND %s
        ORDER BY r.race_date, r.id
    """, (PERIOD_START, PERIOD_END))
    races = pd.DataFrame(cur.fetchall())
    race_ids = races['race_id'].tolist()
    cur.execute("""
        SELECT race_id, boat_number, player_class, win_rate_2, local_win_rate_2,
               avg_st, motor_win_rate_2, exhibition_time, approach_course,
               is_new_motor, parts_changed, weight
        FROM boats WHERE race_id = ANY(%s) ORDER BY race_id, boat_number
    """, (race_ids,))
    boats = pd.DataFrame(cur.fetchall())
    conn.close()
    return races, boats


def build_features(races, boats):
    races = races.copy()
    boats_p = boats.pivot(index='race_id', columns='boat_number',
                          values=['player_class', 'exhibition_time',
                                  'avg_st', 'motor_win_rate_2',
                                  'approach_course', 'parts_changed',
                                  'is_new_motor', 'weight', 'local_win_rate_2'])
    boats_p.columns = [f'{a}_b{b}' for a, b in boats_p.columns]
    races = races.merge(boats_p, left_on='race_id', right_index=True, how='left')
    exh_cols = [f'exhibition_time_b{i}' for i in range(1, 7)]
    races['exh_rank_b1'] = races[exh_cols].rank(axis=1, method='min').iloc[:, 0]
    races['exh_rank_b4'] = races[exh_cols].rank(axis=1, method='min').iloc[:, 3]
    races['exh_b1_minus_mean_others'] = races['exhibition_time_b1'] - races[
        [f'exhibition_time_b{i}' for i in range(2, 7)]].mean(axis=1)
    races['dow'] = pd.to_datetime(races['race_date']).dt.dayofweek
    races['month'] = pd.to_datetime(races['race_date']).dt.month
    # 着順フラグ (各艇 × 1/2/3 着)
    for boat in range(1, 7):
        races[f'win1_b{boat}'] = (races['result_1st'] == boat).astype(int)
        races[f'win2_b{boat}'] = (races['result_2nd'] == boat).astype(int)
        races[f'win3_b{boat}'] = (races['result_3rd'] == boat).astype(int)
    return races


HYPOTHESES = [
    ('H01', '1号艇 A1', lambda r: r['player_class_b1'] == 'A1'),
    ('H02', '1号艇 A級 + 残り全 B級',
     lambda r: (r['player_class_b1'].isin(['A1','A2'])) &
        r['player_class_b2'].str.startswith('B', na=False) &
        r['player_class_b3'].str.startswith('B', na=False) &
        r['player_class_b4'].str.startswith('B', na=False) &
        r['player_class_b5'].str.startswith('B', na=False) &
        r['player_class_b6'].str.startswith('B', na=False)),
    ('H03', '6号艇 A級', lambda r: r['player_class_b6'].isin(['A1','A2'])),
    ('H04', '1号艇 motor_win_rate_2 > 0.40', lambda r: r['motor_win_rate_2_b1'] > 0.40),
    ('H05', '1号艇 motor_win_rate_2 < 0.25', lambda r: r['motor_win_rate_2_b1'] < 0.25),
    ('H06', '1号艇 parts_changed', lambda r: r['parts_changed_b1'] == True),
    ('H07', '1号艇 is_new_motor', lambda r: r['is_new_motor_b1'] == True),
    ('H08', '1号艇 展示 1番時計', lambda r: r['exh_rank_b1'] == 1),
    ('H09', '1号艇 展示 vs 他平均 +0.10秒以上遅', lambda r: r['exh_b1_minus_mean_others'] > 0.10),
    ('H10', '4号艇 展示 1番時計', lambda r: r['exh_rank_b4'] == 1),
    ('H11', '1号艇 avg_st < 0.13', lambda r: r['avg_st_b1'] < 0.13),
    ('H12', '1号艇 approach_course = 1', lambda r: r['approach_course_b1'] == 1),
    ('H13', '4号艇 前づけ 1-3 入り',
     lambda r: r['approach_course_b4'].notna() & (r['approach_course_b4'] < 4)),
    ('H14', '風速 5m+', lambda r: r['wind_speed'] >= 5),
    ('H15', '風速 < 1', lambda r: r['wind_speed'] < 1),
    ('H16', '波高 > 4', lambda r: r['wave_height'] > 4),
    ('H17', '波高 = 0', lambda r: r['wave_height'] == 0),
    ('H18', 'subtitle 優勝戦', lambda r: r['race_subtitle'].str.contains('優勝戦', na=False)),
    ('H19', 'subtitle 準優', lambda r: r['race_subtitle'].str.contains('準優', na=False)),
    ('H20', '初日', lambda r: r['day_label'] == '初日'),
    ('H21', '最終日/優勝戦', lambda r: r['day_label'].isin(['最終日','優勝戦'])),
    ('H22', '土日', lambda r: r['dow'].isin([5, 6])),
    ('H23', '夏季 (5-8月)', lambda r: r['month'].isin([5, 6, 7, 8])),
    ('H24', '戸田 V2', lambda r: r['venue_id'] == 2),
    ('H25', '桐生 V1', lambda r: r['venue_id'] == 1),
    ('H26', '浜名湖 V6', lambda r: r['venue_id'] == 6),
    ('H27', '1号艇 weight > 55', lambda r: r['weight_b1'] > 55),
    ('H28', '1号艇 weight < 50', lambda r: r['weight_b1'] < 50),
    ('H29', '1号艇 local_win_rate_2 >= 0.35', lambda r: r['local_win_rate_2_b1'] >= 0.35),
    ('H30', '1号艇 local_win_rate_2 < 0.20', lambda r: r['local_win_rate_2_b1'] < 0.20),
]


def fdr_bh(pvals, alpha=0.10):
    p = np.array(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n+1) / n) * alpha
    below = ranked <= thresholds
    max_k = -1
    for i in range(n):
        if below[i]:
            max_k = i
    sig = np.zeros(n, dtype=bool)
    if max_k >= 0:
        sig[order[:max_k+1]] = True
    return sig


def main():
    logger.info(f"30 仮説 6 艇×3着 拡張 {PERIOD_START}〜{PERIOD_END}")
    races, boats = fetch_data()
    logger.info(f"races: {len(races)}, boats: {len(boats)}")
    races = build_features(races, boats)

    base_rates = {}
    for boat in range(1, 7):
        for pos in [1, 2, 3]:
            col = f'win{pos}_b{boat}'
            base_rates[(boat, pos)] = races[col].mean()

    # 各仮説 × 18 metric
    all_results = []
    for h_id, h_desc, filter_fn in HYPOTHESES:
        mask = filter_fn(races).fillna(False).astype(bool)
        n_cond = int(mask.sum())
        if n_cond < 30:
            continue  # 標本不足は skip
        n_base = len(races) - n_cond
        for boat in range(1, 7):
            for pos in [1, 2, 3]:
                col = f'win{pos}_b{boat}'
                p_cond = races.loc[mask, col].mean()
                p_base = races.loc[~mask, col].mean() if n_base > 0 else 0
                hits_cond = int(races.loc[mask, col].sum())
                hits_base = int(races.loc[~mask, col].sum())
                # Fisher's exact
                a, b = hits_cond, n_cond - hits_cond
                c, d = hits_base, n_base - hits_base
                try:
                    _, pval = fisher_exact([[a, b], [c, d]])
                except Exception:
                    pval = 1.0
                lift = (p_cond - p_base) * 100
                all_results.append({
                    'h_id': h_id, 'desc': h_desc,
                    'boat': boat, 'pos': pos,
                    'n_cond': n_cond,
                    'p_cond': p_cond * 100, 'p_base': p_base * 100,
                    'lift': lift, 'pval': pval,
                    'metric': f'B{boat} {pos}着率',
                })

    df = pd.DataFrame(all_results)
    # FDR
    sig_mask = fdr_bh(df['pval'].values, alpha=0.10)
    df['sig'] = sig_mask
    df['abs_lift'] = df['lift'].abs()

    lines = []
    lines.append("# 30 仮説 × 6 艇 × 3 着 拡張 screening\n\n")
    lines.append(f"対象: {PERIOD_START}〜{PERIOD_END} (races {len(races)})\n")
    lines.append("評価軸: 各 (仮説 × 艇 × 着位置) で条件付き確率 vs ベースライン\n")
    lines.append("Fisher's exact + FDR (BH, α=0.10) 補正、標本数 < 30 は除外\n\n")

    # 最重要セクション: 1着率では出なかったが 2着 or 3着で出た signal
    lines.append("## 🌟 新発見: 1着率では出なかったが 2/3着率で signal\n\n")
    lines.append("各仮説で 1着率 lift が |5pt| 未満だったのに、2着率 or 3着率で\n")
    lines.append("|5pt| 以上の lift + FDR 有意なものを抽出。\n\n")

    h_with_1st_weak = set()
    for h_id, _, _ in HYPOTHESES:
        h_rows = df[df['h_id'] == h_id]
        if h_rows.empty:
            continue
        win1_rows = h_rows[h_rows['pos'] == 1]
        if win1_rows.empty:
            continue
        if (win1_rows['abs_lift'] >= 5).any():
            continue  # 1 着率で既に signal あり
        h_with_1st_weak.add(h_id)
    new_findings = df[
        df['h_id'].isin(h_with_1st_weak) &
        (df['pos'].isin([2, 3])) &
        (df['abs_lift'] >= 5) &
        df['sig']
    ].sort_values('abs_lift', ascending=False)

    if new_findings.empty:
        lines.append("該当なし (1着率で見落としていた signal は無し)\n\n")
    else:
        lines.append(f"**{len(new_findings)} 件発見**:\n\n")
        lines.append("| h_id | 仮説 | metric | n | p_cond | p_base | lift |\n|---|---|---|---|---|---|---|\n")
        for _, r in new_findings.iterrows():
            lines.append(f"| {r['h_id']} | {r['desc']} | {r['metric']} | {r['n_cond']} | "
                         f"{r['p_cond']:.1f}% | {r['p_base']:.1f}% | **{r['lift']:+.1f}** |\n")

    # 全結果: 仮説別に lift |≥5| かつ FDR 有意な metric を表示
    lines.append("\n## 仮説別 強い signal (|lift| ≥ 5pt + FDR ✅)\n\n")
    for h_id, h_desc, _ in HYPOTHESES:
        h_rows = df[(df['h_id'] == h_id) & (df['abs_lift'] >= 5) & df['sig']]
        if h_rows.empty:
            continue
        h_rows = h_rows.sort_values('abs_lift', ascending=False)
        lines.append(f"\n### {h_id} {h_desc}\n\n")
        lines.append("| boat | 着 | n | p_cond | p_base | lift | pval |\n|---|---|---|---|---|---|---|\n")
        for _, r in h_rows.iterrows():
            lines.append(f"| B{r['boat']} | {r['pos']}着 | {r['n_cond']} | "
                         f"{r['p_cond']:.1f}% | {r['p_base']:.1f}% | "
                         f"**{r['lift']:+.1f}** | {r['pval']:.4f} |\n")

    # ベースライン (参考)
    lines.append("\n## 全体ベースライン (各艇 × 1/2/3着)\n\n")
    lines.append("| boat | 1着率 | 2着率 | 3着率 |\n|---|---|---|---|\n")
    for boat in range(1, 7):
        lines.append(f"| B{boat} | {base_rates[(boat,1)]*100:.1f}% | "
                     f"{base_rates[(boat,2)]*100:.1f}% | {base_rates[(boat,3)]*100:.1f}% |\n")

    lines.append("\n## 重要な留意 (CLAUDE.md 批判プロトコル準拠)\n\n")
    lines.append("- これは **データ集計のみ**。QMC 予測との突合は別途 `49_qmc_vs_empirical.py` で実施\n")
    lines.append("- 強い signal も「QMC の `compute_ratings_early` で既処理 signal」と重複していないか要確認\n")
    lines.append("- 採用判断は岩下さんが下す。本レポートは論点提供まで\n")
    lines.append("- 中穴狙い戦略では「2/3 着 signal を取りに行く代替買い目」を検討すべき (skip より)\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")
    logger.info(f"新発見 (1着率 弱 × 2/3着率 強 signal): {len(new_findings)} 件")


if __name__ == '__main__':
    main()
