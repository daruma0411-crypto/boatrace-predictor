"""30 仮説 data driven screening (mc3 全戦略、FDR 補正、hold-out 確認)

入力: bets (mc3 系) + races + boats + race_titles
評価軸:
  - 1号艇 1着率 (条件下 vs ベースライン)
  - mc3 strategies 合算 ROI (該当条件下 vs 全体)

統計:
  - Fisher's exact (proportion test)
  - Benjamini-Hochberg FDR (α=0.10)
  - Hold-out: 2026-04-11〜05-04 (train) → 2026-05-05〜05-15 (test)

出力: analysis/reports/hypothesis_screening.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / 'hypothesis_screening.md'

PERIOD_START = date(2026, 4, 11)
TRAIN_END = date(2026, 5, 4)
TEST_START = date(2026, 5, 5)
PERIOD_END = date(2026, 5, 15)

MC3_STRATEGIES = ['mc3_early_race', 'mc3_venue_focus', 'mc3_venue_focus_r2', 'mc3_venue_focus_r4']

VENUE_NAMES = {1:'桐生',2:'戸田',3:'江戸川',4:'平和島',5:'多摩川',6:'浜名湖',
               7:'蒲郡',8:'常滑',9:'津',10:'三国',11:'びわこ',12:'住之江',
               13:'尼崎',14:'鳴門',15:'丸亀',16:'児島',17:'宮島',18:'徳山',
               19:'下関',20:'若松',21:'芦屋',22:'福岡',23:'唐津',24:'大村'}


def fetch_races_with_boats():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
               r.result_1st, r.wind_speed, r.wind_direction, r.wave_height,
               r.temperature, r.water_temperature, r.payout_sanrentan,
               rt.title AS race_title, rt.subtitle AS race_subtitle, rt.day_label
        FROM races r
        LEFT JOIN race_titles rt ON r.id = rt.race_id
        WHERE r.is_finished = true
          AND r.result_1st IS NOT NULL
          AND r.race_date BETWEEN %s AND %s
        ORDER BY r.race_date, r.id
    """, (PERIOD_START, PERIOD_END))
    races = pd.DataFrame(cur.fetchall())
    if races.empty:
        return None, None, None
    race_ids = races['race_id'].tolist()
    cur.execute("""
        SELECT race_id, boat_number, player_class, win_rate, win_rate_2,
               local_win_rate, local_win_rate_2, avg_st,
               motor_win_rate_2, boat_win_rate_2, weight, exhibition_time,
               approach_course, is_new_motor, tilt, parts_changed
        FROM boats WHERE race_id = ANY(%s) ORDER BY race_id, boat_number
    """, (race_ids,))
    boats = pd.DataFrame(cur.fetchall())
    cur.execute("""
        SELECT b.race_id, b.amount::float AS stake, b.return_amount::float AS payout,
               b.is_hit, b.strategy_type, r.race_date
        FROM bets b JOIN races r ON b.race_id = r.id
        WHERE b.strategy_type = ANY(%s) AND r.race_date BETWEEN %s AND %s
    """, (MC3_STRATEGIES, PERIOD_START, PERIOD_END))
    bets = pd.DataFrame(cur.fetchall())
    conn.close()
    return races, boats, bets


def build_race_features(races, boats):
    """各レースに boats から導出した特徴量を追加"""
    races = races.copy()
    boats_p = boats.pivot(index='race_id', columns='boat_number',
                          values=['player_class', 'win_rate_2', 'local_win_rate_2',
                                  'avg_st', 'motor_win_rate_2', 'exhibition_time',
                                  'approach_course', 'is_new_motor', 'parts_changed',
                                  'weight'])
    # flatten columns
    boats_p.columns = [f'{a}_b{b}' for a, b in boats_p.columns]
    races = races.merge(boats_p, left_on='race_id', right_index=True, how='left')
    # 展示タイム ranking (1=fastest)
    exh_cols = [f'exhibition_time_b{i}' for i in range(1, 7)]
    races['exh_rank_b1'] = races[exh_cols].rank(axis=1, method='min').iloc[:, 0]
    races['exh_rank_b4'] = races[exh_cols].rank(axis=1, method='min').iloc[:, 3]
    # 1号艇 vs 平均他艇 展示タイム差
    races['exh_b1_minus_mean_others'] = races['exhibition_time_b1'] - races[
        ['exhibition_time_b2','exhibition_time_b3','exhibition_time_b4',
         'exhibition_time_b5','exhibition_time_b6']].mean(axis=1)
    races['win1'] = (races['result_1st'] == 1).astype(int)
    races['win3'] = (races['result_1st'] == 3).astype(int)
    races['win4'] = (races['result_1st'] == 4).astype(int)
    races['dow'] = pd.to_datetime(races['race_date']).dt.dayofweek
    races['month'] = pd.to_datetime(races['race_date']).dt.month
    return races


HYPOTHESES = [
    # (id, description, lambda races -> mask, target_metric_col, baseline_label)
    ('H01', '1号艇 A1 vs A2 (全レース)', lambda r: r['player_class_b1'] == 'A1', 'win1', 'all'),
    ('H02', '1号艇 A級 + 残り全 B級', lambda r: (r['player_class_b1'].isin(['A1','A2'])) &
        (r['player_class_b2'].str.startswith('B', na=False)) & (r['player_class_b3'].str.startswith('B', na=False)) &
        (r['player_class_b4'].str.startswith('B', na=False)) & (r['player_class_b5'].str.startswith('B', na=False)) &
        (r['player_class_b6'].str.startswith('B', na=False)), 'win1', 'all'),
    ('H03', '6号艇 A級 (レア)', lambda r: r['player_class_b6'].isin(['A1','A2']), 'win1', 'all'),
    ('H04', '1号艇 motor_win_rate_2 > 0.40', lambda r: r['motor_win_rate_2_b1'] > 0.40, 'win1', 'all'),
    ('H05', '1号艇 motor_win_rate_2 < 0.25', lambda r: r['motor_win_rate_2_b1'] < 0.25, 'win1', 'all'),
    ('H06', '1号艇 parts_changed=true', lambda r: r['parts_changed_b1'] == True, 'win1', 'all'),
    ('H07', '1号艇 is_new_motor=true', lambda r: r['is_new_motor_b1'] == True, 'win1', 'all'),
    ('H08', '1号艇 展示タイム 1番時計', lambda r: r['exh_rank_b1'] == 1, 'win1', 'all'),
    ('H09', '1号艇 展示タイム vs 他平均 +0.10 秒以上遅', lambda r: r['exh_b1_minus_mean_others'] > 0.10, 'win1', 'all'),
    ('H10', '4号艇 展示タイム 1番時計', lambda r: r['exh_rank_b4'] == 1, 'win4', 'all_b4'),
    ('H11', '1号艇 avg_st < 0.13', lambda r: r['avg_st_b1'] < 0.13, 'win1', 'all'),
    ('H12', '1号艇 approach_course = 1 (前づけなし)', lambda r: r['approach_course_b1'] == 1, 'win1', 'all'),
    ('H13', '4号艇 が前づけして 1-3 になった', lambda r: r['approach_course_b4'].notna() & (r['approach_course_b4'] < 4), 'win1', 'all'),
    ('H14', '風速 5m+', lambda r: r['wind_speed'] >= 5, 'win1', 'all'),
    ('H15', '風速 < 1 (無風)', lambda r: r['wind_speed'] < 1, 'win1', 'all'),
    ('H16', '波高 > 4cm', lambda r: r['wave_height'] > 4, 'win1', 'all'),
    ('H17', '波高 = 0', lambda r: r['wave_height'] == 0, 'win1', 'all'),
    ('H18', 'race_title「優勝戦」含む', lambda r: r['race_subtitle'].str.contains('優勝戦', na=False), 'win1', 'all'),
    ('H19', 'race_title「準優」含む', lambda r: r['race_subtitle'].str.contains('準優', na=False), 'win1', 'all'),
    ('H20', 'day_label「初日」', lambda r: r['day_label'] == '初日', 'win1', 'all'),
    ('H21', 'day_label「最終日」or「優勝戦」', lambda r: r['day_label'].isin(['最終日','優勝戦']), 'win1', 'all'),
    ('H22', '土日開催', lambda r: r['dow'].isin([5, 6]), 'win1', 'all'),
    ('H23', '夏季 (5-8月)', lambda r: r['month'].isin([5, 6, 7, 8]), 'win1', 'all'),
    ('H24', '戸田 (V2) 3号艇 1着', lambda r: r['venue_id'] == 2, 'win3', 'all_b3'),
    ('H25', '桐生 (V1) 4号艇 1着', lambda r: r['venue_id'] == 1, 'win4', 'all_b4'),
    ('H26', '浜名湖 (V6) 全体', lambda r: r['venue_id'] == 6, 'win1', 'all'),
    ('H27', '1号艇 weight > 55kg', lambda r: r['weight_b1'] > 55, 'win1', 'all'),
    ('H28', '1号艇 weight < 50kg', lambda r: r['weight_b1'] < 50, 'win1', 'all'),
    ('H29', '1号艇 local_win_rate_2 高 (>= 0.35)', lambda r: r['local_win_rate_2_b1'] >= 0.35, 'win1', 'all'),
    ('H30', '1号艇 local_win_rate_2 低 (< 0.20)', lambda r: r['local_win_rate_2_b1'] < 0.20, 'win1', 'all'),
]


def baseline_rate(races, baseline_label):
    if baseline_label == 'all':
        return races['win1'].sum(), len(races)
    if baseline_label == 'all_b3':
        return races['win3'].sum(), len(races)
    if baseline_label == 'all_b4':
        return races['win4'].sum(), len(races)
    return 0, 0


def fdr_bh(pvals, alpha=0.10):
    """Benjamini-Hochberg FDR. Returns boolean array of significant."""
    p = np.array(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n+1) / n) * alpha
    below = ranked <= thresholds
    # 最大 k で below=True
    max_k = -1
    for i in range(n):
        if below[i]:
            max_k = i
    sig = np.zeros(n, dtype=bool)
    if max_k >= 0:
        sig_order = order[:max_k+1]
        sig[sig_order] = True
    return sig


def evaluate_hypothesis(races, h_id, desc, filter_fn, metric_col, baseline_label):
    mask = filter_fn(races).fillna(False).astype(bool)
    n_cond = int(mask.sum())
    if n_cond == 0:
        return None
    n_hit_cond = int(races.loc[mask, metric_col].sum())
    n_hit_base, n_base = baseline_rate(races, baseline_label)
    if n_base == 0:
        return None
    p_cond = n_hit_cond / n_cond
    p_base = n_hit_base / n_base
    # Fisher's exact (条件あり vs 条件なし)
    a = n_hit_cond  # 条件あり、当
    b = n_cond - n_hit_cond  # 条件あり、外
    c = n_hit_base - n_hit_cond  # 条件なし、当
    d = (n_base - n_cond) - c  # 条件なし、外
    try:
        _, pval = fisher_exact([[a, b], [c, d]])
    except Exception:
        pval = 1.0
    return {
        'id': h_id, 'desc': desc, 'n_cond': n_cond,
        'p_cond': p_cond * 100, 'p_base': p_base * 100,
        'lift_pt': (p_cond - p_base) * 100, 'pval': pval,
    }


def split_hold_out(races):
    train = races[pd.to_datetime(races['race_date']).dt.date <= TRAIN_END]
    test = races[pd.to_datetime(races['race_date']).dt.date >= TEST_START]
    return train, test


def main():
    logger.info(f"30 仮説スクリーニング {PERIOD_START} 〜 {PERIOD_END}")
    races, boats, bets = fetch_races_with_boats()
    if races is None:
        raise SystemExit("data なし")
    logger.info(f"races: {len(races)}, boats: {len(boats)}, mc3 bets: {len(bets)}")
    races = build_race_features(races, boats)
    train, test = split_hold_out(races)
    logger.info(f"train (〜{TRAIN_END}): {len(train)}, test ({TEST_START}〜): {len(test)}")

    train_results = []
    for h in HYPOTHESES:
        r = evaluate_hypothesis(train, *h)
        if r:
            train_results.append(r)
    pvals = [r['pval'] for r in train_results]
    sig_mask = fdr_bh(pvals, alpha=0.10)
    for i, r in enumerate(train_results):
        r['sig_train'] = bool(sig_mask[i])

    # hold-out: 同じ条件で test に適用
    test_results = {}
    for h in HYPOTHESES:
        r = evaluate_hypothesis(test, *h)
        if r:
            test_results[r['id']] = r

    lines = []
    lines.append("# 30 仮説スクリーニング (mc3 全戦略、FDR α=0.10、hold-out)\n\n")
    lines.append(f"対象: {PERIOD_START} 〜 {PERIOD_END} (races {len(races)})\n")
    lines.append(f"Train: 〜{TRAIN_END} ({len(train)} races) / Test: {TEST_START}〜 ({len(test)} races)\n\n")

    lines.append("## 結果 (lift_pt 降順)\n\n")
    lines.append("| ID | 仮説 | n_train | 条件P% | base% | **lift pt** | pval | **FDR** | n_test | test lift |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(train_results, key=lambda x: -x['lift_pt']):
        sig_mark = '✅' if r['sig_train'] else ''
        tres = test_results.get(r['id'])
        t_n = tres['n_cond'] if tres else 0
        t_lift = f"{tres['lift_pt']:+.1f}" if tres else '-'
        t_confirm = ''
        if tres and r['lift_pt'] > 0 and tres['lift_pt'] > 0:
            t_confirm = '🟢'
        elif tres and r['lift_pt'] > 0 and tres['lift_pt'] <= 0:
            t_confirm = '🔴'
        lines.append(f"| {r['id']} | {r['desc']} | {r['n_cond']} | {r['p_cond']:.1f}% | "
                     f"{r['p_base']:.1f}% | **{r['lift_pt']:+.1f}** | {r['pval']:.4f} | "
                     f"{sig_mark} | {t_n} | {t_lift}{t_confirm} |\n")

    # 採用候補 (FDR有意 + hold-out で同方向 + lift>=5pt)
    lines.append("\n## 採用候補 (FDR ✅ + hold-out 🟢 + |lift|>=5pt)\n\n")
    candidates = []
    for r in train_results:
        if not r['sig_train']:
            continue
        if abs(r['lift_pt']) < 5:
            continue
        tres = test_results.get(r['id'])
        if not tres:
            continue
        if (r['lift_pt'] > 0 and tres['lift_pt'] <= 0) or (r['lift_pt'] < 0 and tres['lift_pt'] >= 0):
            continue
        candidates.append((r, tres))
    if not candidates:
        lines.append("条件を満たす仮説なし\n")
    else:
        lines.append("| ID | 仮説 | train lift | test lift | 採用方針 |\n|---|---|---|---|---|\n")
        for r, t in sorted(candidates, key=lambda x: -abs(x[0]['lift_pt'])):
            direction = '取る (1着率高)' if r['lift_pt'] > 0 else '避ける (1着率低)'
            lines.append(f"| {r['id']} | {r['desc']} | {r['lift_pt']:+.1f}pt | {t['lift_pt']:+.1f}pt | {direction} |\n")

    # mc3 戦略の ROI に対する影響 (該当条件の bets ROI vs 全体)
    lines.append("\n## mc3 戦略 ROI 補足 (採用候補の条件下 ROI)\n\n")
    if not candidates:
        lines.append("候補なし、スキップ\n")
    else:
        # bets と races 結合
        race_with_bets = races.merge(bets[['race_id', 'stake', 'payout', 'is_hit']],
                                      on='race_id', how='left')
        all_bets_total_stake = bets['stake'].sum() if len(bets) else 0
        all_bets_total_payout = bets['payout'].sum() if len(bets) else 0
        baseline_roi = all_bets_total_payout / all_bets_total_stake * 100 if all_bets_total_stake else 0
        lines.append(f"全 mc3 bets baseline ROI: **{baseline_roi:.1f}%** ({len(bets)} bets)\n\n")
        lines.append("| ID | 仮説 | 該当 bets | ROI | vs baseline |\n|---|---|---|---|---|\n")
        for r, _ in candidates:
            h = next(h for h in HYPOTHESES if h[0] == r['id'])
            mask = h[2](races).fillna(False).astype(bool)
            target_ids = races.loc[mask, 'race_id'].tolist()
            sub_bets = bets[bets['race_id'].isin(target_ids)]
            if len(sub_bets) == 0:
                lines.append(f"| {r['id']} | {r['desc']} | 0 | - | - |\n")
                continue
            sub_stake = sub_bets['stake'].sum()
            sub_payout = sub_bets['payout'].sum()
            sub_roi = sub_payout / sub_stake * 100 if sub_stake else 0
            diff = sub_roi - baseline_roi
            lines.append(f"| {r['id']} | {r['desc']} | {len(sub_bets)} | {sub_roi:.1f}% | {diff:+.1f}pt |\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
