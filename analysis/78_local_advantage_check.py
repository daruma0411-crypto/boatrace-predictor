"""地元 (local_win_rate_2) signal の未取り込み edge 検出 (Path D)

仮説: 「競艇は地元が強い」は market 周知だが、V10/specialist が完全に学習しきれず
未取り込みの edge が残っているか確認。

検証手順:
  各 venue test (2026-05) で:
    - 1号艇 local_win_rate_2 を 5 分位 (Q1-Q5) に分割
    - 各分位での actual 1号艇 1着率 を集計
    - V10 / specialist の予測 1号艇 1着率 と比較
    - bias (predicted - actual) が分位別で大きく異なるか確認
  + 同様に各艇 (B2-B6) の local 別 hit パターン

判定:
  顕著な未取り込み edge (>5pt bias 偏り) があれば Path A (feature 拡張) に進む価値
  なければ shadow 並走へ

出力: analysis/reports/78_local_advantage.md
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import date
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import lightgbm as lgb

from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
VENUE_PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
SPECIALISTS_DIR = ROOT / 'models' / 'specialists'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '78_local_advantage.md'

TEST_START = date(2026, 5, 1)

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

# 13 functional venues
FUNCTIONAL = [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24]


def load_specialists():
    return {vid: lgb.Booster(model_file=str(SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt'))
            for vid in VENUE_NAMES if (SPECIALISTS_DIR / f'lightgbm_v{vid:02d}_1st.txt').exists()}


def main():
    logger.info("地元 signal 未取り込み edge 検出 (Path D)")
    venue_preds = pickle.load(open(VENUE_PRED_PATH, 'rb'))
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    specialists = load_specialists()

    # 各 venue test races を抽出 + features 生成
    venue_test_data = {}
    for tvid, preds in venue_preds.items():
        if tvid not in FUNCTIONAL:
            continue
        records = []
        for rid, p in preds.items():
            d = date.fromisoformat(p['race_date'])
            if d < TEST_START:
                continue
            try:
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                # local_win_rate_2 を 6 艇分抽出
                local_rates = []
                for b in p['boats']:
                    lr = b.get('local_win_rate_2', 0) or 0
                    local_rates.append(float(lr))
                records.append({
                    'rid': rid, 'features': features,
                    'v10_probs': np.array(p['probs_1st']),
                    'local_rates': local_rates,
                    'result_1st': p['result_1st'],
                })
            except Exception:
                continue
        venue_test_data[tvid] = records

    # 分析 1: 1号艇 local_win_rate_2 分位別の hit pattern
    lines = []
    lines.append("# 地元 (local_win_rate_2) signal 未取り込み edge 検出 (Path D)\n\n")
    lines.append("仮説: 「競艇は地元が強い」は V10/specialist の `local_win_rate_2` 特徴量で\n")
    lines.append("既に学習されている。だが完全には捕捉していない可能性。\n")
    lines.append("各 venue test (2026-05) で 1号艇 `local_win_rate_2` 5 分位別に hit / 予測 を集計。\n\n")

    lines.append("## 1. 1号艇 local_win_rate_2 分位別 (13 venues 集計)\n\n")
    all_records = []
    for tvid, recs in venue_test_data.items():
        for r in recs:
            all_records.append({
                'venue': tvid,
                'b1_local': r['local_rates'][0],
                'v10_b1_pred': r['v10_probs'][0],
                'actual_b1_win': 1 if r['result_1st'] == 1 else 0,
            })

    if all_records:
        b1_locals = np.array([r['b1_local'] for r in all_records])
        # 5 分位 (quintile) thresholds
        q_thresholds = np.percentile(b1_locals, [20, 40, 60, 80])
        # Bin
        def get_bin(lr):
            for i, th in enumerate(q_thresholds):
                if lr <= th:
                    return f"Q{i+1} (≤{th:.1f})"
            return f"Q5 (>{q_thresholds[-1]:.1f})"

        bin_stats = defaultdict(lambda: {'n': 0, 'actual_hits': 0, 'v10_sum': 0.0, 'local_sum': 0.0})
        for r in all_records:
            b = get_bin(r['b1_local'])
            bin_stats[b]['n'] += 1
            bin_stats[b]['actual_hits'] += r['actual_b1_win']
            bin_stats[b]['v10_sum'] += r['v10_b1_pred']
            bin_stats[b]['local_sum'] += r['b1_local']

        lines.append("| 分位 | n | 平均 local rate | actual 1号艇 1着率 | V10 予測 1号艇 1着率 | bias (V10 - actual) |\n|---|---|---|---|---|---|\n")
        for b in ['Q1 (≤{:.1f})'.format(q_thresholds[0]),
                  'Q2 (≤{:.1f})'.format(q_thresholds[1]),
                  'Q3 (≤{:.1f})'.format(q_thresholds[2]),
                  'Q4 (≤{:.1f})'.format(q_thresholds[3]),
                  'Q5 (>{:.1f})'.format(q_thresholds[-1])]:
            s = bin_stats.get(b)
            if not s or s['n'] == 0:
                continue
            actual_rate = s['actual_hits'] / s['n'] * 100
            v10_rate = s['v10_sum'] / s['n'] * 100
            local_mean = s['local_sum'] / s['n']
            bias = v10_rate - actual_rate
            lines.append(f"| {b} | {s['n']} | {local_mean:.2f} | {actual_rate:.2f}% | {v10_rate:.2f}% | {bias:+.2f}pt |\n")

        # 全体平均
        total_n = sum(s['n'] for s in bin_stats.values())
        total_actual = sum(s['actual_hits'] for s in bin_stats.values()) / total_n * 100 if total_n else 0
        total_v10 = sum(s['v10_sum'] for s in bin_stats.values()) / total_n * 100 if total_n else 0
        lines.append(f"\n全体: n={total_n}, actual 1号艇 1着率 {total_actual:.2f}%, V10 予測 {total_v10:.2f}%, bias {total_v10-total_actual:+.2f}pt\n")

    # 分析 2: venue 別
    lines.append("\n## 2. venue 別 1号艇 local 分位 hit (top/bottom)\n\n")
    lines.append("各 venue で test races の 1号艇 local_win_rate_2 を上下半分に分割し、\n")
    lines.append("V10 bias の違いを確認。\n\n")
    lines.append("| venue | name | n_test | mean local | top half (高 local) | bottom half (低 local) | bias 差 |\n|---|---|---|---|---|---|---|\n")
    for vid in FUNCTIONAL:
        recs = venue_test_data.get(vid, [])
        if len(recs) < 20:
            continue
        locals_arr = np.array([r['local_rates'][0] for r in recs])
        median = np.median(locals_arr)
        top = [r for r in recs if r['local_rates'][0] > median]
        bot = [r for r in recs if r['local_rates'][0] <= median]
        if not top or not bot:
            continue
        # 各 half の V10 bias
        def half_stats(arr):
            n = len(arr)
            actual_rate = sum(1 for r in arr if r['result_1st'] == 1) / n * 100
            v10_rate = sum(r['v10_probs'][0] for r in arr) / n * 100
            return v10_rate - actual_rate
        top_bias = half_stats(top)
        bot_bias = half_stats(bot)
        bias_diff = top_bias - bot_bias
        flag = '⚠️' if abs(bias_diff) > 10 else ''
        lines.append(f"| {vid} | {VENUE_NAMES[vid]} | {len(recs)} | {locals_arr.mean():.2f} | "
                     f"{top_bias:+.2f}pt | {bot_bias:+.2f}pt | **{bias_diff:+.2f}pt {flag}** |\n")

    # 分析 3: 全 6 艇の local advantage signal
    lines.append("\n## 3. 全 6 艇の local advantage signal\n\n")
    lines.append("各艇で local_win_rate_2 vs general win_rate_2 の差 = local advantage。\n")
    lines.append("当地で本来の実力以上に走る選手の signal。\n\n")

    # 各艇の local advantage = local_win_rate_2 - win_rate_2 を集計
    # 注: win_rate_2 は features に含まれるが、records に直接保持していないので boats から再抽出
    venue_local_adv = defaultdict(list)
    for tvid, preds in venue_preds.items():
        if tvid not in FUNCTIONAL:
            continue
        for rid, p in preds.items():
            d = date.fromisoformat(p['race_date'])
            if d < TEST_START:
                continue
            for i, b in enumerate(p['boats']):
                lr = b.get('local_win_rate_2', 0) or 0
                gr = b.get('win_rate_2', 0) or 0
                if lr > 0 and gr > 0:
                    venue_local_adv[(tvid, i+1)].append({
                        'local_adv': lr - gr,
                        'is_win': 1 if p['result_1st'] == i+1 else 0,
                    })

    lines.append("### 各艇の local advantage 高低別 1着率 (13 venues 集計)\n\n")
    lines.append("local advantage = local_win_rate_2 - win_rate_2 (当地ボーナス)\n\n")
    for boat in range(1, 7):
        # boat 別に全 venue 集計
        all_for_boat = []
        for (vid, bn), arr in venue_local_adv.items():
            if bn == boat:
                all_for_boat.extend(arr)
        if len(all_for_boat) < 50:
            continue
        adv_arr = np.array([r['local_adv'] for r in all_for_boat])
        wins_arr = np.array([r['is_win'] for r in all_for_boat])
        # 上下半分
        median = np.median(adv_arr)
        top_idx = adv_arr > median
        bot_idx = adv_arr <= median
        top_rate = wins_arr[top_idx].mean() * 100 if top_idx.sum() else 0
        bot_rate = wins_arr[bot_idx].mean() * 100 if bot_idx.sum() else 0
        diff = top_rate - bot_rate
        lines.append(f"- B{boat}: n={len(all_for_boat)}, 全体 1着率 {wins_arr.mean()*100:.2f}%, "
                     f"上位 local adv {top_rate:.2f}%, 下位 {bot_rate:.2f}%, 差 **{diff:+.2f}pt**\n")

    # 判定
    lines.append("\n## 判定: 未取り込み edge の有無\n\n")
    # 全体 bias diff (top vs bottom) の絶対値が大きい venue 数
    big_venues = []
    for vid in FUNCTIONAL:
        recs = venue_test_data.get(vid, [])
        if len(recs) < 20:
            continue
        locals_arr = np.array([r['local_rates'][0] for r in recs])
        median = np.median(locals_arr)
        top = [r for r in recs if r['local_rates'][0] > median]
        bot = [r for r in recs if r['local_rates'][0] <= median]
        if not top or not bot:
            continue
        def half_stats(arr):
            n = len(arr)
            actual = sum(1 for r in arr if r['result_1st'] == 1) / n * 100
            v10 = sum(r['v10_probs'][0] for r in arr) / n * 100
            return v10 - actual
        diff = half_stats(top) - half_stats(bot)
        if abs(diff) > 10:
            big_venues.append((vid, diff))

    if big_venues:
        lines.append(f"- ⚠️ **{len(big_venues)} venues で V10 bias が高 local / 低 local 帯で 10pt 以上差**\n")
        lines.append(f"  → 未取り込み edge の可能性、Path A (feature 拡張) 検討価値あり\n")
        for vid, diff in big_venues:
            lines.append(f"  - {VENUE_NAMES[vid]} (v{vid}): bias 差 {diff:+.2f}pt\n")
    else:
        lines.append(f"- 🟢 全 venue で V10 bias 差 < 10pt\n")
        lines.append(f"  → V10 が local signal を十分に取り込み、追加 edge 期待薄\n")
        lines.append(f"  → Shadow 並走に進む筋\n")

    lines.append("\n## 留意事項\n\n")
    lines.append("- local_win_rate_2 は V10 学習 features に含まれる (76dim の一部)\n")
    lines.append("- specialist LightGBM の feature importance で `B6_local_win_rate_2` のみ 14 位\n")
    lines.append("- 本分析は test n=70-170/venue で検出力限界、forward で再確認すべき\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
