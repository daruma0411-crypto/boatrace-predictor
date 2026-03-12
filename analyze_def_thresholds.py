"""D/E/F戦略フィルター閾値分析スクリプト

本日の全レースに対してモデル予測を実行し、
エントロピー分布・アンサンブル一致率・乖離度を計測する。
"""
import sys
import os
import math
import time
import logging
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scraper import _get_session, scrape_racelist, scrape_odds_3t, scrape_beforeinfo
from src.predictor import RealtimePredictor, EnsemblePredictor
from src.betting import _calculate_entropy, _check_ensemble_agreement

logging.basicConfig(level=logging.WARNING)

session = _get_session()
predictor = RealtimePredictor('models/boatrace_model.pth')
ensemble = EnsemblePredictor()

today = date.today()

# 全場のR1を試行して開催場を特定
print("=== 開催場の特定 ===")
active_venues = []
for vid in range(1, 25):
    boats = scrape_racelist(session, today, vid, 1)
    if boats and len(boats) == 6:
        active_venues.append(vid)
    time.sleep(0.3)

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}
names = [str(vid) + "(" + VENUE_NAMES.get(vid, "?") + ")" for vid in active_venues]
print(f"開催場: {names}")

# 各場のR1-R12をサンプリング（全部やると重いのでR1,R6,R12）
sample_races = [1, 4, 6, 8, 10, 12]
entropies = []
ensemble_results = []  # (agreed, top_boats)
divergence_ratios = []

total = 0
for vid in active_venues:
    for rno in sample_races:
        boats = scrape_racelist(session, today, vid, rno)
        if not boats or len(boats) != 6:
            time.sleep(0.3)
            continue

        # 展示データ
        try:
            exh = scrape_beforeinfo(session, today, vid, rno)
        except Exception:
            exh = None

        # race_data構築
        race_data = {
            'venue_id': vid,
            'race_number': rno,
            'race_date': today,
        }
        if exh and 'weather' in exh:
            race_data.update(exh['weather'])

        boats_data = []
        for i, b in enumerate(boats):
            bd = {
                'boat_number': i + 1,
                'racer_name': b.get('name', ''),
                'racer_class': b.get('class', 'B2'),
                'win_rate': b.get('win_rate', 0),
                'two_quinella_rate': b.get('two_quinella_rate', 0),
                'avg_start_timing': b.get('avg_st', 0.15),
                'weight': b.get('weight', 52),
                'local_win_rate': b.get('local_win_rate', 0),
                'local_two_quinella_rate': b.get('local_two_rate', 0),
                'motor_two_quinella_rate': b.get('motor_2rate', 0),
                'boat_two_quinella_rate': b.get('boat_2rate', 0),
            }
            if exh and 'boats' in exh:
                for eb in exh['boats']:
                    if eb.get('boat_number') == i + 1:
                        bd['exhibition_time'] = eb.get('exhibition_time', 0)
                        bd['tilt'] = eb.get('tilt', 0)
                        break
            boats_data.append(bd)

        # シングルモデル予測
        try:
            pred = predictor.predict(race_data, boats_data)
        except Exception as e:
            print(f"  予測エラー {vid}-R{rno}: {e}")
            time.sleep(0.3)
            continue

        probs_1st = pred['probs_1st']
        entropy = _calculate_entropy(probs_1st)
        entropies.append(entropy)

        top_boat = max(range(6), key=lambda i: probs_1st[i]) + 1
        top_prob = max(probs_1st)

        # アンサンブル予測
        try:
            ens_preds = ensemble.predict_all(race_data, boats_data)
            agreed, agreed_boat = _check_ensemble_agreement(ens_preds)
            ensemble_results.append({
                'agreed': agreed,
                'top_boats': [max(range(6), key=lambda i: p['probs_1st'][i]) + 1 for p in ens_preds],
                'venue': vid, 'race': rno,
            })
        except Exception as e:
            ensemble_results.append({'agreed': False, 'top_boats': [], 'venue': vid, 'race': rno})

        # オッズ取得 → 乖離度
        try:
            odds = scrape_odds_3t(session, today, vid, rno)
            if odds:
                from itertools import permutations
                for perm in permutations(range(6), 3):
                    a, b, c = perm
                    combo = f"{a+1}-{b+1}-{c+1}"
                    model_p = probs_1st[a] * probs_1st[b] * probs_1st[c]  # 簡易推定
                    raw_odds = odds.get(combo, 0)
                    if raw_odds > 1.0:
                        market_p = 1.0 / raw_odds
                        if market_p > 0:
                            div = model_p / market_p
                            if div > 1.0:
                                divergence_ratios.append(div)
        except Exception:
            pass

        total += 1
        print(f"  場{vid:2d} R{rno:2d}: H={entropy:.3f} top={top_boat}号艇({top_prob:.1%}) ens={'一致' if ensemble_results[-1]['agreed'] else '不一致'}({ensemble_results[-1]['top_boats']})")
        time.sleep(0.5)

print(f"\n=== 分析結果 ({total}レース) ===\n")

# エントロピー分布
if entropies:
    entropies.sort()
    print("【エントロピー分布 (D/F用)】")
    print(f"  最小: {min(entropies):.3f}")
    print(f"  最大: {max(entropies):.3f}")
    print(f"  中央値: {entropies[len(entropies)//2]:.3f}")
    print(f"  平均: {sum(entropies)/len(entropies):.3f}")
    print()
    for threshold in [2.0, 2.1, 2.2, 2.3, 2.4, 2.45, 2.5]:
        passed = sum(1 for h in entropies if h < threshold)
        pct = passed / len(entropies) * 100
        print(f"  H < {threshold}: {passed}/{len(entropies)} ({pct:.0f}%)")

print()

# アンサンブル一致率
if ensemble_results:
    agreed_count = sum(1 for r in ensemble_results if r['agreed'])
    print("【アンサンブル一致率 (E用)】")
    print(f"  全一致: {agreed_count}/{len(ensemble_results)} ({agreed_count/len(ensemble_results)*100:.0f}%)")

    # 3/4一致、2/4一致も計測
    from collections import Counter
    majority_3 = 0
    majority_2 = 0
    for r in ensemble_results:
        if r['top_boats']:
            c = Counter(r['top_boats'])
            most_common_count = c.most_common(1)[0][1]
            if most_common_count >= 3:
                majority_3 += 1
            if most_common_count >= 2:
                majority_2 += 1
    print(f"  3/4一致: {majority_3}/{len(ensemble_results)} ({majority_3/len(ensemble_results)*100:.0f}%)")
    print(f"  2/4一致: {majority_2}/{len(ensemble_results)} ({majority_2/len(ensemble_results)*100:.0f}%)")

print()

# 乖離度分布
if divergence_ratios:
    divergence_ratios.sort()
    print("【市場乖離度分布 (C/F用)】")
    print(f"  サンプル数: {len(divergence_ratios)}")
    print(f"  中央値: {divergence_ratios[len(divergence_ratios)//2]:.2f}")
    for threshold in [1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
        passed = sum(1 for d in divergence_ratios if d >= threshold)
        pct = passed / len(divergence_ratios) * 100
        print(f"  div >= {threshold}: {passed}/{len(divergence_ratios)} ({pct:.1f}%)")

print("\n=== 推奨閾値 ===")
if entropies:
    # 20-30%のレースが通るラインを推奨
    for target_pct in [20, 30, 50]:
        idx = int(len(entropies) * target_pct / 100)
        if idx < len(entropies):
            print(f"  D: エントロピー閾値 {entropies[idx]:.2f} → 約{target_pct}%のレースが通過")
