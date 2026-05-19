"""24 会場 outcome patterns + 物理特性で venue 類似度集計

戸田類似 venue のランキングを data で算出。
67 (sequential venue addition) のための候補リストを作る。

特徴量:
  data-driven:
    - 1号艇 1着率
    - 6 艇 1着率分布 (6dim)
    - 風強時 (>=5m/s) と 弱時 (<2m/s) の 1着艇分布の差 (6dim)
    - クラス別 (A1/A2/B1) 1号艇 1着率 (3dim)
    - 平均 payout (1dim)
  physical (手作業):
    - 水質 (淡水/海水/汽水)
    - 戸田類似度 (主観初期値)

出力: analysis/reports/66_venue_clustering.md
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
REPORT_PATH = ROOT / 'analysis' / 'reports' / '66_venue_clustering.md'

TODA_VENUE_ID = 2

# 会場名と物理特性 (boatrace.jp + 一般知識ベース、岩下さん確認推奨)
VENUE_INFO = {
    1:  {'name': '桐生',   'water': '淡水', 'note': '群馬の谷風・夜風強い、狭水面寄り'},
    2:  {'name': '戸田',   'water': '淡水', 'note': '全国最狭水面、横風強、1マーク振り、硬い水面'},
    3:  {'name': '江戸川', 'water': '汽水', 'note': '河川・潮の影響、波荒い、難水面'},
    4:  {'name': '平和島', 'water': '海水', 'note': '東京湾、うねり、1号艇弱め'},
    5:  {'name': '多摩川', 'water': '淡水', 'note': '河川、静水面、1号艇強い'},
    6:  {'name': '浜名湖', 'water': '汽水', 'note': '広水面、潮の影響中'},
    7:  {'name': '蒲郡',   'water': '汽水', 'note': 'ナイター、静水寄り'},
    8:  {'name': '常滑',   'water': '海水', 'note': '伊勢湾、強風、波あり'},
    9:  {'name': '津',     'water': '汽水', 'note': '伊勢湾、潮あり'},
    10: {'name': '三国',   'water': '淡水', 'note': '北陸、強風寄り、1号艇弱め'},
    11: {'name': 'びわこ', 'water': '淡水', 'note': '湖、標高高、空気薄い'},
    12: {'name': '住之江', 'water': '淡水', 'note': 'ナイター、屋根あり風影響小'},
    13: {'name': '尼崎',   'water': '淡水', 'note': '静水寄り'},
    14: {'name': '鳴門',   'water': '海水', 'note': '潮、海風、波あり'},
    15: {'name': '丸亀',   'water': '海水', 'note': 'ナイター、瀬戸内、潮あり'},
    16: {'name': '児島',   'water': '海水', 'note': '瀬戸内、波高、難水面'},
    17: {'name': '宮島',   'water': '海水', 'note': '潮、強風、波あり'},
    18: {'name': '徳山',   'water': '海水', 'note': '瀬戸内、静水寄り'},
    19: {'name': '下関',   'water': '海水', 'note': 'ナイター、潮あり'},
    20: {'name': '若松',   'water': '海水', 'note': 'ナイター、内海'},
    21: {'name': '芦屋',   'water': '淡水', 'note': '女子戦多い、1号艇最強'},
    22: {'name': '福岡',   'water': '海水', 'note': '博多湾、強風'},
    23: {'name': '唐津',   'water': '淡水', 'note': '内海寄り、波小'},
    24: {'name': '大村',   'water': '海水', 'note': '海上、静水、1号艇最強'},
}


def fetch_venue_stats():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id AS race_id, r.venue_id, r.result_1st,
               r.wind_speed, r.payout_sanrentan,
               b.boat_number, b.player_class
        FROM races r
        JOIN boats b ON b.race_id = r.id AND b.boat_number = 1
        WHERE r.race_date >= '2025-06-01' AND r.result_1st IS NOT NULL
          AND r.payout_sanrentan IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    logger.info("24 venue クラスタリング")
    rows = fetch_venue_stats()
    logger.info(f"races: {len(rows)}")

    # venue 別集計
    venue_data = defaultdict(list)
    for r in rows:
        venue_data[r['venue_id']].append(r)

    venue_features = {}
    for vid, rs in venue_data.items():
        n = len(rs)
        if n < 100:
            continue
        # 1着艇 分布
        boat_1st_dist = np.zeros(6)
        for r in rs:
            boat_1st_dist[r['result_1st'] - 1] += 1
        boat_1st_dist /= n
        # 風強・弱 1着艇分布
        wind_strong = [r for r in rs if (r['wind_speed'] or 0) >= 5]
        wind_weak = [r for r in rs if (r['wind_speed'] or 0) < 2]
        ws_dist = np.zeros(6)
        ww_dist = np.zeros(6)
        for r in wind_strong:
            ws_dist[r['result_1st'] - 1] += 1
        for r in wind_weak:
            ww_dist[r['result_1st'] - 1] += 1
        ws_dist = ws_dist / len(wind_strong) if wind_strong else np.zeros(6)
        ww_dist = ww_dist / len(wind_weak) if wind_weak else np.zeros(6)
        wind_shift = ws_dist - ww_dist  # 風強で各艇 1着率がどう動くか
        # クラス別 (1号艇 が A1/A2/B1 の時の 1号艇 1着率)
        cls_b1 = {'A1': [], 'A2': [], 'B1': []}
        for r in rs:
            cls = r.get('player_class', 'B1')
            if cls in cls_b1:
                cls_b1[cls].append(1 if r['result_1st'] == 1 else 0)
        cls_rates = [np.mean(cls_b1[c]) if cls_b1[c] else 0.0 for c in ['A1', 'A2', 'B1']]
        # payout 平均
        payouts = [r['payout_sanrentan'] for r in rs if r['payout_sanrentan']]
        avg_payout = float(np.mean(payouts)) if payouts else 0
        info = VENUE_INFO.get(vid, {'name': str(vid), 'water': '?'})
        venue_features[vid] = {
            'venue_id': vid,
            'name': info['name'],
            'water': info['water'],
            'note': info.get('note', ''),
            'n': n,
            'b1_1st': boat_1st_dist[0] * 100,
            'b2_1st': boat_1st_dist[1] * 100,
            'b3_1st': boat_1st_dist[2] * 100,
            'b4_1st': boat_1st_dist[3] * 100,
            'b5_1st': boat_1st_dist[4] * 100,
            'b6_1st': boat_1st_dist[5] * 100,
            'wind_shift_b1': wind_shift[0] * 100,
            'wind_shift_b4': wind_shift[3] * 100,
            'cls_a1': cls_rates[0] * 100,
            'cls_a2': cls_rates[1] * 100,
            'cls_b1': cls_rates[2] * 100,
            'avg_payout': avg_payout,
        }

    # 戸田との距離計算 (シンプル euclidean on key features)
    toda = venue_features[TODA_VENUE_ID]
    feature_keys = ['b1_1st', 'b2_1st', 'b3_1st', 'b4_1st', 'b5_1st', 'b6_1st',
                    'wind_shift_b1', 'wind_shift_b4', 'cls_a1', 'cls_a2', 'cls_b1']
    # 正規化なしの euclidean (粗い)
    distances = {}
    for vid, vf in venue_features.items():
        if vid == TODA_VENUE_ID:
            continue
        d = 0
        for k in feature_keys:
            d += (vf[k] - toda[k]) ** 2
        # 水質ボーナス: 同水質なら -5 distance
        if vf['water'] == toda['water']:
            d *= 0.7  # 30% 近く
        distances[vid] = (d ** 0.5, vf)

    sorted_dist = sorted(distances.items(), key=lambda x: x[1][0])

    # レポート
    lines = []
    lines.append("# 24 会場 outcome patterns + 物理特性まとめ (戸田類似度)\n\n")
    lines.append(f"対象: 2025-06 以降 finished races (戸田: n={toda['n']})\n")
    lines.append("距離 = key features (1着艇分布 6 + 風強弱 shift 2 + クラス別 1号艇 3) Euclidean\n")
    lines.append("水質同等は ×0.7 (補正)\n\n")

    lines.append("## 全会場 outcome stats\n\n")
    lines.append("| venue | name | 水質 | n | 1号艇% | 2号% | 3号% | 4号% | 5号% | 6号% | 風強→1号 shift | A1時1号% | 平均payout | 備考 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for vid in sorted(venue_features.keys()):
        v = venue_features[vid]
        lines.append(f"| {vid} | {v['name']} | {v['water']} | {v['n']} | "
                     f"{v['b1_1st']:.1f}% | {v['b2_1st']:.1f}% | {v['b3_1st']:.1f}% | "
                     f"{v['b4_1st']:.1f}% | {v['b5_1st']:.1f}% | {v['b6_1st']:.1f}% | "
                     f"{v['wind_shift_b1']:+.1f}pt | {v['cls_a1']:.1f}% | "
                     f"¥{v['avg_payout']:,.0f} | {v['note']} |\n")

    lines.append("\n## 戸田類似度ランキング (top 10)\n\n")
    lines.append("| rank | venue | name | 水質 | 距離 | 1号艇% (vs 戸田 43.2%) | 6号艇% | 備考 |\n|---|---|---|---|---|---|---|---|\n")
    for i, (vid, (d, v)) in enumerate(sorted_dist[:10]):
        lines.append(f"| {i+1} | {vid} | {v['name']} | {v['water']} | {d:.2f} | "
                     f"{v['b1_1st']:.1f}% | {v['b6_1st']:.1f}% | {v['note']} |\n")

    lines.append("\n## 戸田と最も似ていない (top 5)\n\n")
    lines.append("| venue | name | 距離 | 1号艇% | 備考 |\n|---|---|---|---|---|\n")
    for vid, (d, v) in sorted_dist[-5:]:
        lines.append(f"| {vid} | {v['name']} | {d:.2f} | {v['b1_1st']:.1f}% | {v['note']} |\n")

    # 67 のための候補リスト
    lines.append("\n## 67 (sequential add) 候補ランキング\n\n")
    lines.append("戸田 LightGBM (65 結果 ROI -35.85%) に 1 つずつ追加して再訓練。\n")
    lines.append("最も類似度高い venue から順に試行:\n\n")
    for i, (vid, (d, v)) in enumerate(sorted_dist[:5]):
        lines.append(f"{i+1}. **{v['name']} (venue {vid})** — 距離 {d:.2f}, 1号艇率 {v['b1_1st']:.1f}%, {v['water']}, {v['note']}\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")


if __name__ == '__main__':
    main()
