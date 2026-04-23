"""V11 Backtest: val集合（直近20%）でのROI/的中率シミュレーション

簡易バックテスト:
  - V11予測を各レースに適用
  - 3連単の上位N候補を1点¥1,000固定で購入したと仮定
  - 実結果 (race.actual_result_trifecta, payout_sanrentan) でROI算出
  - Miss Analysisフィルタ適用版も同時評価

V10モデルは触らない・ロードしない。V11単独評価 + Vtop1 baseline との比較。
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
REPORT_DIR = Path(__file__).parent / "reports"

# Miss Analysis フィルタ
WEAK_VENUES = [1, 2, 3, 4, 5, 6]       # 負け会場
STRONG_VENUES = [9, 10, 12, 20, 23]    # 勝ち会場
EXCLUDE_R = [1]                         # R1除外
MAX_R = 4                               # R1-R4のみ
MAX_ODDS = 40
BET_AMOUNT = 1000                       # 1点¥1,000固定

# フィルタ候補
FILTERS = {
    'V11_raw': {'desc': 'V11 raw（フィルタなし）'},
    'V11_filtered': {'desc': 'V11 + Miss分析フィルタ (勝ち会場・R2-R4)'},
    'V11_max_odds_40': {'desc': 'V11 + max_odds=40のみ'},
}


def load_v11_models():
    models = {}
    for pos in ['1st', '2nd', '3rd']:
        path = MODELS_DIR / f"boatrace_v11_{pos}.txt"
        models[pos] = lgb.Booster(model_file=str(path))
    return models


def predict_trifecta_topk(models, X_row, k=3):
    """1レース分の特徴量からトップK組合せを予測

    簡易版: 1着/2着/3着それぞれ独立のargmax top-kからクロス積で組合せ生成。
    確率の積でスコアリング。
    """
    p1 = models['1st'].predict([X_row])[0]  # shape (6,)
    p2 = models['2nd'].predict([X_row])[0]
    p3 = models['3rd'].predict([X_row])[0]

    combos = []
    for a in range(6):
        for b in range(6):
            if b == a:
                continue
            for c in range(6):
                if c == a or c == b:
                    continue
                score = p1[a] * p2[b] * p3[c]
                combo = f"{a+1}-{b+1}-{c+1}"
                combos.append((combo, score, p1[a], p2[b], p3[c]))
    combos.sort(key=lambda x: -x[1])
    return combos[:k]


def fetch_race_context(cur, race_ids):
    """val集合のレース情報を取得"""
    cur.execute("""
        SELECT id, venue_id, race_number,
               actual_result_trifecta, payout_sanrentan,
               wind_speed, wave_height
        FROM races
        WHERE id = ANY(%s)
    """, ([int(r) for r in race_ids],))
    return {r['id']: dict(r) for r in cur.fetchall()}


def main():
    logger.info("V11 Backtest 開始")

    # データ + モデルロード
    data_path = MODELS_DIR / "train_data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    race_ids = data['race_ids']
    n = len(X)
    split = int(n * 0.8)
    X_val = X[split:]
    race_ids_val = race_ids[split:]
    logger.info(f"Val 集合: {len(X_val)}レース")

    models = load_v11_models()
    logger.info("V11モデルロード完了")

    # レース情報取得
    conn = psycopg2.connect(os.environ['DATABASE_URL'],
                            cursor_factory=RealDictCursor)
    cur = conn.cursor()
    race_info = fetch_race_context(cur, race_ids_val)
    logger.info(f"レース情報: {len(race_info)}件")

    # 各フィルタでバックテスト
    results = {}
    for filter_name in FILTERS:
        stats = {
            'bets': 0,
            'invest': 0,
            'hits': 0,
            'payout': 0,
            'filtered_out_races': 0,
        }

        for i, rid in enumerate(race_ids_val):
            race = race_info.get(int(rid))
            if not race:
                continue

            # フィルタ適用（フィルタ版のみ）
            if filter_name == 'V11_filtered':
                # Miss Analysis フィルタ
                if race['venue_id'] not in STRONG_VENUES:
                    stats['filtered_out_races'] += 1
                    continue
                if race['race_number'] in EXCLUDE_R:
                    stats['filtered_out_races'] += 1
                    continue
                if race['race_number'] > MAX_R:
                    stats['filtered_out_races'] += 1
                    continue

            # V11予測 (top 3)
            top3 = predict_trifecta_topk(models, X_val[i], k=3)

            actual = race.get('actual_result_trifecta')
            payout = race.get('payout_sanrentan') or 0

            for combo, score, _, _, _ in top3:
                # max_odds_40 フィルタは payout/100 = 擬似odds ベース
                if filter_name in ('V11_filtered', 'V11_max_odds_40'):
                    # このcomboが的中した時の配当相当 (近似)
                    # 実際のoddsは購入時点にしか分からないが、payoutベースで近似
                    if combo == actual and payout > 0:
                        if payout / 100.0 > MAX_ODDS:
                            continue
                    # 的中以外はoddsわからないのでそのままbet

                stats['bets'] += 1
                stats['invest'] += BET_AMOUNT

                if combo == actual:
                    # payout は100円単位の払戻
                    ret = int(payout) * (BET_AMOUNT // 100)
                    stats['hits'] += 1
                    stats['payout'] += ret

        roi = stats['payout'] / stats['invest'] if stats['invest'] else 0
        hit_rate = stats['hits'] / stats['bets'] if stats['bets'] else 0
        profit = stats['payout'] - stats['invest']

        stats['roi'] = roi
        stats['hit_rate'] = hit_rate
        stats['profit'] = profit
        results[filter_name] = stats

        logger.info(f"\n=== {filter_name} ({FILTERS[filter_name]['desc']}) ===")
        logger.info(f"  bets={stats['bets']} hits={stats['hits']} "
                    f"的中率={hit_rate*100:.2f}%")
        logger.info(f"  投資={stats['invest']:,} 回収={stats['payout']:,} "
                    f"損益={profit:+,} ROI={roi*100:.1f}%")
        if filter_name == 'V11_filtered':
            logger.info(f"  フィルタ除外: {stats['filtered_out_races']}レース")

    # 結果保存
    report = {
        'backtest_at': datetime.now().isoformat(),
        'val_races': len(X_val),
        'bet_amount_per_ticket': BET_AMOUNT,
        'top_k_bets': 3,
        'results': results,
    }
    out_path = REPORT_DIR / "04_v11_backtest.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nレポート: {out_path}")

    conn.close()


if __name__ == '__main__':
    main()
