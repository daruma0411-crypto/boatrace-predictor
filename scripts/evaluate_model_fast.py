"""モデル精度評価スクリプト（高速バッチ版）

全レース+boats を一括取得し、バッチ推論で高速評価する。
元の evaluate_model.py の 1件ずつクエリ版は 53k レースで数時間かかるため、
こちらを使用する。
"""
import sys
import os
import logging
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load_model
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def evaluate():
    model = load_model('models/boatrace_model.pth')
    model.eval()
    feature_engineer = FeatureEngineer()
    device = torch.device('cpu')

    # === 1. 全データを一括取得 ===
    logger.info("データ一括取得中...")

    with get_db_connection() as conn:
        cur = conn.cursor()

        # レース一括取得（天候データ含む）
        cur.execute("""
            SELECT id, venue_id, race_date,
                   result_1st, result_2nd, result_3rd,
                   payout_sanrentan,
                   wind_speed, wind_direction, temperature,
                   wave_height, water_temperature
            FROM races
            WHERE result_1st IS NOT NULL AND status = 'finished'
            ORDER BY race_date
        """)
        races = cur.fetchall()
        logger.info(f"レース取得: {len(races):,}件")

        # 対象レースIDリスト
        race_ids = [r['id'] for r in races]

        # boats 一括取得（tilt/parts_changed含む）
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()
        logger.info(f"ボート取得: {len(all_boats):,}件")

    # レースIDごとにboatsをグループ化
    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    # === 2. 特徴量一括生成 ===
    logger.info("特徴量生成中...")
    features_list = []
    valid_races = []

    for race in races:
        boats = boats_by_race.get(race['id'], [])
        if len(boats) != 6:
            continue

        race_data = {
            'venue_id': race['venue_id'],
            'month': race['race_date'].month,
            'distance': 1800,
            'wind_speed': race.get('wind_speed') or 0,
            'wind_direction': race.get('wind_direction') or 'calm',
            'temperature': race.get('temperature') or 20,
            'wave_height': race.get('wave_height') or 0,
            'water_temperature': race.get('water_temperature') or 20,
        }

        try:
            features = feature_engineer.transform(race_data, boats)
            features_list.append(features)
            valid_races.append(race)
        except Exception:
            continue

    total = len(valid_races)
    logger.info(f"有効レース: {total:,}件\n")

    # === 3. バッチ推論 ===
    logger.info("バッチ推論中...")
    X = torch.FloatTensor(np.array(features_list)).to(device)

    BATCH_SIZE = 4096
    all_probs_1st = []
    all_probs_2nd = []
    all_probs_3rd = []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch = X[i:i + BATCH_SIZE]
            out_1st, out_2nd, out_3rd = model(batch)
            all_probs_1st.append(torch.softmax(out_1st, dim=1).numpy())
            all_probs_2nd.append(torch.softmax(out_2nd, dim=1).numpy())
            all_probs_3rd.append(torch.softmax(out_3rd, dim=1).numpy())

    probs_1st = np.concatenate(all_probs_1st)
    probs_2nd = np.concatenate(all_probs_2nd)
    probs_3rd = np.concatenate(all_probs_3rd)
    logger.info("推論完了\n")

    # === 4. メトリクス計算 ===
    pred_1st_arr = np.argmax(probs_1st, axis=1) + 1
    pred_2nd_arr = np.argmax(probs_2nd, axis=1) + 1
    pred_3rd_arr = np.argmax(probs_3rd, axis=1) + 1

    actual_1st_arr = np.array([r['result_1st'] for r in valid_races])
    actual_2nd_arr = np.array([r['result_2nd'] for r in valid_races])
    actual_3rd_arr = np.array([r['result_3rd'] for r in valid_races])
    payouts_arr = np.array([r['payout_sanrentan'] or 0 for r in valid_races])

    # 1着正解
    correct_1st = np.sum(pred_1st_arr == actual_1st_arr)

    # Top-2, Top-3
    top2_indices = np.argsort(probs_1st, axis=1)[:, -2:]
    top3_indices = np.argsort(probs_1st, axis=1)[:, -3:]
    top2_1st = sum(
        actual_1st_arr[i] - 1 in top2_indices[i] for i in range(total)
    )
    top3_1st = sum(
        actual_1st_arr[i] - 1 in top3_indices[i] for i in range(total)
    )

    # 3連単的中
    correct_trifecta = np.sum(
        (pred_1st_arr == actual_1st_arr) &
        (pred_2nd_arr == actual_2nd_arr) &
        (pred_3rd_arr == actual_3rd_arr)
    )

    # 1号艇バイアス
    boat1_actual_wins = np.sum(actual_1st_arr == 1)
    boat1_predicted_wins = np.sum(pred_1st_arr == 1)

    # 回収率シミュレーション（均等買い）
    trifecta_mask = (
        (pred_1st_arr == actual_1st_arr) &
        (pred_2nd_arr == actual_2nd_arr) &
        (pred_3rd_arr == actual_3rd_arr)
    )
    total_bet = total * 100
    total_payout = int(np.sum(payouts_arr[trifecta_mask]))

    # ケリー基準シミュレーション
    kelly_bet = 0
    kelly_payout = 0
    for i in range(total):
        p_trifecta = float(
            probs_1st[i, pred_1st_arr[i] - 1] *
            probs_2nd[i, pred_2nd_arr[i] - 1] *
            probs_3rd[i, pred_3rd_arr[i] - 1]
        )
        payout = payouts_arr[i]
        if payout > 0 and p_trifecta > 0:
            odds = payout / 100.0
            edge = p_trifecta * odds - 1
            if edge > 0:
                kelly_f = min(0.05, edge / odds * 0.5)
                bet_amount = int(kelly_f * 10000)
                if bet_amount >= 100:
                    kelly_bet += bet_amount
                    if trifecta_mask[i]:
                        kelly_payout += int(bet_amount * odds)

    # === 5. 結果出力 ===
    logger.info(f"--- 基本精度 ---")
    logger.info(f"1着予測 accuracy: {correct_1st}/{total} = {correct_1st/total*100:.1f}% (理論値 16.7%)")
    logger.info(f"1着 Top-2 accuracy: {top2_1st}/{total} = {top2_1st/total*100:.1f}% (理論値 33.3%)")
    logger.info(f"1着 Top-3 accuracy: {top3_1st}/{total} = {top3_1st/total*100:.1f}% (理論値 50.0%)")
    logger.info(f"3連単的中率: {correct_trifecta}/{total} = {correct_trifecta/total*100:.2f}% (理論値 0.14%)")
    logger.info(f"")

    logger.info(f"--- 1号艇バイアス ---")
    logger.info(f"実際の1号艇1着率: {boat1_actual_wins}/{total} = {boat1_actual_wins/total*100:.1f}%")
    logger.info(f"モデルの1号艇1着予測率: {boat1_predicted_wins}/{total} = {boat1_predicted_wins/total*100:.1f}%")
    logger.info(f"")

    logger.info(f"--- 回収率シミュレーション ---")
    logger.info(f"均等買い: 投資{total_bet:,}円 → 回収{total_payout:,}円 = {total_payout/total_bet*100:.1f}%")
    if kelly_bet > 0:
        logger.info(f"ケリー基準: 投資{kelly_bet:,}円 → 回収{kelly_payout:,}円 = {kelly_payout/kelly_bet*100:.1f}%")
    else:
        logger.info(f"ケリー基準: ベットなし (エッジのある機会なし)")
    logger.info(f"")

    # 予測確率分布
    logger.info(f"--- 予測確率分布 (1着) ---")
    for boat_num in range(1, 7):
        mean_p = probs_1st[:, boat_num - 1].mean()
        actual_rate = np.sum(actual_1st_arr == boat_num) / total
        logger.info(f"  {boat_num}号艇: 予測平均 {mean_p*100:.1f}% / 実績 {actual_rate*100:.1f}%")

    # 前回比較用サマリー
    logger.info(f"\n=== サマリー ===")
    logger.info(f"データ数: {total:,}")
    logger.info(f"1着精度: {correct_1st/total*100:.1f}%")
    logger.info(f"3連単的中: {correct_trifecta/total*100:.2f}%")
    logger.info(f"1号艇バイアス: {boat1_predicted_wins/total*100:.1f}%")
    logger.info(f"均等ROI: {total_payout/total_bet*100:.1f}%")
    if kelly_bet > 0:
        logger.info(f"ケリーROI: {kelly_payout/kelly_bet*100:.1f}%")
    logger.info(f"\n=== 評価完了 ===")


if __name__ == '__main__':
    evaluate()
