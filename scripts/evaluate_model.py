"""モデル精度評価スクリプト

1着予測accuracy, 3連単的中率, 1号艇バイアス, 回収率シミュレーション
"""
import sys
import os
import logging
import numpy as np
import torch
from collections import Counter

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

    # 全レースデータ取得
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.payout_sanrentan
            FROM races r
            WHERE r.result_1st IS NOT NULL AND r.status = 'finished'
            ORDER BY r.race_date
        """)
        races = cur.fetchall()

    logger.info(f"=== 精度評価: {len(races)}レース ===\n")

    # 評価用メトリクス
    correct_1st = 0
    correct_trifecta = 0
    top2_1st = 0
    top3_1st = 0
    total = 0
    boat1_actual_wins = 0
    boat1_predicted_wins = 0

    # 回収率シミュレーション
    total_bet = 0
    total_payout = 0
    kelly_bet = 0
    kelly_payout = 0

    # 予測確率の蓄積
    all_probs_1st = []
    actual_1st_list = []

    with get_db_connection() as conn:
        cur = conn.cursor()

        for race in races:
            cur.execute("""
                SELECT * FROM boats WHERE race_id = %s ORDER BY boat_number
            """, (race['id'],))
            boats = cur.fetchall()

            if len(boats) != 6:
                continue

            race_data = {
                'venue_id': race['venue_id'],
                'month': race['race_date'].month,
                'distance': 1800,
                'wind_speed': 0,
                'wind_direction': 'calm',
                'temperature': 20,
            }
            boats_data = [dict(b) for b in boats]

            try:
                features = feature_engineer.transform(race_data, boats_data)
            except Exception:
                continue

            x = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                out_1st, out_2nd, out_3rd = model(x)

            probs_1st = torch.softmax(out_1st, dim=1).squeeze().numpy()
            probs_2nd = torch.softmax(out_2nd, dim=1).squeeze().numpy()
            probs_3rd = torch.softmax(out_3rd, dim=1).squeeze().numpy()

            pred_1st = np.argmax(probs_1st) + 1  # 0-indexed → 1-indexed
            pred_2nd = np.argmax(probs_2nd) + 1
            pred_3rd = np.argmax(probs_3rd) + 1

            actual_1st = race['result_1st']
            actual_2nd = race['result_2nd']
            actual_3rd = race['result_3rd']
            payout = race['payout_sanrentan'] or 0

            total += 1
            all_probs_1st.append(probs_1st)
            actual_1st_list.append(actual_1st)

            # 1着予測正解
            if pred_1st == actual_1st:
                correct_1st += 1

            # Top-2, Top-3
            top2 = np.argsort(probs_1st)[-2:][::-1] + 1
            top3 = np.argsort(probs_1st)[-3:][::-1] + 1
            if actual_1st in top2:
                top2_1st += 1
            if actual_1st in top3:
                top3_1st += 1

            # 3連単的中
            if pred_1st == actual_1st and pred_2nd == actual_2nd and pred_3rd == actual_3rd:
                correct_trifecta += 1

            # 1号艇統計
            if actual_1st == 1:
                boat1_actual_wins += 1
            if pred_1st == 1:
                boat1_predicted_wins += 1

            # 均等買いシミュレーション（1レース100円）
            total_bet += 100
            if pred_1st == actual_1st and pred_2nd == actual_2nd and pred_3rd == actual_3rd:
                total_payout += payout  # 払戻金は100円あたり

            # ケリー基準シミュレーション
            # 3連単確率 = P(1st) * P(2nd|1st) * P(3rd|1st,2nd) の近似
            p_trifecta = float(probs_1st[pred_1st - 1] *
                               probs_2nd[pred_2nd - 1] *
                               probs_3rd[pred_3rd - 1])
            if payout > 0 and p_trifecta > 0:
                odds = payout / 100.0
                edge = p_trifecta * odds - 1
                if edge > 0:
                    kelly_f = min(0.05, edge / odds * 0.5)  # ハーフケリー, max 5%
                    bet_amount = int(kelly_f * 10000)  # 仮想資金1万円
                    if bet_amount >= 100:
                        kelly_bet += bet_amount
                        if (pred_1st == actual_1st and
                                pred_2nd == actual_2nd and
                                pred_3rd == actual_3rd):
                            kelly_payout += int(bet_amount * odds)

    # 結果出力
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

    # 予測確率の分布分析
    all_probs = np.array(all_probs_1st)
    logger.info(f"--- 予測確率分布 (1着) ---")
    for i in range(6):
        boat_num = i + 1
        mean_p = all_probs[:, i].mean()
        actual_rate = sum(1 for a in actual_1st_list if a == boat_num) / total
        logger.info(f"  {boat_num}号艇: 予測平均 {mean_p*100:.1f}% / 実績 {actual_rate*100:.1f}%")

    logger.info(f"\n=== 評価完了 ===")


if __name__ == '__main__':
    evaluate()
