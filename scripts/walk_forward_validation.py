"""Walk-Forward検証: 真のOOS (Out-of-Sample) 回収率を算出

時系列順に「訓練→検証→テスト」をスライドし、
モデルが未来を見ていない状態での予測精度・回収率を測定。

使い方:
    DATABASE_URL=xxx python scripts/walk_forward_validation.py
    DATABASE_URL=xxx python scripts/walk_forward_validation.py --folds 8 --test-months 1
"""
import sys
import os
import logging
import pickle
import numpy as np
import torch
from collections import Counter
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss
from src.features import FeatureEngineer
from src.database import get_db_connection
from scripts.train_model import compute_class_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_data_with_dates():
    """全データを日付付きで取得"""
    feature_engineer = FeatureEngineer()

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date, r.race_number,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.payout_sanrentan,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.status = 'finished'
              AND r.result_1st IS NOT NULL
              AND r.wind_speed IS NOT NULL
              AND r.payout_sanrentan IS NOT NULL
            ORDER BY r.race_date, r.race_number
        """)
        races = cur.fetchall()
        logger.info(f"レース取得: {len(races):,}件")

        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor, tilt, parts_changed
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()

    from collections import defaultdict
    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    X_list, y1_list, y2_list, y3_list = [], [], [], []
    dates_list = []
    payouts_list = []

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
            X_list.append(features)
            y1_list.append(race['result_1st'] - 1)
            y2_list.append(race['result_2nd'] - 1)
            y3_list.append(race['result_3rd'] - 1)
            dates_list.append(race['race_date'])
            payouts_list.append(race['payout_sanrentan'] or 0)
        except Exception:
            continue

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y1_list, dtype=np.int64),
        np.array(y2_list, dtype=np.int64),
        np.array(y3_list, dtype=np.int64),
        dates_list,
        np.array(payouts_list, dtype=np.float32),
    )


def train_and_evaluate(X_train, y1_train, y2_train, y3_train,
                       X_test, y1_test, y2_test, y3_test,
                       payouts_test,
                       epochs=60, lr=0.0005, gamma=2.0):
    """1フォールドの訓練→評価"""
    from torch.utils.data import DataLoader, TensorDataset

    # StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # クラス重み
    cw_1st = compute_class_weights(y1_train, smoothing=0.3)
    cw_2nd = compute_class_weights(y2_train, smoothing=0.7)
    cw_3rd = compute_class_weights(y3_train, smoothing=0.7)

    device = torch.device('cpu')
    input_dim = X_train.shape[1]
    hidden_dims = [512, 256, 128] if input_dim > 50 else [256, 128, 64]

    model = BoatraceMultiTaskModel(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.15
    ).to(device)
    criterion = BoatraceMultiTaskLoss(
        class_weights_1st=cw_1st, class_weights_2nd=cw_2nd,
        class_weights_3rd=cw_3rd, gamma=gamma, label_smoothing_1st=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_s),
        torch.LongTensor(y1_train), torch.LongTensor(y2_train),
        torch.LongTensor(y3_train),
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    # 学習
    model.train()
    for epoch in range(epochs):
        for bx, by1, by2, by3 in loader:
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, (by1, by2, by3))
            loss.backward()
            optimizer.step()

    # テスト評価
    model.eval()
    x_test_t = torch.FloatTensor(X_test_s)
    with torch.no_grad():
        out_1st, out_2nd, out_3rd = model(x_test_t)

    probs_1st = torch.softmax(out_1st, dim=1).numpy()
    pred_1st = probs_1st.argmax(axis=1)

    # 1着精度
    acc_1st = (pred_1st == y1_test).mean() * 100

    # 1着予測分布
    pred_dist = Counter(pred_1st.tolist())
    dist_str = " ".join(
        f"{i+1}号艇:{pred_dist.get(i,0)/len(pred_1st)*100:.1f}%"
        for i in range(6)
    )

    # 三連単の仮想回収率（top1予測を100円ベット想定）
    probs_2nd = torch.softmax(out_2nd, dim=1).numpy()
    probs_3rd = torch.softmax(out_3rd, dim=1).numpy()

    total_bet = 0
    total_payout = 0
    for idx in range(len(y1_test)):
        # top1三連単
        p1 = probs_1st[idx].argmax()
        remaining_2 = [j for j in range(6) if j != p1]
        p2_probs = np.array([probs_2nd[idx][j] for j in remaining_2])
        p2 = remaining_2[p2_probs.argmax()]
        remaining_3 = [j for j in range(6) if j != p1 and j != p2]
        p3_probs = np.array([probs_3rd[idx][j] for j in remaining_3])
        p3 = remaining_3[p3_probs.argmax()]

        total_bet += 100
        if (p1 == y1_test[idx] and p2 == y2_test[idx]
                and p3 == y3_test[idx]):
            total_payout += payouts_test[idx]

    roi = total_payout / total_bet * 100 if total_bet > 0 else 0

    return {
        'acc_1st': acc_1st,
        'pred_dist': dist_str,
        'roi_top1': roi,
        'total_bet': total_bet,
        'total_payout': total_payout,
        'n_test': len(y1_test),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward検証')
    parser.add_argument('--folds', type=int, default=6,
                        help='検証フォールド数')
    parser.add_argument('--test-months', type=int, default=1,
                        help='テスト期間（月数）')
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()

    logger.info("=== Walk-Forward検証 ===")
    logger.info(f"  folds={args.folds}, test_months={args.test_months}")

    X, y1, y2, y3, dates, payouts = load_data_with_dates()
    logger.info(f"データ: {len(X):,}件, 期間: {dates[0]} ~ {dates[-1]}")

    # 日付からフォールドを生成
    unique_dates = sorted(set(dates))
    total_days = (unique_dates[-1] - unique_dates[0]).days
    test_days = args.test_months * 30

    results = []
    for fold in range(args.folds):
        # テスト期間: 末尾からfold番目のtest_days
        test_end_idx = len(unique_dates) - fold * (test_days // 2)
        test_start_idx = max(0, test_end_idx - test_days)

        if test_start_idx <= 0 or test_end_idx <= test_start_idx:
            break

        test_start_date = unique_dates[test_start_idx]
        test_end_date = unique_dates[min(test_end_idx, len(unique_dates) - 1)]

        # データ分割
        train_mask = np.array([d < test_start_date for d in dates])
        test_mask = np.array([
            test_start_date <= d <= test_end_date for d in dates
        ])

        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            logger.warning(f"Fold {fold+1}: データ不足 "
                           f"(train={train_mask.sum()}, test={test_mask.sum()})")
            continue

        logger.info(
            f"\nFold {fold+1}: 訓練=~{test_start_date} ({train_mask.sum():,}件), "
            f"テスト={test_start_date}~{test_end_date} ({test_mask.sum():,}件)"
        )

        result = train_and_evaluate(
            X[train_mask], y1[train_mask], y2[train_mask], y3[train_mask],
            X[test_mask], y1[test_mask], y2[test_mask], y3[test_mask],
            payouts[test_mask],
            epochs=args.epochs,
        )
        result['fold'] = fold + 1
        result['test_period'] = f"{test_start_date}~{test_end_date}"
        results.append(result)

        logger.info(
            f"  1着精度: {result['acc_1st']:.1f}%, "
            f"Top1 ROI: {result['roi_top1']:.1f}%"
        )
        logger.info(f"  予測分布: {result['pred_dist']}")

    # サマリ
    if results:
        logger.info("\n=== Walk-Forward検証サマリ ===")
        avg_acc = np.mean([r['acc_1st'] for r in results])
        avg_roi = np.mean([r['roi_top1'] for r in results])
        logger.info(f"平均1着精度: {avg_acc:.1f}%")
        logger.info(f"平均Top1 ROI: {avg_roi:.1f}%")
        for r in results:
            logger.info(
                f"  Fold {r['fold']}: {r['test_period']} "
                f"acc={r['acc_1st']:.1f}% roi={r['roi_top1']:.1f}% "
                f"n={r['n_test']}"
            )


if __name__ == '__main__':
    main()
