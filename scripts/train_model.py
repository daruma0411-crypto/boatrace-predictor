"""モデル訓練スクリプト: クラス重み付き + 一括データ取得 + Early Stopping"""
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss, save_model
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_class_weights(labels, num_classes=6, smoothing=0.3):
    """出現頻度の逆数ベースのクラス重みを計算

    smoothing: 0.0=完全逆数, 1.0=均等重み
        0.3 は穴番を約3倍重視しつつ、1号艇も完全無視しない設定

    例 (54kデータ):
        1号艇: 54.6% → weight ≈ 0.56
        2号艇: 13.5% → weight ≈ 1.35
        3号艇: 12.7% → weight ≈ 1.43
        4号艇: 10.1% → weight ≈ 1.60
        5号艇:  5.9% → weight ≈ 2.11
        6号艇:  3.1% → weight ≈ 2.79
    """
    counts = Counter(labels.tolist())
    total = len(labels)

    # 逆頻度重み: w_i = total / (num_classes * count_i)
    raw_weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        raw_weights.append(total / (num_classes * count))

    raw_weights = np.array(raw_weights, dtype=np.float32)

    # スムージング: smoothed = (1 - s) * raw + s * 1.0
    smoothed = (1 - smoothing) * raw_weights + smoothing * np.ones(num_classes)

    # 正規化（平均1.0に）
    smoothed = smoothed / smoothed.mean()

    return torch.FloatTensor(smoothed)


def load_training_data_fast(years=3):
    """過去N年分のレースデータをDB一括取得→特徴量変換"""
    feature_engineer = FeatureEngineer()
    cutoff_date = datetime.now() - timedelta(days=365 * years)

    logger.info("データ一括取得中...")

    with get_db_connection() as conn:
        cur = conn.cursor()

        # レース一括取得（天候データ含む）
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.race_date >= %s AND r.status = 'finished'
              AND r.result_1st IS NOT NULL
            ORDER BY r.race_date
        """, (cutoff_date.date(),))
        races = cur.fetchall()
        logger.info(f"レース取得: {len(races):,}件")

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

    # レースIDごとにグループ化
    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    logger.info("特徴量生成中...")
    X_list = []
    y1_list = []
    y2_list = []
    y3_list = []

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
        except Exception as e:
            continue

    if not X_list:
        logger.warning("訓練データが0件です")
        return None, None, None, None

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int64)
    y2 = np.array(y2_list, dtype=np.int64)
    y3 = np.array(y3_list, dtype=np.int64)

    logger.info(f"訓練データ: {len(X):,}件, 特徴量次元: {X.shape[1]}")
    return X, y1, y2, y3


def train(epochs=100, batch_size=256, lr=0.001, patience=10,
          weight_smoothing=0.3):
    """モデル訓練（クラス重み + Early Stopping）

    Args:
        weight_smoothing: クラス重みの平滑化パラメータ
            0.0 = 完全な逆頻度重み（穴番を極端に重視）
            0.3 = デフォルト（穴番を適度に重視）
            1.0 = 均等重み（旧来と同じ）
    """
    logger.info("=== モデル訓練開始 ===")
    logger.info(f"クラス重みスムージング: {weight_smoothing}")

    X, y1, y2, y3 = load_training_data_fast()
    if X is None:
        logger.error("訓練データがありません。")
        return

    # === クラス重み計算 ===
    cw_1st = compute_class_weights(y1, smoothing=weight_smoothing)
    cw_2nd = compute_class_weights(y2, smoothing=weight_smoothing)
    cw_3rd = compute_class_weights(y3, smoothing=weight_smoothing)

    logger.info(f"1着クラス重み: {['%.2f' % w for w in cw_1st.tolist()]}")
    logger.info(f"  (1号艇={cw_1st[0]:.2f}, 6号艇={cw_1st[5]:.2f}, 比率={cw_1st[5]/cw_1st[0]:.1f}x)")

    # 訓練/検証 分割 (8:2)
    n = len(X)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X[train_idx]),
        torch.LongTensor(y1[train_idx]),
        torch.LongTensor(y2[train_idx]),
        torch.LongTensor(y3[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X[val_idx]),
        torch.LongTensor(y1[val_idx]),
        torch.LongTensor(y2[val_idx]),
        torch.LongTensor(y3[val_idx]),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"デバイス: {device}")

    model = BoatraceMultiTaskModel().to(device)
    criterion = BoatraceMultiTaskLoss(
        class_weights_1st=cw_1st.to(device),
        class_weights_2nd=cw_2nd.to(device),
        class_weights_3rd=cw_3rd.to(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 訓練
        model.train()
        train_loss = 0
        for batch_x, batch_y1, batch_y2, batch_y3 in train_loader:
            batch_x = batch_x.to(device)
            targets = (
                batch_y1.to(device),
                batch_y2.to(device),
                batch_y3.to(device),
            )
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y1, batch_y2, batch_y3 in val_loader:
                batch_x = batch_x.to(device)
                targets = (
                    batch_y1.to(device),
                    batch_y2.to(device),
                    batch_y3.to(device),
                )
                outputs = model(batch_x)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, metadata={
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'class_weights_1st': cw_1st.tolist(),
                'weight_smoothing': weight_smoothing,
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early Stopping: epoch {epoch+1}, "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

    logger.info("=== モデル訓練完了 ===")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoothing', type=float, default=0.3,
                        help='クラス重みスムージング (0.0=逆頻度, 1.0=均等)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        weight_smoothing=args.smoothing,
    )
