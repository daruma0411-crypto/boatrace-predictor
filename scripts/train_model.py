"""モデル訓練スクリプト: 過去3年データ、Early Stopping"""
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss, save_model
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(years=3):
    """過去N年分のレースデータをDBから取得し特徴量に変換"""
    feature_engineer = FeatureEngineer()
    cutoff_date = datetime.now() - timedelta(days=365 * years)

    X_list = []
    y1_list = []
    y2_list = []
    y3_list = []

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd
            FROM races r
            WHERE r.race_date >= %s AND r.status = 'finished'
              AND r.result_1st IS NOT NULL
            ORDER BY r.race_date
        """, (cutoff_date.date(),))
        races = cur.fetchall()

        logger.info(f"訓練対象レース: {len(races)}件")

        for race in races:
            cur.execute("""
                SELECT * FROM boats
                WHERE race_id = %s
                ORDER BY boat_number
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
                X_list.append(features)

                # 実際の着順ラベル（1始まり→0始まり）
                y1_list.append(race['result_1st'] - 1)
                y2_list.append(race['result_2nd'] - 1)
                y3_list.append(race['result_3rd'] - 1)
            except Exception as e:
                logger.warning(f"特徴量変換エラー (race_id={race['id']}): {e}")
                continue

    if not X_list:
        logger.warning("訓練データが0件です")
        return None, None, None, None

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int64)
    y2 = np.array(y2_list, dtype=np.int64)
    y3 = np.array(y3_list, dtype=np.int64)

    logger.info(f"訓練データ: {len(X)}件, 特徴量次元: {X.shape[1]}")
    return X, y1, y2, y3


def train(epochs=100, batch_size=256, lr=0.001, patience=10):
    """モデル訓練（Early Stopping付き）"""
    logger.info("=== モデル訓練開始 ===")

    X, y1, y2, y3 = load_training_data()
    if X is None:
        logger.error("訓練データがありません。先にcollect_historical_data.pyを実行してください。")
        return

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
    criterion = BoatraceMultiTaskLoss()
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
    train()
