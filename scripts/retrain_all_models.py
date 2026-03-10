"""全モデル一括再学習スクリプト

メインモデル + 3つのアンサンブルモデルを208次元で統一再学習する。
各モデルはクラス重みスムージングパラメータが異なる:
  - boatrace_model.pth:      smoothing=0.3 (デフォルト)
  - boatrace_model_s05.pth:  smoothing=0.5
  - boatrace_model_s07.pth:  smoothing=0.7
  - boatrace_model_s085.pth: smoothing=0.85

使い方:
    DATABASE_URL=xxx python scripts/retrain_all_models.py
    DATABASE_URL=xxx python scripts/retrain_all_models.py --epochs 80 --patience 15
"""
import sys
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss, save_model
from scripts.train_model import load_training_data_fast, compute_class_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODELS = [
    {'path': 'models/boatrace_model.pth', 'smoothing': 0.3},
    {'path': 'models/boatrace_model_s05.pth', 'smoothing': 0.5},
    {'path': 'models/boatrace_model_s07.pth', 'smoothing': 0.7},
    {'path': 'models/boatrace_model_s085.pth', 'smoothing': 0.85},
]


def train_one_model(X_train, y1_train, y2_train, y3_train,
                    X_val, y1_val, y2_val, y3_val,
                    smoothing, save_path,
                    epochs=100, batch_size=256, lr=0.001, patience=10):
    """1モデルを訓練して保存"""
    logger.info(f"=== {save_path} (smoothing={smoothing}) ===")

    cw_1st = compute_class_weights(y1_train, smoothing=smoothing)
    cw_2nd = compute_class_weights(y2_train, smoothing=smoothing)
    cw_3rd = compute_class_weights(y3_train, smoothing=smoothing)

    logger.info(f"  class_weights_1st: {['%.2f' % w for w in cw_1st.tolist()]}")

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y1_train),
        torch.LongTensor(y2_train),
        torch.LongTensor(y3_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y1_val),
        torch.LongTensor(y2_val),
        torch.LongTensor(y3_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_train.shape[1]
    model = BoatraceMultiTaskModel(input_dim=input_dim).to(device)
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
        model.train()
        train_loss = 0
        for batch_x, batch_y1, batch_y2, batch_y3 in train_loader:
            batch_x = batch_x.to(device)
            targets = (batch_y1.to(device), batch_y2.to(device), batch_y3.to(device))
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y1, batch_y2, batch_y3 in val_loader:
                batch_x = batch_x.to(device)
                targets = (batch_y1.to(device), batch_y2.to(device), batch_y3.to(device))
                outputs = model(batch_x)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, path=save_path, metadata={
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'class_weights_1st': cw_1st.tolist(),
                'weight_smoothing': smoothing,
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early Stop epoch {epoch+1}, best_val={best_val_loss:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

    logger.info(f"  完了: best_val_loss={best_val_loss:.4f}")
    return best_val_loss


def main():
    import argparse
    parser = argparse.ArgumentParser(description='全モデル一括再学習')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    logger.info("=== 全モデル一括再学習 ===")

    # データ読み込み（1回だけ）
    X, y1, y2, y3 = load_training_data_fast()
    if X is None:
        logger.error("訓練データなし。collect_parallel.py でデータ収集してください。")
        return

    logger.info(f"データ: {len(X):,}件, 次元: {X.shape[1]}")

    # 共通の訓練/検証分割
    n = len(X)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y1_train, y1_val = y1[train_idx], y1[val_idx]
    y2_train, y2_val = y2[train_idx], y2[val_idx]
    y3_train, y3_val = y3[train_idx], y3[val_idx]

    results = []
    for m in MODELS:
        val_loss = train_one_model(
            X_train, y1_train, y2_train, y3_train,
            X_val, y1_val, y2_val, y3_val,
            smoothing=m['smoothing'],
            save_path=m['path'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        results.append({'path': m['path'], 'smoothing': m['smoothing'], 'val_loss': val_loss})

    logger.info("\n=== 全モデル再学習完了 ===")
    for r in results:
        logger.info(f"  {r['path']}: smoothing={r['smoothing']}, val_loss={r['val_loss']:.4f}")

    # 次元確認
    for m in MODELS:
        cp = torch.load(m['path'], map_location='cpu', weights_only=False)
        logger.info(f"  {m['path']}: input_dim={cp.get('input_dim')}")


if __name__ == '__main__':
    main()
