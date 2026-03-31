"""荒れ専門モデル訓練: 1号艇が勝たないレースに特化した5クラス分類

Model B: 2-6号艇のどれが1着になるかを予測。
1号艇が勝つレース(55%)を訓練データから除外し、
残り45%の「荒れレース」だけで学習。

モデルの全容量を2-6号艇の区別に投入するため、
通常モデル(Model A)より荒れレースでの予測精度が高い。

使い方:
    DATABASE_URL=xxx python scripts/train_model_are.py
"""
import sys
import os
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss, save_model
from src.features import FeatureEngineer
from scripts.train_model import load_training_data_fast, compute_class_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = 'models/boatrace_model_are.pth'
SCALER_PATH = 'models/feature_scaler_are.pkl'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='荒れ専門モデル訓練 (5クラス)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    args = parser.parse_args()

    logger.info("=== 荒れ専門モデル訓練 (Model B: 5クラス) ===")

    # 全データ読み込み
    X, y1, y2, y3 = load_training_data_fast()
    if X is None:
        logger.error("訓練データなし")
        return

    logger.info(f"全データ: {len(X):,}件")

    # 1号艇が勝ったレースを除外
    are_mask = y1 != 0  # 0-indexed: 0=1号艇
    X_are = X[are_mask]
    y1_are = y1[are_mask]
    y2_are = y2[are_mask]
    y3_are = y3[are_mask]
    logger.info(f"荒れレース (1号艇以外が1着): {len(X_are):,}件 "
                f"({len(X_are)/len(X)*100:.1f}%)")

    # ラベルリマッピング: 2号艇→0, 3号艇→1, ..., 6号艇→4
    # 1着: 元のboat_idx(1-5) → 新idx(0-4)
    y1_remap = y1_are - 1  # 1→0, 2→1, 3→2, 4→3, 5→4

    # 2着/3着: 1号艇(idx=0)も含む6クラスのまま
    # → 5クラスにリマップ（1号艇が2着/3着に来る場合もある）
    # 方針: 2着/3着は6クラスのまま維持（1号艇が2-3着に来ることはある）

    # 1着分布確認
    dist = Counter(y1_remap.tolist())
    logger.info("1着分布 (リマップ後):")
    for i in range(5):
        boat = i + 2  # 2号艇〜6号艇
        pct = dist.get(i, 0) / len(y1_remap) * 100
        logger.info(f"  {boat}号艇 (idx={i}): {pct:.1f}%")

    # 時系列分割
    n = len(X_are)
    split = int(n * 0.8)
    train_idx = list(range(0, split))
    val_idx = list(range(split, n))
    logger.info(f"時系列分割: 訓練={len(train_idx):,}件, 検証={len(val_idx):,}件")

    # StandardScaler
    scaler = StandardScaler()
    X_are[train_idx] = scaler.fit_transform(X_are[train_idx])
    X_are[val_idx] = scaler.transform(X_are[val_idx])

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"StandardScaler保存: {SCALER_PATH}")

    # クラス重み (5クラス)
    cw_1st = compute_class_weights(y1_remap[train_idx], num_classes=5, smoothing=0.3)
    cw_2nd = compute_class_weights(y2_are[train_idx], num_classes=6, smoothing=0.7)
    cw_3rd = compute_class_weights(y3_are[train_idx], num_classes=6, smoothing=0.7)
    logger.info(f"1着クラス重み (5cls): {['%.2f' % w for w in cw_1st.tolist()]}")

    # データセット
    train_ds = TensorDataset(
        torch.FloatTensor(X_are[train_idx]),
        torch.LongTensor(y1_remap[train_idx]),
        torch.LongTensor(y2_are[train_idx]),
        torch.LongTensor(y3_are[train_idx]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_are[val_idx]),
        torch.LongTensor(y1_remap[val_idx]),
        torch.LongTensor(y2_are[val_idx]),
        torch.LongTensor(y3_are[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    device = torch.device('cpu')
    input_dim = X_are.shape[1]
    hidden_dims = [512, 256, 128] if input_dim > 50 else [256, 128, 64]

    # 1着ヘッド: 5クラス, 2着/3着ヘッド: 6クラス（1号艇が2-3着に来ることがある）
    model = BoatraceMultiTaskModel(
        input_dim=input_dim, hidden_dims=hidden_dims,
        num_boats=5, dropout=args.dropout,
    ).to(device)

    # 2着/3着ヘッドを6クラスに差し替え
    model.head_2nd = torch.nn.Linear(hidden_dims[-1], 6).to(device)
    model.head_3rd = torch.nn.Linear(hidden_dims[-1], 6).to(device)
    model.num_boats = 5  # 1着用

    from src.models import FocalLoss

    criterion_1st = FocalLoss(gamma=args.focal_gamma, class_weights=cw_1st.to(device),
                              label_smoothing=args.label_smoothing)
    criterion_2nd = FocalLoss(gamma=args.focal_gamma, class_weights=cw_2nd.to(device))
    criterion_3rd = FocalLoss(gamma=args.focal_gamma, class_weights=cw_3rd.to(device))
    task_weights = [1.0, 0.7, 0.5]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.7
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for bx, by1, by2, by3 in train_loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            out_1st, out_2nd, out_3rd = model(bx)
            loss = (task_weights[0] * criterion_1st(out_1st, by1.to(device)) +
                    task_weights[1] * criterion_2nd(out_2nd, by2.to(device)) +
                    task_weights[2] * criterion_3rd(out_3rd, by3.to(device)))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct_1st = 0
        total = 0
        with torch.no_grad():
            for bx, by1, by2, by3 in val_loader:
                bx = bx.to(device)
                out_1st, out_2nd, out_3rd = model(bx)
                loss = (task_weights[0] * criterion_1st(out_1st, by1.to(device)) +
                        task_weights[1] * criterion_2nd(out_2nd, by2.to(device)) +
                        task_weights[2] * criterion_3rd(out_3rd, by3.to(device)))
                val_loss += loss.item()
                pred = out_1st.argmax(dim=1)
                correct_1st += (pred == by1).sum().item()
                total += by1.size(0)
        val_loss /= len(val_loader)
        val_acc = correct_1st / total * 100
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}: train={train_loss:.4f} "
                        f"val={val_loss:.4f} acc_1st={val_acc:.1f}% lr={lr_now:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, path=MODEL_PATH, metadata={
                'version': 'v5_are_specialist',
                'model_type': 'are',
                'num_classes_1st': 5,
                'num_classes_2nd3rd': 6,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'val_loss': val_loss,
                'val_acc_1st': val_acc,
                'label_smoothing': args.label_smoothing,
                'focal_gamma': args.focal_gamma,
                'boat_mapping': '2号艇→0, 3号艇→1, 4号艇→2, 5号艇→3, 6号艇→4',
            })
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early Stop epoch {epoch+1}, "
                            f"best_val={best_val_loss:.4f}, acc={val_acc:.1f}%")
                break

    logger.info(f"\n=== 訓練完了 ===")
    logger.info(f"  best_val_loss={best_val_loss:.4f}")
    logger.info(f"  保存先: {MODEL_PATH}")
    logger.info(f"  スケーラー: {SCALER_PATH}")


if __name__ == '__main__':
    main()
