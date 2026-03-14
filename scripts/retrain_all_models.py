"""全モデル一括再学習スクリプト v2: Focal Loss + 1着均等重み

4つのアンサンブルモデルを異なる Focal Loss gamma で訓練:
  - boatrace_model.pth:      gamma=2.0 (標準)
  - boatrace_model_s05.pth:  gamma=1.5 (マイルド)
  - boatrace_model_s07.pth:  gamma=2.5 (やや強め)
  - boatrace_model_s085.pth: gamma=3.0 (アグレッシブ)

v2変更点:
  - 1着: クラス重みなし (均等) → 3号艇バイアス解消
  - 2着/3着: smoothing=0.7 (軽い補正)
  - FocalLoss (gamma可変) → メリハリのある確率分布
  - Dropout: 0.15
  - lr=0.0005, patience=15, scheduler patience=8/factor=0.7

使い方:
    DATABASE_URL=xxx python scripts/retrain_all_models.py
    DATABASE_URL=xxx python scripts/retrain_all_models.py --epochs 120 --patience 20
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

# v2: gamma でアンサンブル多様性を確保
MODELS = [
    {'path': 'models/boatrace_model.pth', 'gamma': 2.0, 'label': '標準'},
    {'path': 'models/boatrace_model_s05.pth', 'gamma': 1.5, 'label': 'マイルド'},
    {'path': 'models/boatrace_model_s07.pth', 'gamma': 2.5, 'label': 'やや強め'},
    {'path': 'models/boatrace_model_s085.pth', 'gamma': 3.0, 'label': 'アグレッシブ'},
]


def train_one_model(X_train, y1_train, y2_train, y3_train,
                    X_val, y1_val, y2_val, y3_val,
                    gamma, save_path, label='',
                    epochs=100, batch_size=256, lr=0.0005,
                    patience=15, dropout=0.15,
                    weight_smoothing_2nd3rd=0.7):
    """1モデルを訓練して保存"""
    logger.info(f"=== {save_path} (gamma={gamma}, {label}) ===")

    # 1着: クラス重みなし (均等)
    cw_1st = None
    # 2着/3着: 軽い補正
    cw_2nd = compute_class_weights(y2_train, smoothing=weight_smoothing_2nd3rd)
    cw_3rd = compute_class_weights(y3_train, smoothing=weight_smoothing_2nd3rd)

    logger.info(f"  1着: 均等重み, 2着/3着: smoothing={weight_smoothing_2nd3rd}")
    logger.info(f"  class_weights_2nd: {['%.2f' % w for w in cw_2nd.tolist()]}")

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
    # v3: NN縮小 (43次元入力には[512,256,128]は過大)
    if input_dim <= 50:
        hidden_dims = [256, 128, 64]
    else:
        hidden_dims = [512, 256, 128]
    logger.info(f"  モデル構成: input={input_dim}, hidden={hidden_dims}")
    model = BoatraceMultiTaskModel(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
    ).to(device)
    criterion = BoatraceMultiTaskLoss(
        class_weights_1st=None,
        class_weights_2nd=cw_2nd.to(device),
        class_weights_3rd=cw_3rd.to(device),
        gamma=gamma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.7
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
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
        correct_1st = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y1, batch_y2, batch_y3 in val_loader:
                batch_x = batch_x.to(device)
                targets = (batch_y1.to(device), batch_y2.to(device), batch_y3.to(device))
                outputs = model(batch_x)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                pred_1st = outputs[0].argmax(dim=1)
                correct_1st += (pred_1st == targets[0]).sum().item()
                total += targets[0].size(0)
        val_loss /= len(val_loader)
        val_acc = correct_1st / total * 100
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_model(model, path=save_path, metadata={
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_1st': val_acc,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'class_weights_1st': 'uniform',
                'weight_smoothing_2nd3rd': weight_smoothing_2nd3rd,
                'focal_gamma': gamma,
                'dropout': dropout,
                'version': 'v3_feature_selection',
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early Stop epoch {epoch+1}, "
                            f"best_val={best_val_loss:.4f}, acc_1st={best_val_acc:.1f}%")
                break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Epoch {epoch+1}: train={train_loss:.4f} "
                        f"val={val_loss:.4f} acc_1st={val_acc:.1f}% lr={current_lr:.6f}")

    logger.info(f"  完了: best_val_loss={best_val_loss:.4f}, best_acc_1st={best_val_acc:.1f}%")
    return best_val_loss, best_val_acc


def main():
    import argparse
    parser = argparse.ArgumentParser(description='全モデル一括再学習 v2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--smoothing-2nd3rd', type=float, default=0.7)
    args = parser.parse_args()

    logger.info("=== 全モデル一括再学習 v2 (Focal Loss) ===")
    logger.info(f"  lr={args.lr}, patience={args.patience}, dropout={args.dropout}")

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
        val_loss, val_acc = train_one_model(
            X_train, y1_train, y2_train, y3_train,
            X_val, y1_val, y2_val, y3_val,
            gamma=m['gamma'],
            save_path=m['path'],
            label=m['label'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dropout=args.dropout,
            weight_smoothing_2nd3rd=args.smoothing_2nd3rd,
        )
        results.append({
            'path': m['path'], 'gamma': m['gamma'],
            'label': m['label'], 'val_loss': val_loss,
            'val_acc': val_acc,
        })

    logger.info("\n=== 全モデル再学習完了 ===")
    for r in results:
        logger.info(f"  {r['path']}: gamma={r['gamma']} ({r['label']}), "
                    f"val_loss={r['val_loss']:.4f}, acc_1st={r['val_acc']:.1f}%")

    # 次元確認
    for m in MODELS:
        cp = torch.load(m['path'], map_location='cpu', weights_only=False)
        meta = cp.get('metadata', {})
        logger.info(f"  {m['path']}: input_dim={cp.get('input_dim')}, "
                    f"dropout={cp.get('dropout')}, version={meta.get('version')}")


if __name__ == '__main__':
    main()
