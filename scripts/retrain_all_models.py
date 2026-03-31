"""全モデル一括再学習スクリプト v3: StandardScaler正規化 + 時系列分割

4つのアンサンブルモデルを異なる Focal Loss gamma で訓練:
  - boatrace_model.pth:      gamma=2.0 (標準)
  - boatrace_model_s05.pth:  gamma=1.5 (マイルド)
  - boatrace_model_s07.pth:  gamma=2.5 (やや強め)
  - boatrace_model_s085.pth: gamma=3.0 (アグレッシブ)

v3変更点:
  - StandardScaler正規化 (train_model.pyと共通スケーラー)
  - 時系列分割 (ランダム→時系列順80/20)

使い方:
    DATABASE_URL=xxx python scripts/retrain_all_models.py
    DATABASE_URL=xxx python scripts/retrain_all_models.py --epochs 120 --patience 20
"""
import sys
import os
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss, save_model
from scripts.train_model import load_training_data_fast, compute_class_weights, SCALER_PATH

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
                    weight_smoothing_1st=0.3,
                    weight_smoothing_2nd3rd=0.7,
                    label_smoothing=0.1):
    """1モデルを訓練して保存"""
    logger.info(f"=== {save_path} (gamma={gamma}, {label}) ===")

    # 1着: 逆頻度クラス重み (1号艇バイアス対策)
    cw_1st = compute_class_weights(y1_train, smoothing=weight_smoothing_1st)
    # 2着/3着: 軽い補正
    cw_2nd = compute_class_weights(y2_train, smoothing=weight_smoothing_2nd3rd)
    cw_3rd = compute_class_weights(y3_train, smoothing=weight_smoothing_2nd3rd)

    logger.info(f"  1着: 逆頻度 smoothing={weight_smoothing_1st}, "
                f"label_smoothing={label_smoothing}")
    logger.info(f"  class_weights_1st: {['%.2f' % w for w in cw_1st.tolist()]}")
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
        class_weights_1st=cw_1st.to(device),
        class_weights_2nd=cw_2nd.to(device),
        class_weights_3rd=cw_3rd.to(device),
        gamma=gamma,
        label_smoothing_1st=label_smoothing,
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
                'class_weights_1st': f'inverse_freq_s{weight_smoothing_1st}',
                'label_smoothing': label_smoothing,
                'weight_smoothing_2nd3rd': weight_smoothing_2nd3rd,
                'focal_gamma': gamma,
                'dropout': dropout,
                'version': 'v5_scaled_timesplit',
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
    parser.add_argument('--smoothing-1st', type=float, default=0.3,
                        help='1着クラス重みスムージング (0.3=強め)')
    parser.add_argument('--smoothing-2nd3rd', type=float, default=0.7)
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label Smoothing (1着ヘッド)')
    args = parser.parse_args()

    logger.info("=== 全モデル一括再学習 v3 (逆頻度重み + Label Smoothing) ===")
    logger.info(f"  lr={args.lr}, patience={args.patience}, dropout={args.dropout}")
    logger.info(f"  1st smoothing={args.smoothing_1st}, label_smoothing={args.label_smoothing}")

    # データ読み込み（1回だけ）
    X, y1, y2, y3 = load_training_data_fast()
    if X is None:
        logger.error("訓練データなし。collect_parallel.py でデータ収集してください。")
        return

    logger.info(f"データ: {len(X):,}件, 次元: {X.shape[1]}")

    # 時系列分割 (80/20) — データはrace_date昇順で取得済み
    n = len(X)
    split = int(n * 0.8)
    train_idx = list(range(0, split))
    val_idx = list(range(split, n))
    logger.info(f"時系列分割: 訓練={len(train_idx):,}件 (古い80%), "
                f"検証={len(val_idx):,}件 (新しい20%)")

    # StandardScaler: 訓練データでfit、検証データにはtransformのみ
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])

    # スケーラー保存（推論時に同じ変換を適用）
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"StandardScaler保存: {SCALER_PATH}")
    logger.info(f"  mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    logger.info(f"  std range:  [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")

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
            weight_smoothing_1st=args.smoothing_1st,
            weight_smoothing_2nd3rd=args.smoothing_2nd3rd,
            label_smoothing=args.label_smoothing,
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

    # === Isotonic Regression キャリブレーション ===
    # メインモデルの検証データ予測でキャリブレータをfit
    logger.info("\n=== Isotonic Regression キャリブレーション ===")
    from sklearn.isotonic import IsotonicRegression
    from src.models import load_model

    device = torch.device('cpu')
    main_model = load_model(MODELS[0]['path'], device)

    X_val_tensor = torch.FloatTensor(X_val)
    with torch.no_grad():
        out_1st, out_2nd, out_3rd = main_model(X_val_tensor)

    probs_1st = torch.softmax(out_1st, dim=1).numpy()
    probs_2nd = torch.softmax(out_2nd, dim=1).numpy()
    probs_3rd = torch.softmax(out_3rd, dim=1).numpy()

    # 各クラスごとにIsotonic Regressionをfit（1着/2着/3着 × 6クラス = 18個）
    calibrators = {'1st': [], '2nd': [], '3rd': []}
    for pos_name, probs_pos, y_pos in [
        ('1st', probs_1st, y1_val),
        ('2nd', probs_2nd, y2_val),
        ('3rd', probs_3rd, y3_val),
    ]:
        for cls_idx in range(6):
            # 二値化: このクラスかどうか
            y_binary = (y_pos == cls_idx).astype(float)
            p_cls = probs_pos[:, cls_idx]

            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(p_cls, y_binary)
            calibrators[pos_name].append(iso)

            # キャリブレーション効果: 平均予測確率 vs 実際の頻度
            mean_pred = p_cls.mean()
            actual_freq = y_binary.mean()
            cal_pred = iso.predict(p_cls).mean()
            logger.info(
                f"  {pos_name} {cls_idx+1}号艇: "
                f"モデル平均={mean_pred:.4f}, 実際={actual_freq:.4f}, "
                f"補正後={cal_pred:.4f}"
            )

    # キャリブレータ保存
    calibrator_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'models', 'calibrators.pkl'
    )
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrators, f)
    logger.info(f"キャリブレータ保存: {calibrator_path}")


if __name__ == '__main__':
    main()
