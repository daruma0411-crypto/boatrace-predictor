"""V10 を base に過去年4月の pkl データで fine-tune

10_finetune_v10_2.py の派生版。DB ではなく pkl から読み込む READ-ONLY。
extract_training_data_from_pkl.py で生成した train_data_*_04.pkl を使う。

入力: analysis/models_v11/v10_april_finetune/train_data_{years}_04.pkl
出力: analysis/models_v11/v10_april_finetune/boatrace_model_v10_april.pth

V10 側 (models/) は一切触らない。
"""
import os
import sys
import json
import pickle
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models import BoatraceMultiTaskModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_MODEL_PATH = Path("models/boatrace_model.pth")
V10_SCALER_PATH = Path("models/feature_scaler.pkl")

OUT_DIR = Path(__file__).parent / "models_v11" / "v10_april_finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "boatrace_model_v10_april.pth"
SCALER_PATH = OUT_DIR / "feature_scaler_v10_april.pkl"
LOG_PATH = REPORT_DIR / "finetune_v10_april.json"


class SampleWeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else None
        )

    def forward(self, logits, targets, sample_weights=None):
        nc = logits.size(1)
        lp = F.log_softmax(logits, dim=1)
        probs = torch.exp(lp)
        if self.label_smoothing > 0:
            ts = F.one_hot(targets, num_classes=nc).float()
            ts = (1 - self.label_smoothing) * ts + self.label_smoothing / nc
        else:
            ts = F.one_hot(targets, num_classes=nc).float()
        pt = (probs * ts).sum(dim=1)
        fw = (1 - pt) ** self.gamma
        alpha = self.class_weights[targets] if self.class_weights is not None else 1.0
        loss_ps = -(ts * lp).sum(dim=1)
        loss = alpha * fw * loss_ps
        if sample_weights is not None:
            return (loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        return loss.mean()


def compute_class_weights(labels, num_classes=6, smoothing=0.7):
    counts = Counter(labels.tolist())
    total = len(labels)
    raw = np.array([total / (num_classes * counts.get(i, 1))
                    for i in range(num_classes)], dtype=np.float32)
    w = np.power(raw, smoothing)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def load_pkl_training_data(train_data_path: Path):
    """extract_training_data_from_pkl.py の出力を読み込む"""
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y1 = data['y_1st'].astype(np.int64)
    y2 = data['y_2nd'].astype(np.int64)
    y3 = data['y_3rd'].astype(np.int64)
    wts = data['weights']
    logger.info(f"  読込: {train_data_path.name}")
    logger.info(f"  shape: X={X.shape} y1={y1.shape} weights mean={wts.mean():.2f}")
    return X, y1, y2, y3, wts, data.get('race_dates')


def finetune(train_data_path: Path, epochs=20, batch_size=128,
             lr=0.00005, patience=8, focal_gamma=2.0, label_smoothing=0.1):
    """V10 pth baseline で 4月データに fine-tune"""
    logger.info("=== V10 April Fine-tune 開始 ===")
    logger.info(f"  baseline: {V10_MODEL_PATH}")
    logger.info(f"  訓練データ: {train_data_path}")
    logger.info(f"  fine-tune lr={lr} patience={patience} epochs={epochs}")

    shutil.copy(V10_SCALER_PATH, SCALER_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    v10_state = torch.load(V10_MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=v10_state['input_dim'],
        hidden_dims=v10_state['hidden_dims'],
        num_boats=v10_state['num_boats'],
        dropout=v10_state['dropout'],
    )
    model.load_state_dict(v10_state['model_state_dict'])
    logger.info(f"  V10 重みロード済み (val_acc_1st={v10_state['metadata'].get('val_acc_1st', 0):.1f}%)")

    X, y1, y2, y3, wts, race_dates = load_pkl_training_data(train_data_path)
    if X.shape[1] != v10_state['input_dim']:
        raise ValueError(f"次元不一致: X={X.shape[1]} V10={v10_state['input_dim']}")

    X_scaled = scaler.transform(X).astype(np.float32)

    # 時系列80/20 分割（race_dates でソート前提）
    if race_dates is not None:
        order = np.argsort(race_dates)
        X_scaled, y1, y2, y3, wts = X_scaled[order], y1[order], y2[order], y3[order], wts[order]
    n = len(X_scaled)
    split = int(n * 0.8)

    cw1 = compute_class_weights(y1, smoothing=0.3)
    cw2 = compute_class_weights(y2, smoothing=0.7)
    cw3 = compute_class_weights(y3, smoothing=0.7)

    X_tr = torch.FloatTensor(X_scaled[:split])
    X_va = torch.FloatTensor(X_scaled[split:])
    y1_tr, y1_va = torch.LongTensor(y1[:split]), torch.LongTensor(y1[split:])
    y2_tr, y2_va = torch.LongTensor(y2[:split]), torch.LongTensor(y2[split:])
    y3_tr, y3_va = torch.LongTensor(y3[:split]), torch.LongTensor(y3[split:])
    w_tr = torch.FloatTensor(wts[:split])
    w_va = torch.FloatTensor(wts[split:])

    train_ds = TensorDataset(X_tr, y1_tr, y2_tr, y3_tr, w_tr)
    val_ds = TensorDataset(X_va, y1_va, y2_va, y3_va, w_va)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"  device: {device}, train={len(X_tr)} val={len(X_va)}")

    l1 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw1.to(device),
                                  label_smoothing=label_smoothing).to(device)
    l2 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw2.to(device)).to(device)
    l3 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw3.to(device)).to(device)
    tw = [1.0, 0.7, 0.5]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    # V10 baseline 計測
    model.eval()
    base_val = 0; base_correct = 0; base_total = 0
    with torch.no_grad():
        for bx, by1, by2, by3, bw in val_loader:
            bx, by1, by2, by3, bw = bx.to(device), by1.to(device), by2.to(device), by3.to(device), bw.to(device)
            out = model(bx)
            loss = tw[0]*l1(out[0], by1, bw) + tw[1]*l2(out[1], by2, bw) + tw[2]*l3(out[2], by3, bw)
            base_val += loss.item()
            pred = out[0].argmax(dim=1)
            base_correct += (pred == by1).sum().item()
            base_total += by1.size(0)
    base_val /= len(val_loader)
    base_acc = base_correct / base_total * 100
    logger.info(f"  V10 baseline val_loss={base_val:.4f} val_acc_1st={base_acc:.1f}%")

    best_val = base_val
    best_acc = base_acc
    patience_counter = 0
    history = [{'epoch': 0, 'val_loss': base_val, 'val_acc_1st': base_acc, 'note': 'V10 baseline'}]
    saved_any = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for bx, by1, by2, by3, bw in train_loader:
            bx, by1, by2, by3, bw = bx.to(device), by1.to(device), by2.to(device), by3.to(device), bw.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = tw[0]*l1(out[0], by1, bw) + tw[1]*l2(out[1], by2, bw) + tw[2]*l3(out[2], by3, bw)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0; correct = 0; total = 0
        with torch.no_grad():
            for bx, by1, by2, by3, bw in val_loader:
                bx, by1, by2, by3, bw = bx.to(device), by1.to(device), by2.to(device), by3.to(device), bw.to(device)
                out = model(bx)
                loss = tw[0]*l1(out[0], by1, bw) + tw[1]*l2(out[1], by2, bw) + tw[2]*l3(out[2], by3, bw)
                val_loss += loss.item()
                pred = out[0].argmax(dim=1)
                correct += (pred == by1).sum().item()
                total += by1.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total * 100
        sched.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                       'val_acc_1st': val_acc})

        lr_now = opt.param_groups[0]['lr']
        logger.info(f"Ep {epoch}/{epochs}: train={train_loss:.4f} val={val_loss:.4f} "
                    f"acc={val_acc:.1f}% (Δbase={val_loss-base_val:+.4f}) lr={lr_now:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_acc = val_acc
            patience_counter = 0
            saved_any = True
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims,
                'num_boats': model.num_boats,
                'dropout': model.dropout,
                'metadata': {
                    'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                    'val_acc_1st': val_acc, 'version': 'v10_april_finetuned',
                    'baseline': 'v10', 'v10_baseline_val_loss': base_val,
                    'train_data': str(train_data_path),
                },
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stop: epoch {epoch}")
                break

    logger.info("")
    logger.info("=== 結果 ===")
    logger.info(f"  V10 baseline:    val_loss={base_val:.4f} val_acc_1st={base_acc:.1f}%")
    logger.info(f"  V10 April best:  val_loss={best_val:.4f} val_acc_1st={best_acc:.1f}%")
    if saved_any:
        logger.info(f"  → V10 baseline より改善（{best_val-base_val:+.4f}）→ 保存: {MODEL_PATH}")
    else:
        logger.info(f"  → V10 baseline を超えなかった（保存せず）")

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'trained_at': datetime.now().isoformat(),
                   'train_data': str(train_data_path),
                   'baseline_val_loss': base_val,
                   'baseline_val_acc_1st': base_acc,
                   'best_val_loss': best_val,
                   'best_val_acc_1st': best_acc,
                   'improved': saved_any,
                   'history': history}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True,
                        help='extract_training_data_from_pkl.py の出力 pkl パス')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00005)
    args = parser.parse_args()
    finetune(Path(args.train_data), epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr)
