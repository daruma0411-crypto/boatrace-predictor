"""戸田特化 NN warm-start fine-tune (案 X Option A、真の Option D 代替)

V10 (全国 model) を初期値、戸田 2271 races で継続学習。
全国知識を温存しつつ戸田特性を追加。

設計:
- 76dim features (V10 と同じ、特徴量蓄積を活用)
- V10 architecture [512, 256, 128]
- V10 weights を初期値 (warm start)
- Adam lr=5e-5 (V10 訓練時より低め、過学習防止)
- weight_decay=1e-4 (L2 reg)
- batch_size=32
- max 20 epoch、early stopping (patience=3, val loss)
- Train: 2025-06〜2026-03 (n=~1900)
- Val:   2026-04 (n=~170)
- Test (hold-out): 2026-05 (n=~135) — 訓練に絶対触らない

出力: models/boatrace_model_toda.pth + analysis/reports/63_toda_finetune.md
"""
import os
import sys
import pickle
import logging
import copy
from pathlib import Path
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import BoatraceMultiTaskModel, load_model, save_model
from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PRED_PATH = ROOT / 'analysis' / 'toda_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
MASK_PATH = ROOT / 'models' / 'feature_mask_208.npy'
V10_PATH = ROOT / 'models' / 'boatrace_model.pth'
OUT_MODEL_PATH = ROOT / 'models' / 'boatrace_model_toda.pth'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '63_toda_finetune.md'

VAL_START = date(2026, 4, 1)
TEST_START = date(2026, 5, 1)

# Hyperparameters
LR = 5e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_features(predictions):
    """戸田 predictions から (X, y_1st, y_2nd, y_3rd, dates) を構築.

    FeatureEngineer.TOTAL_DIM=76 (既に削減済み)、mask 適用は不要。
    """
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    Xs = []
    y1, y2, y3 = [], [], []
    dates = []
    n_err = 0
    for rid, p in predictions.items():
        try:
            features = fe.transform(p['race_data'], p['boats'])
            features = scaler.transform(features.reshape(1, -1)).flatten()
            Xs.append(features)
            y1.append(p['result_1st'] - 1)  # 1-indexed → 0-indexed
            y2.append(p['result_2nd'] - 1)
            y3.append(p['result_3rd'] - 1)
            dates.append(date.fromisoformat(p['race_date']))
        except Exception as e:
            n_err += 1
            if n_err < 3:
                logger.warning(f"feature build error {rid}: {e}")
    logger.info(f"feature build err: {n_err}")
    X = np.array(Xs, dtype=np.float32)
    y1 = np.array(y1, dtype=np.int64)
    y2 = np.array(y2, dtype=np.int64)
    y3 = np.array(y3, dtype=np.int64)
    return X, y1, y2, y3, dates


def split_by_date(X, y1, y2, y3, dates):
    """train (〜2026-03) / val (2026-04) / test (2026-05) 分割."""
    train_idx = [i for i, d in enumerate(dates) if d < VAL_START]
    val_idx = [i for i, d in enumerate(dates) if VAL_START <= d < TEST_START]
    test_idx = [i for i, d in enumerate(dates) if d >= TEST_START]
    return {
        'train': (X[train_idx], y1[train_idx], y2[train_idx], y3[train_idx]),
        'val':   (X[val_idx],   y1[val_idx],   y2[val_idx],   y3[val_idx]),
        'test':  (X[test_idx],  y1[test_idx],  y2[test_idx],  y3[test_idx]),
    }


def train(model, train_data, val_data, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    Xtr, y1tr, y2tr, y3tr = train_data
    Xv,  y1v,  y2v,  y3v  = val_data

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(y1tr),
                      torch.LongTensor(y2tr), torch.LongTensor(y3tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        tr_losses = []
        for X, y1, y2, y3 in train_loader:
            optimizer.zero_grad()
            o1, o2, o3 = model(X)
            loss = criterion(o1, y1) + criterion(o2, y2) + criterion(o3, y3)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            o1, o2, o3 = model(torch.FloatTensor(Xv))
            val_loss = (criterion(o1, torch.LongTensor(y1v)) +
                        criterion(o2, torch.LongTensor(y2v)) +
                        criterion(o3, torch.LongTensor(y3v))).item()
            # 1着 accuracy
            val_acc = (o1.argmax(dim=1).numpy() == y1v).mean()

        tr_loss_mean = float(np.mean(tr_losses))
        history.append({
            'epoch': epoch + 1, 'train_loss': tr_loss_mean,
            'val_loss': val_loss, 'val_acc': float(val_acc),
        })
        logger.info(f"epoch {epoch+1}: train_loss={tr_loss_mean:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        o1, o2, o3 = model(torch.FloatTensor(X))
        p1 = torch.softmax(o1, dim=1).numpy()
        p2 = torch.softmax(o2, dim=1).numpy()
        p3 = torch.softmax(o3, dim=1).numpy()
    return p1, p2, p3


def main():
    logger.info("戸田 NN warm-start fine-tune")
    with open(PRED_PATH, 'rb') as f:
        predictions = pickle.load(f)
    logger.info(f"戸田 predictions: {len(predictions)}")

    logger.info("Features 構築")
    X, y1, y2, y3, dates = build_features(predictions)
    logger.info(f"X shape: {X.shape}")

    splits = split_by_date(X, y1, y2, y3, dates)
    for k, v in splits.items():
        logger.info(f"{k}: {len(v[0])} races")

    # V10 warm-start
    logger.info(f"V10 weights load: {V10_PATH}")
    model = load_model(str(V10_PATH))
    model.train()  # 学習モード

    # V10 baseline (戸田 train での val accuracy)
    p1_v10, _, _ = predict_probs(model, splits['val'][0])
    v10_val_acc = (p1_v10.argmax(axis=1) == splits['val'][1]).mean()
    logger.info(f"V10 val (戸田 2026-04) 1着 acc: {v10_val_acc:.4f}")

    # Fine-tune
    logger.info(f"Fine-tune start (lr={LR}, weight_decay={WEIGHT_DECAY}, max_epochs={MAX_EPOCHS})")
    model, history, best_val = train(model, splits['train'], splits['val'])

    # 保存
    save_model(model, str(OUT_MODEL_PATH), metadata={
        'venue_id': 2,
        'venue_name': 'Toda',
        'train_period': '2025-06 〜 2026-03',
        'val_period': '2026-04',
        'n_train': len(splits['train'][0]),
        'n_val': len(splits['val'][0]),
        'base_model': 'V10 (boatrace_model.pth)',
        'hyperparams': {'lr': LR, 'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE, 'patience': PATIENCE},
        'history': history,
    })

    # Test (hold-out) evaluation
    Xte, y1te, y2te, y3te = splits['test']
    p1_toda, _, _ = predict_probs(model, Xte)
    # V10 baseline (戸田 test)
    v10 = load_model(str(V10_PATH))
    v10.eval()
    p1_v10_test, _, _ = predict_probs(v10, Xte)

    toda_acc = (p1_toda.argmax(axis=1) == y1te).mean()
    v10_acc = (p1_v10_test.argmax(axis=1) == y1te).mean()

    # 1着 NN-only top-1 accuracy / log-loss
    eps = 1e-8
    toda_logloss = -np.log(np.clip(p1_toda[np.arange(len(y1te)), y1te], eps, 1)).mean()
    v10_logloss = -np.log(np.clip(p1_v10_test[np.arange(len(y1te)), y1te], eps, 1)).mean()

    # boat-level mean prediction vs actual (calibration check)
    actual_by_boat = [(y1te == b).mean() * 100 for b in range(6)]
    v10_pred_by_boat = [p1_v10_test[:, b].mean() * 100 for b in range(6)]
    toda_pred_by_boat = [p1_toda[:, b].mean() * 100 for b in range(6)]

    # Report
    lines = []
    lines.append("# 戸田 NN warm-start fine-tune (案 X Option A)\n\n")
    lines.append(f"Train: 2025-06〜2026-03 (n={len(splits['train'][0])})\n")
    lines.append(f"Val:   2026-04 (n={len(splits['val'][0])})\n")
    lines.append(f"Test (hold-out): 2026-05 (n={len(splits['test'][0])})\n\n")
    lines.append(f"Hyperparameters: lr={LR}, weight_decay={WEIGHT_DECAY}, "
                 f"batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}\n\n")

    lines.append("## 訓練 history\n\n")
    lines.append("| epoch | train_loss | val_loss | val 1着 acc |\n|---|---|---|---|\n")
    for h in history:
        lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['val_acc']:.4f} |\n")
    lines.append(f"\n**best val_loss**: {best_val:.4f}\n")

    lines.append("\n## Test (hold-out 2026-05) NN-only metrics\n\n")
    lines.append("| model | 1着 top-1 acc | 1着 log-loss |\n|---|---|---|\n")
    lines.append(f"| V10 baseline | {v10_acc:.4f} | {v10_logloss:.4f} |\n")
    lines.append(f"| Toda fine-tune | {toda_acc:.4f} | {toda_logloss:.4f} |\n")
    lines.append(f"\n**改善幅**: acc {(toda_acc-v10_acc)*100:+.2f}pt, log-loss {toda_logloss-v10_logloss:+.4f}\n")

    lines.append("\n## Test 期間 boat-level calibration\n\n")
    lines.append("| boat | actual rate% | V10 mean pred% | Toda mean pred% | V10 bias | Toda bias |\n|---|---|---|---|---|---|\n")
    for b in range(6):
        v10_bias = v10_pred_by_boat[b] - actual_by_boat[b]
        toda_bias = toda_pred_by_boat[b] - actual_by_boat[b]
        lines.append(f"| {b+1} | {actual_by_boat[b]:.2f}% | {v10_pred_by_boat[b]:.2f}% | "
                     f"{toda_pred_by_boat[b]:.2f}% | {v10_bias:+.2f}pt | {toda_bias:+.2f}pt |\n")

    lines.append("\n## 判定 ヒント\n\n")
    lines.append(f"- V10 → Toda fine-tune で **1号艇 bias** が改善したか? (戸田は 1号艇 -10pt 構造)\n")
    lines.append(f"- 全 boat の log-loss が低下 → calibration 改善\n")
    lines.append(f"- 次は Toda model + QMC で backtest ROI 算出 (analysis/64)\n")
    lines.append(f"- 結論は岩下さんに委ね、shadow 並走必須\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート: {REPORT_PATH}")
    logger.info(f"モデル: {OUT_MODEL_PATH}")


if __name__ == '__main__':
    main()
