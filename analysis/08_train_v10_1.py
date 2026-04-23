"""V10.1 訓練: V10 NNアーキテクチャ継承 + Miss Analysis sample weight

V10 (models/boatrace_model.pth) は一切変更せず、V10と同じ構造で
追加情報（Miss Analysis ベースのsample weight）を与えて再学習する。

保存先: analysis/models_v11/v10_1/
  - boatrace_model_v10_1.pth  (NN weights)
  - feature_scaler_v10_1.pkl   (StandardScaler)
  - training_log.json          (メトリクス)

V10 の models/ 配下には一切書き込まない。
"""
import os
import sys
import json
import pickle
import math
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "models_v11" / "v10_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).parent / "reports"

MODEL_PATH = OUT_DIR / "boatrace_model_v10_1.pth"
SCALER_PATH = OUT_DIR / "feature_scaler_v10_1.pkl"
LOG_PATH = REPORT_DIR / "08_v10_1_training.json"

# === Miss Analysis 重み (mild preset: 04-22 backtest で最良の設定) ===
WEAK_VENUES = [1, 2, 3, 4, 5, 6]
STRONG_VENUES = [9, 10, 12, 20, 23]
WEAK_WIND = (0, 2)
STRONG_WIND = (3, 5)

WEIGHT_PRESET_MILD = {
    'weak_venue': 0.7, 'r1': 0.5, 'weak_wind': 1.2,
    'strong_venue': 2.0, 'strong_wind': 1.5,
}


def compute_sample_weight(race):
    """Miss Analysis ベースのサンプル重み"""
    w = 1.0
    p = WEIGHT_PRESET_MILD
    vid = race.get('venue_id')
    if vid in WEAK_VENUES:
        w *= p['weak_venue']
    elif vid in STRONG_VENUES:
        w *= p['strong_venue']
    if race.get('race_number') == 1:
        w *= p['r1']
    wind = race.get('wind_speed')
    if wind is not None:
        if STRONG_WIND[0] <= wind <= STRONG_WIND[1]:
            w *= p['strong_wind']
        elif WEAK_WIND[0] <= wind <= WEAK_WIND[1]:
            w *= p['weak_wind']
    return max(w, 0.1)


class SampleWeightedFocalLoss(nn.Module):
    """FocalLoss + Label Smoothing + per-sample weight サポート版

    src.models.FocalLoss の拡張。reduction='none' で per-sample loss を計算し、
    sample_weight を適用してから平均を取る。
    """
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else None
        )

    def forward(self, logits, targets, sample_weights=None):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            targets_smooth = F.one_hot(targets, num_classes=num_classes).float()
            targets_smooth = (1.0 - self.label_smoothing) * targets_smooth + \
                             self.label_smoothing / num_classes
        else:
            targets_smooth = F.one_hot(targets, num_classes=num_classes).float()

        pt = (probs * targets_smooth).sum(dim=1)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.class_weights is not None:
            alpha = self.class_weights[targets]
        else:
            alpha = 1.0

        loss_per_sample = -(targets_smooth * log_probs).sum(dim=1)
        loss = alpha * focal_weight * loss_per_sample

        if sample_weights is not None:
            # weighted mean
            return (loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        return loss.mean()


def compute_class_weights(labels, num_classes=6, smoothing=0.7):
    """V10同等の逆頻度クラス重み"""
    counts = Counter(labels.tolist())
    total = len(labels)
    raw_weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        raw_weights.append(total / (num_classes * count))
    raw_weights = np.array(raw_weights, dtype=np.float32)
    weights = np.power(raw_weights, smoothing)
    weights = weights / weights.mean()  # 平均を1に正規化
    return torch.tensor(weights, dtype=torch.float32)


def load_training_data():
    """V10 の load_training_data_fast と同等のデータ取得 + race情報も保持

    Returns:
        X, y1, y2, y3, weights, races_meta
    """
    feature_engineer = FeatureEngineer()
    logger.info("訓練データ取得中...")

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date, r.race_number,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.is_finished = true
              AND r.actual_result_trifecta IS NOT NULL
              AND r.result_1st IS NOT NULL
              AND r.result_2nd IS NOT NULL
              AND r.result_3rd IS NOT NULL
              AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date ASC, r.id ASC
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
                   approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()
        logger.info(f"ボート取得: {len(all_boats):,}件")

    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    X_list, y1_list, y2_list, y3_list = [], [], [], []
    weights_list = []
    race_ids_list = []

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
        except Exception:
            continue
        X_list.append(features)
        y1_list.append(race['result_1st'] - 1)
        y2_list.append(race['result_2nd'] - 1)
        y3_list.append(race['result_3rd'] - 1)
        weights_list.append(compute_sample_weight(race))
        race_ids_list.append(race['id'])

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int64)
    y2 = np.array(y2_list, dtype=np.int64)
    y3 = np.array(y3_list, dtype=np.int64)
    weights = np.array(weights_list, dtype=np.float32)

    logger.info(f"訓練データ: {len(X):,}件, {X.shape[1]}次元")
    logger.info(f"weight統計: min={weights.min():.3f} "
                f"max={weights.max():.3f} mean={weights.mean():.3f} "
                f"std={weights.std():.3f}")
    return X, y1, y2, y3, weights, race_ids_list


def train(epochs=100, batch_size=256, lr=0.0005, patience=15,
          weight_smoothing_1st=0.3, weight_smoothing_2nd3rd=0.7,
          focal_gamma=2.0, dropout=0.15, label_smoothing=0.1):
    """V10.1 訓練（V10と同じハイパラ、sample_weight のみ追加）"""
    logger.info("=== V10.1 訓練開始 ===")
    logger.info(f"preset: MILD (Miss Analysis mild weights)")
    logger.info(f"  lr={lr} patience={patience} focal_gamma={focal_gamma} "
                f"dropout={dropout} label_smoothing={label_smoothing}")

    X, y1, y2, y3, weights, race_ids = load_training_data()

    # クラス重み
    cw_1st = compute_class_weights(y1, smoothing=weight_smoothing_1st)
    cw_2nd = compute_class_weights(y2, smoothing=weight_smoothing_2nd3rd)
    cw_3rd = compute_class_weights(y3, smoothing=weight_smoothing_2nd3rd)
    logger.info(f"1着 class_w: {['%.2f' % w for w in cw_1st.tolist()]}")

    # 時系列 80/20 分割
    n = len(X)
    split = int(n * 0.8)

    # Scaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[:split] = scaler.fit_transform(X_scaled[:split])
    X_scaled[split:] = scaler.transform(X_scaled[split:])

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler保存: {SCALER_PATH}")

    # Tensor化
    X_tr = torch.FloatTensor(X_scaled[:split])
    X_va = torch.FloatTensor(X_scaled[split:])
    y1_tr, y2_tr, y3_tr = torch.LongTensor(y1[:split]), torch.LongTensor(y2[:split]), torch.LongTensor(y3[:split])
    y1_va, y2_va, y3_va = torch.LongTensor(y1[split:]), torch.LongTensor(y2[split:]), torch.LongTensor(y3[split:])
    w_tr = torch.FloatTensor(weights[:split])
    w_va = torch.FloatTensor(weights[split:])

    train_ds = TensorDataset(X_tr, y1_tr, y2_tr, y3_tr, w_tr)
    val_ds = TensorDataset(X_va, y1_va, y2_va, y3_va, w_va)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    input_dim = X.shape[1]
    hidden_dims = [256, 128, 64] if input_dim <= 50 else [512, 256, 128]
    # V10 と同じ構造
    if input_dim == 76:
        hidden_dims = [256, 128, 64]
    logger.info(f"モデル: input={input_dim}, hidden={hidden_dims}")
    model = BoatraceMultiTaskModel(input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    dropout=dropout).to(device)

    # weighted FocalLoss
    loss_1st = SampleWeightedFocalLoss(gamma=focal_gamma,
                                        class_weights=cw_1st.to(device),
                                        label_smoothing=label_smoothing).to(device)
    loss_2nd = SampleWeightedFocalLoss(gamma=focal_gamma,
                                        class_weights=cw_2nd.to(device)).to(device)
    loss_3rd = SampleWeightedFocalLoss(gamma=focal_gamma,
                                        class_weights=cw_3rd.to(device)).to(device)
    task_weights = [1.0, 0.7, 0.5]  # V10と同じ

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.7
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for bx, by1, by2, by3, bw in train_loader:
            bx, by1, by2, by3, bw = bx.to(device), by1.to(device), by2.to(device), by3.to(device), bw.to(device)
            optimizer.zero_grad()
            out = model(bx)
            l1 = loss_1st(out[0], by1, bw)
            l2 = loss_2nd(out[1], by2, bw)
            l3 = loss_3rd(out[2], by3, bw)
            loss = task_weights[0]*l1 + task_weights[1]*l2 + task_weights[2]*l3
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        correct_1st = 0; total = 0
        with torch.no_grad():
            for bx, by1, by2, by3, bw in val_loader:
                bx, by1, by2, by3, bw = bx.to(device), by1.to(device), by2.to(device), by3.to(device), bw.to(device)
                out = model(bx)
                l1 = loss_1st(out[0], by1, bw)
                l2 = loss_2nd(out[1], by2, bw)
                l3 = loss_3rd(out[2], by3, bw)
                loss = task_weights[0]*l1 + task_weights[1]*l2 + task_weights[2]*l3
                val_loss += loss.item()
                pred = out[0].argmax(dim=1)
                correct_1st += (pred == by1).sum().item()
                total += by1.size(0)
        val_loss /= len(val_loader)
        val_acc = correct_1st / total * 100
        scheduler.step(val_loss)

        history.append({
            'epoch': epoch+1, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc_1st': val_acc
        })

        if (epoch+1) % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} val_acc_1st={val_acc:.1f}% lr={lr_now:.6f}")

        # Early stopping + save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            state = {
                'model_state_dict': model.state_dict(),
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims,
                'num_boats': model.num_boats,
                'dropout': model.dropout,
                'metadata': {
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc_1st': val_acc,
                    'version': 'v10_1_miss_weighted',
                    'weight_preset': 'mild',
                },
            }
            torch.save(state, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping: epoch {epoch+1}, "
                            f"best_val_loss={best_val_loss:.4f}")
                break

    # Summary
    final_log = {
        'trained_at': datetime.now().isoformat(),
        'version': 'v10_1_miss_weighted',
        'total_samples': n,
        'train_samples': split,
        'val_samples': n - split,
        'hyperparams': {
            'lr': lr, 'batch_size': batch_size, 'focal_gamma': focal_gamma,
            'dropout': dropout, 'label_smoothing': label_smoothing,
            'weight_preset': 'mild',
        },
        'best_val_loss': best_val_loss,
        'history': history,
    }
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_log, f, ensure_ascii=False, indent=2)
    logger.info(f"ログ保存: {LOG_PATH}")
    logger.info(f"モデル: {MODEL_PATH}")


if __name__ == '__main__':
    train()
