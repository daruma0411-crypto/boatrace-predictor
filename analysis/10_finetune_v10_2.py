"""V10.2: V10 pth を baseline に Miss Analysis weight で fine-tune

V10 (models/boatrace_model.pth) の学習済み重みを**起点**として、
Miss Analysis の示唆 (weight_preset='mild') を sample weight で乗せて
低 lr で追加訓練する。V10 の 34,712件分の知識を保持したまま、
会場・R番号・風速のバイアスを調整する。

保存先: analysis/models_v11/v10_2/
  - boatrace_model_v10_2.pth
  - feature_scaler_v10_2.pkl (V10 と同一をコピー)

V10 側の models/ は一切変更しない (READ-ONLY)。
"""
import os
import sys
import json
import pickle
import logging
import shutil
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

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_MODEL_PATH = Path("models/boatrace_model.pth")       # READ-ONLY
V10_SCALER_PATH = Path("models/feature_scaler.pkl")      # READ-ONLY

OUT_DIR = Path(__file__).parent / "models_v11" / "v10_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).parent / "reports"

MODEL_PATH = OUT_DIR / "boatrace_model_v10_2.pth"
SCALER_PATH = OUT_DIR / "feature_scaler_v10_2.pkl"
LOG_PATH = REPORT_DIR / "10_v10_2_training.json"

# Miss Analysis mild preset
WEAK_VENUES = [1, 2, 3, 4, 5, 6]
STRONG_VENUES = [9, 10, 12, 20, 23]
WEIGHT_PRESET = {
    'weak_venue': 0.7, 'r1': 0.5, 'weak_wind': 1.2,
    'strong_venue': 2.0, 'strong_wind': 1.5,
}


def compute_sample_weight(race):
    w = 1.0
    p = WEIGHT_PRESET
    vid = race.get('venue_id')
    if vid in WEAK_VENUES: w *= p['weak_venue']
    elif vid in STRONG_VENUES: w *= p['strong_venue']
    if race.get('race_number') == 1: w *= p['r1']
    wind = race.get('wind_speed')
    if wind is not None:
        if 3 <= wind <= 5: w *= p['strong_wind']
        elif 0 <= wind <= 2: w *= p['weak_wind']
    return max(w, 0.1)


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


def load_training_data():
    fe = FeatureEngineer()
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
              AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date ASC, r.id ASC
        """)
        races = cur.fetchall()
        logger.info(f"レース: {len(races):,}件")
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

    boats_by = defaultdict(list)
    for b in all_boats:
        boats_by[b['race_id']].append(dict(b))

    X, y1, y2, y3, wts = [], [], [], [], []
    for race in races:
        boats = boats_by.get(race['id'], [])
        if len(boats) != 6: continue
        rd = {'venue_id': race['venue_id'], 'month': race['race_date'].month,
              'distance': 1800,
              'wind_speed': race.get('wind_speed') or 0,
              'wind_direction': race.get('wind_direction') or 'calm',
              'temperature': race.get('temperature') or 20,
              'wave_height': race.get('wave_height') or 0,
              'water_temperature': race.get('water_temperature') or 20}
        try:
            f = fe.transform(rd, boats)
        except Exception: continue
        X.append(f)
        y1.append(race['result_1st'] - 1)
        y2.append(race['result_2nd'] - 1)
        y3.append(race['result_3rd'] - 1)
        wts.append(compute_sample_weight(race))

    X = np.array(X, dtype=np.float32)
    y1 = np.array(y1, dtype=np.int64)
    y2 = np.array(y2, dtype=np.int64)
    y3 = np.array(y3, dtype=np.int64)
    wts = np.array(wts, dtype=np.float32)
    logger.info(f"有効: {len(X)}件 {X.shape[1]}dim weight mean={wts.mean():.2f}")
    return X, y1, y2, y3, wts


def finetune(epochs=20, batch_size=256, lr=0.0001, patience=8,
             focal_gamma=2.0, label_smoothing=0.1):
    """V10 pth baseline で fine-tune"""
    logger.info("=== V10.2 Fine-tune 開始 ===")
    logger.info(f"  baseline: {V10_MODEL_PATH}")
    logger.info(f"  fine-tune lr={lr} patience={patience} epochs={epochs}")

    # V10 scaler を流用（そのままコピー）
    shutil.copy(V10_SCALER_PATH, SCALER_PATH)
    logger.info(f"  scaler流用: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # V10 model をロード (baseline)
    v10_state = torch.load(V10_MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=v10_state['input_dim'],
        hidden_dims=v10_state['hidden_dims'],
        num_boats=v10_state['num_boats'],
        dropout=v10_state['dropout'],
    )
    model.load_state_dict(v10_state['model_state_dict'])
    logger.info(f"  V10 重みロード済み (val_acc_1st={v10_state['metadata'].get('val_acc_1st', 0):.1f}%)")

    # データ
    X, y1, y2, y3, wts = load_training_data()
    X_scaled = scaler.transform(X).astype(np.float32)

    # 時系列80/20 分割
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
    logger.info(f"  device: {device}")

    l1 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw1.to(device),
                                  label_smoothing=label_smoothing).to(device)
    l2 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw2.to(device)).to(device)
    l3 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw3.to(device)).to(device)
    tw = [1.0, 0.7, 0.5]

    # 低 lr (fine-tune) + weight decay で overfit 抑制
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    # === V10 ベースライン val_loss を先に測定 ===
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

    best_val = base_val  # V10 の val_loss より低下したら採用
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
                    'val_acc_1st': val_acc, 'version': 'v10_2_finetuned',
                    'baseline': 'v10', 'v10_baseline_val_loss': base_val,
                    'weight_preset': 'mild',
                },
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stop: epoch {epoch}")
                break

    logger.info("")
    logger.info("=== 結果 ===")
    logger.info(f"  V10 baseline: val_loss={base_val:.4f} val_acc_1st={base_acc:.1f}%")
    logger.info(f"  V10.2 best:   val_loss={best_val:.4f} val_acc_1st={best_acc:.1f}%")
    if saved_any:
        logger.info(f"  → V10 baseline より改善（{best_val-base_val:+.4f}）")
        logger.info(f"  → 保存: {MODEL_PATH}")
    else:
        logger.info(f"  → V10 baseline を超えなかった（保存せず）")

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'trained_at': datetime.now().isoformat(),
                   'baseline_val_loss': base_val,
                   'baseline_val_acc_1st': base_acc,
                   'best_val_loss': best_val,
                   'best_val_acc_1st': best_acc,
                   'improved': saved_any,
                   'history': history}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    finetune()
