"""V10.2 変種グリッド fine-tune (X3)

V10 pth baseline を起点に、8変種を fine-tune する。
探索軸: weight_preset × focal_gamma × label_smoothing × fine-tune epochs

保存先: analysis/models_v11/v10_2_variants/<name>/
"""
import os
import sys
import json
import pickle
import shutil
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

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_MODEL_PATH = Path("models/boatrace_model.pth")
V10_SCALER_PATH = Path("models/feature_scaler.pkl")
OUT_ROOT = Path(__file__).parent / "models_v11" / "v10_2_variants"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).parent / "reports"

# Miss Analysis weight presets
WEAK_VENUES = [1, 2, 3, 4, 5, 6]
STRONG_VENUES = [9, 10, 12, 20, 23]
WEIGHT_PRESETS = {
    'none':    {'weak_venue': 1.0, 'r1': 1.0, 'weak_wind': 1.0, 'strong_venue': 1.0, 'strong_wind': 1.0},
    'mild':    {'weak_venue': 0.7, 'r1': 0.5, 'weak_wind': 1.2, 'strong_venue': 2.0, 'strong_wind': 1.5},
    'strong':  {'weak_venue': 0.3, 'r1': 0.3, 'weak_wind': 1.8, 'strong_venue': 3.0, 'strong_wind': 2.0},
}

# 変種定義 (的中率UP狙い)
VARIANTS = [
    # weight 軸
    {'name': 'v10_2_none',     'weight_preset': 'none',   'focal_gamma': 2.0, 'label_smoothing': 0.1, 'lr': 0.0001, 'epochs': 20},
    {'name': 'v10_2_mild',     'weight_preset': 'mild',   'focal_gamma': 2.0, 'label_smoothing': 0.1, 'lr': 0.0001, 'epochs': 20},  # 既存V10.2
    {'name': 'v10_2_strong',   'weight_preset': 'strong', 'focal_gamma': 2.0, 'label_smoothing': 0.1, 'lr': 0.0001, 'epochs': 20},
    # gamma 軸 (Focal強度)
    {'name': 'v10_2_gamma0',   'weight_preset': 'mild',   'focal_gamma': 0.0, 'label_smoothing': 0.1, 'lr': 0.0001, 'epochs': 20},  # CE相当
    {'name': 'v10_2_gamma3',   'weight_preset': 'mild',   'focal_gamma': 3.0, 'label_smoothing': 0.1, 'lr': 0.0001, 'epochs': 20},
    # label_smoothing 軸
    {'name': 'v10_2_ls0',      'weight_preset': 'mild',   'focal_gamma': 2.0, 'label_smoothing': 0.0, 'lr': 0.0001, 'epochs': 20},
    # lr 軸
    {'name': 'v10_2_lr_hi',    'weight_preset': 'mild',   'focal_gamma': 2.0, 'label_smoothing': 0.1, 'lr': 0.0005, 'epochs': 20},
    # epochs 軸 (学習深く)
    {'name': 'v10_2_long',     'weight_preset': 'mild',   'focal_gamma': 2.0, 'label_smoothing': 0.1, 'lr': 0.00005, 'epochs': 50},
]


def compute_sample_weight(race, preset_name):
    p = WEIGHT_PRESETS[preset_name]
    w = 1.0
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


# グローバルキャッシュ（訓練データは全変種で共通）
_DATA_CACHE = None


def load_training_data():
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    fe = FeatureEngineer()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date, r.race_number,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.is_finished = true AND r.actual_result_trifecta IS NOT NULL
              AND r.result_1st IS NOT NULL AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date ASC, r.id ASC
        """)
        races = cur.fetchall()
        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3, local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3, boat_win_rate_2,
                   weight, exhibition_time, approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()

    boats_by = defaultdict(list)
    for b in all_boats:
        boats_by[b['race_id']].append(dict(b))

    X, y1, y2, y3, race_meta = [], [], [], [], []
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
        race_meta.append(dict(race))

    X = np.array(X, dtype=np.float32)
    y1 = np.array(y1, dtype=np.int64)
    y2 = np.array(y2, dtype=np.int64)
    y3 = np.array(y3, dtype=np.int64)
    _DATA_CACHE = (X, y1, y2, y3, race_meta)
    return _DATA_CACHE


def finetune_one(variant):
    name = variant['name']
    out_dir = OUT_ROOT / name
    out_dir.mkdir(exist_ok=True)
    logger.info(f"\n=== {name} ===")
    logger.info(f"  preset={variant['weight_preset']} gamma={variant['focal_gamma']} "
                f"ls={variant['label_smoothing']} lr={variant['lr']} epochs={variant['epochs']}")

    # scaler は V10 のをコピー（特徴量正規化は同一に保つ）
    scaler_path = out_dir / "feature_scaler.pkl"
    shutil.copy(V10_SCALER_PATH, scaler_path)
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)

    # V10 baseline load
    v10 = torch.load(V10_MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=v10['input_dim'], hidden_dims=v10['hidden_dims'],
        num_boats=v10['num_boats'], dropout=v10['dropout'])
    model.load_state_dict(v10['model_state_dict'])

    X, y1, y2, y3, race_meta = load_training_data()
    X_scaled = scaler.transform(X).astype(np.float32)
    wts = np.array([compute_sample_weight(r, variant['weight_preset'])
                    for r in race_meta], dtype=np.float32)

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

    train_loader = DataLoader(TensorDataset(X_tr, y1_tr, y2_tr, y3_tr, w_tr),
                              batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y1_va, y2_va, y3_va, w_va),
                            batch_size=256)

    device = torch.device('cpu')
    model = model.to(device)

    l1 = SampleWeightedFocalLoss(gamma=variant['focal_gamma'],
                                  class_weights=cw1.to(device),
                                  label_smoothing=variant['label_smoothing']).to(device)
    l2 = SampleWeightedFocalLoss(gamma=variant['focal_gamma'],
                                  class_weights=cw2.to(device)).to(device)
    l3 = SampleWeightedFocalLoss(gamma=variant['focal_gamma'],
                                  class_weights=cw3.to(device)).to(device)
    tw = [1.0, 0.7, 0.5]

    opt = torch.optim.Adam(model.parameters(), lr=variant['lr'], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    # Baseline measure
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

    best_val = base_val; best_acc = base_acc; patience = 0; saved = False

    for epoch in range(1, variant['epochs'] + 1):
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

        if val_loss < best_val:
            best_val = val_loss
            best_acc = val_acc
            patience = 0
            saved = True
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims,
                'num_boats': model.num_boats,
                'dropout': model.dropout,
                'metadata': {
                    'epoch': epoch, 'val_loss': val_loss, 'val_acc_1st': val_acc,
                    'variant': variant, 'baseline_val_loss': base_val,
                    'baseline_val_acc_1st': base_acc,
                },
            }, out_dir / 'model.pth')
        else:
            patience += 1
            if patience >= 8: break

    logger.info(f"  best: val_loss={best_val:.4f} val_acc={best_acc:.1f}% "
                f"(Δbase={best_val-base_val:+.4f} / {best_acc-base_acc:+.1f}pt) saved={saved}")
    return {
        'name': name, 'variant': variant,
        'baseline_val_loss': base_val, 'baseline_val_acc_1st': base_acc,
        'best_val_loss': best_val, 'best_val_acc_1st': best_acc,
        'saved': saved,
    }


def main():
    logger.info(f"X3 変種 fine-tune: {len(VARIANTS)}個")
    results = []
    for v in VARIANTS:
        try:
            results.append(finetune_one(v))
        except Exception as e:
            logger.error(f"{v['name']} 失敗: {e}")
            results.append({'name': v['name'], 'variant': v, 'error': str(e)})

    out = {'trained_at': datetime.now().isoformat(), 'variants': results}
    out_path = REPORT_DIR / "15_variants_training.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n=== 全変種完了 ===")
    logger.info(f"{'変種':<20} {'val_loss':>10} {'val_acc':>8} {'baseline':>10}")
    for r in results:
        if 'error' in r:
            logger.info(f"{r['name']:<20} FAILED")
        else:
            logger.info(f"{r['name']:<20} {r['best_val_loss']:>10.4f} "
                        f"{r['best_val_acc_1st']:>7.1f}% "
                        f"{r['baseline_val_loss']:>10.4f}")
    logger.info(f"\nレポート: {out_path}")


if __name__ == '__main__':
    main()
