"""V10 → April-tuned: 過去年4月データ (2024-04 + 2025-04) で V10 を fine-tune

10_finetune_v10_2.py をベースに以下だけ改造:
  - load_training_data() を **pkl 直読み** に変更 (本番DBには触れない)
  - 入力: analysis/historical_data/2024_04/ + 2025_04/ の 4 種類 pkl
  - 出力先: analysis/models_v11/v10_april_finetune/

学習ロジック・loss・schedule は 10_finetune_v10_2.py と同一。
V10 (models/boatrace_model.pth) と V10 scaler は **READ-ONLY**。
"""
import os
import sys
import json
import pickle
import logging
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

V10_MODEL_PATH = Path("models/boatrace_model.pth")    # READ-ONLY
V10_SCALER_PATH = Path("models/feature_scaler.pkl")   # READ-ONLY

OUT_DIR = Path(__file__).parent / "models_v11" / "v10_april_finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "boatrace_model_april.pth"
SCALER_PATH = OUT_DIR / "feature_scaler_april.pkl"
LOG_PATH = REPORT_DIR / "18_april_finetune.json"

HIST_DIR = Path(__file__).parent / "historical_data"
DEFAULT_YEARS_MONTHS = [(2024, 4), (2025, 4)]

# Miss Analysis weight preset (10_finetune_v10_2 と同一)
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


def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_training_data_from_pkl(years_months):
    """過去年4月の pkl 4種を読み込んで、DB 版と同じ形式で返す。

    Returns:
      X (n,dim) np.float32
      y1, y2, y3 (n,) np.int64
      wts (n,) np.float32
    """
    fe = FeatureEngineer()
    races = []
    boats_by_key = {}    # key=(date,venue,race_number) → list[boat dict]
    weather_by_key = {}  # key → weather dict
    bi_by_key = {}       # key → beforeinfo boats merge 用 dict

    for year, month in years_months:
        d = HIST_DIR / f'{year}_{month:02d}'
        if not d.exists():
            logger.warning(f'shard 無し: {d}')
            continue

        racelist = _load_pkl(d / 'racelist.pkl')
        results = _load_pkl(d / 'result.pkl')
        beforeinfo = _load_pkl(d / 'beforeinfo.pkl')

        # beforeinfo の boats と weather を key で索引
        for r in beforeinfo:
            k = (str(r['race_date']), r['venue_id'], r['race_number'])
            weather_by_key[k] = r.get('weather', {}) or {}
            bi_by_key[k] = {b['boat_number']: b for b in r.get('boats', [])}

        # result を key で索引
        result_by_key = {(str(r['race_date']), r['venue_id'], r['race_number']): r
                         for r in results}

        # racelist を主軸に、result + beforeinfo を join
        for rl in racelist:
            k = (str(rl['race_date']), rl['venue_id'], rl['race_number'])
            res = result_by_key.get(k)
            if res is None:
                continue  # 中止レース等
            if not (res.get('result_1st') and res.get('result_2nd') and res.get('result_3rd')):
                continue

            # boats: racelist の選手情報 + beforeinfo の展示情報をマージ
            bi_boats = bi_by_key.get(k, {})
            merged_boats = []
            for b in rl['boats']:
                bn = b['boat_number']
                bi = bi_boats.get(bn, {})
                merged = {**b}
                # beforeinfo 上書き (当日体重・展示タイム・チルト)
                if bi.get('weight') is not None:
                    merged['weight'] = bi['weight']
                merged['exhibition_time'] = bi.get('exhibition_time')
                merged['tilt'] = bi.get('tilt')
                merged['approach_course'] = bi.get('approach_course')
                merged['parts_changed'] = bi.get('parts_changed', False)
                merged['is_new_motor'] = False  # pkl にない情報、デフォルト
                merged_boats.append(merged)
            if len(merged_boats) != 6:
                continue

            weather = weather_by_key.get(k, {})
            races.append({
                'race_date': rl['race_date'],
                'venue_id': rl['venue_id'],
                'race_number': rl['race_number'],
                'wind_speed': weather.get('wind_speed') or 0,
                'wind_direction': weather.get('wind_direction') or 'calm',
                'temperature': weather.get('temperature') or 20,
                'wave_height': weather.get('wave_height') or 0,
                'water_temperature': weather.get('water_temperature') or 20,
                'result_1st': int(res['result_1st']),
                'result_2nd': int(res['result_2nd']),
                'result_3rd': int(res['result_3rd']),
                'payout_sanrentan': res.get('payout_sanrentan') or 0,
                '_boats': merged_boats,
            })

    logger.info(f'pkl 読込完了: 有効レース {len(races)} 件 (期間: {years_months})')

    # 時系列ソート (race_date)
    races.sort(key=lambda r: (str(r['race_date']), r['venue_id'], r['race_number']))

    X, y1, y2, y3, wts = [], [], [], [], []
    skipped = 0
    for race in races:
        boats = race['_boats']
        # FeatureEngineer 入力形式
        rd = {
            'venue_id': race['venue_id'],
            'month': int(str(race['race_date']).split('-')[1]),
            'distance': 1800,
            'wind_speed': race['wind_speed'],
            'wind_direction': race['wind_direction'],
            'temperature': race['temperature'],
            'wave_height': race['wave_height'],
            'water_temperature': race['water_temperature'],
        }
        try:
            f = fe.transform(rd, boats)
        except Exception as e:
            skipped += 1
            continue
        b1 = race['result_1st'] - 1
        b2 = race['result_2nd'] - 1
        b3 = race['result_3rd'] - 1
        if not (0 <= b1 <= 5 and 0 <= b2 <= 5 and 0 <= b3 <= 5):
            skipped += 1
            continue
        X.append(f)
        y1.append(b1)
        y2.append(b2)
        y3.append(b3)
        wts.append(compute_sample_weight(race))

    if skipped:
        logger.info(f'特徴量生成 skip: {skipped} 件')

    X = np.array(X, dtype=np.float32)
    y1 = np.array(y1, dtype=np.int64)
    y2 = np.array(y2, dtype=np.int64)
    y3 = np.array(y3, dtype=np.int64)
    wts = np.array(wts, dtype=np.float32)
    logger.info(f'有効サンプル: {len(X)} 件 / {X.shape[1] if len(X) else 0} dim '
                f'/ weight mean={wts.mean():.2f}')
    return X, y1, y2, y3, wts


def finetune(epochs=20, batch_size=256, lr=0.0001, patience=8,
             focal_gamma=2.0, label_smoothing=0.1,
             years_months=None):
    if years_months is None:
        years_months = DEFAULT_YEARS_MONTHS

    logger.info('=== April Fine-tune 開始 ===')
    logger.info(f'  baseline: {V10_MODEL_PATH}')
    logger.info(f'  期間: {years_months}')
    logger.info(f'  lr={lr} patience={patience} epochs={epochs}')

    # V10 scaler を流用
    shutil.copy(V10_SCALER_PATH, SCALER_PATH)
    logger.info(f'  scaler流用: {SCALER_PATH}')
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # V10 model load
    v10_state = torch.load(V10_MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=v10_state['input_dim'],
        hidden_dims=v10_state['hidden_dims'],
        num_boats=v10_state['num_boats'],
        dropout=v10_state['dropout'],
    )
    model.load_state_dict(v10_state['model_state_dict'])
    logger.info(f"  V10 重みロード済 (val_acc_1st={v10_state['metadata'].get('val_acc_1st', 0):.1f}%)")

    X, y1, y2, y3, wts = load_training_data_from_pkl(years_months)
    if len(X) == 0:
        logger.error('訓練データゼロ — 中止')
        return
    X_scaled = scaler.transform(X).astype(np.float32)

    # 時系列80/20 (4月データなのでデータ内最後の数日が val)
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
    logger.info(f'  device: {device}')

    l1 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw1.to(device),
                                  label_smoothing=label_smoothing).to(device)
    l2 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw2.to(device)).to(device)
    l3 = SampleWeightedFocalLoss(gamma=focal_gamma, class_weights=cw3.to(device)).to(device)
    tw = [1.0, 0.7, 0.5]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    # baseline (V10) val_loss 測定
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
    logger.info(f'  V10 baseline val_loss={base_val:.4f} val_acc_1st={base_acc:.1f}%')

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
        logger.info(f'Ep {epoch}/{epochs}: train={train_loss:.4f} val={val_loss:.4f} '
                    f'acc={val_acc:.1f}% (Δbase={val_loss-base_val:+.4f}) lr={lr_now:.6f}')

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
                    'val_acc_1st': val_acc, 'version': 'v10_april_finetune',
                    'baseline': 'v10', 'v10_baseline_val_loss': base_val,
                    'years_months': years_months,
                    'weight_preset': 'mild',
                },
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stop: epoch {epoch}')
                break

    logger.info('')
    logger.info('=== 結果 ===')
    logger.info(f'  V10 baseline:  val_loss={base_val:.4f} val_acc_1st={base_acc:.1f}%')
    logger.info(f'  April-tuned:   val_loss={best_val:.4f} val_acc_1st={best_acc:.1f}%')
    if saved_any:
        logger.info(f'  → V10 baseline より改善 ({best_val-base_val:+.4f})')
        logger.info(f'  → 保存: {MODEL_PATH}')
    else:
        logger.info(f'  → V10 baseline を超えなかった (保存せず)')

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'trained_at': datetime.now().isoformat(),
                   'baseline_val_loss': base_val,
                   'baseline_val_acc_1st': base_acc,
                   'best_val_loss': best_val,
                   'best_val_acc_1st': best_acc,
                   'improved': saved_any,
                   'years_months': years_months,
                   'history': history}, f, ensure_ascii=False, indent=2)
    logger.info(f'  ログ: {LOG_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=8)
    args = parser.parse_args()
    finetune(epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, patience=args.patience)
