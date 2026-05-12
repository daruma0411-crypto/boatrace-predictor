# Phase A (A4 + A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #4 改善ロードマップの Phase A から A4 (race title 在庫確認) と A1 (V10 キャリブレーター再 fit + hold-out 評価) を実装。

**Architecture:** 2 つの独立した analysis スクリプトを新規追加。本番 DB は READ-ONLY、本番 `calibrators.pkl` は非破壊。A1 は時系列 hold-out (fit: 2026-02〜03、評価: 2026-04) で汎化性能を測る。

**Tech Stack:** Python 3.x、PostgreSQL (本番 DB)、PyTorch (V10 推論)、scikit-learn (IsotonicRegression)、既存 `src.database` / `src.features` / `src.models` モジュール

**Spec:** `docs/superpowers/specs/2026-05-12-phase-a-recalibration-design.md`

**Issue:** https://github.com/daruma0411-crypto/boatrace-predictor/issues/4

---

## File Structure

| Path | Action | Purpose |
|---|---|---|
| `analysis/19_race_title_inventory.py` | Create | A4: races テーブル schema 動的検出 + 月別 title 充足率集計 + scraped raw 走査 |
| `analysis/20_recalibrate_v10.py` | Create | A1: V10 で raw probability 取得 → isotonic fit (fit 期間) + hold-out 評価 |
| `analysis/reports/race_title_inventory.md` | Generated | A4 出力 |
| `models/calibrators_v2.pkl` | Generated | A1 fit 出力 (本番非投入) |
| `analysis/reports/calibration_v2_eval.md` | Generated | A1 評価出力 |

既存ファイルは一切変更しない (新規追加のみ)。

---

## Task 1: A4 - race title 在庫確認スクリプト

**Files:**
- Create: `analysis/19_race_title_inventory.py`
- Generated: `analysis/reports/race_title_inventory.md`

### - [ ] Step 1: スクリプト全体を作成

`analysis/19_race_title_inventory.py` を新規作成。内容は以下の全文：

```python
"""race title 在庫確認 (A4 of Phase A roadmap, Issue #4)

本番 DB の races テーブルから title 候補列を動的検出し、月別の
充足率を集計。analysis/historical_data/ の scraped raw も並走走査。

READ-ONLY 厳守 (DB に対して SELECT のみ)。
出力: analysis/reports/race_title_inventory.md
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).parent / "reports" / "race_title_inventory.md"
HISTORICAL_DIR = Path(__file__).parent / "historical_data"
TITLE_CANDIDATE_COLS = ['race_title', 'race_name', 'title', 'subtitle', 'race_subtitle']


def detect_title_columns():
    """races テーブルの全カラムを取得し、title 候補列を検出"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'races'
            ORDER BY ordinal_position
        """)
        all_cols = [(r['column_name'], r['data_type']) for r in cur.fetchall()]
    detected = [c for c, _ in all_cols if c in TITLE_CANDIDATE_COLS]
    logger.info(f"races 全カラム: {len(all_cols)} 件")
    logger.info(f"title 候補列検出: {detected}")
    return all_cols, detected


def aggregate_db_inventory(detected_cols):
    """検出された title 列の月別充足率を集計"""
    monthly = {}
    for col in detected_cols:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT to_char(r.race_date, 'YYYY-MM') AS ym,
                       COUNT(*) AS total,
                       COUNT(NULLIF(r.{col}, '')) AS filled
                FROM races r
                WHERE r.race_date BETWEEN '2024-01-01' AND '2026-04-30'
                GROUP BY ym
                ORDER BY ym
            """)
            monthly[col] = [(r['ym'], r['total'], r['filled']) for r in cur.fetchall()]
    return monthly


def aggregate_historical_inventory():
    """analysis/historical_data/ 配下の JSON/PKL を走査"""
    if not HISTORICAL_DIR.exists():
        logger.warning(f"{HISTORICAL_DIR} なし、historical 走査スキップ")
        return {}
    result = defaultdict(lambda: {'total': 0, 'with_title': 0, 'keys_seen': set()})

    def scan_items(items, bucket):
        for it in items:
            if not isinstance(it, dict):
                continue
            result[bucket]['total'] += 1
            result[bucket]['keys_seen'].update(it.keys())
            if any(k in it and it[k] for k in ['title', 'race_title', 'race_name']):
                result[bucket]['with_title'] += 1

    for path in HISTORICAL_DIR.rglob('*.json'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"skip {path}: {e}")
            continue
        scan_items(data if isinstance(data, list) else [data], str(path.parent.name))

    for path in HISTORICAL_DIR.rglob('*.pkl'):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"skip {path}: {e}")
            continue
        scan_items(data if isinstance(data, list) else [data], str(path.parent.name))

    return dict(result)


def make_report(all_cols, detected, monthly, historical):
    lines = []
    lines.append("# race title 在庫確認レポート\n\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n")
    lines.append("対象: Phase A の A4 (Issue #4)\n\n")
    lines.append("## 1. races テーブル schema\n\n")
    lines.append(f"- 全カラム数: {len(all_cols)}\n")
    detected_str = ', '.join(detected) if detected else 'なし'
    lines.append(f"- title 候補列検出 (`{TITLE_CANDIDATE_COLS}`): **{detected_str}**\n\n")
    lines.append("### 全カラム一覧\n\n| カラム名 | 型 |\n|---|---|\n")
    for c, t in all_cols:
        lines.append(f"| {c} | {t} |\n")
    lines.append("\n## 2. 月別 title 充足率 (本番 DB)\n")
    if not detected:
        lines.append("\ntitle 候補列が races テーブルに存在しない。**A3 (スクレイピング拡張) 発火必要**。\n")
    else:
        for col, rows in monthly.items():
            lines.append(f"\n### 列 `{col}`\n\n| 年月 | total | filled | 充足率 |\n|---|---|---|---|\n")
            for ym, total, filled in rows:
                ratio = (filled / total * 100) if total else 0
                lines.append(f"| {ym} | {total} | {filled} | {ratio:.1f}% |\n")
    lines.append("\n## 3. scraped historical_data の在庫\n\n")
    if not historical:
        lines.append("historical_data なし\n")
    else:
        lines.append("| バケツ | 件数 | title 含む | 充足率 | 観測キー (先頭 8) |\n|---|---|---|---|---|\n")
        for bucket, info in sorted(historical.items()):
            t = info['total']
            wt = info['with_title']
            ratio = (wt / t * 100) if t else 0
            keys_str = ", ".join(sorted(info['keys_seen'])[:8])
            lines.append(f"| {bucket} | {t} | {wt} | {ratio:.1f}% | {keys_str} |\n")
    lines.append("\n## 4. 判定\n\n")
    if detected and monthly:
        recent_ratios = []
        for col, rows in monthly.items():
            for ym, total, filled in rows:
                if ym in ('2026-02', '2026-03', '2026-04') and total:
                    recent_ratios.append(filled / total * 100)
        avg_recent = sum(recent_ratios) / len(recent_ratios) if recent_ratios else 0
        verdict = "A3 スキップ可能" if avg_recent >= 95 else "A3 (スクレイピング拡張) 発火必要"
        lines.append(f"直近3ヶ月 (2026-02 〜 2026-04) の平均充足率: **{avg_recent:.1f}%**\n\n")
        lines.append(f"判定: **{verdict}** (基準: 95%)\n")
    else:
        lines.append("title 候補列が DB に無いため A3 発火必要\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {REPORT_PATH}")


def main():
    logger.info("A4 race title 在庫確認 開始")
    all_cols, detected = detect_title_columns()
    monthly = aggregate_db_inventory(detected)
    historical = aggregate_historical_inventory()
    make_report(all_cols, detected, monthly, historical)
    logger.info("完了")


if __name__ == '__main__':
    main()
```

### - [ ] Step 2: スクリプトを実行

実行コマンド (workspace ルートから):

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python analysis/19_race_title_inventory.py
```

期待: エラーなく完走し、ログに「レポート出力: ...race_title_inventory.md」が出る。所要時間 5-30 秒。

### - [ ] Step 3: 出力レポートを検証

```bash
cat analysis/reports/race_title_inventory.md | head -80
```

検証ポイント (どれか1つでも欠けたら STOP して原因を報告):
- 「## 1. races テーブル schema」セクションが存在
- 全カラム一覧テーブルに行がある (races のカラム数 ≥ 10)
- 「## 2. 月別 title 充足率」セクションが存在
- 「## 4. 判定」セクションに直近3ヶ月の平均充足率と判定が記載されている

### - [ ] Step 4: コミット

```bash
git add analysis/19_race_title_inventory.py analysis/reports/race_title_inventory.md
git commit -m "feat(analysis): A4 race title 在庫確認スクリプト追加と初回レポート"
```

---

## Task 2: A1 - V10 キャリブレーター再 fit + hold-out 評価

**Files:**
- Create: `analysis/20_recalibrate_v10.py`
- Generated: `models/calibrators_v2.pkl`
- Generated: `analysis/reports/calibration_v2_eval.md`
- Read-only reference: `analysis/12_calibrate_v10_2.py` (V10.2 用の既存スクリプト、構造をベースに改変)
- Read-only reference: `models/calibrators.pkl` (本番、評価時に旧版として比較)

### - [ ] Step 1: スクリプト全体を作成

`analysis/20_recalibrate_v10.py` を新規作成。内容は以下の全文：

```python
"""V10 キャリブレーター再 fit (A1 of Phase A roadmap, Issue #4)

- fit 区間: 2026-02-01 〜 2026-03-31 (2 ヶ月)
- hold-out 評価: 2026-04-01 〜 2026-04-30 (1 ヶ月)
- 出力 1: models/calibrators_v2.pkl (本番非投入、shadow 検証用)
- 出力 2: analysis/reports/calibration_v2_eval.md
- 本番 models/calibrators.pkl は touch しない (旧版として読み込み比較のみ)
"""
import os
import sys
import pickle
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from src.models import BoatraceMultiTaskModel
from src.features import FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "boatrace_model.pth"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"
OLD_CAL_PATH = ROOT / "models" / "calibrators.pkl"
NEW_CAL_PATH = ROOT / "models" / "calibrators_v2.pkl"
EVAL_PATH = ROOT / "analysis" / "reports" / "calibration_v2_eval.md"

FIT_START = '2026-02-01'
FIT_END = '2026-03-31'
EVAL_START = '2026-04-01'
EVAL_END = '2026-04-30'


def load_model_and_scaler():
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state['input_dim'], hidden_dims=state['hidden_dims'],
        num_boats=state['num_boats'], dropout=state['dropout'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def fetch_races(date_from, date_to):
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
              AND r.race_date BETWEEN %s AND %s
            ORDER BY r.race_date ASC, r.id ASC
        """, (date_from, date_to))
        races = cur.fetchall()
        race_ids = [r['id'] for r in races]
        if not race_ids:
            return [], {}
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
    return races, boats_by


def predict_batch(races, boats_by, model, scaler, fe):
    """V10 推論を実行し raw probability + labels を返す"""
    probs_1 = np.zeros((len(races), 6), dtype=np.float32)
    probs_2 = np.zeros((len(races), 6), dtype=np.float32)
    probs_3 = np.zeros((len(races), 6), dtype=np.float32)
    labels_1, labels_2, labels_3 = [], [], []
    valid_idx = []
    for i, race in enumerate(races):
        boats = boats_by.get(race['id'], [])
        if len(boats) != 6:
            continue
        rd = {'venue_id': race['venue_id'], 'month': race['race_date'].month,
              'distance': 1800,
              'wind_speed': race.get('wind_speed') or 0,
              'wind_direction': race.get('wind_direction') or 'calm',
              'temperature': race.get('temperature') or 20,
              'wave_height': race.get('wave_height') or 0,
              'water_temperature': race.get('water_temperature') or 20}
        try:
            f = fe.transform(rd, boats)
        except Exception:
            continue
        f = scaler.transform(f.reshape(1, -1))
        X = torch.FloatTensor(f)
        with torch.no_grad():
            out = model(X)
        probs_1[i] = F.softmax(out[0], dim=1).numpy()[0]
        probs_2[i] = F.softmax(out[1], dim=1).numpy()[0]
        probs_3[i] = F.softmax(out[2], dim=1).numpy()[0]
        labels_1.append(race['result_1st'] - 1)
        labels_2.append(race['result_2nd'] - 1)
        labels_3.append(race['result_3rd'] - 1)
        valid_idx.append(i)
        if len(valid_idx) % 500 == 0:
            logger.info(f"  予測 {len(valid_idx)} 件...")
    valid_idx = np.array(valid_idx)
    return (probs_1[valid_idx], probs_2[valid_idx], probs_3[valid_idx],
            np.array(labels_1), np.array(labels_2), np.array(labels_3))


def fit_position(probs, labels, pos_name):
    calibrators = []
    for cls in range(6):
        p = probs[:, cls]
        y = (labels == cls).astype(np.float32)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso.fit(p, y)
        calibrators.append(iso)
        logger.info(f"  {pos_name}[{cls+1}号艇]: pred_mean={p.mean():.4f} actual_mean={y.mean():.4f}")
    return calibrators


def apply_calibrators(probs, calibrators):
    out = np.zeros_like(probs)
    for cls in range(6):
        out[:, cls] = calibrators[cls].predict(probs[:, cls])
    return out


def reliability_bins(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for cls in range(6):
        p = probs[:, cls]
        y = (labels == cls).astype(np.float32)
        row = []
        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (p >= bins[b]) & (p < bins[b + 1])
            else:
                mask = (p >= bins[b]) & (p <= bins[b + 1])
            if mask.sum() == 0:
                row.append((bins[b], bins[b+1], 0, np.nan, np.nan))
            else:
                row.append((bins[b], bins[b+1], int(mask.sum()),
                            float(p[mask].mean()), float(y[mask].mean())))
        result.append(row)
    return result


def kpi_07_08_gap(reliability):
    """0.7-0.8 bin の |pred - hit| 平均 (pt) を返す。サンプル無しは除外"""
    gaps = []
    for cls_row in reliability:
        for lo, hi, n, pred, hit in cls_row:
            if abs(lo - 0.7) < 1e-6 and n > 0 and not np.isnan(hit):
                gaps.append(abs(pred - hit) * 100)
    return float(np.mean(gaps)) if gaps else float('nan')


def fit_calibrators():
    logger.info(f"=== A1 V10 calibrator 再 fit (fit: {FIT_START} 〜 {FIT_END}) ===")
    model, scaler = load_model_and_scaler()
    fe = FeatureEngineer()
    races, boats_by = fetch_races(FIT_START, FIT_END)
    logger.info(f"fit 区間レース取得: {len(races)} 件")
    p1, p2, p3, y1, y2, y3 = predict_batch(races, boats_by, model, scaler, fe)
    logger.info(f"有効 fit サンプル: {len(p1)}")
    cal_1 = fit_position(p1, y1, '1st')
    cal_2 = fit_position(p2, y2, '2nd')
    cal_3 = fit_position(p3, y3, '3rd')
    out = {
        '1st': cal_1, '2nd': cal_2, '3rd': cal_3,
        'fitted_at': datetime.now().isoformat(),
        'n_samples': int(len(p1)),
        'fit_period': f"{FIT_START} 〜 {FIT_END}",
        'source': 'V10 (boatrace_model.pth) Phase A A1 recalibration',
    }
    with open(NEW_CAL_PATH, 'wb') as f:
        pickle.dump(out, f)
    logger.info(f"保存: {NEW_CAL_PATH}")


def write_eval_report(results, n_samples):
    lines = []
    lines.append("# V10 キャリブレーター v2 評価レポート\n\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n\n")
    lines.append(f"- fit 期間: {FIT_START} 〜 {FIT_END}\n")
    lines.append(f"- hold-out 評価期間: {EVAL_START} 〜 {EVAL_END}\n")
    lines.append(f"- 評価有効サンプル数: {n_samples}\n\n")

    def fmt(x):
        if x is None:
            return "-"
        try:
            if np.isnan(x):
                return "-"
        except TypeError:
            return "-"
        return f"{x:.3f}"

    for pos in ['1st', '2nd', '3rd']:
        lines.append(f"## {pos} 着\n\n")
        for cls in range(6):
            lines.append(f"### {cls+1}号艇\n\n")
            lines.append("| bin | n | raw pred | raw hit | 旧 pred | 旧 hit | 新 pred | 新 hit |\n|---|---|---|---|---|---|---|---|\n")
            for b in range(10):
                raw_row = results[pos]['raw'][cls][b]
                old_row = results[pos]['old'][cls][b]
                new_row = results[pos]['new'][cls][b]
                lo, hi, n_b, raw_p, raw_h = raw_row
                _, _, _, old_p, old_h = old_row
                _, _, _, new_p, new_h = new_row
                bin_label = f"{lo:.1f}-{hi:.1f}"
                lines.append(f"| {bin_label} | {n_b} | {fmt(raw_p)} | {fmt(raw_h)} | {fmt(old_p)} | {fmt(old_h)} | {fmt(new_p)} | {fmt(new_h)} |\n")
            lines.append("\n")
    lines.append("## KPI: 0.7-0.8 帯ズレ (1着 全艇平均)\n\n")
    gap_raw = kpi_07_08_gap(results['1st']['raw'])
    gap_old = kpi_07_08_gap(results['1st']['old'])
    gap_new = kpi_07_08_gap(results['1st']['new'])
    lines.append("| 版 | |pred - hit| (pt) |\n|---|---|\n")
    lines.append(f"| raw (補正なし) | {gap_raw:.2f} |\n")
    lines.append(f"| 旧 calibrators.pkl | {gap_old:.2f} |\n")
    lines.append(f"| 新 calibrators_v2.pkl | {gap_new:.2f} |\n\n")
    if not np.isnan(gap_new) and gap_new <= 3.0:
        verdict = "KPI 達成 (±3pt 以内)"
    elif not np.isnan(gap_new) and gap_new < gap_old:
        verdict = "KPI 未達 (改善はしているが ±3pt 超え)"
    else:
        verdict = "KPI 未達 (旧版より悪化または同等)"
    lines.append(f"判定: **{verdict}** (新版ズレ {gap_new:.2f} pt)\n")
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {EVAL_PATH}")


def evaluate_calibrators():
    logger.info(f"=== A1 評価 (hold-out: {EVAL_START} 〜 {EVAL_END}) ===")
    model, scaler = load_model_and_scaler()
    fe = FeatureEngineer()
    with open(OLD_CAL_PATH, 'rb') as f:
        old_cal = pickle.load(f)
    with open(NEW_CAL_PATH, 'rb') as f:
        new_cal = pickle.load(f)
    races, boats_by = fetch_races(EVAL_START, EVAL_END)
    logger.info(f"評価区間レース取得: {len(races)} 件")
    p1, p2, p3, y1, y2, y3 = predict_batch(races, boats_by, model, scaler, fe)
    logger.info(f"有効評価サンプル: {len(p1)}")
    results = {}
    for pos_name, p, y in [('1st', p1, y1), ('2nd', p2, y2), ('3rd', p3, y3)]:
        p_old = apply_calibrators(p, old_cal[pos_name])
        p_new = apply_calibrators(p, new_cal[pos_name])
        results[pos_name] = {
            'raw': reliability_bins(p, y),
            'old': reliability_bins(p_old, y),
            'new': reliability_bins(p_new, y),
        }
    write_eval_report(results, int(len(p1)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['fit', 'eval', 'all'], default='all')
    args = parser.parse_args()
    if args.step in ('fit', 'all'):
        fit_calibrators()
    if args.step in ('eval', 'all'):
        evaluate_calibrators()
```

### - [ ] Step 2: fit を実行

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python analysis/20_recalibrate_v10.py --step fit
```

期待: ログに以下が出る (所要時間 10-30 分目安):
- `fit 区間レース取得: <数千> 件`
- `有効 fit サンプル: <数千>`
- `1st[1号艇]: pred_mean=... actual_mean=...` (各艇 18 行)
- `保存: ...models/calibrators_v2.pkl`

エラー時の対処:
- `pkg-config` / DB 接続失敗 → `.env` の `DATABASE_URL` を確認
- `feature_scaler.pkl` not found → workspace ルートで実行しているか確認
- 推論で `RuntimeError` → `analysis/12_calibrate_v10_2.py` の同じ箇所と差分を比較

### - [ ] Step 3: fit 出力 (calibrators_v2.pkl) を検証

```bash
python - <<'EOF'
import pickle
with open('models/calibrators_v2.pkl', 'rb') as f:
    new = pickle.load(f)
with open('models/calibrators.pkl', 'rb') as f:
    old = pickle.load(f)
print('NEW keys:', sorted(new.keys()))
print('OLD keys:', sorted(old.keys()))
for k in ('1st', '2nd', '3rd'):
    print(f'{k}: new len={len(new[k])} old len={len(old[k])}')
print('NEW meta:', {k: new[k] for k in new if k not in ("1st","2nd","3rd")})
EOF
```

期待出力:
- `NEW keys` に `'1st', '2nd', '3rd'` を含む
- 各ポジションで `len(new[k]) == 6` かつ `len(old[k]) == 6` (一致)
- `NEW meta` に `fitted_at`, `n_samples`, `fit_period`, `source` が含まれる

NG なら STOP して構造ミスマッチを報告。

### - [ ] Step 4: 評価を実行

```bash
python analysis/20_recalibrate_v10.py --step eval
```

期待: 所要時間 5-15 分目安。ログに `レポート出力: ...calibration_v2_eval.md` が出る。

### - [ ] Step 5: 評価レポートを検証 + KPI 判定を確認

```bash
cat analysis/reports/calibration_v2_eval.md | head -120
```

検証ポイント:
- 「fit 期間: 2026-02-01 〜 2026-03-31」「hold-out 評価期間: 2026-04-01 〜 2026-04-30」が記載されている
- 評価有効サンプル数が ≥ 数百 (4月の1ヶ月分)
- 1st/2nd/3rd × 1-6号艇の reliability テーブル (18 個) が全て出力されている
- 「## KPI: 0.7-0.8 帯ズレ」セクションに raw / 旧 / 新 の3行が入っている
- 「判定: **<KPI 達成 or 未達>**」が末尾にある

**KPI 判定後の方針**:
- 達成 (±3pt 以内): Phase A 残タスク (A2/A3 — A3 は A4 結果次第) に進める準備完了として完了。
- 未達でも旧版より改善: 進捗として記録、A2 と並走で評価期間を広げるか検討 (今セッションでは方針変更しない、報告のみ)
- 未達かつ旧版同等以下: STOP して原因分析 (期間が短すぎ? V10 でない推論バグ? feature drift?)

### - [ ] Step 6: コミット

```bash
git add analysis/20_recalibrate_v10.py models/calibrators_v2.pkl analysis/reports/calibration_v2_eval.md
git commit -m "feat(analysis): A1 V10 キャリブレーター再 fit + hold-out 評価"
```

---

## Self-Review

- **Spec 網羅性**: spec のスコープ (A4 + A1) → 各 1 タスク。成功条件 (race_title_inventory.md / calibrators_v2.pkl / calibration_v2_eval.md) → 各タスクの「出力検証」ステップで担保。非破壊性 → 既存ファイルは一切 modify しない、DB は SELECT のみ。
- **プレースホルダ**: スクリプト全文を Step 1 に埋め込み済み。"TBD", "TODO" なし。
- **型整合**: `fetch_races` / `predict_batch` / `apply_calibrators` / `reliability_bins` / `kpi_07_08_gap` の I/O は同じ Task 内で一貫。`calibrators_v2.pkl` の構造 (`{'1st': [iso×6], '2nd': [...], '3rd': [...]}`) は spec と一致、Step 3 の検証で構造差を catch。
- **粒度**: Task 1 は 4 ステップ、Task 2 は 6 ステップ。スクリプト作成は 1 ステップで大きいが、関数単位に分けると context が分散して逆に読みづらいので全文を 1 ステップに集約。実行・検証・コミットを各 step に分けて回せる。

---

## 完了後の状態

- `analysis/reports/race_title_inventory.md` に A3 発火可否の判定が記載される
- `models/calibrators_v2.pkl` が生成される (本番非投入)
- `analysis/reports/calibration_v2_eval.md` に KPI 判定が記載される
- Phase A 残タスク (A2 P7 shadow、A3 スクレイピング拡張 — 必要なら) に進める状態
