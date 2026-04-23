# Analysis ディレクトリ

V10 再学習の方針を data-driven に決めるための分析スクリプト群。

## 絶対ルール
- **V10本番（mc_early_race）に影響を与えない**
- production コード (`src/`, `config/`, `scripts/teleboat_purchaser.py`) は一切変更しない
- DB は READ-ONLY のみ（SELECT のみ、INSERT/UPDATE/DELETE 禁止）
- 新規モデルは別 strategy_type で並走、既存を置換しない

## フェーズ

| 分析 | ファイル | 目的 |
|---|---|---|
| A: Miss Pattern | `01_miss_pattern.py` | どのセグメントで崩れてるか |
| B: 特徴量寄与度 | `02_feature_importance.py` | 何の特徴量が効いてるか |
| C: Calibration | `03_calibration.py` | 予測確率と実績のずれ |
| D: 時系列重み | `04_temporal_weight.py` | 直近データの扱い |

## 実行

```
cd boatrace-predictor
PYTHONIOENCODING=utf-8 PGCLIENTENCODING=UTF8 venv/Scripts/python.exe analysis/01_miss_pattern.py
```

レポートは `analysis/reports/` に出力。
