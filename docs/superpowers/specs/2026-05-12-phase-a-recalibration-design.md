---
date: 2026-05-12
topic: Phase A 部分着手 (A4 + A1)
issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
status: approved
risk: 🟢 通常 (影響:小 / READ-ONLY)
---

# Phase A 部分着手 設計: race title 在庫確認 + V10 キャリブレーター再 fit

## 背景

Issue #4 改善ロードマップの Phase A (Week 1-2) のうち、本セッションで A4 と A1 のみ着手する。残りの A2 (P7 shadow 追加) と A3 (race title スクレイピング拡張) は別セッションで扱う。

A2 を切り離す理由: GCP scheduler 介入を伴い、🟠 強化リスクなので影響分析を別途実施する必要がある。
A3 を切り離す理由: A4 の在庫確認結果次第で実施要否が決まるため。

## スコープ

| 項目 | 内容 |
|---|---|
| 範囲 | A4 (race title 在庫確認) と A1 (V10 キャリブレーター再 fit) |
| 範囲外 | A2, A3, Phase B 以降 |
| 本番影響 | ゼロ。全工程 READ-ONLY、出力は新規ファイルのみ。本番 `models/calibrators.pkl` 非破壊 |
| リスク判定 | 🟢 通常 |

## 成功条件

1. **A4 完了**: `analysis/reports/race_title_inventory.md` に以下が記載されている
   - races テーブルの title 候補列の存在有無
   - 月別 (2024-01〜2026-04) の title 充足率
   - scraped raw データの title 出現率
   - 判定: 直近3ヶ月の title 充足率が ≥95% であれば A3 スキップ可、未満なら A3 発火
2. **A1 完了**: `models/calibrators_v2.pkl` が生成され、`analysis/reports/calibration_v2_eval.md` に以下が記載されている
   - before (`calibrators.pkl`) / after (`calibrators_v2.pkl`) の reliability 比較テーブル
   - 0.7-0.8 帯の実 hit 率とのズレ (V10 既知の課題 -10.6pt が改善されたか)
   - ロードマップ KPI 「±3pt 以内」の達成可否

## A4: race title 在庫確認

### スクリプト

`analysis/19_race_title_inventory.py` (新規)

### 処理フロー

1. 本番 DB に READ-ONLY 接続
2. `information_schema.columns` を SELECT し、`races` テーブルの全カラム名を取得
3. title 候補 (`race_title`, `race_name`, `title`, `subtitle` 等) を検出
4. 検出された各列について月別 (2024-01〜2026-04) に NULL/空文字率を集計
5. `analysis/historical_data/` 配下の `*.json` / `*.pkl` を走査し、辞書キーに `title` / `race_title` が含まれるか + 充足率を集計
6. 結果を `analysis/reports/race_title_inventory.md` に保存

### 判定ロジック

```
直近3ヶ月 (2026-02 〜 2026-04) の title 充足率:
  ≥ 95% → A3 (スクレイピング拡張) スキップ可、Phase B に進める
  < 95% → A3 発火、scraper.py 拡張が必要
```

### 工数見積もり

1.0 - 1.5 時間

## A1: V10 キャリブレーター再 fit

### スクリプト

`analysis/20_recalibrate_v10.py` (新規、`analysis/12_calibrate_v10_2.py` をベース)

### 処理フロー

1. `models/boatrace_model.pth` + `models/feature_scaler.pkl` をロード
2. 本番 DB から SELECT (READ-ONLY):
   ```sql
   SELECT r.id, r.venue_id, r.race_date, r.race_number,
          r.result_1st, r.result_2nd, r.result_3rd,
          r.wind_speed, r.wind_direction, r.temperature,
          r.wave_height, r.water_temperature
   FROM races r
   WHERE r.is_finished = true
     AND r.actual_result_trifecta IS NOT NULL
     AND r.result_1st IS NOT NULL
     AND r.wind_speed IS NOT NULL
     AND r.race_date BETWEEN '2026-02-01' AND '2026-04-30'
   ORDER BY r.race_date ASC, r.id ASC
   ```
3. **時系列 split**:
   - **fit 区間 (train)**: 2026-02-01 〜 2026-03-31 (2 ヶ月)
   - **評価区間 (hold-out)**: 2026-04-01 〜 2026-04-30 (1 ヶ月)
   - 理由: 同じ期間で fit + 評価すると in-sample で当然ズレ 0 に近づき、KPI「±3pt」判定が無意味化する。時系列 hold-out で汎化性能を測る
4. `FeatureEngineer` で 76dim 特徴量を生成
5. V10 で推論し、ポジション×艇 (3×6 = 18系列) ごとに raw probability を蓄積
6. **fit 区間のデータのみ**で `IsotonicRegression` を 18 個 fit
7. `calibrators.pkl` と同じ構造で `models/calibrators_v2.pkl` に保存:
   ```python
   {
     '1st': [iso_0, iso_1, iso_2, iso_3, iso_4, iso_5],
     '2nd': [...],
     '3rd': [...]
   }
   ```

### 評価フロー

別関数 `evaluate_calibrators()`:
1. **評価区間 (2026-04-01〜04-30) の hold-out データのみ**を使用 (fit に使わなかったデータ)
2. その期間で V10 推論を実行し raw probability を取得
3. `calibrators.pkl` (旧) と `calibrators_v2.pkl` (新) をそれぞれ適用して補正後確率を得る
4. 補正後確率を 10 bin (0.0-0.1, 0.1-0.2, ..., 0.9-1.0) × position (1st/2nd/3rd) で分割し、各 bin の平均予測確率と実 hit 率を比較
5. 0.7-0.8 帯 (bin 7-8) の |予測 - 実 hit| を before/after で対比し、KPI「±3pt 以内」達成可否を判定
6. `analysis/reports/calibration_v2_eval.md` に以下を出力:
   - 評価期間と件数
   - bin × position × {旧, 新} のテーブル (予測平均 / 実 hit / ズレ pt)
   - 0.7-0.8 帯ズレの before/after 数値
   - KPI 判定結果

### 出力ファイル

- `models/calibrators_v2.pkl` (本番非投入、shadow 検証用)
- `analysis/reports/calibration_v2_eval.md`

### 工数見積もり

2.0 - 3.0 時間

## 非破壊性とロールバック

- DB は `SELECT` のみ。INSERT/UPDATE/DELETE 厳禁 (memory `feedback_boatrace_db_isolation.md` 準拠)
- 本番 `models/calibrators.pkl` は touch しない
- 本番投入は Phase A 全体完了後に別判断
- ロールバック手順: 生成ファイルを `rm` するだけ

## テスト

- A4: 出力 markdown が生成され、月別表に値が入っていれば成功
- A1: `calibrators_v2.pkl` が `pickle.load()` 可能 + キー構造が `calibrators.pkl` と一致 + 評価レポートに before/after 数値が入っていれば成功
- 単体テスト追加は不要 (調査・再学習スクリプトのため)

## 関連

- Issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
- memory: `project_boatrace_improvement_roadmap.md`
- 既存類似スクリプト: `analysis/12_calibrate_v10_2.py` (V10.2 用、これをベースに改変)
- 既存 calibrator: `models/calibrators.pkl` (V10 当時のもの、本番運用中)
