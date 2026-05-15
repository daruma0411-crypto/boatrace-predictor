---
date: 2026-05-15
topic: Phase D 資金管理パラメータ最適化
issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
status: approved
risk: 🟢 通常 (analysis 完結、READ-ONLY DB、本番介入なし)
depends_on: 現状本番 P6 (mc3_venue_focus_r4) 設定
unlocks: Phase D の本番反映 (別セッション、shadow 並走後)
---

# Phase D 資金管理パラメータ最適化 設計

## 背景

サードパーティレビュー ([vault](C:\Users\iwashita.AKGNET\Documents\Obsidian Vault\30_decisions\2026-05-15-third-party-review-phase-d-direction.md)) で「**期待値 E = p × o フィルタ + 分数ケリー 1/10〜1/4**」が推奨された。

調査の結果、boatrace-predictor は既に EV + Kelly の枠組みを実装済み (`src/betting.py`, `config/betting_config.json`)。ただし本番運用中の P6 (`mc3_venue_focus_r4`) は:

- `kelly_fraction: 0.0625` (1/16 ケリー、推奨 1/10〜1/4 より過剰保守)
- `min_expected_value: 0.0` (フィルタ事実上なし、推奨 ≥ 1.0 を未活用)
- `kelly_prob_gain: 1.0` (動的化未活用)

これらの「未活用機能」を backtest で組合せ最適化する。本番介入なし、analysis 完結。

## スコープ

| 項目 | 内容 |
|---|---|
| 範囲 | P6 (mc3_venue_focus_r4) を対象に `(kelly_fraction, min_expected_value)` の組合せ backtest 比較 (D0+D1)、動的 `kelly_prob_gain` の効果検証 (D2) |
| 範囲外 | 本番設定変更、shadow デプロイ、新戦略創出、Phase B 特徴量の組込み、新モデル訓練 |
| 本番影響 | ゼロ |
| リスク判定 | 🟢 通常 |

## 成功条件

1. 6 組合せ (kelly × min_ev) backtest が完了、`analysis/reports/phase_d_grid.md` に比較表が出る
2. 現状 P6 default より ROI 改善する組合せが 1 つ以上発見される (失敗時は撤退判定明示)
3. 動的 kelly_prob_gain (entropy 別) backtest 結果が `analysis/reports/phase_d_dynamic.md` に記録される
4. 最終的に「**本番反映候補となるパラメータセット (or 「現状維持推奨」)** 」が明示される

## 既存システムの確認

### betting.py の挙動 (`src/betting.py:735-`)
- `_apply_common_kelly_strategy()` が `min_expected_value` で EV フィルタ
- Kelly 計算: f = (p × o - 1) / (o - 1) を `kelly_fraction` 倍
- `odds_discount_factor` でオッズスリップ補正済み (0.92 + `use_dynamic_discount=true`)
- `kelly_prob_gain` で確率ブースト可能 (1.0=なし, 1.5=覚醒, 2.0=)

### 既存 backtest フレームワーク参考
- `analysis/17_backtest_apr2026.py` (Issue #3 で作成、April 2026 backtest)
- 同フレームワークを流用、戦略パラメータだけ差し替え

## グリッドサーチ

### D0 + D1 (合計 6 組合せ)

| # | kelly_fraction | min_expected_value | 備考 |
|---|---|---|---|
| 1 | 0.0625 | 0.0 | **現状 P6** (baseline) |
| 2 | 0.0625 | 1.0 | EV フィルタのみ追加 |
| 3 | 0.10 | 0.0 | Kelly 増額のみ |
| 4 | 0.10 | 1.0 | EV + Kelly 増 |
| 5 | 0.10 | 1.1 | EV 厳しめ |
| 6 | 0.20 | 1.0 | Kelly 大幅増 + EV |

### D2 (動的 kelly_prob_gain)

D0/D1 ベスト組合せ固定後、`kelly_prob_gain` を entropy 別に変動:

| 条件 | gain |
|---|---|
| entropy < 1.5 (高確信) | 1.5 |
| 1.5 ≤ entropy < 2.0 | 1.2 |
| entropy ≥ 2.0 (低確信、通常) | 1.0 |

(entropy は 3 連単確率分布のエントロピー、`config` の `filter_type='entropy'` 関連)

## 期間とデータ

- **backtest 期間: 2026-03-01 〜 2026-04-30** (2 ヶ月、~8,000 races)
- 2026-02 は `is_finished=false` で skip
- データソース: 既存 `races` + `boats` + `odds_*` テーブル (READ-ONLY)
- 各 race で V10 推論 + (kelly, min_ev) で bets 生成 → `result_*` と照合して PnL

## 評価指標

各組合せで以下を出力:

| 指標 | 説明 |
|---|---|
| **ROI** | 主要指標、(総 PnL + 元 stake) / 元 stake × 100 |
| **総 bets 数** | EV フィルタ強化で減少リスク |
| **総 stake** | 累計ベット額 |
| **総 PnL** | 絶対損益 (¥) |
| **Sharpe (日次)** | リスク調整リターン (mean(daily_pnl) / std(daily_pnl)) |
| **最大ドローダウン** | 累計 PnL の最大下振れ (¥) |
| **勝率 (bets 単位)** | 補助 |

## ファイル構造

| Path | Action | 用途 | 推定行数 |
|---|---|---|---|
| `analysis/40_phase_d_backtest.py` | Create | D0+D1 (6 組合せ) backtest | ~300 |
| `analysis/41_phase_d_dynamic_kelly.py` | Create | D2 動的 gain backtest | ~200 |
| `analysis/reports/phase_d_grid.md` | Generated | 6 組合せ比較表 + 最大ドローダウン推移 |
| `analysis/reports/phase_d_dynamic.md` | Generated | 動的 gain 結果と最終提言 |

## 撤退ライン

| 条件 | 行動 |
|---|---|
| 全 6 組合せ ROI が現状 P6 以下 | Phase D 撤退、現状維持。フレームワーク or サードパーティ仮説の検証失敗 |
| ベスト組合せでも bets 数が 90% 減 | EV フィルタ強すぎ、min_ev 0.5 等の中間値を追加検討 |
| ベスト組合せの MDD > ¥80,000 (元 bankroll ¥200,000 の 40%) | kelly_fraction 高すぎ、低めにシフト |
| 動的 gain で ROI 改善が +5pt 以下 | D3 (動的 kelly) 撤退、固定値で本番反映 |

## 非破壊性とロールバック

- 本番 `config/betting_config.json` の P6 (`mc3_venue_focus_r4`) は **一切変更しない**
- 別途 `analysis/` 内で組合せを試行し、レポートに結果を残す
- 本番反映の判断と実装は別セッション (Phase D 終了後の shadow 並走を経て)

## テスト

- 単体テスト: backtest スクリプトは確率 + EV → bets 数 + Kelly stake 計算が正しいか fixture race で検証
- 統合テスト: P6 default で backtest を回し、現状 memory に記載の ROI 224% (シミュ値) と数値が近いか確認 (期間が違うので完全一致は期待しない)

## 関連

- Issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
- vault: `30_decisions/2026-05-15-third-party-review-phase-d-direction.md` (サードパーティ推奨)
- vault: `30_decisions/2026-05-15-phase-c-retreat-v10-tie.md` (Phase C 撤退)
- 既存 backtest 参考: `analysis/17_backtest_apr2026.py`
- 既存 betting ロジック: `src/betting.py:735-` (`_apply_common_kelly_strategy`)
- 戦略設定: `config/betting_config.json` (P6 `mc3_venue_focus_r4`)
