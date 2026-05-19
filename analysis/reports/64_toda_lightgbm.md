# 戸田 LightGBM (案 X Option A 改、NN を LightGBM に置換)

Train: 2025-06〜2026-03 (n=1968), Val: 2026-04 (n=168), Test (hold-out): 2026-05 (n=135)

LightGBM multi-class (6 boats), num_leaves=31, lr=0.05, early stopping (val multi_logloss)

## 訓練結果

| task | best_iter | val multi_logloss |
|---|---|---|
| 1着 | 23 | 1.4065 |
| 2着 | 13 | 1.7114 |
| 3着 | 7 | 1.7674 |

## Test (hold-out 2026-05) NN-only vs LightGBM 比較

| model | 1着 top-1 acc | 1着 log-loss |
|---|---|---|
| V10 baseline | 0.3407 | 1.5642 |
| **戸田 LightGBM** | **0.3704** | **1.5655** |

**改善幅**: acc +2.96pt, log-loss +0.0013

## Test 期間 boat-level calibration

| boat | actual | V10 pred | LightGBM pred | V10 bias | LightGBM bias |
|---|---|---|---|---|---|
| 1 | 30.37% | 52.54% | 45.79% | +22.17pt | +15.42pt |
| 2 | 18.52% | 14.52% | 17.80% | -4.00pt | -0.72pt |
| 3 | 21.48% | 12.37% | 15.01% | -9.12pt | -6.47pt |
| 4 | 17.78% | 11.37% | 11.88% | -6.41pt | -5.90pt |
| 5 | 8.15% | 6.02% | 6.35% | -2.13pt | -1.80pt |
| 6 | 3.70% | 3.19% | 3.17% | -0.51pt | -0.54pt |

## Feature importance (1着 model top 15)

| rank | feature | gain |
|---|---|---|
| 1 | B1_win_rate_3 | 1542 |
| 2 | B1_win_rate_2 | 1099 |
| 3 | B2_win_rate_3 | 745 |
| 4 | B2_win_rate_2 | 670 |
| 5 | B3_exhibit_time_diff | 662 |
| 6 | B3_win_rate_2 | 645 |
| 7 | B4_win_rate_3 | 613 |
| 8 | B5_win_rate_2 | 578 |
| 9 | B4_exhibit_time_diff | 549 |
| 10 | B4_win_rate_2 | 548 |
| 11 | B5_win_rate_3 | 535 |
| 12 | B1_weight_diff | 526 |
| 13 | B6_exhibit_time_diff | 514 |
| 14 | B6_local_win_rate_2 | 499 |
| 15 | B1_boat_win_rate_2 | 495 |

## V10 NN warm-start (63) vs LightGBM 比較 (Test)

| model | 1着 top-1 acc | 1着 log-loss |
|---|---|---|
| V10 baseline | 0.3556 | 1.5060 |
| V10 + Toda fine-tune (NN) | 0.3778 | 1.5102 |
| **戸田 LightGBM** | **0.3704** | **1.5655** |

## 判定 ヒント

- 🟡 LightGBM が V10 を僅かに上回る、効果限定
- 次は LightGBM + QMC で backtest ROI 算出 (analysis/65)
- 結論は岩下さんに委ね、shadow 並走必須
