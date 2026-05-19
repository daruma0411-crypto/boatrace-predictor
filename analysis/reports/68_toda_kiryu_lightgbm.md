# 戸田 + 桐生 LightGBM (sequential venue add Step 1)

Train: 戸田 1968 + 桐生 2130 = 4098 races
Val: 戸田 168, Test (hold-out): 戸田 135

## 訓練結果

- 1着 best iter: **25**
- val multi_logloss: **1.3700**

(参考 64 戸田単独: best iter 23, val logloss 1.4065)

## NN-only metrics on 戸田 hold-out (2026-05)

- top-1 acc: 0.3481
- 1着 log-loss: 1.5728

## Boat-level calibration (戸田 hold-out)

| boat | actual | LightGBM (戸田+桐生) pred | bias |
|---|---|---|---|
| 1 | 30.37% | 47.02% | +16.65pt |
| 2 | 18.52% | 15.68% | -2.84pt |
| 3 | 21.48% | 13.50% | -7.98pt |
| 4 | 17.78% | 13.43% | -4.35pt |
| 5 | 8.15% | 7.17% | -0.98pt |
| 6 | 3.70% | 3.20% | -0.51pt |

## QMC backtest 比較 (戸田 hold-out 2026-05)

| 戦略 | n | top-3 hit% | 投資 | 回収 | PnL | ROI |
|---|---|---|---|---|---|---|
| V10 (pkl) | 135 | 11.85% | ¥40,500 | ¥23,400 | ¥-17,100 | -42.22% |
| LightGBM 戸田単独 (65) | 135 | 13.33% | ¥40,500 | ¥25,980 | ¥-14,520 | -35.85% |
| **LightGBM 戸田+桐生** | 135 | 12.59% | ¥40,500 | ¥26,710 | ¥-13,790 | -34.05% |

**改善幅 (戸田+桐生 vs 戸田単独)**: ROI +1.80pt
**改善幅 (戸田+桐生 vs V10)**: ROI +8.17pt

## 自動判定

- 🟡 **桐生 追加で +1.80pt** (撤退ライン +5pt 未達) → 効果限定

## 留意事項

- Test n=135 races は検出力ぎりぎり、改善幅は noise の可能性
- 桐生 train 全期間 (2025-06〜2026-04) は test 期間 (2026-05) を含まず、leakage なし
- ROI 改善は top-3 全部購入 proxy、本番 Kelly/EV filter とは別
