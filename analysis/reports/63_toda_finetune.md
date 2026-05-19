# 戸田 NN warm-start fine-tune (案 X Option A)

Train: 2025-06〜2026-03 (n=1968)
Val:   2026-04 (n=168)
Test (hold-out): 2026-05 (n=135)

Hyperparameters: lr=5e-05, weight_decay=0.0001, batch_size=32, max_epochs=20, patience=3

## 訓練 history

| epoch | train_loss | val_loss | val 1着 acc |
|---|---|---|---|
| 1 | 4.8601 | 4.9550 | 0.4405 |
| 2 | 4.7884 | 4.9453 | 0.4405 |
| 3 | 4.7617 | 4.9304 | 0.4524 |
| 4 | 4.7355 | 4.9135 | 0.4464 |
| 5 | 4.6932 | 4.9094 | 0.4464 |
| 6 | 4.6756 | 4.9003 | 0.4524 |
| 7 | 4.6493 | 4.9038 | 0.4524 |
| 8 | 4.6359 | 4.8932 | 0.4524 |
| 9 | 4.6115 | 4.8964 | 0.4583 |
| 10 | 4.6115 | 4.8872 | 0.4464 |
| 11 | 4.5783 | 4.8942 | 0.4464 |
| 12 | 4.5577 | 4.8923 | 0.4583 |
| 13 | 4.5575 | 4.8956 | 0.4524 |

**best val_loss**: 4.8872

## Test (hold-out 2026-05) NN-only metrics

| model | 1着 top-1 acc | 1着 log-loss |
|---|---|---|
| V10 baseline | 0.3556 | 1.5060 |
| Toda fine-tune | 0.3778 | 1.5102 |

**改善幅**: acc +2.22pt, log-loss +0.0042

## Test 期間 boat-level calibration

| boat | actual rate% | V10 mean pred% | Toda mean pred% | V10 bias | Toda bias |
|---|---|---|---|---|---|
| 1 | 30.37% | 26.05% | 42.77% | -4.32pt | +12.40pt |
| 2 | 18.52% | 16.36% | 17.64% | -2.15pt | -0.87pt |
| 3 | 21.48% | 17.03% | 15.80% | -4.45pt | -5.68pt |
| 4 | 17.78% | 16.58% | 12.47% | -1.20pt | -5.31pt |
| 5 | 8.15% | 12.77% | 6.98% | +4.63pt | -1.17pt |
| 6 | 3.70% | 11.20% | 4.33% | +7.50pt | +0.63pt |

## 判定 ヒント

- V10 → Toda fine-tune で **1号艇 bias** が改善したか? (戸田は 1号艇 -10pt 構造)
- 全 boat の log-loss が低下 → calibration 改善
- 次は Toda model + QMC で backtest ROI 算出 (analysis/64)
- 結論は岩下さんに委ね、shadow 並走必須
