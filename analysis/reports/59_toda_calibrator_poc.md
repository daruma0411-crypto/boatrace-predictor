# 戸田 calibrator PoC (案 X Phase 1)

Train: 2026-03 (n=133), Test: 2026-04 + 2026-05 (n=303)

## Test 期間 backtest (forward 検証)

| 戦略 | n | top-1 hit% | top-3 hit% | 投資 (¥) | 回収 (¥) | PnL | ROI |
|---|---|---|---|---|---|---|---|
| V10 raw | 303 | 3.63% | 12.87% | ¥90,900 | ¥60,880 | ¥-30,020 | -33.03% |
| V10 + 戸田 calibrator | 303 | 4.29% | 10.56% | ¥90,900 | ¥58,340 | ¥-32,560 | -35.82% |

**改善幅 (Test)**: top-3 hit -2.31pt, ROI -2.79pt

## Train 期間 backtest (overfit check)

| 戦略 | n | top-3 hit% | ROI |
|---|---|---|---|
| Train V10 raw | 133 | 15.04% | -28.45% |
| Train V10 + cal | 133 | 19.55% | +6.34% |

## Calibrator が NN 出力をどう変えたか

Test 期間で boat-1 raw prob と calibrated prob の差を集計:

| boat | raw mean% | calibrated mean% | actual rate% | raw bias | cal bias |
|---|---|---|---|---|---|
| 1 | 52.34% | 61.31% | 38.94% | +13.40pt | +22.36pt |
| 2 | 13.55% | 15.90% | 18.15% | -4.60pt | -2.25pt |
| 3 | 12.30% | 10.73% | 18.81% | -6.52pt | -8.08pt |
| 4 | 11.96% | 6.61% | 14.52% | -2.57pt | -7.91pt |
| 5 | 6.47% | 6.80% | 6.93% | -0.46pt | -0.13pt |
| 6 | 3.38% | 2.66% | 2.64% | +0.74pt | +0.02pt |

## 自動判定 (CLAUDE.md 採用基準)

- 🔴 **calibrator 効果なし or 悪化** (ROI -2.79pt) → 凍結
- overfit check: Train ROI +6.34% vs Test ROI -35.82% (gap +42.16pt)
  - ⚠️ Train-Test gap 大、overfit 疑い

## 留意事項

- Train n=133 races は IsotonicRegression にギリギリ充足、calibrator 精度に限界
- Test n=304 races (forward) で +5pt 以上の ROI 改善が安定して出れば実用候補
- top-3 全部購入 proxy は Kelly/EV filter 不含、本番 ROI とは別
- 効果ありなら、次に Phase 2 (QMC 係数調整) と組み合わせ
- 結論は岩下さん判断、shadow 並走必須
