# 唐津 / 平和島 / 尼崎 recipe 細密化 (1 ラウンド)

各 target で K × own_weight × 引き算 component の 96 recipes を val 上で評価。
Phase 1 val top 5 → Phase 2 test hold-out 評価。


## venue 23 (唐津)

val: 210 races, test: 132 races
V10 test ROI: -2.58%
現状 best (73 R_top5_sim_avg): test ROI +50.98%

### val top 5 → test ROI

| rank | recipe | val ROI | test ROI | 現状比 |
|---|---|---|---|---|
| 1 | K2_own1.0_sub_opp3x0.2 | +5.87% | -34.62% | -85.60pt 🔴 |
| 2 | K3_own2.0_sub_none | +5.63% | -20.96% | -71.94pt 🔴 |
| 3 | K2_own1.0_sub_opp3x0.1 | +4.70% | -33.16% | -84.14pt 🔴 |
| 4 | K5_own1.5_sub_opp3x0.3 | +4.65% | -25.58% | -76.56pt 🔴 |
| 5 | K4_own1.5_sub_opp3x0.1 | +4.62% | +56.39% | +5.41pt 🟢 |

**val best (K2_own1.0_sub_opp3x0.2) 細密化結果**:
- test ROI: -34.62%
- 現状比改善: -85.60pt
- 🔴 **現状維持** (改善なし)

## venue 4 (平和島)

val: 192 races, test: 130 races
V10 test ROI: -37.05%
現状 best (73 R_top5_sim_avg): test ROI -1.59%

### val top 5 → test ROI

| rank | recipe | val ROI | test ROI | 現状比 |
|---|---|---|---|---|
| 1 | K7_own1.5_sub_none | -4.41% | +1.69% | +3.28pt 🟢 |
| 2 | K7_own1.5_sub_opp3x0.1 | -4.41% | +1.69% | +3.28pt 🟢 |
| 3 | K7_own1.5_sub_opp3x0.2 | -6.30% | +1.69% | +3.28pt 🟢 |
| 4 | K7_own1.5_sub_opp3x0.3 | -6.30% | +1.69% | +3.28pt 🟢 |
| 5 | K7_own1.0_sub_none | -9.58% | -1.59% | +0.00pt 🟢 |

**val best (K7_own1.5_sub_none) 細密化結果**:
- test ROI: +1.69%
- 現状比改善: +3.28pt
- 🟡 微改善、updating の価値判定要

## venue 13 (尼崎)

val: 144 races, test: 77 races
V10 test ROI: -43.68%
現状 best (74 R-_own-opp3x0.2): test ROI -5.41%

### val top 5 → test ROI

| rank | recipe | val ROI | test ROI | 現状比 |
|---|---|---|---|---|
| 1 | K2_own1.0_sub_opp3x0.3 | +27.20% | +24.46% | +29.87pt 🟢 |
| 2 | K3_own3.0_sub_opp3x0.2 | +14.37% | +18.74% | +24.15pt 🟢 |
| 3 | K2_own3.0_sub_none | +13.75% | +30.74% | +36.15pt 🟢 |
| 4 | K2_own3.0_sub_opp3x0.1 | +13.75% | +27.23% | +32.64pt 🟢 |
| 5 | K3_own1.5_sub_none | +12.94% | +18.48% | +23.89pt 🟢 |

**val best (K2_own1.0_sub_opp3x0.3) 細密化結果**:
- test ROI: +24.46%
- 現状比改善: +29.87pt
- 🟢 **採用更新候補** (+5pt 以上)

## 全体判定

- 細密化で +5pt 以上更新: **1/3 venues**
- これで「改善余地探索」は完了、これ以上は forward (2026-06+) 待ち

## 留意事項

- val/test 分離方式 (73/74 と同じ厳密性)
- 1 ラウンドのみ、これ以上の細密化は test set 上の過剰最適化
- 採用候補は shadow 並走 2 週間で forward 検証必須
