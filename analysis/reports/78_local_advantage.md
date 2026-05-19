# 地元 (local_win_rate_2) signal 未取り込み edge 検出 (Path D)

仮説: 「競艇は地元が強い」は V10/specialist の `local_win_rate_2` 特徴量で
既に学習されている。だが完全には捕捉していない可能性。
各 venue test (2026-05) で 1号艇 `local_win_rate_2` 5 分位別に hit / 予測 を集計。

## 1. 1号艇 local_win_rate_2 分位別 (13 venues 集計)

| 分位 | n | 平均 local rate | actual 1号艇 1着率 | V10 予測 1号艇 1着率 | bias (V10 - actual) |
|---|---|---|---|---|---|
| Q1 (≤19.8) | 314 | 8.17 | 36.31% | 40.65% | +4.35pt |
| Q2 (≤32.0) | 315 | 26.12 | 38.10% | 45.96% | +7.86pt |
| Q3 (≤41.0) | 316 | 36.70 | 50.32% | 55.41% | +5.09pt |
| Q4 (≤50.0) | 329 | 45.85 | 59.88% | 60.81% | +0.93pt |
| Q5 (>50.0) | 295 | 58.33 | 71.19% | 63.60% | -7.59pt |

全体: n=1569, actual 1号艇 1着率 50.99%, V10 予測 53.23%, bias +2.24pt

## 2. venue 別 1号艇 local 分位 hit (top/bottom)

各 venue で test races の 1号艇 local_win_rate_2 を上下半分に分割し、
V10 bias の違いを確認。

| venue | name | n_test | mean local | top half (高 local) | bottom half (低 local) | bias 差 |
|---|---|---|---|---|---|---|
| 1 | 桐生 | 122 | 36.84 | -15.57pt | +14.38pt | **-29.95pt ⚠️** |
| 2 | 戸田 | 135 | 33.33 | +22.89pt | +21.46pt | **+1.43pt ** |
| 3 | 江戸川 | 78 | 46.10 | +6.62pt | +9.31pt | **-2.69pt ** |
| 4 | 平和島 | 130 | 34.08 | -1.66pt | +4.72pt | **-6.38pt ** |
| 7 | 蒲郡 | 138 | 29.82 | -0.04pt | -5.24pt | **+5.20pt ** |
| 10 | 三国 | 132 | 35.01 | +5.03pt | +10.47pt | **-5.44pt ** |
| 12 | 住之江 | 115 | 36.68 | -12.49pt | -14.28pt | **+1.79pt ** |
| 13 | 尼崎 | 77 | 38.52 | -20.94pt | -3.76pt | **-17.18pt ⚠️** |
| 14 | 鳴門 | 167 | 30.95 | +1.01pt | +12.33pt | **-11.32pt ⚠️** |
| 16 | 児島 | 151 | 33.56 | -0.06pt | +15.50pt | **-15.55pt ⚠️** |
| 22 | 福岡 | 120 | 34.76 | -11.16pt | -4.62pt | **-6.54pt ** |
| 23 | 唐津 | 132 | 35.65 | -6.90pt | +1.35pt | **-8.26pt ** |
| 24 | 大村 | 72 | 36.49 | -3.17pt | +16.18pt | **-19.35pt ⚠️** |

## 3. 全 6 艇の local advantage signal

各艇で local_win_rate_2 vs general win_rate_2 の差 = local advantage。
当地で本来の実力以上に走る選手の signal。

### 各艇の local advantage 高低別 1着率 (13 venues 集計)

local advantage = local_win_rate_2 - win_rate_2 (当地ボーナス)

- B1: n=1450, 全体 1着率 51.72%, 上位 local adv 53.24%, 下位 50.21%, 差 **+3.03pt**
- B2: n=1430, 全体 1着率 15.66%, 上位 local adv 17.93%, 下位 13.41%, 差 **+4.52pt**
- B3: n=1427, 全体 1着率 15.00%, 上位 local adv 16.41%, 下位 13.59%, 差 **+2.82pt**
- B4: n=1387, 全体 1着率 11.39%, 上位 local adv 11.40%, 下位 11.38%, 差 **+0.02pt**
- B5: n=1339, 全体 1着率 6.72%, 上位 local adv 7.50%, 下位 5.95%, 差 **+1.54pt**
- B6: n=1261, 全体 1着率 3.49%, 上位 local adv 3.97%, 下位 3.01%, 差 **+0.96pt**

## 判定: 未取り込み edge の有無

- ⚠️ **5 venues で V10 bias が高 local / 低 local 帯で 10pt 以上差**
  → 未取り込み edge の可能性、Path A (feature 拡張) 検討価値あり
  - 桐生 (v1): bias 差 -29.95pt
  - 尼崎 (v13): bias 差 -17.18pt
  - 鳴門 (v14): bias 差 -11.32pt
  - 児島 (v16): bias 差 -15.55pt
  - 大村 (v24): bias 差 -19.35pt

## 留意事項

- local_win_rate_2 は V10 学習 features に含まれる (76dim の一部)
- specialist LightGBM の feature importance で `B6_local_win_rate_2` のみ 14 位
- 本分析は test n=70-170/venue で検出力限界、forward で再確認すべき
