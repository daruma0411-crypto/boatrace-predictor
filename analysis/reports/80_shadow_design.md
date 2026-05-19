# Shadow 戦略 v1 設計 + 2026-05 再現確認

13 functional venues の venue 別 best approach を統合した shadow strategy。

## venue 別 strategy mapping

| venue | name | approach | type | ROI 期待値 (期待: 累積分析より) |
|---|---|---|---|---|
| 1 | 桐生 | 73 R+_V10x0.6+own_x0.4 | recipe | +9.34% |
| 2 | 戸田 | specialist_82 | specialist_82 | -5.43% |
| 3 | 江戸川 | specialist_82 | specialist_82 | +15.09% |
| 4 | 平和島 | 73 R+_top5_sim_avg | recipe | -1.59% |
| 7 | 蒲郡 | specialist_76 | specialist_76 | +10.65% |
| 10 | 三国 | 73 R+_top2_sim_avg | recipe | -28.13% |
| 12 | 住之江 | specialist_76 | specialist_76 | +17.68% |
| 13 | 尼崎 | 75 K2_own1.0_sub_opp3x0.3 | recipe_75 | +24.46% |
| 14 | 鳴門 | pool | pool | -11.34% |
| 16 | 児島 | pool | pool | -29.36% |
| 22 | 福岡 | specialist_82 | specialist_82 | +36.64% |
| 23 | 唐津 | pool | pool | +66.72% |
| 24 | 大村 | 73 R+_own_x2+functional | recipe | -41.20% |

## 2026-05 再現確認 (各 venue で shadow ROI vs 期待値)

| venue | name | n | V10 ROI | shadow ROI | 期待 ROI | 再現性 |
|---|---|---|---|---|---|---|
| 1 | 桐生 | 122 | -8.28% | +9.34% | +9.34% | +0.00pt ✅ |
| 2 | 戸田 | 135 | -42.22% | -5.43% | -5.43% | -0.00pt ✅ |
| 3 | 江戸川 | 78 | -16.15% | +15.09% | +15.09% | -0.00pt ✅ |
| 4 | 平和島 | 130 | -37.05% | -1.59% | -1.59% | +0.00pt ✅ |
| 7 | 蒲郡 | 138 | -16.55% | +10.65% | +10.65% | +0.00pt ✅ |
| 10 | 三国 | 132 | -39.55% | -28.13% | -28.13% | -0.00pt ✅ |
| 12 | 住之江 | 115 | -2.93% | +17.68% | +17.68% | +0.00pt ✅ |
| 13 | 尼崎 | 77 | -43.68% | +24.46% | +24.46% | -0.00pt ✅ |
| 14 | 鳴門 | 167 | -45.31% | -11.34% | -11.34% | +0.00pt ✅ |
| 16 | 児島 | 151 | -58.04% | -29.36% | -29.36% | +0.00pt ✅ |
| 22 | 福岡 | 120 | -12.31% | +36.64% | +36.64% | -0.00pt ✅ |
| 23 | 唐津 | 132 | -2.58% | +66.72% | +66.72% | -0.00pt ✅ |
| 24 | 大村 | 72 | -56.62% | -41.20% | -41.20% | -0.00pt ✅ |

## 13 venues 統合集計

- 統合 n: 1569
- shadow 投資合計: ¥470,700
- shadow 回収合計: ¥491,580
- shadow ROI: **+4.44%**
- V10 baseline ROI (同 races): **-29.45%**
- shadow vs V10 差: **+33.89pt**

## 留意事項

- 13 venues のみ shadow 戦略採用、11 venues は V10 baseline 維持
- 各 venue の再現性 (shadow ROI vs 期待) は実装の整合性チェック
- 2026-05 単月 hold-out、forward (2026-06+) で再検証必須
- 採用候補は production scheduler 統合 → shadow テーブル記録 → 2 週間 forward 検証
