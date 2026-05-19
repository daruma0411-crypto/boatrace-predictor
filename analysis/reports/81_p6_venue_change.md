# P6 venue list 変更時の影響シミュレーション

**評価**: test 2026-05 hold-out、top-3 picks proxy ROI (300¥/race 投資)
**shadow strategy**: 80 で確定した venue 別 best approach (recipe/pool/specialist)
**V10 baseline**: 各 venue で V10 NN raw probs + QMC top-3 picks

## venue list 比較

| set | venues |
|---|---|
| **P6 current (10)** | v2,v4,v5,v6,v9,v10,v12,v13,v17,v23 |
| Functional 13 | v1,v2,v3,v4,v7,v10,v12,v13,v14,v16,v22,v23,v24 |
| Union 15 (P6 ∪ Functional) | v1,v2,v3,v4,v5,v6,v7,v9,v10,v12,v13,v14,v16,v17,v22,v23,v24 |

- 重複: v2,v4,v10,v12,v13,v23 (6 venues)
- P6 のみ: v5,v6,v9,v17 (4 venues)
- Functional のみ: v1,v3,v7,v14,v16,v22,v24 (7 venues)

## 4 シナリオ 統合 ROI 比較

| シナリオ | n_races | 投資 ¥ | 回収 ¥ | PnL | ROI |
|---|---|---|---|---|---|
| A. P6 current × V10 | 1156 | ¥346,800 | ¥275,484 | ¥-71,316 | **-20.56%** |
| B. P6 current × shadow | 1156 | ¥346,800 | ¥359,029 | ¥+12,229 | **+3.53%** |
| C. Functional 13 × shadow | 1569 | ¥470,700 | ¥491,581 | ¥+20,881 | **+4.44%** |
| D. Union 15 × shadow | 2004 | ¥601,200 | ¥610,098 | ¥+8,898 | **+1.48%** |
| 参考: 全 24 venues × V10 | 2904 | ¥871,200 | ¥690,318 | ¥-180,882 | **-20.76%** |
| 参考: 全 24 venues × shadow (13 で shadow、他 V10) | 2904 | ¥871,200 | ¥849,847 | ¥-21,353 | **-2.45%** |

### A. P6 current × V10 venue 内訳

| venue | name | n | ROI | source |
|---|---|---|---|---|
| 2 | 戸田 | 135 | -42.22% | V10 |
| 4 | 平和島 | 130 | -37.05% | V10 |
| 5 | 多摩川 | 96 | -19.31% | V10 |
| 6 | 浜名湖 | 113 | +36.28% | V10 |
| 9 | 津 | 127 | -34.20% | V10 |
| 10 | 三国 | 132 | -39.55% | V10 |
| 12 | 住之江 | 115 | -2.93% | V10 |
| 13 | 尼崎 | 77 | -43.68% | V10 |
| 17 | 宮島 | 99 | -19.16% | V10 |
| 23 | 唐津 | 132 | -2.58% | V10 |

### B. P6 current × shadow venue 内訳

| venue | name | n | ROI | source |
|---|---|---|---|---|
| 2 | 戸田 | 135 | -5.43% | shadow |
| 4 | 平和島 | 130 | -1.59% | shadow |
| 5 | 多摩川 | 96 | -19.31% | V10 |
| 6 | 浜名湖 | 113 | +36.28% | V10 |
| 9 | 津 | 127 | -34.20% | V10 |
| 10 | 三国 | 132 | -28.13% | shadow |
| 12 | 住之江 | 115 | +17.68% | shadow |
| 13 | 尼崎 | 77 | +24.46% | shadow |
| 17 | 宮島 | 99 | -19.16% | V10 |
| 23 | 唐津 | 132 | +66.72% | shadow |

### C. Functional 13 × shadow venue 内訳

| venue | name | n | ROI | source |
|---|---|---|---|---|
| 1 | 桐生 | 122 | +9.34% | shadow |
| 2 | 戸田 | 135 | -5.43% | shadow |
| 3 | 江戸川 | 78 | +15.09% | shadow |
| 4 | 平和島 | 130 | -1.59% | shadow |
| 7 | 蒲郡 | 138 | +10.65% | shadow |
| 10 | 三国 | 132 | -28.13% | shadow |
| 12 | 住之江 | 115 | +17.68% | shadow |
| 13 | 尼崎 | 77 | +24.46% | shadow |
| 14 | 鳴門 | 167 | -11.34% | shadow |
| 16 | 児島 | 151 | -29.36% | shadow |
| 22 | 福岡 | 120 | +36.64% | shadow |
| 23 | 唐津 | 132 | +66.72% | shadow |
| 24 | 大村 | 72 | -41.20% | shadow |

### D. Union 15 × shadow venue 内訳

| venue | name | n | ROI | source |
|---|---|---|---|---|
| 1 | 桐生 | 122 | +9.34% | shadow |
| 2 | 戸田 | 135 | -5.43% | shadow |
| 3 | 江戸川 | 78 | +15.09% | shadow |
| 4 | 平和島 | 130 | -1.59% | shadow |
| 5 | 多摩川 | 96 | -19.31% | V10 |
| 6 | 浜名湖 | 113 | +36.28% | V10 |
| 7 | 蒲郡 | 138 | +10.65% | shadow |
| 9 | 津 | 127 | -34.20% | V10 |
| 10 | 三国 | 132 | -28.13% | shadow |
| 12 | 住之江 | 115 | +17.68% | shadow |
| 13 | 尼崎 | 77 | +24.46% | shadow |
| 14 | 鳴門 | 167 | -11.34% | shadow |
| 16 | 児島 | 151 | -29.36% | shadow |
| 17 | 宮島 | 99 | -19.16% | V10 |
| 22 | 福岡 | 120 | +36.64% | shadow |
| 23 | 唐津 | 132 | +66.72% | shadow |
| 24 | 大村 | 72 | -41.20% | shadow |

## 比較サマリ (場変更の影響)

- **A → B (P6 維持、内部 shadow 化)**: -20.56% → +3.53% (+24.09pt)
- **A → C (venue list を functional 13 に変更 + shadow)**: -20.56% → +4.44% (+25.00pt)
- **A → D (venue list を union 15 に変更 + shadow)**: -20.56% → +1.48% (+22.04pt)
- **B → C (venue list 変更の純粋効果)**: +3.53% → +4.44% (+0.91pt)

## 注意事項

- top-3 picks proxy ROI は **Kelly/EV filter なし**、P6 実 production ROI とは別物
- P6 実 production は max_odds=80, skip_56=True, kelly_fraction=0.0625 等で picks 絞り
- proxy ROI は venue list + 予測精度の比較指標、絶対 ROI ではない
- 真の影響は production 投入 + actual purchase data で検証必須

## 推奨判定

- proxy ROI 最大: **C. Functional 13 × shadow** (+4.44%)
- **venue list 変更 + shadow** は明確な改善 (+25.00pt)
