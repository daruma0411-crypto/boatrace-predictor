# V11.5 ミニシミュレーション v2 レポート
生成日時: 2026-05-15T07:18:54.040370
v1 で AUC -0.023 だったので、V10 全 18 prob + 小グリッドサーチで再判定

- Train: 2026-03-01〜03-22 (1907) — 2026-02 は DB 未登録
- Val: 2026-03-23〜03-31 (1332)
- Test: 2026-04 (4320)

## グリッドサーチ結果 (test AUC、val AUC で best 選定)

| params | val AUC | test AUC | test Brier |
|---|---|---|---|
| leaves=15 lr=0.05 min_data=50 | 0.6880 | 0.6861 | 0.2213 |
| leaves=15 lr=0.02 min_data=50 | 0.6889 | 0.6886 | 0.2208 |
| leaves=31 lr=0.05 min_data=50 | 0.6809 | 0.6812 | 0.2226 |
| leaves=31 lr=0.02 min_data=30 | 0.6758 | 0.6749 | 0.2240 |
| leaves=63 lr=0.02 min_data=50 | 0.6799 | 0.6801 | 0.2229 |
| leaves=7 lr=0.05 min_data=100 ✅ | 0.6978 | 0.6941 | 0.2188 |

## hold-out 比較 (val-best モデル)

| 指標 | V10 (calibrated) | V11.5 v2 (best) | 差 |
|---|---|---|---|
| AUC | 0.6942 | 0.6941 | -0.0001 |
| Brier (低いほど良) | 0.2185 | 0.2188 | +0.0003 |

## SHAP TOP 5 (best モデル)

| rank | feature | mean |shap| |
|---|---|---|
| 1 | `v10_p1_1` | 0.2757 |
| 2 | `v10_p3_1` | 0.1920 |
| 3 | `boat1_skill_gap` | 0.1699 |
| 4 | `v10_p2_1` | 0.0598 |
| 5 | `a_class_consumed` | 0.0588 |

### Phase B 特徴量だけの順位

| rank in PhB | feature | mean |shap| |
|---|---|---|
| 1 | `boat1_skill_gap` | 0.1699 |
| 2 | `a_class_consumed` | 0.0588 |
| 3 | `day_in_meeting` | 0.0251 |
| 4 | `race_category_enc` | 0.0037 |
| 5 | `is_planned_int` | 0.0000 |

## 判定

AUC 差分 -0.0001 → **❌ V10 を上回らず、撤退検討**
