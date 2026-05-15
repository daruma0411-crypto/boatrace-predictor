# V11.5 ミニシミュレーションレポート
生成日時: 2026-05-15T07:07:31.350366
目的: Phase C 本実装前の判定 (V10 + Phase B 特徴量 stacking)

- Train: 2026-03-01 〜 2026-03-22 (1907 races) — 2026-02 は DB に結果未登録のため除外
- Val:   2026-03-23 〜 2026-03-31 (1332 races)
- Test:  2026-04 (4320 races)
- 1号艇 1着 base rate (test): 0.5542

## メトリクス (hold-out 2026-04)

| 指標 | V10 (calibrated) | V11.5 (mini) | 差 |
|---|---|---|---|
| AUC | 0.6942 | 0.6714 | -0.0228 |
| Brier Score (低いほど良) | 0.2185 | 0.2253 | +0.0067 |

## SHAP TOP 5 特徴量寄与度 (平均 |shap|)

| rank | feature | mean |shap| |
|---|---|---|
| 1 | `v10_prob_boat1_raw` | 0.4003 |
| 2 | `boat1_skill_gap` | 0.1662 |
| 3 | `a_class_consumed` | 0.0888 |
| 4 | `day_in_meeting` | 0.0499 |
| 5 | `race_category_enc` | 0.0139 |

## 判定

AUC 差分 -0.0228 → **❌ V10 を上回らず、Phase B 価値疑わしい。撤退検討**
