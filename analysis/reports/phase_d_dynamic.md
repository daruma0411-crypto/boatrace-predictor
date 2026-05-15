# Phase D D2 動的 kelly_prob_gain 結果

Task 1 ベース: P6 default (1/16, EV0)
gain rule: entropy<1.5→1.5, 1.5≤<2.0→1.2, ≥2.0→1.0

## entropy 分布 (cache レース数別)

| 条件 | gain | レース数 | % |
|---|---|---|---|
| entropy<1.5 (高確信) | 1.5 | 0 | 0.0% |
| 1.5≤entropy<2.0 | 1.2 | 1 | 1.5% |
| entropy≥2.0 (通常) | 1.0 | 64 | 98.5% |

## 比較

| 指標 | 静的 gain=1.0 (Task 1 ベスト) | 動的 gain | 差 |
|---|---|---|---|
| n_bets | 191 | 191 | +0 |
| hit_rate | 5.8% | 5.8% | +0.0pt |
| ROI | 189.1% | 189.0% | -0.1pt |
| PnL | ¥+260,700 | ¥+260,500 | ¥-200 |
| Sharpe | 0.336 | 0.336 | -0.000 |
| MDD | ¥60,200 | ¥60,200 | ¥+0 |

## 最終提言

❌ 動的 gain は逆効果 (-0.1pt)、静的 gain 採用

### 本番反映候補のパラメータ (mc3_venue_focus_r4)

- `kelly_fraction`: 0.0625
- `min_expected_value`: 0.0
- `kelly_prob_gain`: 1.0 (静的、現状維持)

**本番反映は別セッションで実施** (shadow 1-2 週並走後)。

## Phase D 総合判定

Task 1 で baseline (P6 default) を超える組合せが見つからず、本 Task 2 でも動的化の効果が小さい場合、**Phase D 全体としては現状の P6 設定を維持** することを推奨。サードパーティ推奨の Kelly 増額・EV フィルタは P6 既存フィルタ (`min_probability=0.005`, `max_odds=80`) との重複により当該データセットでは効果なしと判明。
