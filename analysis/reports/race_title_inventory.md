# race title 在庫確認レポート

生成日時: 2026-05-14T18:40:35.401201
対象: Phase A の A4 (Issue #4)

## 1. races テーブル schema

- 全カラム数: 19
- title 候補列検出 (`['race_title', 'race_name', 'title', 'subtitle', 'race_subtitle']`): **なし**

### 全カラム一覧

| カラム名 | 型 |
|---|---|
| id | integer |
| venue_id | integer |
| race_number | integer |
| race_date | date |
| created_at | timestamp without time zone |
| deadline_time | timestamp with time zone |
| status | character varying |
| result_1st | integer |
| result_2nd | integer |
| result_3rd | integer |
| payout_sanrentan | integer |
| wind_speed | real |
| wind_direction | character varying |
| temperature | real |
| wave_height | real |
| water_temperature | real |
| actual_result_trifecta | character varying |
| payout_amount | integer |
| is_finished | boolean |

## 2. 月別 title 充足率 (本番 DB)

title 候補列が races テーブルに存在しない。**A3 (スクレイピング拡張) 発火必要**。

## 3. scraped historical_data の在庫

| バケツ | 件数 | title 含む | 充足率 | 観測キー (先頭 8) |
|---|---|---|---|---|
| 2024_04 | 17849 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v01-03 | 2064 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v04-06 | 2676 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v07-09 | 2382 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v10-12 | 2231 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v13-15 | 1440 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v16-18 | 1776 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v19-21 | 2880 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2024_04_v22-24 | 2400 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04 | 16940 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v01-03 | 1920 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v04-06 | 2446 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v07-09 | 2474 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v10-12 | 1248 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v13-15 | 2592 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v16-18 | 1532 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v19-21 | 2136 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2025_04_v22-24 | 2592 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04 | 13060 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v01-03 | 1502 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v04-06 | 1620 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v07-09 | 1884 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v10-12 | 1430 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v13-15 | 1656 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v16-18 | 1776 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v19-21 | 1116 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |
| 2026_04_v22-24 | 2076 | 0 | 0.0% | boats, odds, payout_nirentan, payout_sanrentan, race_date, race_number, result_1st, result_2nd |

## 2.5. race_titles テーブル充足率 (A3 後)

| 年月 | total | filled | 充足率 |
|---|---|---|---|
| 2025-06 | 4766 | 0 | 0.0% |
| 2025-07 | 5145 | 0 | 0.0% |
| 2025-08 | 5049 | 0 | 0.0% |
| 2025-09 | 4248 | 0 | 0.0% |
| 2025-10 | 4129 | 0 | 0.0% |
| 2025-11 | 3938 | 0 | 0.0% |
| 2025-12 | 4772 | 0 | 0.0% |
| 2026-01 | 5166 | 0 | 0.0% |
| 2026-02 | 4171 | 4171 | 100.0% |
| 2026-03 | 4661 | 4661 | 100.0% |
| 2026-04 | 4380 | 4380 | 100.0% |

## 4. 判定

直近3ヶ月 race_titles 平均充足率: **100.0%**

判定: **A3 完了 (race_titles 経由で充足)** (基準: 95%)
