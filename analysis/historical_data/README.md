# historical_data/

過去年のレースデータを公式サイト (boatrace.jp) からスクレイピングした pkl ファイル群。

## 目的

- 学習データの過去年範囲拡張（V11 系再学習用）
- backtest シミュレーション用
- 現 production DB には影響しない READ-ONLY なローカルデータ

## 構造

```
historical_data/
├── README.md
├── {year}_{month:02d}/
│   ├── racelist.pkl    # list[dict] 6艇×n_races (出走表)
│   ├── result.pkl      # list[dict] n_races (着順+3連単/2連単 払戻)
│   ├── odds_3t.pkl     # list[dict] n_races (3連単オッズ 120通り)
│   └── skipped.pkl     # list[tuple] 取得失敗ログ
```

## 各 pkl の中身

### racelist.pkl

```python
[
  {
    'race_date': '2024-04-01',
    'venue_id': 1,
    'race_number': 1,
    'boats': [
      {'boat_number': 1, 'player_id': '4246', 'player_class': 'A1',
       'player_name': '...', 'win_rate': 6.5, 'win_rate_2': 45.2, ...},
      ... 6艇分
    ],
  },
  ...
]
```

### result.pkl

```python
[
  {
    'race_date': '2024-04-01', 'venue_id': 1, 'race_number': 1,
    'result_1st': 1, 'result_2nd': 3, 'result_3rd': 2,
    'payout_sanrentan': 1890, 'payout_nirentan': 540,
  },
  ...
]
```

### odds_3t.pkl

```python
[
  {
    'race_date': '2024-04-01', 'venue_id': 1, 'race_number': 1,
    'odds': {'1-2-3': 12.7, '1-2-4': 18.5, ..., '6-5-4': 999.0},  # 120通り
  },
  ...
]
```

## 取得方法 (再生成)

```bash
python analysis/scrape_historical.py --year 2024 --month 4
python analysis/scrape_historical.py --year 2025 --month 4
```

- 1秒/リクエスト で公式サイトに負荷配慮
- 1日終わるごとに中間保存（途中失敗時も復旧可能）
- 再実行時は既存レコードを skip（重複なし）
- 失敗 race は `skipped.pkl` に記録

### 部分実行

```bash
# テスト: 4/1 のみ
python analysis/scrape_historical.py --year 2024 --month 4 --max-days 1

# 途中再開: 4/15 から
python analysis/scrape_historical.py --year 2024 --month 4 --start-day 15
```

## 注意

- `.gitignore` 推奨: pkl ファイルは大きい（数GB）ので git 管理しない
- production code (src/, streamlit_app/) には一切影響しない
- DB にも書き込まない
