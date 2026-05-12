---
date: 2026-05-12
topic: A3 race title スクレイピング拡張
issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
status: approved
risk: 🟡 標準 (新規テーブル + 公式サイトスクレイピング)
depends_on: Phase A A4 (完了、commit 244a2a8)
unlocks: Phase B B1 (企画レース分類器)
---

# A3 race title スクレイピング拡張 設計

## 背景

Issue #4 改善ロードマップ Phase A の A4 (race title 在庫確認) を実装した結果、本番 `races` テーブルに title 候補列は存在せず、scraped historical data にも title キーがゼロ件であることが判明した (commit 244a2a8 / `analysis/reports/race_title_inventory.md`)。

A3 はこのデータ欠如を埋めるためのスクレイピング拡張。Phase B B1 (race title → 企画レース分類器) の前提条件である。

## スコープ

| 項目 | 内容 |
|---|---|
| 範囲 | 別テーブル `race_titles` 作成 + scraper 関数追加 + 増分 cron + 3 ヶ月 backfill |
| 範囲外 | 本番 `races` テーブル変更、scheduler.py 拡張、B1 分類器実装 |
| 本番影響 | 新規テーブル追加のみ。本番アプリ (Streamlit / scheduler / purchaser) は `race_titles` を参照しないため無影響 |
| リスク判定 | 🟡 標準 |

## 成功条件

1. `race_titles` テーブルが本番 DB に作成され、index `idx_race_titles_title` が存在
2. backfill (2026-02-01 〜 2026-04-30) 実行後、`race_titles` の充足率 ≥95% (races テーブルの同期間レース数比)
3. GCP daily cron 登録され、翌日のレース分が自動取得される
4. `analysis/19_race_title_inventory.py` (A4 のスクリプト) を `race_titles` 対応に更新して再実行、判定が「A3 完了」になる

## アーキテクチャ

```
GCP VM 34.85.123.74
  └ /opt/race-title-scraper/scrape_race_titles.py  (cron 09:00 JST daily)
       ├ SELECT id, race_date, venue_id, race_number FROM races
       │   WHERE race_date = CURRENT_DATE  (READ-ONLY)
       └ for each race:
            ├ scrape_race_title(session, race_date, venue_id, race_number)
            │   → BS4 parse from boatrace.jp/owpc/pc/race/racelist?...
            └ INSERT INTO race_titles (race_id, title, scraped_at) ON CONFLICT (race_id) DO UPDATE

ローカル one-shot (workspace)
  └ scripts/backfill_race_titles.py
       ├ for race_date in 2026-02-01 〜 2026-04-30:
       │   ├ SELECT races WHERE race_date = ?
       │   └ for each race: scrape_race_title() → UPSERT
       ├ レート制限: 0.5s sleep / request
       └ 進捗ログ + UPSERT による再開可能性
```

## ファイル構造

| Path | Action | 用途 | 推定行数 |
|---|---|---|---|
| `src/scraper.py` | Modify | 末尾に `scrape_race_title()` 関数追加 (既存 `scrape_racelist` パターン準拠) | +40 |
| `src/database.py` | Modify | `CREATE TABLE IF NOT EXISTS race_titles ...` + index 追加 | +15 |
| `scripts/scrape_race_titles.py` | Create | 増分 cron 用、当日分 daily | ~120 |
| `scripts/backfill_race_titles.py` | Create | 期間 backfill (`--from`, `--to` 引数) | ~150 |
| `tests/test_scrape_race_title.py` | Create | fixture HTML での単体テスト | ~80 |
| `tests/fixtures/racelist_sample.html` | Create | テスト用 fixture (boatrace.jp の保存版) | - |

`scripts/` ディレクトリは既存。`tests/` は新規作成 (現状 boatrace-predictor は pytest 構成なし)。pytest 公式設定 (`pytest.ini` 等) は導入せず、テストファイルだけ追加。`pytest tests/` または `python -m unittest tests/test_scrape_race_title.py` のどちらでも実行可能な書き方にする (assert ベース、fixture は通常の Python ファイル import)。

## DB スキーマ

```sql
CREATE TABLE IF NOT EXISTS race_titles (
    race_id INTEGER PRIMARY KEY REFERENCES races(id) ON DELETE CASCADE,
    title TEXT,
    scraped_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_race_titles_title ON race_titles(title);
```

- `race_id` を PK + FK にすることで重複防止と参照整合性を担保
- `ON DELETE CASCADE`: races 行が削除された場合に追従 (運用上 races は削除されない想定だが防御的)
- `title` は TEXT NULL (公式サイトに title が無い古いレース等を想定)
- `scraped_at` で再 scrape 履歴を追跡

## scrape_race_title() 関数仕様

```python
def scrape_race_title(session, race_date, venue_id, race_number):
    """boatrace 公式サイトから race title を取得

    Args:
        session: requests.Session
        race_date: datetime.date
        venue_id: int (1-24)
        race_number: int (1-12)

    Returns:
        str | None: title 文字列。取得失敗 / title 無し時は None
    """
    url = f"{BASE_URL}/racelist"
    params = {'rno': race_number, 'jcd': f'{venue_id:02d}', 'hd': race_date.strftime('%Y%m%d')}
    # ...
    # title は <h2 class="heading2_titleName">企画レース名</h2> のような場所にある (要確認)
    # 戻り値: title text or None
```

注: title の正確な HTML セレクタは **実装時に boatrace.jp の生 HTML を 1 ページ取得して特定する** (`requests.get(url)` → fixture として `tests/fixtures/racelist_sample.html` に保存 → BS4 で title 要素を発掘)。`scrape_racelist` (既存) と同じ URL を使うが、本 spec では別関数として独立させて影響を局所化する。

## データフロー詳細

### 増分 cron (`scrape_race_titles.py`)

1. 環境変数から DB URL 取得 (`.env`)
2. SELECT 当日の races (`race_date = CURRENT_DATE`) — READ-ONLY
3. for each race:
   - try: `scrape_race_title()` → title 取得
   - except: ログ警告、continue (race 単位で失敗を許容)
   - INSERT INTO race_titles ... ON CONFLICT (race_id) DO UPDATE SET title=EXCLUDED.title, scraped_at=NOW()
   - 0.5s sleep
4. 完了サマリ (取得成功 / 失敗 / 0件レース) をログ出力

### backfill (`backfill_race_titles.py`)

1. CLI 引数: `--from YYYY-MM-DD --to YYYY-MM-DD` (デフォルト 2026-02-01 / 2026-04-30)
2. for race_date in 範囲:
   - SELECT races WHERE race_date = ?
   - 増分 cron と同じ処理
   - 日次サマリログ
3. UPSERT のため中断時は同じコマンドで再開可能 (既取得分はスキップ)

## 非破壊性とロールバック

- 本番 `races` テーブル: SELECT のみ。INSERT/UPDATE/DELETE/ALTER 一切なし
- 新規 `race_titles` テーブル: 本番アプリ (app.py / scheduler.py / purchaser) は参照しないため、追加しても運用無影響
- ロールバック手順:
  1. `DROP INDEX idx_race_titles_title;`
  2. `DROP TABLE race_titles;`
  3. GCP cron 削除: `crontab -e`
  4. workspace のスクリプト削除 (`git revert`)

## テスト

| テスト | 種別 | 内容 |
|---|---|---|
| `test_scrape_race_title_with_title` | unit | fixture HTML (title あり) を BS4 parse、期待 title 文字列が返る |
| `test_scrape_race_title_without_title` | unit | fixture HTML (title なし) で None が返る |
| `test_scrape_race_title_malformed` | unit | 不正 HTML で例外を投げず None |
| Smoke (manual) | integration | backfill を 1 日 (`--from 2026-04-30 --to 2026-04-30`) だけ実行、race_titles 行数 = races 同日件数 |
| A4 再実行 | regression | 19_race_title_inventory.py 更新版で `race_titles` を scan、判定「A3 完了」 |

## リスク・対処

| リスク | 対処 |
|---|---|
| 公式サイト負荷 (1 日 120 req) | 0.5s sleep/req → 60s/day。負荷無視できる |
| HTML 構造変更 | `scrape_racelist` と同じ BS4 セレクタパターン使用、変更時は既存スクレイパーも壊れるので即検出される |
| GCP cron 失敗 | ログ `/tmp/race_titles.log` 残置、weekly 確認 (Phase E 月次レポートで自動化予定) |
| backfill 中断 | UPSERT で再開可能、進捗ログで再開 race_date を特定 |
| DB 接続失敗 | リトライ 3 回 (psycopg2 標準)、最終失敗時はログ + 非 zero exit code |
| title 取得不能なレース (公式サイト未掲載期間) | title=NULL で記録、`scraped_at` を残すことで再 scrape 不要を判定 |

## デプロイ手順

GCP cron 登録は実装計画では **操作者 (岩下) が手動 SSH で実施する手順**として記載する。Claude は SSH コマンド一式と検証コマンドを提供するが、自動実行はしない。理由:
- 本番 VM への直接介入のため、ユーザー監視下で実施するのが安全
- 認証情報 (`.env`) は GCP 上に別途配置必要

手順 (実装計画で詳細化):
1. ローカルから GCP に scripts/scrape_race_titles.py + 依存物を scp
2. GCP で `.env` (DB URL) を `/opt/race-title-scraper/.env` に配置
3. `crontab -e` で `0 0 * * * cd /opt/race-title-scraper && python3 scrape_race_titles.py >> /tmp/race_titles.log 2>&1` を追加 (UTC で 00:00 = JST 09:00)
4. 翌日のログで動作確認

## 関連

- Issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
- A4 spec/results: `docs/superpowers/specs/2026-05-12-phase-a-recalibration-design.md`, `analysis/reports/race_title_inventory.md`
- vault: `30_decisions/2026-05-12-phase-a-a4-a1-findings.md`
- 既存スクレイパー参照: `src/scraper.py:103 scrape_racelist`
