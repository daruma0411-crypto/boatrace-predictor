# A3 race title スクレイピング拡張 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** boatrace 公式サイトから race title (企画レース名等) を取得し新規 `race_titles` テーブルに格納。3 ヶ月 backfill + GCP daily cron で増分取得。Phase B B1 の前提条件を整える。

**Architecture:** 別テーブル方式で本番 `races` テーブル非変更。`src/scraper.py` に offline テスト可能な `scrape_race_title()` を追加し、cron スクリプトと backfill スクリプトを `scripts/` 配下に独立配置。GCP cron 登録は手動 SSH 手順を文書化。

**Tech Stack:** Python 3.x、requests + BeautifulSoup4 (既存 scraper パターン)、PostgreSQL (Railway 本番)、GCP cron。

**Spec:** `docs/superpowers/specs/2026-05-12-a3-race-title-scraping-design.md`

**Issue:** https://github.com/daruma0411-crypto/boatrace-predictor/issues/4

---

## File Structure

| Path | Action | 責務 |
|---|---|---|
| `src/scraper.py` | Modify (+~50 lines, end of file) | `scrape_race_title()` 関数追加 |
| `src/database.py` | Modify (~+10 lines in `init_database()`) | `race_titles` テーブル + index の CREATE 文追加 |
| `tests/__init__.py` | Create (空ファイル) | テストパッケージ宣言 |
| `tests/test_scrape_race_title.py` | Create | scrape_race_title() の単体テスト (fixture HTML 使用) |
| `tests/fixtures/racelist_sample.html` | Create | 実 boatrace.jp HTML を一回キャプチャしたもの |
| `tests/fixtures/racelist_no_title.html` | Create | title 抽出不能パターンの synthetic fixture |
| `scripts/scrape_race_titles.py` | Create | 増分 cron 用、当日分のみ scrape |
| `scripts/backfill_race_titles.py` | Create | 期間 backfill (`--from --to`) |
| `docs/runbooks/a3-gcp-cron-deploy.md` | Create | GCP cron 登録の手動手順書 |

---

## Task 1: HTML fixture 取得とセレクタ特定

**Files:**
- Create: `tests/fixtures/racelist_sample.html`
- Create: `tests/fixtures/racelist_no_title.html`

このタスクはコードを書く前の探索フェーズ。実 HTML を取得して title 要素の正確なセレクタを特定する。

- [ ] **Step 1: 実 racelist ページを取得して保存**

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
mkdir -p tests/fixtures
python - <<'EOF'
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
s = requests.Session()
s.verify = False
s.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
# 直近の典型的レース (適当な venue / race / date) — 公式ページが残っている期間を選ぶ
url = "https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=02&hd=20260430"
r = s.get(url, timeout=15)
print(f"Status: {r.status_code}")
print(f"Length: {len(r.text)}")
with open('tests/fixtures/racelist_sample.html', 'w', encoding='utf-8') as f:
    f.write(r.text)
print("Saved.")
EOF
```

期待: status 200、length > 10000、ファイル保存成功。失敗時 (404 等) は別の race_date / venue_id を試す。

- [ ] **Step 2: HTML から title 要素を発掘**

```bash
python - <<'EOF'
from bs4 import BeautifulSoup
with open('tests/fixtures/racelist_sample.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')
# 候補セレクタを片端から試す
candidates = [
    ('h2.heading2_titleName', soup.select('h2.heading2_titleName')),
    ('.heading2_titleName', soup.select('.heading2_titleName')),
    ('.title16_titleName', soup.select('.title16_titleName')),
    ('.title16_titleDetail__add2020', soup.select('.title16_titleDetail__add2020')),
    ('h3', soup.find_all('h3')[:3] if soup.find_all('h3') else []),
]
for selector, found in candidates:
    print(f"[{selector}] {len(found)} 件")
    for el in found[:3]:
        text = el.get_text(strip=True)
        print(f"  -> {text[:80]!r}")
EOF
```

確認: どのセレクタが「企画レース名」「グレード」「日次タイトル」に該当するかを目視で判別。例 (実行結果から判別する想定):
- `h2.heading2_titleName` などにレース大会名 (例: "サンライズ V")
- 他にデイリータイトル (例: "予選 5日目")

**判別後、テスト用に正解 title 文字列をメモしておく** (Step 4 で使う)。

- [ ] **Step 3: title 無しケースの synthetic fixture 作成**

```bash
python - <<'EOF'
# 最小の HTML を手動で組む。title 要素ありの構造を真似て中身だけ空にする
html = '''<html><head><title>Boat Race</title></head>
<body>
<div class="contentsFrame1_inner">
<h2 class="heading2_titleName"></h2>
<table><tbody><tr><td>placeholder</td></tr></tbody></table>
</div>
</body></html>'''
with open('tests/fixtures/racelist_no_title.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Saved.")
EOF
```

Step 2 で特定したセレクタ名を使うこと (上の `heading2_titleName` は仮)。

- [ ] **Step 4: 期待 title 文字列を確定**

Step 2 で発掘した title 要素のテキストを `EXPECTED_TITLE` 変数として控える。Task 2 のテストで使う。

例:
```
EXPECTED_TITLE = "サンライズ V"  # Step 2 で確認した実際の値
EXPECTED_SELECTOR = "h2.heading2_titleName"  # Step 2 で動いたセレクタ
```

- [ ] **Step 5: コミット**

```bash
git add tests/fixtures/
git commit -m "test(a3): boatrace.jp racelist HTML fixture を追加 (Phase A A3)"
```

---

## Task 2: scrape_race_title() 実装 (TDD)

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_scrape_race_title.py`
- Modify: `src/scraper.py` (末尾に追加)

- [ ] **Step 1: 空の `tests/__init__.py` を作成**

```bash
touch tests/__init__.py
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_scrape_race_title.py` を新規作成:

```python
"""scrape_race_title() の単体テスト (offline、fixture HTML 使用)

実行:
  pytest tests/test_scrape_race_title.py -v
  または
  python -m unittest tests.test_scrape_race_title
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import _parse_title_from_html

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Task 1 Step 4 で確定した値に置き換えること
EXPECTED_TITLE = "サンライズ V"


class TestScrapeRaceTitle(unittest.TestCase):
    def test_extract_title_from_real_html(self):
        with open(FIXTURE_DIR / "racelist_sample.html", encoding='utf-8') as f:
            html = f.read()
        title = _parse_title_from_html(html)
        # title が空でないこと + Task 1 Step 4 の期待値と一致
        self.assertIsNotNone(title)
        self.assertEqual(title, EXPECTED_TITLE)

    def test_extract_title_from_no_title_html(self):
        with open(FIXTURE_DIR / "racelist_no_title.html", encoding='utf-8') as f:
            html = f.read()
        title = _parse_title_from_html(html)
        # 空文字列の要素は None として扱う
        self.assertIsNone(title)

    def test_extract_title_from_malformed_html(self):
        title = _parse_title_from_html("<html><not closed")
        self.assertIsNone(title)


if __name__ == '__main__':
    unittest.main()
```

`EXPECTED_TITLE` は Task 1 Step 4 で控えた実値を入れる。プレースホルダ放置禁止。

- [ ] **Step 3: テストが失敗することを確認**

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python -m unittest tests.test_scrape_race_title -v
```

期待: `ImportError: cannot import name '_parse_title_from_html' from 'src.scraper'` (関数未定義のため)。

- [ ] **Step 4: `src/scraper.py` の末尾に実装追加**

ファイル末尾 (現在 779 行) の後に追加:

```python


def _parse_title_from_html(html):
    """HTML 文字列から race title を抽出

    Args:
        html: str (boatrace.jp /owpc/pc/race/racelist のレスポンス本文)

    Returns:
        str | None: title 文字列 (前後空白除去)。要素なし / 空文字列 / 不正 HTML 時は None
    """
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        logger.debug(f"BS4 parse error: {e}")
        return None
    # Task 1 Step 2 で確定したセレクタを使うこと
    el = soup.select_one("h2.heading2_titleName")
    if el is None:
        return None
    text = el.get_text(strip=True)
    return text or None


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
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/racelist?rno={race_number}&jcd={venue_id:02d}&hd={hd}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            logger.debug(f"HTTP {r.status_code}: {url}")
            return None
    except Exception as e:
        logger.debug(f"HTTP error {url}: {e}")
        return None
    return _parse_title_from_html(r.text)
```

`"h2.heading2_titleName"` は Task 1 Step 2 で確認した実セレクタに置き換えること。

- [ ] **Step 5: テストが PASS することを確認**

```bash
python -m unittest tests.test_scrape_race_title -v
```

期待: 3 tests, OK。

- [ ] **Step 6: コミット**

```bash
git add tests/__init__.py tests/test_scrape_race_title.py src/scraper.py
git commit -m "feat(scraper): scrape_race_title() 関数追加 + 単体テスト (A3)"
```

---

## Task 3: race_titles テーブル schema 追加 + 本番反映

**Files:**
- Modify: `src/database.py` (`init_database()` 内に CREATE 文追加)

- [ ] **Step 1: `init_database()` の最後の CREATE 文の後に追加**

`src/database.py` の `init_database()` 関数内、最後の `CREATE TABLE IF NOT EXISTS race_processing` の後に挿入:

```python
        cur.execute("""
            CREATE TABLE IF NOT EXISTS race_titles (
                race_id INTEGER PRIMARY KEY REFERENCES races(id) ON DELETE CASCADE,
                title TEXT,
                scraped_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_race_titles_title
            ON race_titles(title)
        """)
        logger.info("race_titles テーブル / index 確認")
```

挿入位置の確認: `init_database()` 関数の最後 (return 文の直前) に置く。

- [ ] **Step 2: 本番 DB に対して `init_database()` を実行**

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()
from src.database import init_database
init_database()
print("OK")
EOF
```

期待: ログに「race_titles テーブル / index 確認」が出力される。エラーなし。

- [ ] **Step 3: テーブルが存在することを直接確認**

```bash
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()
from src.database import get_db_connection
with get_db_connection() as conn:
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'race_titles'
        ORDER BY ordinal_position
    """)
    rows = cur.fetchall()
    print("race_titles columns:")
    for r in rows:
        print(f"  {r['column_name']}: {r['data_type']}")
    cur.execute("""
        SELECT indexname FROM pg_indexes WHERE tablename = 'race_titles'
    """)
    print("indexes:", [r['indexname'] for r in cur.fetchall()])
EOF
```

期待: 3 カラム (race_id, title, scraped_at) + 少なくとも 2 つの index (PK + idx_race_titles_title)。

- [ ] **Step 4: コミット**

```bash
git add src/database.py
git commit -m "feat(db): race_titles テーブル + index 追加 (A3)"
```

---

## Task 4: backfill スクリプト実装

**Files:**
- Create: `scripts/backfill_race_titles.py`

- [ ] **Step 1: スクリプト全体を作成**

`scripts/backfill_race_titles.py` を新規作成:

```python
"""race_titles バックフィル (A3 of Phase A roadmap, Issue #4)

期間内の races に対して race title を順次 scrape し、race_titles に UPSERT する。

実行例:
  python scripts/backfill_race_titles.py --from 2026-02-01 --to 2026-04-30
  python scripts/backfill_race_titles.py --from 2026-04-30 --to 2026-04-30  # smoke
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.scraper import scrape_race_title, _get_session
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

SLEEP_SEC = 0.5


def upsert_title(conn, race_id, title):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO race_titles (race_id, title, scraped_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (race_id) DO UPDATE
        SET title = EXCLUDED.title, scraped_at = NOW()
    """, (race_id, title))


def process_date(session, conn, target_date):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, venue_id, race_number
        FROM races
        WHERE race_date = %s
        ORDER BY venue_id, race_number
    """, (target_date,))
    races = cur.fetchall()
    if not races:
        logger.info(f"  {target_date}: 0 races")
        return 0, 0
    success = 0
    for r in races:
        title = scrape_race_title(session, target_date, r['venue_id'], r['race_number'])
        if title is not None:
            upsert_title(conn, r['id'], title)
            success += 1
        else:
            # title なしレースも記録 (NULL で UPSERT、再 scrape 抑止)
            upsert_title(conn, r['id'], None)
        time.sleep(SLEEP_SEC)
    conn.commit()
    logger.info(f"  {target_date}: {success}/{len(races)} 件取得成功")
    return success, len(races)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='from_', required=True, help='YYYY-MM-DD')
    parser.add_argument('--to', dest='to', required=True, help='YYYY-MM-DD')
    args = parser.parse_args()

    from_date = datetime.strptime(args.from_, '%Y-%m-%d').date()
    to_date = datetime.strptime(args.to, '%Y-%m-%d').date()
    if from_date > to_date:
        raise SystemExit("--from は --to より前である必要")

    session = _get_session()
    total_success, total = 0, 0
    with get_db_connection() as conn:
        d = from_date
        while d <= to_date:
            s, t = process_date(session, conn, d)
            total_success += s
            total += t
            d += timedelta(days=1)
    logger.info(f"=== 完了: {total_success}/{total} 件取得成功 ===")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 1 日だけの smoke テスト**

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python scripts/backfill_race_titles.py --from 2026-04-30 --to 2026-04-30
```

期待: 数十秒で完了 (1 日 ~120 race × 0.5s)。ログに `2026-04-30: <success>/<total> 件取得成功`。success が 0 件なら STOP して Task 1 で特定したセレクタを再検証。

- [ ] **Step 3: smoke テストの結果を DB で確認**

```bash
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()
from src.database import get_db_connection
with get_db_connection() as conn:
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) AS total,
               COUNT(title) AS with_title
        FROM race_titles rt
        JOIN races r ON r.id = rt.race_id
        WHERE r.race_date = '2026-04-30'
    """)
    row = cur.fetchone()
    print(f"2026-04-30: race_titles {row['total']} 件、うち title あり {row['with_title']} 件")
    cur.execute("""
        SELECT rt.title, COUNT(*) AS n
        FROM race_titles rt
        JOIN races r ON r.id = rt.race_id
        WHERE r.race_date = '2026-04-30' AND rt.title IS NOT NULL
        GROUP BY rt.title
        ORDER BY n DESC
        LIMIT 5
    """)
    print("title TOP 5:")
    for r in cur.fetchall():
        print(f"  {r['n']:3d}  {r['title']!r}")
EOF
```

期待: 行数が 2026-04-30 の races 件数 (60-120) と一致、title 充足率 ≥ 80%、TOP 5 に実際のレースタイトル文字列が出る。

- [ ] **Step 4: コミット**

```bash
git add scripts/backfill_race_titles.py
git commit -m "feat(a3): backfill_race_titles.py 追加 + 1 日 smoke 完了"
```

---

## Task 5: 3 ヶ月 backfill 本実行

**Files:**
- なし (実行のみ、出力は DB)

- [ ] **Step 1: 2026-02-01 〜 2026-04-29 を backfill 実行**

(Task 4 で 2026-04-30 は完了済み)

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python scripts/backfill_race_titles.py --from 2026-02-01 --to 2026-04-29 2>&1 | tee /tmp/a3_backfill.log
```

長時間処理 (推定 1.3-1.5 時間)。`run_in_background: true` で起動推奨。

期待: 完了時に `=== 完了: <n_success>/<n_total> 件取得成功 ===` がログ末尾。

- [ ] **Step 2: 充足率を集計**

```bash
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()
from src.database import get_db_connection
with get_db_connection() as conn:
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(DISTINCT r.id) AS races_total,
               COUNT(DISTINCT rt.race_id) AS scraped_total,
               COUNT(DISTINCT CASE WHEN rt.title IS NOT NULL THEN rt.race_id END) AS with_title
        FROM races r
        LEFT JOIN race_titles rt ON r.id = rt.race_id
        WHERE r.race_date BETWEEN '2026-02-01' AND '2026-04-30'
    """)
    row = cur.fetchone()
    rt = row['races_total']
    st = row['scraped_total']
    wt = row['with_title']
    print(f"期間 races: {rt}")
    print(f"scrape 済み:  {st} ({100*st/rt:.1f}%)")
    print(f"title あり:   {wt} ({100*wt/rt:.1f}%)")
    print(f"判定 (95% 基準): {'OK' if 100*wt/rt >= 95 else 'NG'}")
EOF
```

期待: title 充足率 ≥ 95%。未達なら Task 1 のセレクタ精度・URL を再検証して BLOCKED 報告。

- [ ] **Step 3: コミット (実行ログのみ、コードはなし)**

backfill は実行行為であり、新規 commit すべきコード変更はない。Task 5 はコミットなしで完了。

---

## Task 6: 増分 cron スクリプト実装

**Files:**
- Create: `scripts/scrape_race_titles.py`

- [ ] **Step 1: スクリプトを作成**

`scripts/scrape_race_titles.py` を新規作成:

```python
"""race_titles 増分 scrape (A3 of Phase A roadmap, Issue #4)

当日の races に対して race title を scrape し、race_titles に UPSERT する。
GCP daily cron で 09:00 JST (00:00 UTC) に実行する想定。

実行:
  python scripts/scrape_race_titles.py        # 当日 (CURRENT_DATE)
  python scripts/scrape_race_titles.py --date 2026-05-12  # 任意日付
"""
import os
import sys
import logging
import argparse
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.scraper import scrape_race_title, _get_session
from src.database import get_db_connection
from scripts.backfill_race_titles import process_date

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYY-MM-DD (省略時は当日 JST)')
    args = parser.parse_args()
    if args.date:
        target = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target = date.today()
    logger.info(f"対象日付: {target}")
    session = _get_session()
    with get_db_connection() as conn:
        success, total = process_date(session, conn, target)
    if total == 0:
        logger.info(f"races 0 件 ({target})、スクレイプ対象なし")
    else:
        logger.info(f"=== {target}: {success}/{total} 件取得成功 ===")


if __name__ == '__main__':
    main()
```

`process_date` は backfill スクリプトから再利用 (DRY)。

- [ ] **Step 2: ローカルで動作確認 (本日分)**

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python scripts/scrape_race_titles.py
```

期待: 当日のレース件数分 scrape、結果ログ表示。races 0 件の場合は「スクレイプ対象なし」。

- [ ] **Step 3: 任意日付指定でも動作確認**

```bash
python scripts/scrape_race_titles.py --date 2026-04-29
```

期待: 既に backfill 済み (Task 5) の日付なので UPSERT で再実行成功 (scraped_at 更新)。

- [ ] **Step 4: コミット**

```bash
git add scripts/scrape_race_titles.py
git commit -m "feat(a3): scrape_race_titles.py 増分 cron スクリプト追加"
```

---

## Task 7: GCP デプロイ runbook 作成

**Files:**
- Create: `docs/runbooks/a3-gcp-cron-deploy.md`

GCP への自動デプロイは行わない。操作者 (岩下) が SSH して手動実行する手順書を残す。

- [ ] **Step 1: runbook を作成**

`docs/runbooks/a3-gcp-cron-deploy.md` を新規作成:

```markdown
# A3 race title 増分 cron - GCP デプロイ手順

対象 VM: `daruma0411@34.85.123.74`
スクリプト: `scripts/scrape_race_titles.py`
依存: `scripts/backfill_race_titles.py`, `src/scraper.py`, `src/database.py`, `.env`

## 1. ファイルを GCP に配置

ローカルから:

\`\`\`bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
ssh daruma0411@34.85.123.74 "mkdir -p /opt/race-title-scraper/src /opt/race-title-scraper/scripts"
scp src/scraper.py src/database.py daruma0411@34.85.123.74:/opt/race-title-scraper/src/
scp scripts/scrape_race_titles.py scripts/backfill_race_titles.py daruma0411@34.85.123.74:/opt/race-title-scraper/scripts/
ssh daruma0411@34.85.123.74 "touch /opt/race-title-scraper/src/__init__.py /opt/race-title-scraper/scripts/__init__.py"
\`\`\`

## 2. `.env` を配置 (DB URL)

\`\`\`bash
# 本番 DB URL を確認 (memory MEMORY.md 参照)
DB_URL="postgresql://boatrace:..."
ssh daruma0411@34.85.123.74 "echo 'DATABASE_URL=$DB_URL' > /opt/race-title-scraper/.env && chmod 600 /opt/race-title-scraper/.env"
\`\`\`

## 3. 依存パッケージ install

\`\`\`bash
ssh daruma0411@34.85.123.74 "pip3 install --user requests beautifulsoup4 psycopg2-binary python-dotenv"
\`\`\`

## 4. 手動 1 回テスト

\`\`\`bash
ssh daruma0411@34.85.123.74 "cd /opt/race-title-scraper && python3 scripts/scrape_race_titles.py"
\`\`\`

期待: 当日の race title 取得ログが流れる。

## 5. cron 登録

\`\`\`bash
ssh daruma0411@34.85.123.74 "crontab -l > /tmp/cron.bak; (crontab -l 2>/dev/null; echo '0 0 * * * cd /opt/race-title-scraper && python3 scripts/scrape_race_titles.py >> /tmp/race_titles.log 2>&1') | crontab -"
ssh daruma0411@34.85.123.74 "crontab -l | grep race-title"
\`\`\`

UTC 00:00 = JST 09:00 daily。

## 6. 翌日のログ確認

\`\`\`bash
ssh daruma0411@34.85.123.74 "tail -50 /tmp/race_titles.log"
\`\`\`

期待: 翌日 09:00 以降にログが追加され、当日の races 件数分の取得成功が記録されている。

## ロールバック

\`\`\`bash
# cron 削除
ssh daruma0411@34.85.123.74 "crontab -l | grep -v race-title-scraper | crontab -"
# ファイル削除
ssh daruma0411@34.85.123.74 "rm -rf /opt/race-title-scraper"
\`\`\`
\`\`\`

- [ ] **Step 2: コミット**

```bash
git add docs/runbooks/a3-gcp-cron-deploy.md
git commit -m "docs(a3): GCP cron デプロイ手順書追加"
```

- [ ] **Step 3: 操作者 (岩下) に手順実行を依頼**

このタスクは Claude では実行しない。spec 通り、操作者が SSH で実施する。runbook を共有し、実行完了後にステップ 4-6 のログを共有してもらう。

完了報告として spec の「成功条件 3 (GCP daily cron 登録)」をクリアしたかチェックすること。

---

## Task 8: A4 レポート再実行で「A3 完了」判定

**Files:**
- Modify: `analysis/19_race_title_inventory.py` (`race_titles` テーブル対応)

A4 のレポートが「title 候補列なし」と判定したのは races テーブルだけ見ていたから。`race_titles` テーブル対応を追加して再判定する。

- [ ] **Step 1: `analysis/19_race_title_inventory.py` に race_titles 集計を追加**

`detect_title_columns()` の直後、`aggregate_db_inventory()` の前に新しい関数を追加:

```python


def aggregate_race_titles_table():
    """新規 race_titles テーブルの月別充足率を集計"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT to_char(r.race_date, 'YYYY-MM') AS ym,
                   COUNT(*) AS total,
                   COUNT(rt.title) AS filled
            FROM races r
            LEFT JOIN race_titles rt ON r.id = rt.race_id
            WHERE r.race_date BETWEEN '2024-01-01' AND '2026-04-30'
            GROUP BY ym
            ORDER BY ym
        """)
        return [(r['ym'], r['total'], r['filled']) for r in cur.fetchall()]
```

そして `make_report()` の Section 2 の直後に Section 2.5 を追加:

```python
    lines.append("\n## 2.5. race_titles テーブル充足率 (A3 後)\n\n")
    rt_rows = aggregate_race_titles_table()
    if not rt_rows:
        lines.append("race_titles テーブル空 or 未作成\n")
    else:
        lines.append("| 年月 | total | filled | 充足率 |\n|---|---|---|---|\n")
        for ym, total, filled in rt_rows:
            ratio = (filled / total * 100) if total else 0
            lines.append(f"| {ym} | {total} | {filled} | {ratio:.1f}% |\n")
```

`main()` で `aggregate_race_titles_table()` を呼び出す処理は `make_report()` 内に閉じ込めたので main は変更不要。

そして Section 4 「判定」のロジックを更新:

```python
    lines.append("\n## 4. 判定\n\n")
    if detected and monthly:
        # 既存ロジック (races テーブルの title 列がある場合)
        ...
    else:
        # races に title 列なし → race_titles テーブル方式 (A3 後)
        recent_ratios = []
        for ym, total, filled in aggregate_race_titles_table():
            if ym in ('2026-02', '2026-03', '2026-04') and total:
                recent_ratios.append(filled / total * 100)
        if recent_ratios:
            avg_recent = sum(recent_ratios) / len(recent_ratios)
            verdict = "A3 完了 (race_titles 経由で充足)" if avg_recent >= 95 else "A3 不完全 (95% 未達)"
            lines.append(f"直近3ヶ月 race_titles 平均充足率: **{avg_recent:.1f}%**\n\n")
            lines.append(f"判定: **{verdict}** (基準: 95%)\n")
        else:
            lines.append("race_titles 未作成または空。A3 発火必要\n")
```

完全な差し替えコード:

```python
    lines.append("\n## 4. 判定\n\n")
    if detected and monthly:
        recent_ratios = []
        for col, rows in monthly.items():
            for ym, total, filled in rows:
                if ym in ('2026-02', '2026-03', '2026-04') and total:
                    recent_ratios.append(filled / total * 100)
        avg_recent = sum(recent_ratios) / len(recent_ratios) if recent_ratios else 0
        verdict = "A3 スキップ可能" if avg_recent >= 95 else "A3 (スクレイピング拡張) 発火必要"
        lines.append(f"直近3ヶ月 (2026-02 〜 2026-04) の平均充足率 (races 列ベース): **{avg_recent:.1f}%**\n\n")
        lines.append(f"判定: **{verdict}** (基準: 95%)\n")
    else:
        rt_rows = aggregate_race_titles_table()
        recent_ratios = []
        for ym, total, filled in rt_rows:
            if ym in ('2026-02', '2026-03', '2026-04') and total:
                recent_ratios.append(filled / total * 100)
        if recent_ratios:
            avg_recent = sum(recent_ratios) / len(recent_ratios)
            verdict = "A3 完了 (race_titles 経由で充足)" if avg_recent >= 95 else "A3 不完全 (95% 未達)"
            lines.append(f"直近3ヶ月 race_titles 平均充足率: **{avg_recent:.1f}%**\n\n")
            lines.append(f"判定: **{verdict}** (基準: 95%)\n")
        else:
            lines.append("race_titles 未作成または空。A3 発火必要\n")
```

- [ ] **Step 2: スクリプトを再実行**

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python analysis/19_race_title_inventory.py
```

期待: ログに「レポート出力: ...race_title_inventory.md」。

- [ ] **Step 3: レポートで「A3 完了」判定を確認**

```bash
grep -A 5 "## 4. 判定" analysis/reports/race_title_inventory.md
```

期待: 「判定: **A3 完了 (race_titles 経由で充足)**」が出力されている。

- [ ] **Step 4: コミット**

```bash
git add analysis/19_race_title_inventory.py analysis/reports/race_title_inventory.md
git commit -m "feat(analysis): A4 レポートを race_titles 対応に更新、A3 完了判定"
```

---

## Self-Review

- **Spec 網羅**: spec の File Structure (9 ファイル) に対し、Task 1-8 で 8 ファイル作成 + 既存 2 ファイル修正をカバー (`analysis/19_*.py` は spec 範囲外だが Task 8 で更新)。GCP デプロイは spec 通り runbook で担保。
- **プレースホルダ**: Task 1 Step 4 の `EXPECTED_TITLE = "サンライズ V"` と Task 2 Step 4 の `"h2.heading2_titleName"` は **Task 1 Step 2 で確認した実値に置き換える必要**を明示。プレースホルダ放置は禁止と Task 内で念押し済み。
- **型整合**: `scrape_race_title()` の I/O 仕様、`_parse_title_from_html()` のヘルパー、`process_date()` の戻り値 `(success, total)` は Task 4 → Task 6 で一貫使用。
- **粒度**: 各 Task 4-6 steps、5 分以内/step を意識。長時間処理は Task 5 (backfill 1.3h) のみ、これは独立 Task として隔離済み。
- **TDD**: Task 2 で test-first / 失敗確認 / 実装 / pass 確認 のサイクルを明示。

---

## 完了後の状態

- 本番 DB に `race_titles` テーブル + index 作成済み
- 2026-02-01 〜 2026-04-30 の title バックフィル完了 (≥95% 充足)
- GCP cron が daily 09:00 JST で増分取得
- A4 レポートが「A3 完了」と判定
- Phase B B1 (企画レース分類器) の前提条件クリア
