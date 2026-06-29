# オッズ盤台帳（odds-board ledger）v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 予測スケジューラに「判断時の全3連単オッズ盤」を毎レース1行で保存する追記オンリーの仕組みを足し、確率(既存)＋結果(既存)と合わせて後からリプレイできる土台を作る。

**Architecture:** 純粋ロジック(`build_board_row`)＋自己隔離した保存関数(`save_odds_board`)を `src/odds_board.py` に作り、scheduler の `odds_dict` 構築直後に1レース1回呼ぶ(try/except二重保護・`ON CONFLICT DO NOTHING` で複数インスタンスに強い)。研究側は read-only ヘルパ(`analysis/ledger/`)で3テーブルを結合。本番DBの既存テーブルは無改変。

**Tech Stack:** Python 3.11（`venv/Scripts/python.exe`）、psycopg2、PostgreSQL(Railway)。テストは assert ベース、本番DBに**テスト書き込みをしない**（純粋ロジック＋fake conn で検証、実INSERTはデプロイ後に現物確認）。

**前提（実行前に必須）:**
- ブランチは `research/odds-board-ledger`（master から作成済み）。
- 本番DBは研究用途では read-only。**スケジューラの保存処理（本番運用）以外で `race_odds_board` に INSERT しない**。テストは fake conn で行い実DBを汚さない。
- スクリプト実行は `venv/Scripts/python.exe`。
- 仕様書: `docs/superpowers/specs/2026-06-29-odds-board-ledger-design.md`。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/odds_board.py` | 純粋行生成(`build_board_row`)＋自己隔離保存(`save_odds_board`)＋SQL定数 |
| `tests/test_odds_board_build.py` | `build_board_row` 単体（DB不要） |
| `tests/test_odds_board_save.py` | `save_odds_board` 単体（fake conn・成功/例外隔離） |
| `src/scheduler.py`（修正） | `odds_dict` 構築直後にキャプチャ呼び出しを1行追加 |
| `analysis/ledger/__init__.py` | 空 |
| `analysis/ledger/db.py` | read-only 接続（soft_regime 流儀） |
| `analysis/ledger/board.py` | `load_candidate_board`（盤⨝確率⨝結果） |
| `analysis/ledger/test_board.py` | read helper 単体（read-only・空でも壊れない） |
| `analysis/ledger/verify_capture.py` | デプロイ後の現物検証（成功条件ゲート） |

---

## Task 1: src/odds_board.py — 純粋行生成 + SQL定数

**Files:**
- Create: `src/odds_board.py`
- Test: `tests/test_odds_board_build.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_odds_board_build.py`:
```python
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.odds_board import build_board_row

def test_build_board_row_basic():
    odds = {"1-2-3": 45.6, "1-2-4": 88.1, "3-1-5": 12.0}
    rid, j, n = build_board_row(777, odds)
    assert rid == 777
    assert n == 3
    assert json.loads(j) == odds

def test_build_board_row_casts_int():
    rid, j, n = build_board_row("888", {"1-2-3": 10.0})
    assert rid == 888 and n == 1

if __name__ == "__main__":
    test_build_board_row_basic(); test_build_board_row_casts_int(); print("ALL PASS")
```

- [ ] **Step 2: テスト実行（失敗確認）**

Run: `venv/Scripts/python.exe tests/test_odds_board_build.py`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.odds_board'`）

- [ ] **Step 3: 最小実装**

`src/odds_board.py`:
```python
"""オッズ盤台帳: 判断時の全3連単オッズ盤を1レース1行で保存する（追記オンリー・自己隔離）。"""
import json
import logging

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS race_odds_board (
    race_id     integer PRIMARY KEY,
    captured_at timestamptz NOT NULL DEFAULT now(),
    odds_3t     jsonb NOT NULL,
    n_combos    integer NOT NULL
)
"""

INSERT_SQL = """
INSERT INTO race_odds_board (race_id, odds_3t, n_combos)
VALUES (%s, %s, %s)
ON CONFLICT (race_id) DO NOTHING
"""

HEALTH_SQL = "INSERT INTO scheduler_health (status, detail) VALUES (%s, %s)"


def build_board_row(race_id, odds_dict):
    """(race_id:int, odds_json:str, n_combos:int) を返す純粋関数。"""
    return (int(race_id), json.dumps(odds_dict), len(odds_dict))
```

- [ ] **Step 4: テスト実行（成功確認）**

Run: `venv/Scripts/python.exe tests/test_odds_board_build.py`
Expected: `ALL PASS`

- [ ] **Step 5: コミット**

```bash
git add src/odds_board.py tests/test_odds_board_build.py
git commit -m "feat(odds-board): add pure build_board_row + SQL constants"
```

---

## Task 2: src/odds_board.py — save_odds_board（自己隔離・冪等）

**Files:**
- Modify: `src/odds_board.py`（`save_odds_board` を追加）
- Test: `tests/test_odds_board_save.py`

**設計:** 例外を**絶対に呼び出し元へ投げない**（予測・購入を止めない）。fake conn を注入してDBなしで検証。成功時は CREATE→INSERT(ON CONFLICT)→health の3 execute、例外時は False を返す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_odds_board_save.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager
from src.odds_board import save_odds_board

class _FakeCur:
    def __init__(self, log): self.log = log
    def execute(self, sql, params=None): self.log.append((sql.strip().split()[0], params))

class _FakeConn:
    def __init__(self, log): self.log = log
    def cursor(self): return _FakeCur(self.log)

def _factory(log, raise_it=False):
    @contextmanager
    def f():
        if raise_it:
            raise RuntimeError("db down")
        yield _FakeConn(log)
    return f

def test_save_success_executes_insert_and_health():
    log = []
    ok = save_odds_board(101, {"1-2-3": 9.9, "1-2-4": 5.0}, conn_factory=_factory(log))
    assert ok is True
    verbs = [v for v, _ in log]
    assert "CREATE" in verbs and "INSERT" in verbs
    # race_odds_board への INSERT パラメータに n_combos=2 が入る
    inserts = [p for v, p in log if v == "INSERT" and p and p[0] == 101]
    assert inserts and inserts[0][2] == 2

def test_save_empty_odds_returns_false():
    assert save_odds_board(102, {}, conn_factory=_factory([])) is False

def test_save_never_raises_on_db_error():
    # DB が落ちても例外を投げず False（予測・購入を止めない保証）
    assert save_odds_board(103, {"1-2-3": 9.9}, conn_factory=_factory([], raise_it=True)) is False

if __name__ == "__main__":
    test_save_success_executes_insert_and_health()
    test_save_empty_odds_returns_false()
    test_save_never_raises_on_db_error()
    print("ALL PASS")
```

- [ ] **Step 2: テスト実行（失敗確認）**

Run: `venv/Scripts/python.exe tests/test_odds_board_save.py`
Expected: FAIL（`ImportError: cannot import name 'save_odds_board'`）

- [ ] **Step 3: 最小実装（`src/odds_board.py` の末尾に追加）**

```python
def save_odds_board(race_id, odds_dict, conn_factory=None):
    """判断時オッズ盤を1レース1行で保存。例外は投げず bool を返す（本処理を止めない）。"""
    if not odds_dict:
        return False
    if conn_factory is None:
        from src.database import get_db_connection
        conn_factory = get_db_connection
    try:
        row = build_board_row(race_id, odds_dict)
        with conn_factory() as conn:
            cur = conn.cursor()
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(INSERT_SQL, row)
            cur.execute(HEALTH_SQL, ("odds_board_saved", f"race_id={race_id} n={row[2]}"))
        return True
    except Exception as e:
        logger.warning(f"odds_board 保存失敗(本処理に影響なし) race_id={race_id}: {e}")
        try:
            with conn_factory() as conn:
                conn.cursor().execute(
                    HEALTH_SQL, ("odds_board_failed", f"race_id={race_id}: {str(e)[:200]}")
                )
        except Exception:
            pass
        return False
```

- [ ] **Step 4: テスト実行（成功確認）**

Run: `venv/Scripts/python.exe tests/test_odds_board_save.py`
Expected: `ALL PASS`

- [ ] **Step 5: コミット**

```bash
git add src/odds_board.py tests/test_odds_board_save.py
git commit -m "feat(odds-board): add self-isolated idempotent save_odds_board"
```

---

## Task 3: scheduler.py へ配線（odds_dict 構築直後に1レース1回）

**Files:**
- Modify: `src/scheduler.py`（`odds_dict = self._parse_odds(...)` の直後、現状 742 行付近）

**設計:** `save_odds_board` 自体が例外を投げないが、呼び出し側も try/except で二重保護。1レース1回（戦略ループの外）。

- [ ] **Step 1: 修正を入れる**

`src/scheduler.py` の以下の行（`odds_dict` 構築）:
```python
            odds_dict = self._parse_odds(odds_data) if odds_data else {}
```
の**直後**に次を挿入:
```python
            # オッズ盤台帳: 判断時の全オッズ盤を1レース1行で保存（research replay 土台）
            # save_odds_board は self-isolated だが呼び出し側でも二重保護
            try:
                if odds_dict:
                    from src.odds_board import save_odds_board
                    save_odds_board(race['race_id'], odds_dict)
            except Exception as e:
                logger.warning(f"odds_board capture 呼び出し失敗(無視): {e}")
```

- [ ] **Step 2: 構文・import スモークテスト（重い依存を読み込まずに検証）**

Run:
```bash
venv/Scripts/python.exe -c "import ast; ast.parse(open('src/scheduler.py', encoding='utf-8').read()); from src.odds_board import save_odds_board; print('OK')"
```
Expected: `OK`（scheduler.py が構文エラー無し＋`save_odds_board` がインポート可能）

- [ ] **Step 3: 配線箇所の目視確認**

Run: `git diff src/scheduler.py`
Expected: `odds_dict = self._parse_odds(...)` の直後に上記ブロックだけが追加され、他は無変更。

- [ ] **Step 4: コミット**

```bash
git add src/scheduler.py
git commit -m "feat(odds-board): capture odds board once per race in scheduler (isolated)"
```

> 注: スケジューラの実INSERTは本番でしか走らないため、**動作確認は Task 5（デプロイ後の現物検証）**で行う。ここまではローカルで実DBに書き込まない。

---

## Task 4: analysis/ledger — read-only 結合ヘルパ

**Files:**
- Create: `analysis/ledger/__init__.py`（空）
- Create: `analysis/ledger/db.py`
- Create: `analysis/ledger/board.py`
- Test: `analysis/ledger/test_board.py`

**設計:** `race_odds_board ⨝ races ⨝ predictions(marginals)` を読む。確率は model 共通なので 1 レース 1 行に圧縮（`DISTINCT ON`）。デプロイ前は盤データが空なので、テストは「**動いてリストを返す／空でも壊れない／非空なら期待キーを持つ**」を assert。

- [ ] **Step 1: 失敗するテストを書く**

`analysis/ledger/test_board.py`:
```python
from db import connect
from board import load_candidate_board, ROW_KEYS

def test_load_runs_and_shape():
    conn = connect()
    rows = load_candidate_board(conn, "2026-06-01", "2026-12-31")
    assert isinstance(rows, list)              # 動いてリストを返す（空でも可）
    if rows:                                    # 盤データがあれば形を確認
        for k in ROW_KEYS:
            assert k in rows[0], k
        assert isinstance(rows[0]["odds_3t"], dict)
    print("OK n_rows=", len(rows))

if __name__ == "__main__":
    test_load_runs_and_shape(); print("ALL PASS")
```

- [ ] **Step 2: テスト実行（失敗確認）**

Run: `venv/Scripts/python.exe analysis/ledger/test_board.py`
Expected: FAIL（`ModuleNotFoundError: No module named 'db'` または `board`）

- [ ] **Step 3: 最小実装**

`analysis/ledger/db.py`:
```python
import os
import psycopg2
from psycopg2.extras import RealDictCursor

DSN = os.environ.get(
    "BR_DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable",
)


def connect():
    conn = psycopg2.connect(DSN, connect_timeout=25, cursor_factory=RealDictCursor)
    conn.set_session(readonly=True, autocommit=True)
    return conn
```

`analysis/ledger/board.py`:
```python
ROW_KEYS = [
    "race_id", "race_date", "venue_id", "race_number",
    "odds_3t", "n_combos",
    "probabilities_1st", "probabilities_2nd", "probabilities_3rd",
    "result_1st", "result_2nd", "result_3rd", "payout_sanrentan",
]

SQL = """
SELECT DISTINCT ON (ob.race_id)
       ob.race_id, r.race_date, r.venue_id, r.race_number,
       ob.odds_3t, ob.n_combos,
       p.probabilities_1st, p.probabilities_2nd, p.probabilities_3rd,
       r.result_1st, r.result_2nd, r.result_3rd, r.payout_sanrentan
FROM race_odds_board ob
JOIN races r ON r.id = ob.race_id
JOIN predictions p ON p.race_id = ob.race_id
WHERE r.race_date BETWEEN %s AND %s AND r.result_1st IS NOT NULL
ORDER BY ob.race_id
"""


def load_candidate_board(conn, date_from, date_to):
    """盤⨝結果⨝確率(marginals) を 1レース1行で返す（read-only）。"""
    cur = conn.cursor()
    # race_odds_board が未作成の本番初期でも壊れないよう存在確認
    cur.execute("SELECT to_regclass('public.race_odds_board')")
    if cur.fetchone()["to_regclass"] is None:
        return []
    cur.execute(SQL, (date_from, date_to))
    return [dict(r) for r in cur.fetchall()]
```

- [ ] **Step 4: テスト実行（成功確認）**

Run: `venv/Scripts/python.exe analysis/ledger/test_board.py`
Expected: `ALL PASS`（デプロイ前は `OK n_rows= 0` でも可）

- [ ] **Step 5: コミット**

```bash
git add analysis/ledger/__init__.py analysis/ledger/db.py analysis/ledger/board.py analysis/ledger/test_board.py
git commit -m "feat(odds-board): add read-only candidate-board loader"
```

---

## Task 5: デプロイ後の現物検証スクリプト（成功条件ゲート）

**Files:**
- Create: `analysis/ledger/verify_capture.py`

**設計:** 仕様 §2 の成功条件を機械チェック：①当日 `race_odds_board` に行がある ②`n_combos` が概ね 120 ③**同レースの採用 bet のオッズが盤の同じ組み合わせの値と一致**。これは**デプロイ後**に実行（V11.5 のサイレント失敗を二度と起こさないためのゲート）。

- [ ] **Step 1: 実装**

`analysis/ledger/verify_capture.py`:
```python
import sys
from db import connect


def main():
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT to_regclass('public.race_odds_board') AS t")
    if cur.fetchone()["t"] is None:
        print("NG: race_odds_board テーブルが存在しない（未デプロイ/未発火）"); return 1

    cur.execute("""SELECT count(*) n, min(n_combos) mn, max(n_combos) mx,
                          min(captured_at)::date d0, max(captured_at)::date d1
                   FROM race_odds_board""")
    s = cur.fetchone()
    print(f"行数={s['n']} n_combos={s['mn']}〜{s['mx']} 期間={s['d0']}〜{s['d1']}")
    if s["n"] == 0:
        print("NG: 行が0（キャプチャ未発火＝サイレント失敗の疑い）"); return 1

    # 採用bet のオッズ と 盤 の一致クロスチェック（直近で bet がある盤レースを1件）
    cur.execute("""
        SELECT b.race_id, b.combination, b.odds bet_odds, (ob.odds_3t ->> b.combination)::float board_odds
        FROM bets b JOIN race_odds_board ob ON ob.race_id = b.race_id
        WHERE ob.odds_3t ? b.combination
        ORDER BY b.id DESC LIMIT 5
    """)
    rows = cur.fetchall()
    if not rows:
        print("注意: 盤とbetが揃うレースがまだ無い（蓄積待ち）。テーブル/行は健全。")
        return 0
    ok = True
    for r in rows:
        match = r["board_odds"] is not None and abs(r["bet_odds"] - r["board_odds"]) < 0.05 * max(1.0, r["bet_odds"])
        print(f"  race={r['race_id']} {r['combination']} bet={r['bet_odds']} board={r['board_odds']} {'OK' if match else 'MISMATCH'}")
        ok = ok and match
    print("PASS: 盤と採用betのオッズが一致" if ok else "NG: オッズ不一致あり")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: コミット**

```bash
git add analysis/ledger/verify_capture.py
git commit -m "feat(odds-board): add post-deploy capture verification script"
```

- [ ] **Step 3: （デプロイ後）現物検証**

岩下さんが Railway を再デプロイ後に実行:
Run: `venv/Scripts/python.exe analysis/ledger/verify_capture.py`
Expected: 行数>0、n_combos≈120、`PASS: 盤と採用betのオッズが一致`。**ここまで確認できて初めて v1 完了**。

---

## 完了条件
- Task 1,2,4 の `test_*.py` が PASS、Task 3 のスモークテストが `OK`。
- デプロイ後 Task 5 の verify が PASS（行が貯まり始め・n_combos≈120・オッズ一致）。
- **本番運用（予測・購入・既存保存）に一切の悪影響が無い**ことを `scheduler_health` と現物で確認。
- v1 では本番DBへのテスト書き込みを一切行わない（純粋ロジック＋fake conn＋デプロイ後の現物確認で担保）。

## Self-Review（作成者チェック結果）
- **Spec coverage**: §4データフロー→Task3 / §5スキーマ→Task1(SQL)+Task2(実行) / §6安全策(try/except二重・upsert・health)→Task2,3 / §7 read helper→Task4 / §8テスト→各Task / §2成功条件→Task5。全カバー。
- **Placeholder scan**: TBD/TODO無し。各ステップに実コード・実コマンド・期待出力あり。
- **Type consistency**: `build_board_row`→`(int,str,int)` を Task2 が `row[2]=n_combos` で利用。`save_odds_board(race_id, odds_dict, conn_factory=None)` を Task3 が `save_odds_board(race['race_id'], odds_dict)` で呼ぶ。`connect()`(db) / `load_candidate_board`,`ROW_KEYS`(board) を Task4 テスト・Task5 が一貫使用。`race_odds_board(race_id,captured_at,odds_3t,n_combos)` を全 Task で一致。
- **本番書き込み回避**: テストは fake conn（Task2）と read-only（Task4,5）のみ。実INSERTはデプロイ後の本番スケジューラだけ。read helper は未作成テーブルでも `to_regclass` で空を返す。
