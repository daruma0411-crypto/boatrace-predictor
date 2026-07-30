# P改ロールアウト Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** V11の紙運用を撤去し、P改（P＋堅見送り＋低温半額）と P全場（mc_venue_all）を本番スケジューラの紙運用として実装、ダッシュボードに¥200,000スタートで表示する。あわせて台帳1ヶ月分で「P式×全24場」のオフライン一次読みを出す。

**Architecture:** 凍結パラメータは生成スクリプトが `src/softness_params.py` に焼き込み、`src/softness.py`（採点）と `src/p_kai.py`（bet導出・純関数）が使う。レース軟らかさは新テーブル `race_softness` に odds_board と同じ自己隔離パターンで記録し、温度は前日まで7日の日次平均。スケジューラは P の picks から p_kai bets を派生させ通常の保存ループに乗せる。ダッシュボードは既存の STRATEGY_NAMES/ORDER/RESET_DATES 機構に追記。実弾なし（purchaser は DRY のまま）。

**Tech Stack:** Python 3.11（`venv/Scripts/python.exe`）、psycopg2、numpy。既存の `analysis/soft_regime/`（学習ロジック）・`src/odds_board.py`（自己隔離パターン）を再利用。

**前提:**
- ブランチ `feat/p-kai-rollout`（作成済み）。master には触れない（マージは最後）。
- 仕様書: `docs/superpowers/specs/2026-07-30-p-kai-rollout-design.md`
- テストは `venv/Scripts/python.exe <path>` で実行。研究系はDB read-only。**書込みは (a)本番スケジューラ経由 (b)新テーブル `race_softness` へのバックフィルのみ**（既存テーブルへのローカル書込み厳禁）。
- 重要な既知事項: グローバルフィルタ（VENUE_BLACKLIST {7,15,16,18,19,21}・1号艇軸スキップ等）は全戦略に効くため、mc_venue_all の実カバレッジは18場程度になる（仕様通り、驚かないこと）。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `scripts/generate_softness_params.py` | soft_regime の学習を1回実行し `src/softness_params.py` を生成（凍結） |
| `src/softness_params.py` | 自動生成される凍結定数（W/MU/SD/HARD_CUT/TEMP_THR/COLD_SCALE/TEMP_WINDOW） |
| `src/softness.py` | `score_race(boats_data) -> float|None`（本番用の軟らかさ採点） |
| `src/softness_store.py` | `save_race_softness`（自己隔離・冪等）＋ `get_temperature(date)` |
| `scripts/backfill_race_softness.py` | 過去races から race_softness を一括投入（新テーブルのみ） |
| `src/p_kai.py` | `derive_p_kai_bets(p_bets, score, temperature) -> list`（純関数） |
| `src/betting.py`（修正） | ACTIVE_STRATEGIES: `v11_var13` 削除・`mc_venue_all` 追加 |
| `config/betting_config.json`（修正） | `mc_venue_all` エントリ追加（P のクローン、include_venues 無し） |
| `src/scheduler.py`（修正） | 軟らかさ記録＋p_kai 派生（各1ブロック、自己隔離） |
| `streamlit_app/app.py`・`streamlit_app/components/db_utils.py`（修正） | V11非表示・新カード2枚・¥200,000リセット |
| `analysis/venue_expand/replay.py` ほか | 台帳1ヶ月の全場リプレイ＋忠実度ゲート＋REPORT.md |
| `scripts/verify_p_kai_rollout.py` | デプロイ後の現物検証（read-only） |
| `tests/test_softness.py` `tests/test_softness_store.py` `tests/test_p_kai.py` `tests/test_rollout_config.py` | 各単体テスト |

---

## Task 1: 凍結パラメータ生成 + src/softness.py

**Files:** Create `scripts/generate_softness_params.py`, `src/softness_params.py`(生成物), `src/softness.py`; Test `tests/test_softness.py`

- [ ] **Step 1: 生成スクリプトを書く** — `scripts/generate_softness_params.py`:
```python
"""soft_regime の train(〜5/15) 学習を1回実行し、凍結定数を src/softness_params.py に書き出す。
実行: venv/Scripts/python.exe scripts/generate_softness_params.py  (DB read-only)"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "soft_regime"))
from db import connect, SPLIT_DATE, ARE_TRAIN_START
from features import race_features, FEATURE_NAMES
from stage_a_softscore import _payout_map, _matrix

conn = connect()
feats = race_features(conn, ARE_TRAIN_START, "2026-12-31")
payout = _payout_map(conn, ARE_TRAIN_START, "2026-12-31")
ids = [r for r in feats if r in payout]
train_ids = [r for r in ids if str(feats[r]["race_date"]) < SPLIT_DATE]
Xtr, ytr = _matrix(feats, payout, train_ids)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
w, *_ = np.linalg.lstsq(np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]), ytr, rcond=None)
tr_sc = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]) @ w
hard_cut = float(np.percentile(tr_sc, 33.333))
# 温度閾値: 全日の日次平均→前日まで7日平均→train中央値 (p_kai_track.build_model と同一)
from collections import defaultdict
allids = ids
Xall, _ = _matrix(feats, payout, allids)
sca = np.hstack([(Xall-mu)/sd, np.ones((len(Xall),1))]) @ w
day_scores = defaultdict(list)
for i, rid in enumerate(allids):
    day_scores[str(feats[rid]["race_date"])].append(float(sca[i]))
day_mean = {d: float(np.mean(v)) for d, v in day_scores.items()}
days_sorted = sorted(day_mean)
temp = {}
for i, d in enumerate(days_sorted):
    prev = [day_mean[x] for x in days_sorted[max(0, i-7):i]]
    if prev: temp[d] = float(np.mean(prev))
temp_thr = float(np.median([temp[d] for d in days_sorted if d < SPLIT_DATE and d in temp]))

out = os.path.join(ROOT, "src", "softness_params.py")
with open(out, "w", encoding="utf-8") as f:
    f.write('"""自動生成 (scripts/generate_softness_params.py, 2026-07-30 凍結)。手編集禁止。\n'
            'P改の定義変更は別名の新定義として登録すること。"""\n')
    f.write(f"FEATURE_NAMES = {list(FEATURE_NAMES)!r}\n")
    f.write(f"MU = {[float(x) for x in mu]!r}\n")
    f.write(f"SD = {[float(x) for x in sd]!r}\n")
    f.write(f"W = {[float(x) for x in w]!r}\n")
    f.write(f"HARD_CUT = {hard_cut!r}\n")
    f.write(f"TEMP_THR = {temp_thr!r}\n")
    f.write("COLD_SCALE = 0.5\nTEMP_WINDOW = 7\n")
print("wrote", out, "hard_cut=", hard_cut, "temp_thr=", temp_thr)
```

- [ ] **Step 2: 実行して生成** — Run: `venv/Scripts/python.exe scripts/generate_softness_params.py`
Expected: `wrote ...src/softness_params.py hard_cut= 7.86... temp_thr= 7.97...`（p_kai_track の 7.863/7.972 と一致すること。大きく違えば STOP して報告）

- [ ] **Step 3: 失敗するテストを書く** — `tests/test_softness.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.softness import score_race
from src.softness_params import HARD_CUT, TEMP_THR

def _boat(n, wr, motor, st):
    return {"boat_number": n, "win_rate": wr, "motor_win_rate_2": motor, "avg_st": st}

def test_score_race_basic():
    boats = [_boat(i+1, 5.0+i*0.3, 35.0+i, 0.15) for i in range(6)]
    s = score_race(boats)
    assert isinstance(s, float)
    assert 7.86 < HARD_CUT < 7.87 and 7.97 < TEMP_THR < 7.98  # 凍結値の再確認

def test_score_race_insufficient_data_returns_none():
    boats = [_boat(1, 5.0, 35.0, 0.15), _boat(2, None, None, None),
             _boat(3, None, None, None), _boat(4, None, None, None),
             _boat(5, None, None, None), _boat(6, None, None, None)]
    assert score_race(boats) is None   # 有効win_rateが4艇未満

def test_score_direction():
    # イン(1号艇)が強く格差が大きい=堅い → 点数は低いはず
    hard = [_boat(1, 8.0, 50.0, 0.12)] + [_boat(i, 4.0, 30.0, 0.18) for i in range(2, 7)]
    soft = [_boat(i, 5.5, 35.0, 0.15) for i in range(1, 7)]
    assert score_race(hard) < score_race(soft)

if __name__ == "__main__":
    test_score_race_basic(); test_score_race_insufficient_data_returns_none(); test_score_direction()
    print("ALL PASS")
```

- [ ] **Step 4: 失敗確認** — Run: `venv/Scripts/python.exe tests/test_softness.py` → `ModuleNotFoundError: No module named 'src.softness'`

- [ ] **Step 5: 実装** — `src/softness.py`（`analysis/soft_regime/features.py` の特徴計算を boats_data 用に移植。順序は FEATURE_NAMES = ["wr_std","wr_top_gap","inside_wr","motor_std","st_std"]）:
```python
"""P改用: レース軟らかさ採点（凍結パラメータ使用）。features.py の計算を boats_data 用に移植。"""
import numpy as np
from src.softness_params import FEATURE_NAMES, MU, SD, W


def _f(v):
    return float(v) if v is not None else float("nan")


def score_race(boats_data):
    """boats_data(本番形式のdictリスト) -> 軟らかさ点数。計算不能なら None。高いほど緩い。"""
    try:
        rows = [( b.get("boat_number"), _f(b.get("win_rate")),
                  _f(b.get("motor_win_rate_2")), _f(b.get("avg_st")) ) for b in boats_data]
        wrs = np.array([x[1] for x in rows], dtype=float)
        valid = wrs[~np.isnan(wrs)]
        if len(rows) < 4 or len(valid) < 4:
            return None
        motors = np.array([x[2] for x in rows], dtype=float)
        sts = np.array([x[3] for x in rows], dtype=float)
        wr_sorted = np.sort(valid)[::-1]
        top_gap = float(wr_sorted[0] - wr_sorted[1]) if len(wr_sorted) >= 2 else 0.0
        inside = [x[1] for x in rows if x[0] == 1]
        inside_wr = float(inside[0]) if inside and not np.isnan(inside[0]) else float(np.nanmean(wrs))
        feat = {
            "wr_std": float(np.nanstd(wrs)),
            "wr_top_gap": top_gap,
            "inside_wr": inside_wr,
            "motor_std": float(np.nanstd(motors)) if not np.isnan(motors).all() else 0.0,
            "st_std": float(np.nanstd(sts)) if not np.isnan(sts).all() else 0.0,
        }
        x = np.array([feat[n] for n in FEATURE_NAMES], dtype=float)
        z = (x - np.array(MU)) / np.array(SD)
        return float(np.dot(np.append(z, 1.0), np.array(W)))
    except Exception:
        return None
```

- [ ] **Step 6: 成功確認** — Run: `venv/Scripts/python.exe tests/test_softness.py` → `ALL PASS`

- [ ] **Step 7: DBスポット照合（忠実度）** — Run:
```bash
venv/Scripts/python.exe -c "
import sys, os; sys.path.insert(0, '.'); sys.path.insert(0, 'analysis/soft_regime')
from db import connect, ARE_TRAIN_START
from features import race_features
from stage_a_softscore import _payout_map, _matrix
import numpy as np
from src.softness import score_race
from src.softness_params import MU, SD, W
from psycopg2.extras import RealDictCursor
conn = connect(); cur = conn.cursor(cursor_factory=RealDictCursor)
cur.execute(\"SELECT race_id FROM race_odds_board ORDER BY race_id DESC LIMIT 5\")
ok = 0
for r in cur.fetchall():
    rid = r['race_id']
    cur.execute('SELECT * FROM boats WHERE race_id=%s ORDER BY boat_number', (rid,))
    boats = [dict(b) for b in cur.fetchall()]
    s = score_race(boats)
    print(rid, s)
    if s is not None: ok += 1
assert ok >= 3
print('OK spot check')"
```
Expected: 5レースのスコアが出て `OK spot check`（値は7〜9近辺の実数）

- [ ] **Step 8: コミット**
```bash
git add scripts/generate_softness_params.py src/softness_params.py src/softness.py tests/test_softness.py
git commit -m "feat(p-kai): frozen softness params generator + production scorer"
```

---

## Task 2: race_softness 保存＋温度（softness_store）＋バックフィル

**Files:** Create `src/softness_store.py`, `scripts/backfill_race_softness.py`; Test `tests/test_softness_store.py`

- [ ] **Step 1: 失敗するテスト** — `tests/test_softness_store.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager
from src.softness_store import save_race_softness, get_temperature

class _Cur:
    def __init__(self, log, rows=None): self.log = log; self._rows = rows or []
    def execute(self, sql, params=None): self.log.append((sql.strip().split()[0].upper(), params))
    def fetchall(self): return self._rows

class _Conn:
    def __init__(self, log, rows=None): self.log = log; self._rows = rows or []
    def cursor(self): return _Cur(self.log, self._rows)

def _factory(log, rows=None, raise_it=False):
    @contextmanager
    def f():
        if raise_it: raise RuntimeError("db down")
        yield _Conn(log, rows)
    return f

def test_save_success_and_never_raises():
    log = []
    assert save_race_softness(1, "2026-07-30", 8.1, conn_factory=_factory(log)) is True
    verbs = [v for v, _ in log]
    assert "CREATE" in verbs and "INSERT" in verbs
    assert save_race_softness(2, "2026-07-30", 8.1, conn_factory=_factory([], raise_it=True)) is False
    assert save_race_softness(3, "2026-07-30", None, conn_factory=_factory([])) is False  # score無しは保存しない

def test_temperature_mean_of_daily_means():
    rows = [{"d": "2026-07-23", "m": 8.0}, {"d": "2026-07-24", "m": 9.0}, {"d": "2026-07-25", "m": 7.0}]
    t = get_temperature("2026-07-30", conn_factory=_factory([], rows=rows))
    assert abs(t - 8.0) < 1e-9

def test_temperature_insufficient_days_returns_none():
    rows = [{"d": "2026-07-29", "m": 8.0}]
    assert get_temperature("2026-07-30", conn_factory=_factory([], rows=rows)) is None
    assert get_temperature("2026-07-30", conn_factory=_factory([], raise_it=True)) is None

if __name__ == "__main__":
    test_save_success_and_never_raises(); test_temperature_mean_of_daily_means()
    test_temperature_insufficient_days_returns_none(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — Run: `venv/Scripts/python.exe tests/test_softness_store.py` → `ImportError`

- [ ] **Step 3: 実装** — `src/softness_store.py`（odds_board と同じ自己隔離パターン）:
```python
"""race_softness の保存(自己隔離・冪等)と市場温度の取得。"""
import logging
from src.softness_params import TEMP_WINDOW

logger = logging.getLogger(__name__)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS race_softness (
    race_id   integer PRIMARY KEY,
    race_date date NOT NULL,
    score     real NOT NULL
)
"""
INSERT_SQL = """
INSERT INTO race_softness (race_id, race_date, score)
VALUES (%s, %s, %s) ON CONFLICT (race_id) DO NOTHING
"""
TEMP_SQL = """
SELECT race_date::text AS d, avg(score) AS m FROM race_softness
WHERE race_date >= (%s::date - %s * interval '1 day') AND race_date < %s::date
GROUP BY race_date
"""


def save_race_softness(race_id, race_date, score, conn_factory=None):
    """1レース1回 upsert。例外は投げない（本処理を止めない）。score None は保存しない。"""
    if score is None:
        return False
    try:
        if conn_factory is None:
            from src.database import get_db_connection
            conn_factory = get_db_connection
        with conn_factory() as conn:
            cur = conn.cursor()
            cur.execute(CREATE_SQL)
            cur.execute(INSERT_SQL, (int(race_id), str(race_date), float(score)))
        return True
    except Exception as e:
        logger.warning(f"race_softness 保存失敗(無視) race_id={race_id}: {e}")
        return False


def get_temperature(target_date, conn_factory=None):
    """前日までの直近 TEMP_WINDOW 日の日次平均スコアの平均。<3日 or 失敗は None。"""
    try:
        if conn_factory is None:
            from src.database import get_db_connection
            conn_factory = get_db_connection
        with conn_factory() as conn:
            cur = conn.cursor()
            cur.execute(TEMP_SQL, (str(target_date), TEMP_WINDOW, str(target_date)))
            rows = cur.fetchall()
        means = [float(r["m"]) for r in rows]
        if len(means) < 3:
            return None
        return sum(means) / len(means)
    except Exception as e:
        logger.warning(f"温度取得失敗(無視) date={target_date}: {e}")
        return None
```

- [ ] **Step 4: 成功確認** — Run: `venv/Scripts/python.exe tests/test_softness_store.py` → `ALL PASS`

- [ ] **Step 5: バックフィルスクリプト** — `scripts/backfill_race_softness.py`（書込みは race_softness のみ）:
```python
"""過去レースの軟らかさを race_softness に一括投入（デプロイ前後に1回実行）。
書込み先は新テーブル race_softness のみ。既存テーブルには一切書かない。"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import psycopg2
from psycopg2.extras import RealDictCursor
from src.softness import score_race
from src.softness_store import CREATE_SQL, INSERT_SQL

DSN = os.environ.get("DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable")
SINCE = "2026-06-15"   # 温度に十分な過去分

conn = psycopg2.connect(DSN, cursor_factory=RealDictCursor)
cur = conn.cursor()
cur.execute(CREATE_SQL)
cur.execute("SELECT id, race_date FROM races WHERE race_date >= %s ORDER BY id", (SINCE,))
races = cur.fetchall()
n_ok = 0
for r in races:
    cur.execute("SELECT * FROM boats WHERE race_id=%s ORDER BY boat_number", (r["id"],))
    boats = [dict(b) for b in cur.fetchall()]
    s = score_race(boats)
    if s is not None:
        cur.execute(INSERT_SQL, (r["id"], str(r["race_date"]), float(s)))
        n_ok += 1
conn.commit()
print(f"backfilled {n_ok}/{len(races)} races since {SINCE}")
```

- [ ] **Step 6: コミット**
```bash
git add src/softness_store.py scripts/backfill_race_softness.py tests/test_softness_store.py
git commit -m "feat(p-kai): race_softness store (isolated) + temperature + backfill script"
```

---

## Task 3: p_kai bet 導出（純関数）

**Files:** Create `src/p_kai.py`; Test `tests/test_p_kai.py`

- [ ] **Step 1: 失敗するテスト** — `tests/test_p_kai.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.p_kai import derive_p_kai_bets
from src.softness_params import HARD_CUT, TEMP_THR

P_BETS = [{"bet_type": "sanrentan", "combination": "1-2-3", "amount": 1300, "odds": 12.0,
           "expected_value": 2.5, "kelly_fraction": 0.01, "strategy_type": "mc_venue_focus"}]

def test_hard_race_skipped():
    assert derive_p_kai_bets(P_BETS, HARD_CUT - 0.1, TEMP_THR + 1) == []

def test_warm_keeps_full_amount():
    out = derive_p_kai_bets(P_BETS, HARD_CUT + 0.5, TEMP_THR + 1)
    assert len(out) == 1 and out[0]["amount"] == 1300
    assert out[0]["strategy_type"] == "p_kai"
    assert P_BETS[0]["strategy_type"] == "mc_venue_focus"   # 元は破壊しない

def test_cold_halves_and_rounds():
    out = derive_p_kai_bets(P_BETS, HARD_CUT + 0.5, TEMP_THR - 1)
    assert out[0]["amount"] == 700     # 1300*0.5=650 → 100円丸め→700
    tiny = [dict(P_BETS[0], amount=100)]
    assert derive_p_kai_bets(tiny, HARD_CUT + 0.5, TEMP_THR - 1) == []  # 50→丸め0→捨て

def test_none_score_keeps_none_temp_warm():
    assert len(derive_p_kai_bets(P_BETS, None, None)) == 1   # 不明は保守的に素通し(等倍)
    assert derive_p_kai_bets([], 9.9, None) == []

if __name__ == "__main__":
    test_hard_race_skipped(); test_warm_keeps_full_amount(); test_cold_halves_and_rounds()
    test_none_score_keeps_none_temp_warm(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — Run: `venv/Scripts/python.exe tests/test_p_kai.py` → `ImportError`

- [ ] **Step 3: 実装** — `src/p_kai.py`:
```python
"""P改: P(mc_venue_focus) の picks に凍結ルールを適用して p_kai bets を導出する純関数。
ルール(2026-07-30凍結): 堅レース(score<=HARD_CUT)見送り / 低温日(temperature<TEMP_THR)は×COLD_SCALE。
score/temperature が None の場合は保守的に「適用しない」(=素通し)。増し張りは存在しない。"""
from src.softness_params import HARD_CUT, TEMP_THR, COLD_SCALE


def derive_p_kai_bets(p_bets, score, temperature):
    if not p_bets:
        return []
    if score is not None and score <= HARD_CUT:
        return []
    scale = COLD_SCALE if (temperature is not None and temperature < TEMP_THR) else 1.0
    out = []
    for b in p_bets:
        amount = int(round(b["amount"] * scale / 100) * 100)
        if amount < 100:
            continue
        nb = dict(b)
        nb["amount"] = amount
        nb["strategy_type"] = "p_kai"
        out.append(nb)
    return out
```

- [ ] **Step 4: 成功確認** — Run: `venv/Scripts/python.exe tests/test_p_kai.py` → `ALL PASS`

- [ ] **Step 5: コミット**
```bash
git add src/p_kai.py tests/test_p_kai.py
git commit -m "feat(p-kai): pure bet-derivation (skip-hard + cold-halving, frozen rules)"
```

---

## Task 4: config/ACTIVE_STRATEGIES（V11撤去・mc_venue_all追加）

**Files:** Modify `src/betting.py`, `config/betting_config.json`; Test `tests/test_rollout_config.py`

- [ ] **Step 1: 失敗するテスト** — `tests/test_rollout_config.py`:
```python
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_active_strategies():
    from src.betting import ACTIVE_STRATEGIES
    assert "v11_var13" not in ACTIVE_STRATEGIES      # V11撤去
    assert "mc_venue_all" in ACTIVE_STRATEGIES       # P全場追加
    assert "mc_venue_focus" in ACTIVE_STRATEGIES     # P は不変
    assert "v11_5_var13" in ACTIVE_STRATEGIES        # V11.5 shadow は継続

def test_config_mc_venue_all():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c = json.load(open(os.path.join(root, "config", "betting_config.json"), encoding="utf-8"))
    strategies = c.get("strategies", c)
    p = strategies["mc_venue_focus"]; a = strategies["mc_venue_all"]
    assert "include_venues" not in a                  # 会場制限なし
    for k in ("min_odds", "max_odds", "kelly_fraction", "filter_type", "max_entropy",
              "skip_56", "use_monte_carlo", "max_recommended_bets"):
        if k in p:
            assert a.get(k) == p.get(k), k            # P と同一フィルタ

if __name__ == "__main__":
    test_active_strategies(); test_config_mc_venue_all(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — Run: `venv/Scripts/python.exe tests/test_rollout_config.py` → AssertionError（v11_var13 がまだ居る）

- [ ] **Step 3: 実装**
  1. `src/betting.py` の `ACTIVE_STRATEGIES`（L254-280 付近）: `'v11_var13',` の行を削除し、`'mc_venue_focus',` の近くに `'mc_venue_all',       # P全場: P同一フィルタ・会場制限なし (得意場発掘 shadow, 2026-07-30)` を追加。`'v11_5_var13'` は残す。
  2. `config/betting_config.json`: `mc_venue_focus` エントリを丸ごとコピーして `mc_venue_all` を作成し、`include_venues` キーを削除、`_comment` を `"P全場: P と同一設定から会場制限を外した得意場発掘 shadow (2026-07-30)"` に変更。他は一切変えない。

- [ ] **Step 4: 成功確認** — Run: `venv/Scripts/python.exe tests/test_rollout_config.py` → `ALL PASS`。さらに smoke:
`venv/Scripts/python.exe -c "import ast; ast.parse(open('src/betting.py',encoding='utf-8').read()); import json; json.load(open('config/betting_config.json',encoding='utf-8')); print('OK')"` → `OK`

- [ ] **Step 5: コミット**
```bash
git add src/betting.py config/betting_config.json tests/test_rollout_config.py
git commit -m "feat(rollout): retire v11_var13 strategy, add mc_venue_all (P clone, all venues)"
```

---

## Task 5: scheduler 配線（軟らかさ記録＋p_kai派生）

**Files:** Modify `src/scheduler.py`

- [ ] **Step 1: 挿入点1（軟らかさ記録）** — `src/scheduler.py` の odds_board capture ブロック（`save_odds_board(race['race_id'], odds_dict)` を含む try ブロック、L743-750 付近）の**直後**に挿入（12スペースインデント）:
```python
            # P改用: レース軟らかさを記録し、当該レースのスコアを保持 (自己隔離)
            softness_score = None
            try:
                from src.softness import score_race
                from src.softness_store import save_race_softness
                softness_score = score_race(boats_data)
                save_race_softness(race['race_id'], race['race_date'], softness_score)
            except Exception as e:
                logger.warning(f"softness 記録失敗(無視): {e}")
```
注意: `race['race_date']` が dict に無い場合は前後のコードを読み、レース日付の取得方法（`race` dict のキー or `now_jst().date()`）を確認して合わせる。`boats_data` は同メソッド内で既に定義済みであることを確認する。

- [ ] **Step 2: 挿入点2（p_kai派生）** — 同ファイルの V11.5 マージブロック終端（`all_bets['v11_5_var13'] = []` が2箇所ある else 節の後、保存ループ `for strategy_type, bets in all_bets.items():` の**直前**）に挿入:
```python
            # P改 (p_kai): P の picks に凍結ルール適用 (堅見送り+低温半額)。自己隔離。
            try:
                from src.p_kai import derive_p_kai_bets
                from src.softness_store import get_temperature
                _temp = get_temperature(now_jst().date())
                all_bets['p_kai'] = derive_p_kai_bets(
                    all_bets.get('mc_venue_focus') or [], softness_score, _temp)
            except Exception as e:
                logger.warning(f"p_kai 導出失敗(無視): {e}")
                all_bets['p_kai'] = []
```
注意: `now_jst` は scheduler で import 済みか確認（`from src.utils import now_jst` 等、ファイル冒頭を確認して既存の日付取得方法に合わせる）。

- [ ] **Step 3: smoke** — Run:
`venv/Scripts/python.exe -c "import ast; ast.parse(open('src/scheduler.py',encoding='utf-8').read()); from src.p_kai import derive_p_kai_bets; from src.softness_store import get_temperature; print('OK')"` → `OK`

- [ ] **Step 4: diff 目視** — Run: `git diff src/scheduler.py` — 上記2ブロックのみ追加、他は無変更であること。

- [ ] **Step 5: コミット**
```bash
git add src/scheduler.py
git commit -m "feat(p-kai): wire softness logging + p_kai derivation into scheduler (isolated)"
```

---

## Task 6: ダッシュボード（V11非表示・新カード2枚・¥200,000リセット）

**Files:** Modify `streamlit_app/app.py`(L254-281 付近), `streamlit_app/components/db_utils.py`(L14 付近)

- [ ] **Step 1: app.py 修正**
  1. `STRATEGY_NAMES`: `'v11_var13': 'V11 (VAR-13)',` をコメントアウト（非表示化コメントの流儀に合わせ `# 'v11_var13': 'V11 (VAR-13)',  # 非表示化 (2026-07-30): 認定P(勝ち)=0%で撤去`）。先頭に追加:
```python
    'p_kai': 'P改: 堅見送り+低温半額',
    'mc_venue_all': 'P全場: 得意場発掘',
```
  2. `STRATEGY_ORDER`: `'v11_var13',` の行を削除し、リスト先頭に `'p_kai', 'mc_venue_all',` を追加。

- [ ] **Step 2: db_utils.py 修正** — `STRATEGY_RESET_DATES`（L14）に追加:
```python
    'p_kai': '2026-07-30',        # P改 ¥200,000 fresh start
    'mc_venue_all': '2026-07-30', # P全場 ¥200,000 fresh start
```
（既存の `'v11_var13': '2026-05-27'` は残す＝過去データの表示整合のため。非表示は app.py 側で行う）

- [ ] **Step 3: smoke** — Run:
`venv/Scripts/python.exe -c "import ast; [ast.parse(open(f,encoding='utf-8').read()) for f in ('streamlit_app/app.py','streamlit_app/components/db_utils.py')]; t=open('streamlit_app/app.py',encoding='utf-8').read(); assert 'p_kai' in t and 'mc_venue_all' in t; assert \"'v11_var13': 'V11\" not in t.split('STRATEGY_NAMES')[1].split('}')[0].replace('# ','#'); print('OK')"` → `OK`（assert が複雑なら「構文OK＋p_kai/mc_venue_all が存在＋STRATEGY_ORDER に v11_var13 が無い」を個別に確認する形に書き換えてよい）

- [ ] **Step 4: コミット**
```bash
git add streamlit_app/app.py streamlit_app/components/db_utils.py
git commit -m "feat(dashboard): hide V11, add P-kai & P-all-venue cards (200k fresh start)"
```

---

## Task 7: 全場リプレイ（オフライン一次読み）

**Files:** Create `analysis/venue_expand/__init__.py`(空), `analysis/venue_expand/replay.py`, `analysis/venue_expand/REPORT.md`(生成物)

**設計:** 台帳(6/30〜)の各レースについて、predictions の probs(marginals)＋boats/races を入力に **本番の `qmc_sanrentan`(MC v1) → `_strategy_kelly`(P config, include_venues 除去)** を再現し、picks を確定配当で決済。会場別に集計。**忠実度ゲート**: P の実bet（5場・同期間）と照合し、レース×買い目の選択一致率を報告（≥70% 期待。オッズ取得タイミング差・bankroll差で完全一致はしない）。

- [ ] **Step 1: 実装** — `analysis/venue_expand/replay.py`（`analysis/combo/loader.py` の `load_races` を再利用。P config は `config/betting_config.json` から読み `include_venues` を除去。`src/betting.py` の kelly 選定関数を offline から呼ぶ。呼び出しシグネチャは実装時に `_strategy_kelly` / `calculate_all_strategies` を読んで確認し、DB依存（当日bet数チェック等）は過去日付では0が返ることを利用する）:
```python
"""P式買い方×全24場のオフライン一次読み (台帳期間・read-only)。
出力: REPORT.md (会場別 n/的中/PnL/ROI + プール + 忠実度ゲート結果 + 仮説どまり注記)"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "combo"))
import numpy as np
from loader import load_races          # 台帳+marginals+boats+races
from src.monte_carlo import qmc_sanrentan
# 実装時確認: P の経路は use_monte_carlo=true & mc_version 無し → MC v1 = qmc_sanrentan(probs_1st, boats_data,...)
# 選定は src/betting.py の kelly 関数 (candidates 構築 L820-987) を import して再利用する。
# import 不可能な内部関数だった場合は、P config の値 (min_odds=5, max_odds=40, entropy<=2.3,
# skip_56, EV>min_ev, kelly>0, top3) を同ファイルの式の通りに写して \"p_replay_select()\" を実装し、
# docstring に「betting.py L820-987 の写し (2026-07-30)」と明記する。
# ... (実装者が上記方針でコードを完成させる。以下の骨格を使う)

def replay_all_venues(date_from="2026-06-30", date_to="2026-12-31"):
    races = load_races(date_from, date_to)
    results = []   # dict(venue, race_id, combo, amount, odds, hit, ret)
    ...
    return results

def fidelity_gate(results, date_from="2026-06-30"):
    """P の実bet(5場)とレース×combo選択を照合し一致率を返す"""
    ...

def write_report(results, gate):
    ...

if __name__ == "__main__":
    res = replay_all_venues()
    gate = fidelity_gate(res)
    write_report(res, gate)
```
※このタスクのみ骨格提示（本番関数の正確なシグネチャに依存するため）。実装者は必ず (1) `src/betting.py` L820-987 を読み、(2) 忠実度ゲートを先に緑にしてから、(3) 全場へ広げること。ゲート一致率 <70% なら REPORT を出さず DONE_WITH_CONCERNS で報告。

- [ ] **Step 2: 実行** — Run: `venv/Scripts/python.exe analysis/venue_expand/replay.py`（数分〜十数分）
Expected: `REPORT.md` 生成。忠実度ゲート一致率と会場別表、および「**1ヶ月・会場別は薄い・仮説どまり（採用判断は mc_venue_all の forward で）**」の注記を含む。

- [ ] **Step 3: コミット**
```bash
git add analysis/venue_expand/
git commit -m "research(venue-expand): offline all-venue replay of P logic (ledger month, hypothesis-only)"
```

---

## Task 8: デプロイ後検証スクリプト＋デプロイ手順

**Files:** Create `scripts/verify_p_kai_rollout.py`

- [ ] **Step 1: 実装** — `scripts/verify_p_kai_rollout.py`（read-only）:
```python
"""P改ロールアウトのデプロイ後検証 (read-only)。実行: venv/Scripts/python.exe scripts/verify_p_kai_rollout.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "ledger"))
from db import connect   # read-only 接続

DEPLOY_AT = "2026-07-30 12:00:00"   # 実デプロイ時刻に合わせて実行時に引数で上書き可

def main():
    import sys as _s
    deploy_at = _s.argv[1] if len(_s.argv) > 1 else DEPLOY_AT
    conn = connect(); cur = conn.cursor()
    ng = 0
    # (a) v11_var13 の新規が止まっている
    cur.execute("SELECT count(*) n FROM predictions WHERE strategy_type='v11_var13' AND created_at > %s", (deploy_at,))
    n = cur.fetchone()["n"]
    print(("OK" if n == 0 else "NG"), f"(a) v11_var13 新規prediction={n} (0であること)"); ng += (n != 0)
    # (b) 新戦略が発火
    for s in ("mc_venue_all", "p_kai"):
        cur.execute("SELECT count(*) n, max(created_at) mx FROM predictions WHERE strategy_type=%s AND created_at > %s", (s, deploy_at))
        r = cur.fetchone()
        print(("OK" if r["n"] > 0 else "NG(蓄積待ちなら翌日再確認)"), f"(b) {s} predictions={r['n']} 最新={r['mx']}"); 
    # (c) p_kai ⊂ P (同レース同comboで金額は等倍or半額)
    cur.execute("""SELECT pk.race_id, pk.combination, pk.amount ka, p.amount pa
                   FROM bets pk JOIN bets p ON p.race_id=pk.race_id AND p.combination=pk.combination
                        AND p.strategy_type='mc_venue_focus'
                   WHERE pk.strategy_type='p_kai' AND pk.created_at > %s LIMIT 20""", (deploy_at,))
    rows = cur.fetchall()
    bad = [r for r in rows if not (abs(float(r["ka"]) - float(r["pa"])) < 1
                                   or abs(float(r["ka"]) - round(float(r["pa"])*0.5/100)*100) < 1)]
    print(("OK" if rows and not bad else ("待ち" if not rows else "NG")),
          f"(c) p_kai⊂P 金額整合 checked={len(rows)} bad={len(bad)}")
    # (d) race_softness が増えている
    cur.execute("SELECT count(*) n, max(race_date) mx FROM race_softness")
    r = cur.fetchone(); print(("OK" if r["n"] > 0 else "NG"), f"(d) race_softness rows={r['n']} 最新={r['mx']}")
    # (e) 既存Pが引き続き発火
    cur.execute("SELECT count(*) n FROM predictions WHERE strategy_type='mc_venue_focus' AND created_at > %s", (deploy_at,))
    print("INFO (e) P predictions after deploy =", cur.fetchone()["n"])
    print("---- 判定: NG が無く (b)(c) が OK になれば合格 (蓄積待ちは翌日再実行) ----")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: コミット**
```bash
git add scripts/verify_p_kai_rollout.py
git commit -m "feat(rollout): post-deploy verification script"
```

- [ ] **Step 3: 全テスト一括**
```bash
for t in tests/test_softness.py tests/test_softness_store.py tests/test_p_kai.py tests/test_rollout_config.py tests/test_odds_board_build.py tests/test_odds_board_save.py; do venv/Scripts/python.exe $t || echo "FAIL: $t"; done
```
Expected: 全て `ALL PASS`（odds_board 既存テストも壊れていないこと）

- [ ] **Step 4: デプロイ手順（コントローラ実施）**
  1. master へマージ・push
  2. 岩下さんに Railway Redeploy を依頼（web/worker、Build 成功確認）
  3. `venv/Scripts/python.exe scripts/backfill_race_softness.py` を実行（race_softness 過去分投入）
  4. 数レース処理後 `venv/Scripts/python.exe scripts/verify_p_kai_rollout.py "<デプロイ時刻>"` で (a)〜(e) 確認
  5. ダッシュボードで P改・P全場カードが ¥200,000 から表示・V11 カードが消えていることを目視

---

## 完了条件
- 全テスト PASS・デプロイ後検証 (a)〜(e) 合格・ダッシュ目視OK。
- REPORT.md（全場一次読み）が仮説どまり注記つきで生成済み。
- 実弾ゼロのまま（purchaser 不変）。実弾投入は forward を見て岩下さんが判断。

## Self-Review（作成者チェック）
- **Spec coverage**: §4.1→T4,T6 / §4.2→T7 / §4.3→T4 / §4.4→T1,T2,T3,T5 / §4.5→T6 / §5→各テスト+T8 / §6ロールバック→ACTIVE_STRATEGIES revert。全カバー。
- **Placeholder scan**: T7 のみ骨格（本番関数シグネチャ依存のため、確認手順・写し先行番地 L820-987・ゲート基準・失敗時挙動を明記）。他は実コード。
- **Type consistency**: `score_race(boats_data)->float|None`(T1) を T2 backfill/T5 が使用。`save_race_softness(race_id,race_date,score,conn_factory)->bool`・`get_temperature(date,conn_factory)->float|None`(T2) を T5/T8 が使用。`derive_p_kai_bets(p_bets,score,temperature)->list`(T3) を T5 が使用。`HARD_CUT/TEMP_THR/COLD_SCALE/TEMP_WINDOW`(T1生成) を T2/T3 が import。戦略名 `p_kai`/`mc_venue_all` は T4/T5/T6/T8 で一致。
