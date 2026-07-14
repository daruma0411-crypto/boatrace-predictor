# 3連単 組み合わせ手法 研究 v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** オッズ盤台帳＋本物QMC結合を使い、「市場の歪み×モデルの組み合わせ順位」を突く3連単手法に out-of-sample のエッジがあるかを、本番同等シミュレータ（¥200,000）で測る read-only 研究パッケージを作る。

**Architecture:** `analysis/combo/` に read-only パッケージを新設。既存の production 関数を**再利用**（`_get_pre_race_data` で DB→入力再構成、`src/monte_carlo.qmc_sanrentan_v3` で本物QMC結合）。小さな責務単位（db/loader/joint/market/diagnose/sim/validate/method/evaluate/report）に分割し、各モジュールに assert ベーステスト。シミュレータは「実 bets を決済して DB残高を再現」できることで忠実度を担保。

**Tech Stack:** Python 3.11（`venv/Scripts/python.exe`）、psycopg2、numpy、scipy（QMC は既存 `src/monte_carlo.py`）。追加依存なし。

**前提（実行前に必須）:**
- ブランチは `research/combo-method`（作成済み）。
- 本番DBは read-only（`analysis/ledger/db.py` の `set_session(readonly=True)` 流儀）。INSERT/UPDATE/DELETE 厳禁。**本番・DB・`betting_config`/`scheduler` は一切変更しない。**
- スクリプト実行は `venv/Scripts/python.exe`。テストは `venv/Scripts/python.exe analysis/combo/test_xxx.py`（自ディレクトリが sys.path[0]）。
- 仕様書: `docs/superpowers/specs/2026-07-14-combo-method-research-design.md`。
- 既知の確定値（sanity 根拠、変動するので幅で判定）: 全盤 blanket 回収 全体≈57%、200倍超帯≈46%、5-40倍帯≈80-84%。モデル proxy 上位1点≈97%。台帳は 6/30 以降のみ。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `analysis/combo/__init__.py` | 空 |
| `analysis/combo/db.py` | read-only 接続 |
| `analysis/combo/loader.py` | 台帳⨝marginals⨝結果 ＋ race_data/boats_data 再構成（QMC入力） |
| `analysis/combo/joint.py` | `qmc_joint`（本物QMC）/ `proxy_joint`（p1×p2×p3） |
| `analysis/combo/market.py` | オッズ盤→de-vig 市場含み確率 |
| `analysis/combo/diagnose.py` | 歪みマップ・モデルvs市場 差マップ・joint較正 |
| `analysis/combo/sim.py` | 本番同等シミュ（¥200,000・実ケリー・上限・丸め・決済） |
| `analysis/combo/validate.py` | 実 bets を sim 決済→DB残高再現（忠実度ゲート）＋既存ベンチ |
| `analysis/combo/method.py` | 事前登録の候補手法（フィルタ→採点→選別） |
| `analysis/combo/evaluate.py` | 手法を sim でリプレイ（フラット＋本番同等）＋train/test |
| `analysis/combo/report.py` → `RESULTS.md` | 全実行→判定レポート生成 |

各 `*.py` に対応する `test_*.py`。

---

## Task 1: db.py + loader.py — 台帳読み込み & QMC入力再構成

**Files:** Create `analysis/combo/__init__.py`(空), `analysis/combo/db.py`, `analysis/combo/loader.py`; Test `analysis/combo/test_loader.py`

- [ ] **Step 1: 失敗するテストを書く** — `analysis/combo/test_loader.py`:
```python
from loader import load_races

def test_load_races_shape():
    races = load_races("2026-06-30", "2026-12-31", limit=50)
    assert isinstance(races, list) and len(races) > 0
    r = races[0]
    for k in ("race_id","race_number","odds","p1","p2","p3","win_combo","race_data","boats_data"):
        assert k in r, k
    assert isinstance(r["odds"], dict) and len(r["odds"]) >= 100   # 全盤
    assert len(r["p1"]) == 6 and len(r["boats_data"]) == 6
    assert r["win_combo"].count("-") == 2
    # race_data/boats_data は QMC入力の形
    assert "venue_id" in r["race_data"]
    assert "exhibition_time" in r["boats_data"][0] and "player_class" in r["boats_data"][0]
    print("OK n=", len(races))

if __name__ == "__main__":
    test_load_races_shape(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_loader.py` → `ModuleNotFoundError: loader`

- [ ] **Step 3: 実装**

`analysis/combo/__init__.py`: 空ファイル。

`analysis/combo/db.py`:
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

`analysis/combo/loader.py`（race_data/boats_data は `src/predictor.py:_get_pre_race_data` と同一キー）:
```python
import json
from db import connect

_RACE_SQL = """
SELECT DISTINCT ON (ob.race_id)
       ob.race_id, ob.odds_3t, r.race_number, r.venue_id, r.race_date,
       r.wind_speed, r.wind_direction, r.temperature, r.wave_height, r.water_temperature,
       r.result_1st, r.result_2nd, r.result_3rd,
       p.probabilities_1st, p.probabilities_2nd, p.probabilities_3rd
FROM race_odds_board ob
JOIN races r ON r.id = ob.race_id
JOIN predictions p ON p.race_id = ob.race_id
WHERE r.race_date BETWEEN %s AND %s
      AND r.result_1st IS NOT NULL AND r.result_2nd IS NOT NULL AND r.result_3rd IS NOT NULL
ORDER BY ob.race_id
"""

_BOATS_SQL = "SELECT * FROM boats WHERE race_id = %s ORDER BY boat_number"


def _vec(v):
    if isinstance(v, str):
        v = json.loads(v)
    return [float(x) for x in v]


def _boats_data(cur, race_id):
    cur.execute(_BOATS_SQL, (race_id,))
    out = []
    for b in cur.fetchall():
        out.append({
            "boat_number": b["boat_number"], "player_class": b["player_class"],
            "win_rate": b["win_rate"], "win_rate_2": b["win_rate_2"], "win_rate_3": b["win_rate_3"],
            "local_win_rate": b["local_win_rate"], "local_win_rate_2": b["local_win_rate_2"],
            "avg_st": b["avg_st"], "motor_win_rate_2": b["motor_win_rate_2"],
            "motor_win_rate_3": b["motor_win_rate_3"], "boat_win_rate_2": b["boat_win_rate_2"],
            "weight": b["weight"], "exhibition_time": b["exhibition_time"],
            "approach_course": b["approach_course"], "is_new_motor": b["is_new_motor"],
            "tilt": b.get("tilt"), "parts_changed": b.get("parts_changed", False),
            "fallback_flag": False,
        })
    return out


def load_races(date_from, date_to, limit=None):
    conn = connect(); cur = conn.cursor()
    cur.execute(_RACE_SQL, (date_from, date_to))
    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        odds = r["odds_3t"] if isinstance(r["odds_3t"], dict) else json.loads(r["odds_3t"])
        if len(odds) < 100:
            continue
        out.append({
            "race_id": r["race_id"], "race_number": r["race_number"],
            "odds": {k: float(v) for k, v in odds.items()},
            "p1": _vec(r["probabilities_1st"]), "p2": _vec(r["probabilities_2nd"]),
            "p3": _vec(r["probabilities_3rd"]),
            "win_combo": f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}",
            "race_data": {
                "venue_id": r["venue_id"], "month": r["race_date"].month, "distance": 1800,
                "wind_speed": r["wind_speed"] or 0, "wind_direction": r["wind_direction"] or "calm",
                "temperature": r["temperature"] or 20, "wave_height": r["wave_height"] or 0,
                "water_temperature": r["water_temperature"] or 20,
            },
            "boats_data": _boats_data(cur, r["race_id"]),
        })
    return out
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_loader.py` → `ALL PASS`

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/__init__.py analysis/combo/db.py analysis/combo/loader.py analysis/combo/test_loader.py
git commit -m "feat(combo): read-only loader (odds board + marginals + QMC inputs)"
```

---

## Task 2: joint.py — 本物QMC結合 & 近似（忠実度ゲート）

**Files:** Create `analysis/combo/joint.py`; Test `analysis/combo/test_joint.py`

**設計:** `qmc_joint` は production の `qmc_sanrentan_v3(probs_1st, boats_data, race_data, race_number)` を**そのまま呼ぶ**（同一入力＝同一結果）。返り値 `{combo: prob}`（sparse）。忠実度テスト＝「その race で実際に bet された combo が、再計算 joint に存在し、それなりの確率を持つ」。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_joint.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root for src.*
from loader import load_races
from joint import qmc_joint, proxy_joint
from db import connect

def test_proxy_joint_basic():
    races = load_races("2026-06-30", "2026-12-31", limit=5)
    j = proxy_joint(races[0])
    assert isinstance(j, dict) and len(j) > 100
    assert all(0 <= v <= 1 for v in j.values())

def test_qmc_joint_reproduces_a_bet():
    # 実際に bet された race を1件見つけ、その combo が QMC joint に載るか
    conn = connect(); cur = conn.cursor()
    cur.execute("""SELECT b.race_id, b.combination FROM bets b
                   JOIN race_odds_board ob ON ob.race_id=b.race_id
                   WHERE b.strategy_type='v11_var13' ORDER BY b.id DESC LIMIT 1""")
    row = cur.fetchone()
    assert row, "v11_var13 の bet が台帳期間にまだ無い（蓄積待ち）"
    races = load_races("2026-06-01", "2026-12-31")
    race = next((r for r in races if r["race_id"] == row["race_id"]), None)
    assert race, "対象raceがloaderに無い"
    j = qmc_joint(race)
    assert isinstance(j, dict) and len(j) > 50
    assert row["combination"] in j, f"bet combo {row['combination']} が QMC joint に無い"
    print("OK qmc_joint combo=", row["combination"], "prob=", round(j[row["combination"]], 4))

if __name__ == "__main__":
    test_proxy_joint_basic(); test_qmc_joint_reproduces_a_bet(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_joint.py` → `ImportError`

- [ ] **Step 3: 実装** — `analysis/combo/joint.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from src.monte_carlo import qmc_sanrentan_v3


def proxy_joint(race):
    """p1×p2×p3 の独立近似（対照用）。{combo: prob}（全120通り、正規化なし）。"""
    p1, p2, p3 = race["p1"], race["p2"], race["p3"]
    out = {}
    for a in range(6):
        for b in range(6):
            if b == a: continue
            for c in range(6):
                if c == a or c == b: continue
                out[f"{a+1}-{b+1}-{c+1}"] = p1[a] * p2[b] * p3[c]
    return out


def qmc_joint(race, seed=12345):
    """production の QMC v3 をそのまま呼ぶ（本物）。{combo: prob}（sparse）。"""
    return qmc_sanrentan_v3(
        np.array(race["p1"]),
        boats_data=race["boats_data"],
        race_data=race["race_data"],
        race_number=race["race_number"],
        seed=seed,
    )
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_joint.py` → `ALL PASS`（`OK qmc_joint combo=...` を含む）。※`v11_var13 の bet がまだ無い` で assert 停止したら、`strategy_type` を bet 実績のある戦略（例 `mc3_venue_focus`）に変えて再確認。

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/joint.py analysis/combo/test_joint.py
git commit -m "feat(combo): real QMC joint (reuse production qmc_sanrentan_v3) + proxy joint"
```

---

## Task 3: market.py — de-vig 市場含み確率

**Files:** Create `analysis/combo/market.py`; Test `analysis/combo/test_market.py`

**設計:** 市場含み確率 = (1/odds) を全 combo で正規化（overround を割る）。最小オッズが最大含み確率。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_market.py`:
```python
from market import implied_probs

def test_implied_sums_to_one_and_favorite_top():
    odds = {"1-2-3": 5.0, "1-2-4": 10.0, "2-1-3": 50.0}
    imp = implied_probs(odds)
    assert abs(sum(imp.values()) - 1.0) < 1e-9
    assert max(imp, key=imp.get) == "1-2-3"     # 最小オッズが最大含み
    assert imp["1-2-3"] > imp["2-1-3"]

if __name__ == "__main__":
    test_implied_sums_to_one_and_favorite_top(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_market.py` → `ImportError`

- [ ] **Step 3: 実装** — `analysis/combo/market.py`:
```python
def implied_probs(odds):
    """{combo: odds} → de-vig した {combo: 市場含み確率}（合計1）。"""
    raw = {c: (1.0 / o) for c, o in odds.items() if o and o > 0}
    s = sum(raw.values())
    if s <= 0:
        return {c: 0.0 for c in odds}
    return {c: v / s for c, v in raw.items()}
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_market.py` → `ALL PASS`

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/market.py analysis/combo/test_market.py
git commit -m "feat(combo): de-vig market implied probabilities"
```

---

## Task 4: diagnose.py — 歪みマップ & モデルvs市場 差マップ

**Files:** Create `analysis/combo/diagnose.py`; Test `analysis/combo/test_diagnose.py`

**設計:** 台帳全レースで (a) オッズ帯別 blanket 回収率（歪みマップ）、(b) モデル(qmc/proxy)上位N点の回収率。既知値を幅で再現。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_diagnose.py`:
```python
from loader import load_races
from diagnose import odds_band_map, model_topN_roi

def test_favorite_longshot_shape():
    races = load_races("2026-06-30", "2026-12-31")
    m = odds_band_map(races)
    # 既知: 200倍超は低回収(<60%)、5-40倍は高め(>70%)、全体<100%
    assert m["200-100000"]["roi"] < 60, m["200-100000"]
    assert m["5-10"]["roi"] > 70 and m["10-20"]["roi"] > 70, m
    print("OK band map:", {k: round(v["roi"]) for k, v in m.items()})

def test_model_topN_beats_blanket():
    races = load_races("2026-06-30", "2026-12-31")
    roi = model_topN_roi(races, joint="proxy", N_list=(1, 3, 5))
    assert roi[1] > roi[5], roi          # 上位ほど高い（単調）
    assert roi[1] > 80, roi              # 上位1点は blanket(57%)を大きく超える
    print("OK model topN(proxy):", {k: round(v) for k, v in roi.items()})

if __name__ == "__main__":
    test_favorite_longshot_shape(); test_model_topN_beats_blanket(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_diagnose.py` → `ImportError`

- [ ] **Step 3: 実装** — `analysis/combo/diagnose.py`:
```python
from joint import qmc_joint, proxy_joint

_EDGES = [1, 5, 10, 20, 40, 80, 200, 100000]


def odds_band_map(races):
    """オッズ帯別 blanket 回収率（全combo 1点ずつ買ったら）。"""
    band = {f"{lo}-{hi}": {"slots": 0, "wins": 0, "ret": 0.0}
            for lo, hi in zip(_EDGES, _EDGES[1:])}
    for r in races:
        win = r["win_combo"]
        for combo, od in r["odds"].items():
            if od <= 1: continue
            for lo, hi in zip(_EDGES, _EDGES[1:]):
                if lo <= od < hi:
                    key = f"{lo}-{hi}"; band[key]["slots"] += 1
                    if combo == win:
                        band[key]["wins"] += 1; band[key]["ret"] += od
                    break
    for k, v in band.items():
        v["roi"] = (v["ret"] / v["slots"] * 100) if v["slots"] else 0.0
        v["hit_rate"] = (v["wins"] / v["slots"] * 100) if v["slots"] else 0.0
    return band


def model_topN_roi(races, joint="qmc", N_list=(1, 3, 5, 10)):
    """各レースでモデル上位N点を1点ずつ買った時の回収率。"""
    jf = qmc_joint if joint == "qmc" else proxy_joint
    acc = {N: [0, 0.0] for N in N_list}   # [n_bet, ret]
    for r in races:
        j = jf(r)
        ranked = sorted(j.items(), key=lambda x: -x[1])
        for N in N_list:
            for combo, _ in ranked[:N]:
                acc[N][0] += 1
                if combo == r["win_combo"]:
                    acc[N][1] += r["odds"].get(combo, 0.0)
    return {N: (ret / nb * 100 if nb else 0.0) for N, (nb, ret) in acc.items()}
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_diagnose.py` → `ALL PASS`

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/diagnose.py analysis/combo/test_diagnose.py
git commit -m "feat(combo): favorite-longshot map + model top-N ROI diagnostics"
```

---

## Task 5: sim.py — 本番同等シミュレータ（¥200,000）

**Files:** Create `analysis/combo/sim.py`; Test `analysis/combo/test_sim.py`

**設計:** 賭け金＝production の kelly 式（`src/betting.py` `_strategy_kelly` L911-934 準拠）: `b=odds-1`, `kelly=(b*p-q)/b`, `amount=bankroll*kelly*kelly_frac`, `[min_bet, max_ticket]` で clamp、100円丸め、レース合計 `max_total` で cut。決済＝当り combo が picks にあれば `amount*odds` 払い戻し。¥200,000 スタートの equity 系列。`settle_only=True` で「与えられた (combo,amount) をそのまま決済」（validate 用）。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_sim.py`:
```python
from sim import stake_kelly, run_sim, DEFAULT_CFG

def test_stake_kelly_positive_edge_bets():
    # p=0.3, odds=5 → b=4, kelly=(4*0.3-0.7)/4=0.125>0 → 賭ける
    amt = stake_kelly(0.30, 5.0, bankroll=200000, cfg=DEFAULT_CFG)
    assert amt >= DEFAULT_CFG["min_bet"] and amt % 100 == 0
    # 負けエッジは0
    assert stake_kelly(0.10, 5.0, bankroll=200000, cfg=DEFAULT_CFG) == 0

def test_run_sim_settle_only_reproduces_given_bets():
    # settle_only: 与えた (combo, amount) を決済。当り1本の払戻を確認
    picks = [
        {"race_id": 1, "win_combo": "1-2-3", "bets": [("1-2-3", 100, 5.0), ("4-5-6", 100, 30.0)]},
        {"race_id": 2, "win_combo": "2-1-3", "bets": [("1-2-3", 100, 5.0)]},
    ]
    res = run_sim(picks, settle_only=True)
    # race1: 投資200, 払戻 100*5=500 → +300 ; race2: 投資100, 払戻0 → -100 ; 計 +200
    assert res["invested"] == 300 and res["returned"] == 500
    assert res["pnl"] == 200 and res["final_bankroll"] == 200000 + 200

if __name__ == "__main__":
    test_stake_kelly_positive_edge_bets(); test_run_sim_settle_only_reproduces_given_bets()
    print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_sim.py` → `ImportError`

- [ ] **Step 3: 実装** — `analysis/combo/sim.py`:
```python
INITIAL_BANKROLL = 200000

# production betting_config の P/P2/V11 共通値（src/betting.py _strategy_kelly 準拠）
DEFAULT_CFG = {
    "kelly_fraction": 0.0625,
    "max_ticket_bet_ratio": 0.008,
    "max_total_bet_ratio": 0.02,
    "min_bet": 100,
    "odds_discount": 0.92,
}


def stake_kelly(prob, odds, bankroll, cfg):
    """production 準拠の kelly 賭け金。負けエッジは0。100円丸め。"""
    d_odds = odds * cfg["odds_discount"]
    b = d_odds - 1.0
    if b < 0.01:
        return 0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0:
        return 0
    amount = bankroll * kelly * cfg["kelly_fraction"]
    max_ticket = bankroll * cfg["max_ticket_bet_ratio"]
    amount = max(cfg["min_bet"], min(max_ticket, amount))
    return int(round(amount / 100) * 100)


def run_sim(per_race, cfg=DEFAULT_CFG, settle_only=False):
    """per_race: [{race_id, win_combo, bets:[(combo,amount,odds)]}]（settle_only）
       equity/ROI/PnL/DD/的中/集中度 を返す。¥200,000 スタート。"""
    bankroll = INITIAL_BANKROLL
    inv = ret = 0.0; n_bets = 0; hits = 0; peak = bankroll; max_dd = 0.0; biggest = 0.0
    for race in per_race:
        win = race["win_combo"]
        race_inv = race_ret = 0.0
        for combo, amount, odds in race["bets"]:
            if amount <= 0: continue
            race_inv += amount; n_bets += 1
            if combo == win:
                pay = amount * odds; race_ret += pay; hits += 1; biggest = max(biggest, pay)
        inv += race_inv; ret += race_ret
        bankroll += (race_ret - race_inv)
        peak = max(peak, bankroll); max_dd = min(max_dd, bankroll - peak)
    return {
        "n_bets": n_bets, "hits": hits, "invested": round(inv), "returned": round(ret),
        "pnl": round(ret - inv), "roi": round(ret / inv * 100, 1) if inv else 0.0,
        "final_bankroll": round(bankroll), "max_drawdown": round(max_dd),
        "top_hit_share": round(biggest / ret * 100, 1) if ret else 0.0,
    }
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_sim.py` → `ALL PASS`

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/sim.py analysis/combo/test_sim.py
git commit -m "feat(combo): production-fidelity simulator (200k, kelly staking, settlement)"
```

---

## Task 6: validate.py — 実bets決済でDB残高を再現（忠実度ゲート）

**Files:** Create `analysis/combo/validate.py`; Test `analysis/combo/test_validate.py`

**設計:** DB の実 bets（P/P2/V11）を `run_sim(settle_only=True)` に流し、**DB上の PnL（Σ return_amount − Σ amount）を許容誤差内で再現**することを確認＝sim の決済機構が本物である証明。同時に既存戦略の実績＝ベンチマーク。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_validate.py`:
```python
from validate import replay_strategy_from_db

def test_sim_reproduces_db_pnl():
    for strat in ("mc_venue_focus", "mc2_venue_focus"):   # P, P2
        db, sim = replay_strategy_from_db(strat, "2026-04-08", "2026-12-31")
        assert db["n"] > 50, (strat, db)
        # sim 決済の PnL が DB 実績 PnL と一致（同じ数字を再構成しているので誤差ほぼ0）
        assert abs(sim["pnl"] - db["pnl"]) <= max(500, abs(db["pnl"]) * 0.02), (strat, db, sim)
        print(f"OK {strat}: DB pnl={db['pnl']} sim pnl={sim['pnl']} (n={db['n']})")

if __name__ == "__main__":
    test_sim_reproduces_db_pnl(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_validate.py` → `ImportError`

- [ ] **Step 3: 実装** — `analysis/combo/validate.py`:
```python
from collections import defaultdict
from db import connect
from sim import run_sim

_SQL = """
SELECT b.race_id, b.combination, b.amount, b.odds,
       coalesce(b.return_amount, b.payout, 0) AS ret, b.is_hit,
       r.result_1st, r.result_2nd, r.result_3rd
FROM bets b JOIN races r ON r.id = b.race_id
WHERE b.strategy_type = %s AND r.race_date BETWEEN %s AND %s
      AND r.result_1st IS NOT NULL
ORDER BY b.id
"""


def replay_strategy_from_db(strat, date_from, date_to):
    """実 bets を sim 決済し、DB実績PnLと突合。(db_summary, sim_summary) を返す。"""
    conn = connect(); cur = conn.cursor()
    cur.execute(_SQL, (strat, date_from, date_to))
    rows = cur.fetchall()
    db_inv = db_ret = 0.0
    by_race = defaultdict(lambda: {"win_combo": None, "bets": []})
    for r in rows:
        db_inv += float(r["amount"]); db_ret += float(r["ret"])
        rid = r["race_id"]
        by_race[rid]["win_combo"] = f"{r['result_1st']}-{r['result_2nd']}-{r['result_3rd']}"
        by_race[rid]["bets"].append((r["combination"], float(r["amount"]), float(r["odds"])))
    per_race = [{"race_id": rid, **v} for rid, v in by_race.items()]
    sim = run_sim(per_race, settle_only=True)
    db = {"n": len(rows), "invested": round(db_inv), "returned": round(db_ret),
          "pnl": round(db_ret - db_inv)}
    return db, sim
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_validate.py` → `ALL PASS`。**もし乖離が出たら sim の決済ロジックを直す（ここが忠実度ゲート）。** 乖離原因の典型: return_amount と odds*amount の丸め差、is_hit と combo 一致の不整合 → sim 側を DB の return_amount 準拠に寄せる。

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/validate.py analysis/combo/test_validate.py
git commit -m "feat(combo): fidelity gate — sim reproduces existing strategies' DB PnL"
```

---

## Task 7: method.py — 事前登録の候補手法

**Files:** Create `analysis/combo/method.py`; Test `analysis/combo/test_method.py`

**設計（事前登録・変更禁止）:** オッズ 5≤odds≤40、採点 `edge = model_prob − 市場含み`、`edge>0` の上位3点を選ぶ。返り値は各レースの `[(combo, model_prob, odds)]`。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_method.py`:
```python
from method import select_combos, CANDIDATE

def test_candidate_params_locked():
    assert CANDIDATE == {"odds_lo": 5.0, "odds_hi": 40.0, "edge_min": 0.0, "top_k": 3}

def test_select_respects_filters():
    race = {"odds": {"1-2-3": 8.0, "1-2-4": 30.0, "2-1-3": 100.0, "3-1-2": 4.0}}
    model = {"1-2-3": 0.20, "1-2-4": 0.10, "2-1-3": 0.05, "3-1-2": 0.40}
    market = {"1-2-3": 0.10, "1-2-4": 0.03, "2-1-3": 0.02, "3-1-2": 0.30}
    picks = select_combos(race, model, market)
    combos = [c for c, _, _ in picks]
    assert "2-1-3" not in combos     # 100倍(>40)は除外
    assert "3-1-2" not in combos     # 4倍(<5)は除外
    assert "1-2-3" in combos and "1-2-4" in combos   # 5-40倍 & edge>0
    assert len(picks) <= 3

if __name__ == "__main__":
    test_candidate_params_locked(); test_select_respects_filters(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_method.py` → `ImportError`

- [ ] **Step 3: 実装** — `analysis/combo/method.py`:
```python
# 事前登録（測定前に確定）。変種は別ファイル/別 run として記録すること。
CANDIDATE = {"odds_lo": 5.0, "odds_hi": 40.0, "edge_min": 0.0, "top_k": 3}


def select_combos(race, model_probs, market_probs, cfg=CANDIDATE):
    """オッズ帯フィルタ → edge=model-market>edge_min → 上位 top_k。
       返り値 [(combo, model_prob, odds)]。"""
    cands = []
    for combo, od in race["odds"].items():
        if not (cfg["odds_lo"] <= od <= cfg["odds_hi"]):
            continue
        mp = model_probs.get(combo, 0.0)
        edge = mp - market_probs.get(combo, 0.0)
        if edge > cfg["edge_min"]:
            cands.append((combo, mp, od, edge))
    cands.sort(key=lambda x: -x[3])
    return [(c, mp, od) for c, mp, od, _ in cands[:cfg["top_k"]]]
```

- [ ] **Step 4: 成功確認** — `venv/Scripts/python.exe analysis/combo/test_method.py` → `ALL PASS`

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/method.py analysis/combo/test_method.py
git commit -m "feat(combo): pre-registered candidate combination method"
```

---

## Task 8: evaluate.py + report.py — 手法評価 & RESULTS.md

**Files:** Create `analysis/combo/evaluate.py`, `analysis/combo/report.py`; Create（生成物）`analysis/combo/RESULTS.md`; Test `analysis/combo/test_report.py`

**設計:** 候補手法を全レースに適用→(a)フラット賭け（各100円）(b)本番同等（`stake_kelly`）で sim。既存P/P2/V11ベンチ・診断・sim検証をまとめ `RESULTS.md` 生成。judgment はプロトコル形式（結論は出すが採否は岩下さん）。

- [ ] **Step 1: 失敗するテスト** — `analysis/combo/test_report.py`:
```python
import os
from report import build_report

def test_report_builds():
    path = build_report("2026-06-30", "2026-12-31")
    assert os.path.exists(path)
    txt = open(path, encoding="utf-8").read()
    for kw in ("歪みマップ", "候補手法", "フラット", "本番同等", "既存", "前向き"):
        assert kw in txt, kw
    print("OK report at", path)

if __name__ == "__main__":
    test_report_builds(); print("ALL PASS")
```

- [ ] **Step 2: 失敗確認** — `venv/Scripts/python.exe analysis/combo/test_report.py` → `ImportError`

- [ ] **Step 3: 実装**

`analysis/combo/evaluate.py`:
```python
from loader import load_races
from joint import qmc_joint
from market import implied_probs
from method import select_combos, CANDIDATE
from sim import run_sim, stake_kelly, DEFAULT_CFG, INITIAL_BANKROLL


def evaluate_method(date_from, date_to, joint_fn=qmc_joint):
    races = load_races(date_from, date_to)
    flat = []; prod = []
    bankroll = INITIAL_BANKROLL
    for r in races:
        model = joint_fn(r); market = implied_probs(r["odds"])
        picks = select_combos(r, model, market, CANDIDATE)
        flat.append({"race_id": r["race_id"], "win_combo": r["win_combo"],
                     "bets": [(c, 100, od) for c, _, od in picks]})
        pbets = []
        for c, mp, od in picks:
            amt = stake_kelly(mp, od, bankroll, DEFAULT_CFG)
            if amt > 0: pbets.append((c, amt, od))
        prod.append({"race_id": r["race_id"], "win_combo": r["win_combo"], "bets": pbets})
        bankroll += sum((a * od if c == r["win_combo"] else -a) for c, a, od in pbets)
    return {"n_races": len(races), "flat": run_sim(flat, settle_only=True),
            "prod": run_sim(prod, settle_only=True)}
```

`analysis/combo/report.py`:
```python
import os
from loader import load_races
from diagnose import odds_band_map, model_topN_roi
from evaluate import evaluate_method
from validate import replay_strategy_from_db

OUT = os.path.join(os.path.dirname(__file__), "RESULTS.md")


def build_report(date_from, date_to):
    races = load_races(date_from, date_to)
    band = odds_band_map(races)
    topN = model_topN_roi(races, joint="proxy", N_list=(1, 3, 5))
    ev = evaluate_method(date_from, date_to)
    L = []
    L.append(f"# 組み合わせ手法 研究 — 中間結果 ({date_from}〜{date_to})\n")
    L.append(f"対象レース={ev['n_races']}（全盤台帳）。**2週間規模の暫定・前向き検証前提。**\n")
    L.append("\n## 市場の歪みマップ（オッズ帯別 blanket 回収率）\n")
    for k, v in band.items():
        L.append(f"- {k}倍: slots={v['slots']} 的中率={v['hit_rate']:.2f}% 回収={v['roi']:.0f}%\n")
    L.append("\n## モデルの組み合わせ力（proxy 上位N点 回収率）\n")
    L.append(f"- 上位1/3/5点: {topN[1]:.0f}% / {topN[3]:.0f}% / {topN[5]:.0f}%（基準 blanket≈57%）\n")
    L.append("\n## 候補手法（5-40倍 × edge>0 上位3点）\n")
    L.append(f"- **フラット**(各100円): ROI {ev['flat']['roi']}% PnL {ev['flat']['pnl']} "
             f"的中{ev['flat']['hits']}/{ev['flat']['n_bets']} 最大1本占{ev['flat']['top_hit_share']}%\n")
    L.append(f"- **本番同等**(¥200,000): ROI {ev['prod']['roi']}% PnL {ev['prod']['pnl']} "
             f"最終残高{ev['prod']['final_bankroll']} maxDD {ev['prod']['max_drawdown']}\n")
    L.append("\n## 既存戦略（同シミュ検証＝ベンチマーク）\n")
    for strat, label in (("mc_venue_focus", "P"), ("mc2_venue_focus", "P2"), ("v11_var13", "V11")):
        try:
            db, sim = replay_strategy_from_db(strat, "2026-04-08", date_to)
            L.append(f"- {label}: DB PnL {db['pnl']} / sim PnL {sim['pnl']}（再現差={sim['pnl']-db['pnl']}）\n")
        except Exception as e:
            L.append(f"- {label}: 取得失敗 {e}\n")
    L.append("\n## 判定（結論は出すが採否は岩下さん）\n")
    passed = ev["flat"]["roi"] > 100 and ev["flat"]["top_hit_share"] < 50
    L.append(f"- フラットROI>100%かつ広く勝つ: **{'該当' if passed else '非該当'}**"
             f"（ROI {ev['flat']['roi']}%, 最大1本占 {ev['flat']['top_hit_share']}%）\n")
    L.append("- **前向き**: 台帳は蓄積中。本レポートは初期窓の一次読み。後続窓で方向一致を要確認。\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(L))
    return OUT
```

- [ ] **Step 4: 成功確認 & 目視** — `venv/Scripts/python.exe analysis/combo/report.py` 実行（`__main__` 追加不要、下記でrun）:
```bash
venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'analysis/combo'); from report import build_report; print(build_report('2026-06-30','2026-12-31'))"
venv/Scripts/python.exe analysis/combo/test_report.py
```
Expected: `RESULTS.md` 生成、`ALL PASS`。生成された `RESULTS.md` を目視。

- [ ] **Step 5: コミット**
```bash
git add analysis/combo/evaluate.py analysis/combo/report.py analysis/combo/test_report.py analysis/combo/RESULTS.md
git commit -m "feat(combo): evaluate candidate + RESULTS.md report (flat/prod, benchmark, verdict)"
```

---

## 完了条件
- 全 `test_*.py` が PASS（特に Task6 の忠実度ゲートが緑＝sim が既存DB実績を再現）。
- `RESULTS.md` に 歪みマップ・モデル上位N・候補手法(フラット/本番同等)・既存ベンチ・判定・前向き注記 が揃う。
- 本番・DB・運用設定への変更ゼロ。
- 結論を踏まえ user に報告。**採否は岩下さん**。前向き窓での再実行手順を明記。

## Self-Review（作成者チェック結果）
- **Spec coverage**: §6 loader→T1 / joint(qmc主・proxy対照)→T2 / market→T3 / diagnose→T4 / sim(¥200k)→T5 / validate(忠実度ゲート＝既存DB再現)→T6 / method(事前登録)→T7 / evaluate(フラット＋本番同等)＋report→T8。§8規律（事前登録・前向き・集中度・言語化）は method 固定＋report の集中度/前向き注記でカバー。
- **Placeholder scan**: TBD/TODO無し。各stepに実コード・実コマンド・期待出力。
- **Type consistency**: `load_races`→race dict(keys: race_id/odds/p1/p2/p3/win_combo/race_data/boats_data) を joint/diagnose/method/evaluate が一貫使用。`qmc_joint/proxy_joint`(joint)、`implied_probs`(market)、`select_combos/CANDIDATE`(method)、`stake_kelly/run_sim/DEFAULT_CFG/INITIAL_BANKROLL`(sim)、`replay_strategy_from_db`(validate) を後続 Task が定義通り使用。`run_sim` の per_race dict(race_id/win_combo/bets:[(combo,amount,odds)]) を sim/validate/evaluate で一致。
- **既知の注意**: Task2 の QMC 再計算は boats_data の展示等が必要。欠損レースは qmc_sanrentan_v3 内で fallback するが、joint が空に近い場合は evaluate 側で picks 0 となり自然にスキップ（除外率は RESULTS に出す拡張は次段）。
