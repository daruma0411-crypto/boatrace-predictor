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
    # QMC v3 は Sobol サンプルに現れた combo のみ返す sparse dict。
    # 本命集中レース(R3等)では ~40 通りに収束するため floor は 30 とする
    # (フィデリティの本丸は下の `row["combination"] in j`)。
    assert isinstance(j, dict) and len(j) > 30
    assert row["combination"] in j, f"bet combo {row['combination']} が QMC joint に無い"
    print("OK qmc_joint combo=", row["combination"], "prob=", round(j[row["combination"]], 4))

if __name__ == "__main__":
    test_proxy_joint_basic(); test_qmc_joint_reproduces_a_bet(); print("ALL PASS")
