from db import connect, SPLIT_DATE
from stage_a_softscore import run_stage_a

def test_stage_a_no_leakage_and_structure():
    conn = connect()
    res = run_stage_a(conn)
    # リーク防止: trainの最大日 < SPLIT_DATE <= testの最小日
    assert res["train_max_date"] < SPLIT_DATE <= res["test_min_date"], res
    # tercile結果が3群そろう
    assert set(res["payout_by_tercile"].keys()) == {"low", "mid", "high"}
    # 決定論（同じ入力→同じスコア）
    res2 = run_stage_a(conn)
    assert res["softscore_top_bottom_payout_ci"] == res2["softscore_top_bottom_payout_ci"]
    print("OK stage_a verdict_payout_monotonic=", res["payout_monotonic"],
          "beats_baseline=", res["beats_calendar_baseline"])

if __name__ == "__main__":
    test_stage_a_no_leakage_and_structure(); print("ALL PASS")
