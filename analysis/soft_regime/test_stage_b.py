from db import connect
from stage_b_calendar import run_stage_b

def test_stage_b_runs_and_reports_gw():
    conn = connect()
    res = run_stage_b(conn)
    assert "payout_gw_vs_nongw" in res
    ci = res["payout_gw_vs_nongw"]  # (lo, hi, mean) GW平均payout - 非GW平均payout
    assert len(ci) == 3
    # GW期のP+P2 ROIは既知で高い（≈227%）。幅で確認。
    assert res["roi_gw"] > 150, res["roi_gw"]
    print("OK stage_b:", res)

if __name__ == "__main__":
    test_stage_b_runs_and_reports_gw(); print("ALL PASS")
