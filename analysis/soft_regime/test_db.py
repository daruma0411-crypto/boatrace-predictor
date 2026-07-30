from db import connect, classify_period, gw_flag, strategy_roi

def test_gw_and_period():
    assert gw_flag("2026-05-03") is True
    assert gw_flag("2026-06-20") is False
    assert classify_period("2026-04-20") == "1_preGW"
    assert classify_period("2026-05-03") == "2_GW"
    assert classify_period("2026-05-30") == "3_dead"
    assert classify_period("2026-06-20") == "4_breakout"

def test_p_roi_band():
    conn = connect()
    s = strategy_roi(conn, "mc_venue_focus", "2026-04-01", "2026-12-31")
    # 既知: P 全期間 ROI は強い黒字。2026-06-24時点≈135.5%、その後のbreakout期勝ち
    # により 2026-06-18 観測で146.2%まで上昇。DB更新で動くため幅広めで「強黒字」を担保。
    assert 125.0 <= s["roi"] <= 160.0, s
    assert s["profit"] > 120000, s
    assert s["n"] > 250, s
    print("OK test_p_roi_band", s)

if __name__ == "__main__":
    test_gw_and_period(); test_p_roi_band(); print("ALL PASS")
