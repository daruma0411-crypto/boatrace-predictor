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
    # 1300*0.5=650 → 100円丸めは production(betting.py) と同じ Python round(銀行丸め):
    # round(6.5)=6 → 600円。四捨五入(700)ではない点に注意(規約統一)。
    assert out[0]["amount"] == 600
    tiny = [dict(P_BETS[0], amount=100)]
    assert derive_p_kai_bets(tiny, HARD_CUT + 0.5, TEMP_THR - 1) == []  # 50→round(0.5)=0→捨て

def test_none_score_keeps_none_temp_warm():
    assert len(derive_p_kai_bets(P_BETS, None, None)) == 1   # 不明は保守的に素通し(等倍)
    assert derive_p_kai_bets([], 9.9, None) == []

if __name__ == "__main__":
    test_hard_race_skipped(); test_warm_keeps_full_amount(); test_cold_halves_and_rounds()
    test_none_score_keeps_none_temp_warm(); print("ALL PASS")
