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
