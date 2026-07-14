from market import implied_probs

def test_implied_sums_to_one_and_favorite_top():
    odds = {"1-2-3": 5.0, "1-2-4": 10.0, "2-1-3": 50.0}
    imp = implied_probs(odds)
    assert abs(sum(imp.values()) - 1.0) < 1e-9
    assert max(imp, key=imp.get) == "1-2-3"     # 最小オッズが最大含み
    assert imp["1-2-3"] > imp["2-1-3"]

if __name__ == "__main__":
    test_implied_sums_to_one_and_favorite_top(); print("ALL PASS")
