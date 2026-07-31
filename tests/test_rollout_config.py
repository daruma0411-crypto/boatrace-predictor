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
