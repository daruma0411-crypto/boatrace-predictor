from sim import stake_kelly, run_sim, DEFAULT_CFG

def test_stake_kelly_positive_edge_bets():
    # p=0.3, odds=5 → b=4, kelly=(4*0.3-0.7)/4=0.125>0 → 賭ける
    amt = stake_kelly(0.30, 5.0, bankroll=200000, cfg=DEFAULT_CFG)
    assert amt >= DEFAULT_CFG["min_bet"] and amt % 100 == 0
    # 負けエッジは0
    assert stake_kelly(0.10, 5.0, bankroll=200000, cfg=DEFAULT_CFG) == 0

def test_run_sim_settle_only_reproduces_given_bets():
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
