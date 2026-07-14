from validate import replay_strategy_from_db

def test_sim_reproduces_db_pnl():
    for strat in ("mc_venue_focus", "mc2_venue_focus"):   # P, P2
        db, sim = replay_strategy_from_db(strat, "2026-04-08", "2026-12-31")
        assert db["n"] > 50, (strat, db)
        # sim 決済(確定配当)の PnL が DB 実績 PnL と一致（同じ実払戻を再構成）
        assert abs(sim["pnl"] - db["pnl"]) <= max(1000, abs(db["pnl"]) * 0.03), (strat, db, sim)
        print(f"OK {strat}: DB pnl={db['pnl']} sim pnl={sim['pnl']} (n={db['n']})")

if __name__ == "__main__":
    test_sim_reproduces_db_pnl(); print("ALL PASS")
