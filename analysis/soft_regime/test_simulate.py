from db import connect
from simulate_policy import simulate

def test_uniform_multiplier_equals_flat():
    conn = connect()
    flat = simulate(conn, {"low": 1.0, "mid": 1.0, "high": 1.0})
    base = simulate(conn, None)  # None = 均等(flat)
    assert abs(flat["roi"] - base["roi"]) < 1e-6, (flat, base)

def test_policy_runs():
    conn = connect()
    pol = simulate(conn, {"low": 0.5, "mid": 1.0, "high": 1.5})
    assert "roi" in pol and "max_drawdown" in pol and "n_bets" in pol
    print("OK policy:", pol)

if __name__ == "__main__":
    test_uniform_multiplier_equals_flat(); test_policy_runs(); print("ALL PASS")
