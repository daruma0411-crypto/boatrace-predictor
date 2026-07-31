import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager
from src.softness_store import save_race_softness, get_temperature

class _Cur:
    def __init__(self, log, rows=None): self.log = log; self._rows = rows or []
    def execute(self, sql, params=None): self.log.append((sql.strip().split()[0].upper(), params))
    def fetchall(self): return self._rows

class _Conn:
    def __init__(self, log, rows=None): self.log = log; self._rows = rows or []
    def cursor(self): return _Cur(self.log, self._rows)

def _factory(log, rows=None, raise_it=False):
    @contextmanager
    def f():
        if raise_it: raise RuntimeError("db down")
        yield _Conn(log, rows)
    return f

def test_save_success_and_never_raises():
    log = []
    assert save_race_softness(1, "2026-07-30", 8.1, conn_factory=_factory(log)) is True
    verbs = [v for v, _ in log]
    assert "CREATE" in verbs and "INSERT" in verbs
    assert save_race_softness(2, "2026-07-30", 8.1, conn_factory=_factory([], raise_it=True)) is False
    assert save_race_softness(3, "2026-07-30", None, conn_factory=_factory([])) is False  # score無しは保存しない

def test_temperature_mean_of_daily_means():
    rows = [{"d": "2026-07-23", "m": 8.0}, {"d": "2026-07-24", "m": 9.0}, {"d": "2026-07-25", "m": 7.0}]
    t = get_temperature("2026-07-30", conn_factory=_factory([], rows=rows))
    assert abs(t - 8.0) < 1e-9

def test_temperature_insufficient_days_returns_none():
    rows = [{"d": "2026-07-29", "m": 8.0}]
    assert get_temperature("2026-07-30", conn_factory=_factory([], rows=rows)) is None
    assert get_temperature("2026-07-30", conn_factory=_factory([], raise_it=True)) is None

if __name__ == "__main__":
    test_save_success_and_never_raises(); test_temperature_mean_of_daily_means()
    test_temperature_insufficient_days_returns_none(); print("ALL PASS")
