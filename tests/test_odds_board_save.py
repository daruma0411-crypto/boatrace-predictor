import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager
from src.odds_board import save_odds_board

class _FakeCur:
    def __init__(self, log): self.log = log
    def execute(self, sql, params=None): self.log.append((sql.strip().split()[0], params))

class _FakeConn:
    def __init__(self, log): self.log = log
    def cursor(self): return _FakeCur(self.log)

def _factory(log, raise_it=False):
    @contextmanager
    def f():
        if raise_it:
            raise RuntimeError("db down")
        yield _FakeConn(log)
    return f

def test_save_success_executes_insert_and_health():
    log = []
    ok = save_odds_board(101, {"1-2-3": 9.9, "1-2-4": 5.0}, conn_factory=_factory(log))
    assert ok is True
    verbs = [v for v, _ in log]
    assert "CREATE" in verbs and "INSERT" in verbs
    inserts = [p for v, p in log if v == "INSERT" and p and p[0] == 101]
    assert inserts and inserts[0][2] == 2

def test_save_empty_odds_returns_false():
    assert save_odds_board(102, {}, conn_factory=_factory([])) is False

def test_save_never_raises_on_db_error():
    assert save_odds_board(103, {"1-2-3": 9.9}, conn_factory=_factory([], raise_it=True)) is False

def test_save_default_factory_used_when_none():
    import types, sys as _sys
    log = []
    fake_mod = types.ModuleType("src.database")
    fake_mod.get_db_connection = _factory(log)
    _sys.modules["src.database"] = fake_mod
    try:
        assert save_odds_board(201, {"1-2-3": 9.9}, conn_factory=None) is True
        assert any(v == "INSERT" for v, _ in log)
    finally:
        _sys.modules.pop("src.database", None)

def test_save_default_factory_import_failure_returns_false():
    import types, sys as _sys
    bad = types.ModuleType("src.database")
    def _boom():
        raise RuntimeError("import-time boom")
    bad.get_db_connection = _boom  # callable that raises when used
    _sys.modules["src.database"] = bad
    try:
        # even if the default factory raises, caller must not see an exception
        assert save_odds_board(202, {"1-2-3": 9.9}, conn_factory=None) is False
    finally:
        _sys.modules.pop("src.database", None)

if __name__ == "__main__":
    test_save_success_executes_insert_and_health()
    test_save_empty_odds_returns_false()
    test_save_never_raises_on_db_error()
    test_save_default_factory_used_when_none()
    test_save_default_factory_import_failure_returns_false()
    print("ALL PASS")
