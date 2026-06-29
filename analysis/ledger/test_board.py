from db import connect
from board import load_candidate_board, ROW_KEYS

def test_load_runs_and_shape():
    conn = connect()
    rows = load_candidate_board(conn, "2026-06-01", "2026-12-31")
    assert isinstance(rows, list)              # 動いてリストを返す（空でも可）
    if rows:                                    # 盤データがあれば形を確認
        for k in ROW_KEYS:
            assert k in rows[0], k
        assert isinstance(rows[0]["odds_3t"], dict)
    print("OK n_rows=", len(rows))

if __name__ == "__main__":
    test_load_runs_and_shape(); print("ALL PASS")
