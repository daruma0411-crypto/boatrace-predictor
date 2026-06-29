import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.odds_board import build_board_row

def test_build_board_row_basic():
    odds = {"1-2-3": 45.6, "1-2-4": 88.1, "3-1-5": 12.0}
    rid, j, n = build_board_row(777, odds)
    assert rid == 777
    assert n == 3
    assert json.loads(j) == odds

def test_build_board_row_casts_int():
    rid, j, n = build_board_row("888", {"1-2-3": 10.0})
    assert rid == 888 and n == 1

if __name__ == "__main__":
    test_build_board_row_basic(); test_build_board_row_casts_int(); print("ALL PASS")
