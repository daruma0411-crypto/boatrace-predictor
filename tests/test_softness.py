import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.softness import score_race
from src.softness_params import HARD_CUT, TEMP_THR

def _boat(n, wr, motor, st):
    return {"boat_number": n, "win_rate": wr, "motor_win_rate_2": motor, "avg_st": st}

def test_score_race_basic():
    boats = [_boat(i+1, 5.0+i*0.3, 35.0+i, 0.15) for i in range(6)]
    s = score_race(boats)
    assert isinstance(s, float)
    assert 7.86 < HARD_CUT < 7.87 and 7.97 < TEMP_THR < 7.98  # 凍結値の再確認

def test_score_race_insufficient_data_returns_none():
    boats = [_boat(1, 5.0, 35.0, 0.15), _boat(2, None, None, None),
             _boat(3, None, None, None), _boat(4, None, None, None),
             _boat(5, None, None, None), _boat(6, None, None, None)]
    assert score_race(boats) is None   # 有効win_rateが4艇未満

def test_score_direction():
    # イン(1号艇)が強く格差が大きい=堅い → 点数は低いはず
    hard = [_boat(1, 8.0, 50.0, 0.12)] + [_boat(i, 4.0, 30.0, 0.18) for i in range(2, 7)]
    soft = [_boat(i, 5.5, 35.0, 0.15) for i in range(1, 7)]
    assert score_race(hard) < score_race(soft)

if __name__ == "__main__":
    test_score_race_basic(); test_score_race_insufficient_data_returns_none(); test_score_direction()
    print("ALL PASS")
