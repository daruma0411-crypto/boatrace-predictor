from loader import load_races

def test_load_races_shape():
    races = load_races("2026-06-30", "2026-12-31", limit=50)
    assert isinstance(races, list) and len(races) > 0
    r = races[0]
    for k in ("race_id","race_number","odds","p1","p2","p3","win_combo","race_data","boats_data"):
        assert k in r, k
    assert isinstance(r["odds"], dict) and len(r["odds"]) >= 100   # 全盤
    assert len(r["p1"]) == 6 and len(r["boats_data"]) == 6
    assert r["win_combo"].count("-") == 2
    assert "venue_id" in r["race_data"]
    assert "exhibition_time" in r["boats_data"][0] and "player_class" in r["boats_data"][0]
    print("OK n=", len(races))

if __name__ == "__main__":
    test_load_races_shape(); print("ALL PASS")
