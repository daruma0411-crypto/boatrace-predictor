from db import connect
from features import race_features, FEATURE_NAMES

def test_features_shape_and_ranges():
    conn = connect()
    feats = race_features(conn, "2026-06-01", "2026-06-10")
    assert len(feats) > 50, len(feats)            # 6/1-10で十分なレース数
    sample = next(iter(feats.values()))
    for name in FEATURE_NAMES:
        assert name in sample, name               # 全特徴が存在
    # win_rate分散は非負、inside_wrは0-9程度
    assert sample["wr_std"] >= 0
    assert 0 <= sample["inside_wr"] <= 12
    assert sample["gw_flag"] in (0, 1)
    print("OK n_races=", len(feats), "feat=", sample)

if __name__ == "__main__":
    test_features_shape_and_ranges(); print("ALL PASS")
