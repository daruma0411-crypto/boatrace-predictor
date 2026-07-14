from loader import load_races
from diagnose import odds_band_map, model_topN_roi

def test_favorite_longshot_shape():
    races = load_races("2026-06-30", "2026-12-31", limit=800)
    m = odds_band_map(races)
    # 既知: 200倍超は低回収(<60%)、5-40倍は高め(>70%)
    assert m["200-100000"]["roi"] < 60, m["200-100000"]
    assert m["5-10"]["roi"] > 70 and m["10-20"]["roi"] > 70, m
    print("OK band map:", {k: round(v["roi"]) for k, v in m.items()})

def test_model_topN_beats_blanket():
    races = load_races("2026-06-30", "2026-12-31", limit=800)
    roi = model_topN_roi(races, joint="proxy", N_list=(1, 3, 5))
    # 頑健な事実: モデル上位N点は blanket 基準(全通り平均≈57%)を大きく超える＝選別にスキルがある。
    # 注: 「上位1点が最良（単調劣化）」は窓依存で頑健でない（初期窓では top5>top1）。
    # 過学習よけのため k の最適順序は主張せず、"blanket 超え" のみを構造的事実として検証する。
    assert min(roi.values()) > 65, roi   # 上位N全帯が blanket(57%)を明確に超える
    assert roi[1] > 75, roi              # 上位1点も blanket を大きく超える
    print("OK model topN(proxy):", {k: round(v) for k, v in roi.items()})

if __name__ == "__main__":
    test_favorite_longshot_shape(); test_model_topN_beats_blanket(); print("ALL PASS")
