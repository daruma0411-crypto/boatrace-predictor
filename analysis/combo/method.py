# 事前登録（測定前に確定）。変種は別ファイル/別 run として記録すること。
CANDIDATE = {"odds_lo": 5.0, "odds_hi": 40.0, "edge_min": 0.0, "top_k": 3}


def select_combos(race, model_probs, market_probs, cfg=CANDIDATE):
    """オッズ帯フィルタ → edge=model-market>edge_min → 上位 top_k。
       返り値 [(combo, model_prob, odds)]。"""
    cands = []
    for combo, od in race["odds"].items():
        if not (cfg["odds_lo"] <= od <= cfg["odds_hi"]):
            continue
        mp = model_probs.get(combo, 0.0)
        edge = mp - market_probs.get(combo, 0.0)
        if edge > cfg["edge_min"]:
            cands.append((combo, mp, od, edge))
    cands.sort(key=lambda x: -x[3])
    return [(c, mp, od) for c, mp, od, _ in cands[:cfg["top_k"]]]
