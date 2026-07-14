from joint import qmc_joint, proxy_joint

_EDGES = [1, 5, 10, 20, 40, 80, 200, 100000]


def odds_band_map(races):
    """オッズ帯別 blanket 回収率（全combo 1点ずつ買ったら）。"""
    band = {f"{lo}-{hi}": {"slots": 0, "wins": 0, "ret": 0.0}
            for lo, hi in zip(_EDGES, _EDGES[1:])}
    for r in races:
        win = r["win_combo"]
        for combo, od in r["odds"].items():
            if od <= 1: continue
            for lo, hi in zip(_EDGES, _EDGES[1:]):
                if lo <= od < hi:
                    key = f"{lo}-{hi}"; band[key]["slots"] += 1
                    if combo == win:
                        band[key]["wins"] += 1; band[key]["ret"] += od
                    break
    for k, v in band.items():
        v["roi"] = (v["ret"] / v["slots"] * 100) if v["slots"] else 0.0
        v["hit_rate"] = (v["wins"] / v["slots"] * 100) if v["slots"] else 0.0
    return band


def model_topN_roi(races, joint="qmc", N_list=(1, 3, 5, 10)):
    """各レースでモデル上位N点を1点ずつ買った時の回収率。"""
    jf = qmc_joint if joint == "qmc" else proxy_joint
    acc = {N: [0, 0.0] for N in N_list}   # [n_bet, ret]
    for r in races:
        j = jf(r)
        ranked = sorted(j.items(), key=lambda x: -x[1])
        for N in N_list:
            for combo, _ in ranked[:N]:
                acc[N][0] += 1
                if combo == r["win_combo"]:
                    acc[N][1] += r["odds"].get(combo, 0.0)
    return {N: (ret / nb * 100 if nb else 0.0) for N, (nb, ret) in acc.items()}
