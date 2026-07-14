import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from src.monte_carlo import qmc_sanrentan_v3


def proxy_joint(race):
    """p1×p2×p3 の独立近似（対照用）。{combo: prob}（全120通り、正規化なし）。"""
    p1, p2, p3 = race["p1"], race["p2"], race["p3"]
    out = {}
    for a in range(6):
        for b in range(6):
            if b == a: continue
            for c in range(6):
                if c == a or c == b: continue
                out[f"{a+1}-{b+1}-{c+1}"] = p1[a] * p2[b] * p3[c]
    return out


def qmc_joint(race, seed=12345):
    """production の QMC v3 をそのまま呼ぶ（本物）。{combo: prob}（sparse）。"""
    return qmc_sanrentan_v3(
        np.array(race["p1"]),
        boats_data=race["boats_data"],
        race_data=race["race_data"],
        race_number=race["race_number"],
        seed=seed,
    )
