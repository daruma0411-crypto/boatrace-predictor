"""P改: P(mc_venue_focus) の picks に凍結ルールを適用して p_kai bets を導出する純関数。
ルール(2026-07-30凍結): 堅レース(score<=HARD_CUT)見送り / 低温日(temperature<TEMP_THR)は×COLD_SCALE。
score/temperature が None の場合は保守的に「適用しない」(=素通し)。増し張りは存在しない。
100円丸めは production betting.py と同じ round()(銀行丸め) を用いる。"""
from src.softness_params import HARD_CUT, TEMP_THR, COLD_SCALE


def derive_p_kai_bets(p_bets, score, temperature):
    if not p_bets:
        return []
    if score is not None and score <= HARD_CUT:
        return []
    scale = COLD_SCALE if (temperature is not None and temperature < TEMP_THR) else 1.0
    out = []
    for b in p_bets:
        amount = int(round(b["amount"] * scale / 100) * 100)
        if amount < 100:
            continue
        nb = dict(b)
        nb["amount"] = amount
        nb["strategy_type"] = "p_kai"
        out.append(nb)
    return out
