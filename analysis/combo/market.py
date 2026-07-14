def implied_probs(odds):
    """{combo: odds} → de-vig した {combo: 市場含み確率}（合計1）。"""
    raw = {c: (1.0 / o) for c, o in odds.items() if o and o > 0}
    s = sum(raw.values())
    if s <= 0:
        return {c: 0.0 for c in odds}
    return {c: v / s for c, v in raw.items()}
