"""本番同等の 3連単シミュレータ（¥200,000 スタート）。

Staking は production の kelly 公式（src/betting.py _strategy_kelly L911-934）に準拠:
    b = discounted_odds - 1
    kelly = (b*p - q) / b
    amount = bankroll * kelly * kelly_fraction
    clamp [min_bet, max_ticket] → 100円丸め
Settlement は的中買い目が picks に含まれれば amount*odds を払戻。
settle_only=True は「与えられた (combo, amount, odds) をそのまま清算する」
（既に張られた bet の検証用 / Task 6 で既存戦略の DB PnL 再現に使う）。
"""

INITIAL_BANKROLL = 200000

# 研究用の保守的デフォルト。式は production の real_kelly 経路(src/betting.py _strategy_kelly)
# 準拠だが kelly_fraction=0.0625 は本番戦略(P=0.125/他=0.20-0.25)の約半分＝未確認エッジを
# 過剰に張らないための保守設定（規律通り）。odds_discount 等の他値は本番と同等。
DEFAULT_CFG = {
    "kelly_fraction": 0.0625,
    "max_ticket_bet_ratio": 0.008,
    "max_total_bet_ratio": 0.02,
    "min_bet": 100,
    "odds_discount": 0.92,
}


def stake_kelly(prob, odds, bankroll, cfg):
    """production 準拠の kelly 賭け金。負けエッジは0。100円丸め。"""
    d_odds = odds * cfg["odds_discount"]
    b = d_odds - 1.0
    if b < 0.01:
        return 0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0:
        return 0
    amount = bankroll * kelly * cfg["kelly_fraction"]
    max_ticket = bankroll * cfg["max_ticket_bet_ratio"]
    amount = max(cfg["min_bet"], min(max_ticket, amount))
    return int(round(amount / 100) * 100)


def run_sim(per_race, cfg=DEFAULT_CFG, settle_only=False):
    """per_race: [{race_id, win_combo, bets:[(combo,amount,odds)]}]（settle_only）
       equity/ROI/PnL/DD/的中/集中度 を返す。¥200,000 スタート。"""
    bankroll = INITIAL_BANKROLL
    inv = ret = 0.0
    n_bets = 0
    hits = 0
    peak = bankroll
    max_dd = 0.0
    biggest = 0.0
    for race in per_race:
        win = race["win_combo"]
        # 当たり目は確定配当(payout_odds)で決済＝本番の実払戻。無ければ bet odds にフォールバック。
        payout_odds = race.get("payout_odds")
        race_inv = race_ret = 0.0
        for combo, amount, odds in race["bets"]:
            if amount <= 0:
                continue
            race_inv += amount
            n_bets += 1
            if combo == win:
                pay = amount * (payout_odds if payout_odds else odds)
                race_ret += pay
                hits += 1
                biggest = max(biggest, pay)
        inv += race_inv
        ret += race_ret
        bankroll += (race_ret - race_inv)
        peak = max(peak, bankroll)
        max_dd = min(max_dd, bankroll - peak)
    return {
        "n_bets": n_bets, "hits": hits, "invested": round(inv), "returned": round(ret),
        "pnl": round(ret - inv), "roi": round(ret / inv * 100, 1) if inv else 0.0,
        "final_bankroll": round(bankroll), "max_drawdown": round(max_dd),
        "top_hit_share": round(biggest / ret * 100, 1) if ret else 0.0,
    }
