from joint import qmc_joint
from market import implied_probs
from method import select_combos, CANDIDATE
from sim import run_sim, stake_kelly, DEFAULT_CFG, INITIAL_BANKROLL


def evaluate_method(races, joint_fn=qmc_joint):
    """事前ロード済み races に候補手法を適用し、フラット/本番同等の sim 結果を返す。"""
    flat = []; prod = []
    bankroll = INITIAL_BANKROLL
    for r in races:
        model = joint_fn(r); market = implied_probs(r["odds"])
        picks = select_combos(r, model, market, CANDIDATE)
        po = r.get("payout_odds")
        flat.append({"race_id": r["race_id"], "win_combo": r["win_combo"],
                     "payout_odds": po, "bets": [(c, 100, od) for c, _, od in picks]})
        pbets = []
        for c, mp, od in picks:
            amt = stake_kelly(mp, od, bankroll, DEFAULT_CFG)
            if amt > 0:
                pbets.append((c, amt, od))
        prod.append({"race_id": r["race_id"], "win_combo": r["win_combo"],
                     "payout_odds": po, "bets": pbets})
        # サイジング用 bankroll も確定配当で更新（当たり目のみ payout_odds、無ければ bet odds）
        bankroll += sum(((a * (po if po else od)) if c == r["win_combo"] else -a)
                        for c, a, od in pbets)
    return {"n_races": len(races), "flat": run_sim(flat, settle_only=True),
            "prod": run_sim(prod, settle_only=True)}
