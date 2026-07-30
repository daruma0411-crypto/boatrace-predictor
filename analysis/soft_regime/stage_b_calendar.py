from db import fetch, connect, gw_flag, STRATS
from util import bootstrap_diff_ci, roi_from_pairs


def _payouts(conn, date_from="2026-04-01", date_to="2026-12-31"):
    sql = """
        select r.race_date, r.payout_sanrentan
        from races r
        where r.race_date between %s and %s and r.payout_sanrentan is not null
              and r.result_1st is not null
    """
    _, rows = fetch(conn, sql, (date_from, date_to))
    return rows


def _strategy_pairs(conn, date_from="2026-04-01", date_to="2026-12-31"):
    sql = """
        select r.race_date, b.amount, coalesce(b.return_amount, b.payout, 0)
        from bets b join races r on b.race_id = r.id
        where b.strategy_type = any(%s) and r.race_date between %s and %s
              and r.result_1st is not null
    """
    _, rows = fetch(conn, sql, (list(STRATS), date_from, date_to))
    return rows


def run_stage_b(conn):
    pay = _payouts(conn)
    gw_pay = [float(p) for d, p in pay if gw_flag(d)]
    non_pay = [float(p) for d, p in pay if not gw_flag(d)]
    ci = bootstrap_diff_ci(gw_pay, non_pay)

    pairs = _strategy_pairs(conn)
    gw_pairs = [(a, r) for d, a, r in pairs if gw_flag(d)]
    non_pairs = [(a, r) for d, a, r in pairs if not gw_flag(d)]
    return {
        "n_races_gw": len(gw_pay), "n_races_nongw": len(non_pay),
        "payout_gw_mean": round(sum(gw_pay) / len(gw_pay), 0) if gw_pay else 0,
        "payout_nongw_mean": round(sum(non_pay) / len(non_pay), 0) if non_pay else 0,
        "payout_gw_vs_nongw": ci,                       # (lo, hi, mean)
        "roi_gw": round(roi_from_pairs(gw_pairs), 1),
        "roi_nongw": round(roi_from_pairs(non_pairs), 1),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_stage_b(connect()), default=str, ensure_ascii=False, indent=2))
