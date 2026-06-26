import os
from db import connect
from stage_b_calendar import run_stage_b
from stage_a_softscore import run_stage_a
from simulate_policy import simulate

OUT = os.path.join(os.path.dirname(__file__), "RESULTS.md")


def build_report(conn):
    b = run_stage_b(conn)
    a = run_stage_a(conn)
    flat = simulate(conn, None)
    policy = simulate(conn, {"low": 0.5, "mid": 1.0, "high": 1.5})

    ci = a["softscore_top_bottom_payout_ci"]
    verdict_detectable = a["payout_monotonic"] and ci[0] > 0 and a["beats_calendar_baseline"]
    verdict = ("事前検知できる" if verdict_detectable else "事前検知できない（後出し専用）")

    lines = []
    lines.append("# 軟相場 事前検知可能性 — 検証結果\n")
    lines.append(f"## 結論: **{verdict}**\n")
    lines.append("主判定: (a)payout単調=%s, (c)top-bottom CI=%s(>0=%s), (d)ベースライン超え=%s\n"
                 % (a["payout_monotonic"], ci, ci[0] > 0, a["beats_calendar_baseline"]))
    lines.append("\n## Stage B（カレンダー・ベースライン）\n")
    lines.append(f"- GW平均payout {b['payout_gw_mean']} vs 非GW {b['payout_nongw_mean']}, 差CI {b['payout_gw_vs_nongw']}\n")
    lines.append(f"- ROI: GW {b['roi_gw']}% vs 非GW {b['roi_nongw']}%\n")
    lines.append("\n## Stage A（事前特徴 softスコア）\n")
    lines.append(f"- train {a['train_n']}本(〜{a['train_max_date']}) / test {a['test_n']}本({a['test_min_date']}〜)\n")
    lines.append(f"- 重み: {a['weights']}\n")
    lines.append(f"- tercile別 平均payout: {a['payout_by_tercile']}（単調={a['payout_monotonic']}）\n")
    lines.append(f"- tercile別 P/P2 ROI(確認): {a['roi_by_tercile']}\n")
    lines.append(f"- 相関 softスコア {a['corr_softscore']} vs カレンダー {a['corr_calendar_baseline']} → 上積み={a['beats_calendar_baseline']}\n")
    lines.append("\n## 運用シミュ（条件付き・test期間）\n")
    lines.append(f"- 均等: ROI {flat['roi']}% profit {flat['profit']} maxDD {flat['max_drawdown']} ({flat['n_bets']}本)\n")
    lines.append(f"- ポリシー(緩×1.5/中×1.0/堅×0.5): ROI {policy['roi']}% profit {policy['profit']} maxDD {policy['max_drawdown']}\n")
    if not verdict_detectable:
        lines.append("\n> 注: 事前検知が否定されたため、運用シミュは参考値。会場/サイジング等の別アプローチへ。\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return OUT


if __name__ == "__main__":
    print("wrote", build_report(connect()))
