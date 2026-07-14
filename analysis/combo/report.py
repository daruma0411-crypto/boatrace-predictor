import os
from loader import load_races
from diagnose import odds_band_map, model_topN_roi
from evaluate import evaluate_method
from validate import replay_strategy_from_db

OUT = os.path.join(os.path.dirname(__file__), "RESULTS.md")


def build_report(date_from, date_to, limit=None):
    races = load_races(date_from, date_to, limit=limit)   # 1回だけロード
    band = odds_band_map(races)
    topN = model_topN_roi(races, joint="proxy", N_list=(1, 3, 5))
    ev = evaluate_method(races)
    L = []
    L.append(f"# 組み合わせ手法 研究 — 中間結果 ({date_from}〜{date_to})\n")
    L.append(f"対象レース={ev['n_races']}（全盤台帳）。**2週間規模の暫定・前向き検証前提。**\n")
    L.append("\n## 市場の歪みマップ（オッズ帯別 blanket 回収率）\n")
    for k, v in band.items():
        L.append(f"- {k}倍: slots={v['slots']} 的中率={v['hit_rate']:.2f}% 回収={v['roi']:.0f}%\n")
    L.append("\n## モデルの組み合わせ力（proxy 上位N点 回収率）\n")
    L.append(f"- 上位1/3/5点: {topN[1]:.0f}% / {topN[3]:.0f}% / {topN[5]:.0f}%（基準 blanket≈57%）\n")
    L.append("\n## 候補手法（5-40倍 × edge>0 上位3点、本物QMC結合）\n")
    L.append(f"- **フラット**(各100円): ROI {ev['flat']['roi']}% PnL {ev['flat']['pnl']} "
             f"的中{ev['flat']['hits']}/{ev['flat']['n_bets']} 最大1本占{ev['flat']['top_hit_share']}%\n")
    L.append(f"- **本番同等**(¥200,000): ROI {ev['prod']['roi']}% PnL {ev['prod']['pnl']} "
             f"最終残高{ev['prod']['final_bankroll']} maxDD {ev['prod']['max_drawdown']}\n")
    # ベンチマークは候補と同じ窓(date_from〜)でフェアに比較する（GW込み全期間で下駄を履かせない）
    L.append(f"\n## 既存戦略（**同じ窓 {date_from}〜** で公平比較＝ベンチマーク）\n")
    for strat, label in (("mc_venue_focus", "P"), ("mc2_venue_focus", "P2"), ("v11_var13", "V11")):
        try:
            db, sim = replay_strategy_from_db(strat, date_from, date_to)
            L.append(f"- {label}: DB PnL {db['pnl']} / sim PnL {sim['pnl']} "
                     f"（n={db['n']}, 再現差={sim['pnl']-db['pnl']}）\n")
        except Exception as e:
            L.append(f"- {label}: 取得失敗 {e}\n")
    L.append("\n## 判定（結論は出すが採否は岩下さん）\n")
    passed = ev["flat"]["roi"] > 100 and ev["flat"]["top_hit_share"] < 50
    L.append(f"- フラットROI>100%かつ広く勝つ: **{'該当' if passed else '非該当'}**"
             f"（ROI {ev['flat']['roi']}%, 最大1本占 {ev['flat']['top_hit_share']}%）\n")
    L.append("- **前向き**: 台帳は蓄積中。本レポートは初期窓の一次読み。後続窓で方向一致を要確認。\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(L))
    return OUT
