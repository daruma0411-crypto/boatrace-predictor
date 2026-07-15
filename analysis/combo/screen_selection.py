"""候補手法の bet を、事前登録した少数仮説(H1-H4)で train/test screening する。

目的: 「広く張った候補(組み合わせ手法)の中に、事前にわかる特徴で儲かる部分集合があるか」を
過学習を避けて探す。前半(train)で見て後半(test)で確認し、**両窓で同じ向きに出た仮説だけ**を残す。

規律(過去4連敗の死因回避):
- 事前登録の少数仮説のみ(結果を見てから特徴を足さない)。
- pre-race 特徴のみ(会場/R番号/オッズ/イン勝率/勝率ばらつき)。
- train/test 分割(race_id昇順=ほぼ時系列で前後半)。両窓一貫のみ採用。
- 2週間規模は薄い→本確定は forward。定期再実行して芽の生死を追う。

実行: venv/Scripts/python.exe analysis/combo/screen_selection.py
(クラウド等 venv 無しの環境では: pip install psycopg2-binary numpy 後に
 python analysis/combo/screen_selection.py を analysis/combo ディレクトリ相当のパスで)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_races
from joint import proxy_joint
from market import implied_probs
from method import select_combos, CANDIDATE

WL = {1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24}  # P系ホワイトリスト(事前宣言)
LEDGER_START = "2026-06-30"


def collect(date_from=LEDGER_START, date_to="2026-12-31"):
    races = load_races(date_from, date_to)
    recs = []
    for r in races:
        m = proxy_joint(r)
        mk = implied_probs(r["odds"])
        po = r.get("payout_odds")
        v = r["race_data"]["venue_id"]
        bd = {b["boat_number"]: b for b in r["boats_data"]}
        wrs = [float(bd[i]["win_rate"]) for i in range(1, 7)
               if i in bd and bd[i]["win_rate"] is not None]
        inside = float(bd[1]["win_rate"]) if 1 in bd and bd[1]["win_rate"] is not None else None
        wr_std = float(np.std(wrs)) if len(wrs) >= 4 else None
        for c, mp, od in select_combos(r, m, mk, CANDIDATE):
            hit = (c == r["win_combo"])
            ret = (100 * po if (hit and po) else 0)
            recs.append(dict(rn=r["race_number"], v=v, od=od, ret=ret, hit=hit,
                             inside=inside, wr_std=wr_std))
    return recs


def _roi(sub):
    inv = 100 * len(sub)
    ret = sum(x["ret"] for x in sub)
    return (ret / inv * 100 if inv else 0.0), len(sub), sum(1 for x in sub if x["hit"])


def run():
    recs = collect()
    n = len(recs)
    mid = n // 2
    tr, te = recs[:mid], recs[mid:]
    im = np.median([x["inside"] for x in recs if x["inside"] is not None])
    sm = np.median([x["wr_std"] for x in recs if x["wr_std"] is not None])
    print(f"候補bet総数={n} (train {len(tr)}/test {len(te)}) inside中央={im:.2f} wr_std中央={sm:.2f}")
    print(f"候補全体ROI: train {_roi(tr)[0]:.0f}% / test {_roi(te)[0]:.0f}%")

    Hs = {
        "H1序盤R1-4": lambda x: x["rn"] <= 4,
        "H2得意場WL": lambda x: x["v"] in WL,
        "H3オッズ5-15": lambda x: 5 <= x["od"] <= 15,
        "H4緩い(内弱+横一線)": lambda x: (x["inside"] is not None and x["wr_std"] is not None
                                          and x["inside"] < im and x["wr_std"] < sm),
    }
    print(f"\n{'仮説':<18}{'train yes/no':>24}{'test yes/no':>24}{'判定':>10}")
    for name, f in Hs.items():
        ty = _roi([x for x in tr if f(x)]); tn = _roi([x for x in tr if not f(x)])
        ey = _roi([x for x in te if f(x)]); en = _roi([x for x in te if not f(x)])
        tr_up = ty[0] > tn[0]; te_up = ey[0] > en[0]
        if tr_up and te_up:
            verdict = "○一貫プラス"
        elif (not tr_up) and (not te_up):
            verdict = "×一貫マイナス"
        else:
            verdict = "△不一致(変動)"
        thin = "(薄)" if min(ty[1], ey[1]) < 60 else ""
        print(f"{name:<18} yes{ty[0]:>4.0f}%(n{ty[1]},{ty[2]}h)/no{tn[0]:>4.0f}%"
              f"   yes{ey[0]:>4.0f}%(n{ey[1]},{ey[2]}h)/no{en[0]:>4.0f}%   {verdict}{thin}")
    print("\n【読み方】○一貫プラス＝両窓で yes>no（芽）。ただし(薄)＝標本過少で信用不可。")
    print("両窓一貫かつ標本が厚い仮説だけが forward 確定の候補。薄い芽はデータ蓄積を待つ。")


if __name__ == "__main__":
    run()
