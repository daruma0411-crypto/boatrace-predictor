"""soft_regime の train(〜5/15) 学習を1回実行し、凍結定数を src/softness_params.py に書き出す。
実行: venv/Scripts/python.exe scripts/generate_softness_params.py  (DB read-only)"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "soft_regime"))
from db import connect, SPLIT_DATE, ARE_TRAIN_START
from features import race_features, FEATURE_NAMES
from stage_a_softscore import _payout_map, _matrix

conn = connect()
feats = race_features(conn, ARE_TRAIN_START, "2026-12-31")
payout = _payout_map(conn, ARE_TRAIN_START, "2026-12-31")
ids = [r for r in feats if r in payout]
train_ids = [r for r in ids if str(feats[r]["race_date"]) < SPLIT_DATE]
Xtr, ytr = _matrix(feats, payout, train_ids)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
w, *_ = np.linalg.lstsq(np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]), ytr, rcond=None)
tr_sc = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]) @ w
hard_cut = float(np.percentile(tr_sc, 33.333))
# 温度閾値: 全日の日次平均→前日まで7日平均→train中央値 (analysis/soft_regime/p_kai_track.py build_model と同一ロジック)
from collections import defaultdict
allids = ids
Xall, _ = _matrix(feats, payout, allids)
sca = np.hstack([(Xall-mu)/sd, np.ones((len(Xall),1))]) @ w
day_scores = defaultdict(list)
for i, rid in enumerate(allids):
    day_scores[str(feats[rid]["race_date"])].append(float(sca[i]))
day_mean = {d: float(np.mean(v)) for d, v in day_scores.items()}
days_sorted = sorted(day_mean)
temp = {}
for i, d in enumerate(days_sorted):
    prev = [day_mean[x] for x in days_sorted[max(0, i-7):i]]
    if prev: temp[d] = float(np.mean(prev))
temp_thr = float(np.median([temp[d] for d in days_sorted if d < SPLIT_DATE and d in temp]))

out = os.path.join(ROOT, "src", "softness_params.py")
with open(out, "w", encoding="utf-8") as f:
    f.write('"""自動生成 (scripts/generate_softness_params.py, 2026-07-30 凍結)。手編集禁止。\n'
            'P改の定義変更は別名の新定義として登録すること。"""\n')
    f.write(f"FEATURE_NAMES = {list(FEATURE_NAMES)!r}\n")
    f.write(f"MU = {[float(x) for x in mu]!r}\n")
    f.write(f"SD = {[float(x) for x in sd]!r}\n")
    f.write(f"W = {[float(x) for x in w]!r}\n")
    f.write(f"HARD_CUT = {hard_cut!r}\n")
    f.write(f"TEMP_THR = {temp_thr!r}\n")
    f.write("COLD_SCALE = 0.5\nTEMP_WINDOW = 7\n")
print("wrote", out, "hard_cut=", hard_cut, "temp_thr=", temp_thr)
