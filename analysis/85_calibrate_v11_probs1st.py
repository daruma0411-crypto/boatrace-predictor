"""V11 probs_1st calibration: boat 別 Isotonic Regression

V11 specialist が boat 1 を過大評価 (本番今日 0.59 vs 実 1着率 50%) する問題を
isotonic で補正。boat 別 6 個の calibrator を fit。

入力 (train):
  2026-04 forward 全 races の V11 probs_1st 予測 vs 実 1着結果 (boat 別 one-hot)

出力:
  models/calibrators_v11/boat_{1..6}.pkl  (sklearn IsotonicRegression)
  models/calibrators_v11/meta.json        (作成日、train sample 数)
  analysis/reports/85_calibration_eval.md (calibration plot / 効果検証)
"""
import os
import sys
import pickle
import json
import logging
from pathlib import Path
from datetime import date
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
SPEC_76_DIR = ROOT / 'models' / 'specialists'
SPEC_82_DIR = ROOT / 'models' / 'specialists_82'
POOL_DIR = ROOT / 'models' / 'pool_models'
V11_CONFIG_PATH = ROOT / 'models' / 'v11_var13_config.json'
CALIBRATOR_DIR = ROOT / 'models' / 'calibrators_v11'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '85_calibration_eval.md'

INCLUDE_VENUES = {1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24}
TRAIN_START = date(2026, 4, 1)
TRAIN_END = date(2026, 4, 30)


def load_v11_config():
    with open(V11_CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


def load_specs():
    spec_76, spec_82, pool_models = {}, {}, {}
    for vid in range(1, 25):
        p1 = SPEC_76_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        p82 = SPEC_82_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        if p1.exists():
            spec_76[vid] = lgb.Booster(model_file=str(p1))
        if p82.exists():
            spec_82[vid] = lgb.Booster(model_file=str(p82))
    cfg = load_v11_config()
    for pool_id, info in cfg.get('pool_models', {}).items():
        path = ROOT / info['file']
        if path.exists():
            pool_models[pool_id] = lgb.Booster(model_file=str(path))
    return spec_76, spec_82, pool_models


def predict_v11_probs1st(venue_id, x_76, x_82, v10, cfg, spec_76, spec_82, pool_models):
    venue_strategies = {int(k): v for k, v in cfg['venue_strategies'].items()}
    venue_distances = {int(k): v for k, v in cfg['venue_distances'].items()}
    if venue_id not in venue_strategies:
        return v10
    s = venue_strategies[venue_id]
    t = s['type']
    if t == 'specialist_76':
        m = spec_76.get(s['venue'])
        return m.predict(x_76, num_iteration=m.best_iteration)[0] if m else v10
    if t == 'specialist_82':
        m = spec_82.get(s['venue'])
        return m.predict(x_82, num_iteration=m.best_iteration)[0] if m else v10
    if t == 'pool':
        m = pool_models.get(s['pool_id'])
        return m.predict(x_76, num_iteration=m.best_iteration)[0] if m else v10
    if t == 'recipe_v10_own':
        m = spec_76.get(venue_id)
        if not m:
            return v10
        own = m.predict(x_76, num_iteration=m.best_iteration)[0]
        return s['v10_weight'] * v10 + s['own_weight'] * own
    if t == 'recipe_top_K_sim':
        target = s['target']
        sim = [d['venue_id'] for d in venue_distances.get(target, [])[:s['K']]]
        members = [target] + sim
        probs = np.zeros(6)
        n = 0
        for v in members:
            m = spec_76.get(v)
            if m:
                probs += m.predict(x_76, num_iteration=m.best_iteration)[0]
                n += 1
        return probs / n if n else v10
    if t == 'recipe_75_sub':
        target = s['target']
        sim = [d['venue_id'] for d in venue_distances.get(target, [])[:s['K']]]
        opp3 = sorted(venue_distances.get(target, []), key=lambda x: -x['distance'])[:3]
        opp_ids = [d['venue_id'] for d in opp3]
        probs = np.zeros(6)
        m = spec_76.get(target)
        if m:
            probs += s['own_w'] * m.predict(x_76, num_iteration=m.best_iteration)[0]
        for v in sim:
            m = spec_76.get(v)
            if m:
                probs += m.predict(x_76, num_iteration=m.best_iteration)[0]
        for v in opp_ids:
            m = spec_76.get(v)
            if m:
                probs += (-s['sub_alpha'] / len(opp_ids)) * m.predict(x_76, num_iteration=m.best_iteration)[0]
        probs = np.clip(probs, 0.001, None)
        return probs / probs.sum()
    if t == 'recipe_own_functional':
        members_w = [(s['target'], s['own_w'])] + [(v, 1.0) for v in s['functional_others']]
        total_w = sum(w for _, w in members_w)
        probs = np.zeros(6)
        for v, w in members_w:
            m = spec_76.get(v)
            if m:
                probs += (w / total_w) * m.predict(x_76, num_iteration=m.best_iteration)[0]
        return probs / probs.sum() if probs.sum() > 0 else v10
    return v10


def main():
    logger.info("V11 probs_1st calibration on 2026-04 (all R + 13 venues)")
    venue_preds = pickle.load(open(PRED_PATH, 'rb'))
    cfg = load_v11_config()
    spec_76, spec_82, pool_models = load_specs()
    fe = FeatureEngineer()
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

    # Collect (predicted_prob_per_boat, actual_winner_one_hot) pairs
    boat_preds = [[] for _ in range(6)]
    boat_actuals = [[] for _ in range(6)]
    skipped = 0
    used_venues = Counter()
    for vid, preds in venue_preds.items():
        if vid not in INCLUDE_VENUES:
            continue
        for rid, p in preds.items():
            try:
                d = date.fromisoformat(p['race_date'])
                if d < TRAIN_START or d > TRAIN_END:
                    continue
                if p.get('result_1st') is None:
                    continue
                features = fe.transform(p['race_data'], p['boats'])
                features = scaler.transform(features.reshape(1, -1)).flatten()
                local_adv = np.array([
                    ((b.get('local_win_rate_2') or 0) - (b.get('win_rate_2') or 0)) / 100.0
                    for b in p['boats']
                ], dtype=np.float32)
                features_82 = np.concatenate([features, local_adv])
                x_76 = features.reshape(1, -1)
                x_82 = features_82.reshape(1, -1)
                v10 = np.array(p['probs_1st'])
                probs = predict_v11_probs1st(vid, x_76, x_82, v10, cfg, spec_76, spec_82, pool_models)
                actual = p['result_1st'] - 1  # 0-indexed
                for b in range(6):
                    boat_preds[b].append(probs[b])
                    boat_actuals[b].append(1 if b == actual else 0)
                used_venues[vid] += 1
            except Exception as e:
                skipped += 1
                continue

    n = len(boat_preds[0])
    logger.info(f"Collected {n} races, skipped {skipped}, used venues: {dict(used_venues)}")
    if n < 500:
        logger.error(f"Sample too small ({n} < 500), aborting")
        return

    # 各 boat 用 IsotonicRegression を fit
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
    calibrators = []
    for b in range(6):
        x = np.array(boat_preds[b], dtype=np.float64)
        y = np.array(boat_actuals[b], dtype=np.float64)
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        ir.fit(x, y)
        path = CALIBRATOR_DIR / f'boat_{b+1}.pkl'
        pickle.dump(ir, open(path, 'wb'))
        calibrators.append(ir)
        logger.info(f"boat {b+1}: n={n}, mean_pred={x.mean():.3f}, mean_actual={y.mean():.3f}, calibrator saved {path.name}")

    # Meta
    meta = {
        'created': '2026-05-26',
        'train_period': f'{TRAIN_START} to {TRAIN_END}',
        'n_samples': int(n),
        'venues': sorted(used_venues.keys()),
        'notes': 'Isotonic Regression per boat (all venues pooled). Use after V11 specialist predict, then renormalize.',
    }
    (CALIBRATOR_DIR / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    # Calibration plot 風レポート (binned reliability)
    def reliability_bins(p, a, bins=10):
        x = np.array(p)
        y = np.array(a)
        bin_edges = np.linspace(0, 1, bins + 1)
        ids = np.digitize(x, bin_edges) - 1
        ids = np.clip(ids, 0, bins - 1)
        out = []
        for i in range(bins):
            mask = ids == i
            if mask.sum() > 0:
                out.append({
                    'bin': f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}',
                    'n': int(mask.sum()),
                    'mean_pred': float(x[mask].mean()),
                    'mean_actual': float(y[mask].mean()),
                })
        return out

    lines = []
    lines.append("# V11 probs_1st calibration evaluation\n\n")
    lines.append(f"Train: 2026-04 (n={n} races), 13 functional venues, R1-R12 全 race\n\n")
    lines.append("## boat 別 calibration 前後\n\n")
    lines.append("| boat | mean_pred (calib 前) | mean_actual | bias | mean_pred (calib 後) |\n|---|---|---|---|---|\n")
    for b in range(6):
        x = np.array(boat_preds[b])
        y = np.array(boat_actuals[b])
        x_cal = calibrators[b].predict(x)
        lines.append(f"| {b+1}号艇 | {x.mean():.3f} | {y.mean():.3f} | "
                     f"{x.mean()-y.mean():+.3f} | {x_cal.mean():.3f} |\n")

    lines.append("\n## boat 1 reliability (calib 前 binned)\n\n")
    lines.append("| pred bin | n | mean_pred | mean_actual | bias |\n|---|---|---|---|---|\n")
    for r in reliability_bins(boat_preds[0], boat_actuals[0]):
        lines.append(f"| {r['bin']} | {r['n']} | {r['mean_pred']:.3f} | {r['mean_actual']:.3f} | "
                     f"{r['mean_pred']-r['mean_actual']:+.3f} |\n")

    lines.append("\n## 期待効果\n\n")
    lines.append("- boat 1 mean_pred 過大評価分が calibrator で補正される\n")
    lines.append("- predictor 適用後は 1着軸 picks 比率 88% → 60% 近辺に低下見込み\n")
    lines.append("- 検証は 2026-05 hold-out で軸分布 + ROI 観察 (analysis/86)\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(''.join(lines), encoding='utf-8')
    logger.info(f"Report: {REPORT_PATH}")
    logger.info("Calibrators saved to models/calibrators_v11/")


if __name__ == '__main__':
    main()
