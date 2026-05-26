"""V11 (現行: QMC v3 + 1着 only specialist) vs V11.5 (QMC v4 + 全 position specialist)
forward simulate on 2026-04 (全月)

各 race で 2 つの戦略を実行し、軸分布 / picks 数 / 的中率 / ROI を比較。
1号艇軸偏重 (V11 = 88.9%) が V11.5 で 60% 近辺に落ち着くか確認。

include_venues = V11 functional 13、R1-R3 限定 (本番設定と同じ)
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import date
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import numpy as np
import lightgbm as lgb

from src.features import FeatureEngineer
from src.monte_carlo import qmc_sanrentan_v3, qmc_sanrentan_v4

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PRED_PATH = ROOT / 'analysis' / 'venue_v10_predictions.pkl'
SCALER_PATH = ROOT / 'models' / 'feature_scaler.pkl'
SPEC_76_DIR = ROOT / 'models' / 'specialists'
SPEC_82_DIR = ROOT / 'models' / 'specialists_82'
POOL_DIR = ROOT / 'models' / 'pool_models'
V11_CONFIG_PATH = ROOT / 'models' / 'v11_var13_config.json'
REPORT_PATH = ROOT / 'analysis' / 'reports' / '84_v11_vs_v115_v4_forward.md'

INCLUDE_VENUES = {1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24}
MAX_RACE = 3
TARGET_START = date(2026, 4, 1)
TARGET_END = date(2026, 4, 30)

# v11_var13 filter 設定 (config と一致)
MAX_ENTROPY = 2.6
MIN_PROBABILITY = 0.005
MIN_ODDS = 5.0
MAX_ODDS = 80.0
MIN_EV = 0.0
MAX_EV = 9999.0
ODDS_DISCOUNT_FACTOR = 0.92
KELLY_FRACTION = 0.0625
MAX_RECOMMENDED_BETS = 3
MIN_BET_AMOUNT = 100
BANKROLL = 200000
MAX_TICKET = int(BANKROLL * 0.008)  # ¥1,600
MAX_TOTAL = int(BANKROLL * 0.02)    # ¥4,000


def load_v11_config():
    import json
    with open(V11_CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)
    return {
        'functional_venues': set(cfg['functional_venues']),
        'venue_strategies': {int(k): v for k, v in cfg['venue_strategies'].items()},
        'venue_distances': {int(k): v for k, v in cfg['venue_distances'].items()},
    }


def load_specialists():
    import json
    spec_76 = {}
    spec_82 = {}
    spec_76_2nd = {}
    spec_76_3rd = {}
    pool_models = {}
    for vid in range(1, 25):
        p1 = SPEC_76_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        p2 = SPEC_76_DIR / f'lightgbm_v{vid:02d}_2nd.txt'
        p3 = SPEC_76_DIR / f'lightgbm_v{vid:02d}_3rd.txt'
        p82 = SPEC_82_DIR / f'lightgbm_v{vid:02d}_1st.txt'
        if p1.exists():
            spec_76[vid] = lgb.Booster(model_file=str(p1))
        if p2.exists():
            spec_76_2nd[vid] = lgb.Booster(model_file=str(p2))
        if p3.exists():
            spec_76_3rd[vid] = lgb.Booster(model_file=str(p3))
        if p82.exists():
            spec_82[vid] = lgb.Booster(model_file=str(p82))
    # pool_models は config の pool_id (日本語含む) を key として load
    with open(V11_CONFIG_PATH, encoding='utf-8') as f:
        full_cfg = json.load(f)
    for pool_id, info in full_cfg.get('pool_models', {}).items():
        path = ROOT / info['file']
        if path.exists():
            pool_models[pool_id] = lgb.Booster(model_file=str(path))
    return spec_76, spec_82, spec_76_2nd, spec_76_3rd, pool_models


def predict_v11_probs_1st(venue_id, x_76, x_82, v10_probs_1st,
                          venue_strategies, venue_distances,
                          spec_76, spec_82, pool_models):
    """V11 v11_var13_predictor のロジックを再現 (venue 別 best approach)"""
    if venue_id not in venue_strategies:
        return v10_probs_1st
    s = venue_strategies[venue_id]
    stype = s['type']

    if stype == 'specialist_76':
        m = spec_76.get(s['venue'])
        return m.predict(x_76, num_iteration=m.best_iteration)[0] if m else v10_probs_1st
    if stype == 'specialist_82':
        m = spec_82.get(s['venue'])
        return m.predict(x_82, num_iteration=m.best_iteration)[0] if m else v10_probs_1st
    if stype == 'pool':
        m = pool_models.get(s['pool_id'])
        return m.predict(x_76, num_iteration=m.best_iteration)[0] if m else v10_probs_1st
    if stype == 'recipe_v10_own':
        v10_w = s['v10_weight']
        own_w = s['own_weight']
        m = spec_76.get(venue_id)
        if not m:
            return v10_probs_1st
        own = m.predict(x_76, num_iteration=m.best_iteration)[0]
        return v10_w * v10_probs_1st + own_w * own
    if stype == 'recipe_top_K_sim':
        target = s['target']
        K = s['K']
        sim = [d['venue_id'] for d in venue_distances.get(target, [])[:K]]
        members = [target] + sim
        probs = np.zeros(6)
        n = 0
        for v in members:
            m = spec_76.get(v)
            if m:
                probs += m.predict(x_76, num_iteration=m.best_iteration)[0]
                n += 1
        return probs / n if n else v10_probs_1st
    if stype == 'recipe_75_sub':
        target = s['target']
        K = s['K']
        own_w = s['own_w']
        sub_alpha = s['sub_alpha']
        sim = [d['venue_id'] for d in venue_distances.get(target, [])[:K]]
        opp3 = sorted(venue_distances.get(target, []), key=lambda x: -x['distance'])[:3]
        opp_ids = [d['venue_id'] for d in opp3]
        probs = np.zeros(6)
        m = spec_76.get(target)
        if m:
            probs += own_w * m.predict(x_76, num_iteration=m.best_iteration)[0]
        for v in sim:
            m = spec_76.get(v)
            if m:
                probs += m.predict(x_76, num_iteration=m.best_iteration)[0]
        for v in opp_ids:
            m = spec_76.get(v)
            if m:
                probs += (-sub_alpha / len(opp_ids)) * m.predict(x_76, num_iteration=m.best_iteration)[0]
        probs = np.clip(probs, 0.001, None)
        s_sum = probs.sum()
        return probs / s_sum if s_sum > 0 else v10_probs_1st
    if stype == 'recipe_own_functional':
        target = s['target']
        own_w = s['own_w']
        others = s['functional_others']
        members_w = [(target, own_w)] + [(v, 1.0) for v in others]
        total_w = sum(w for _, w in members_w)
        probs = np.zeros(6)
        for v, w in members_w:
            m = spec_76.get(v)
            if m:
                probs += (w / total_w) * m.predict(x_76, num_iteration=m.best_iteration)[0]
        s_sum = probs.sum()
        return probs / s_sum if s_sum > 0 else v10_probs_1st
    return v10_probs_1st


def entropy(probs):
    p = np.array(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def synthesize_odds(actual, payout):
    """odds 推定 (簡易): 実払戻のみ既知、他組合せ odds は予測不能 → simulator では
    'odds 推定が必要な戦略' は的中時のみ計算、未的中時は picks の生 probability から
    market 推定する fallback。
    実際の forward では当時の odds.json を読むべきだが、ここでは payout で代替。
    """
    return None  # 未使用


def filter_and_kelly(sanrentan_probs, actual, payout):
    """v11_var13 filter (kelly_fraction=0.0625, max_entropy=2.6, EV>=0 等) を simulate

    odds はリアルタイム取得していないので、market 想定で fair odds (1/prob × overround 0.75) を使う。
    実際の forward 検証では odds.json が必要だが、ここでは picks 数の差を見るのが主目的。
    """
    # 各 combo の market odds 想定 = 1 / prob × overround 0.75 (粗い近似)
    # 払戻から actual の odds を逆算可能、その他は推定
    candidates = []
    for combo, prob in sanrentan_probs.items():
        if prob < MIN_PROBABILITY:
            continue
        # odds 推定: actual のみ既知、他は fair odds × 0.75 (boatrace の控除率 25%)
        if combo == actual and payout:
            raw_odds = payout / 100
        else:
            raw_odds = (1.0 / prob) * 0.75 if prob > 0 else 9999
        if raw_odds < MIN_ODDS or raw_odds > MAX_ODDS:
            continue
        discounted = raw_odds * ODDS_DISCOUNT_FACTOR
        ev = prob * discounted
        if ev < MIN_EV or ev > MAX_EV:
            continue
        # kelly
        if discounted <= 1.0:
            continue
        k = (prob * discounted - 1) / (discounted - 1)
        if k <= 0:
            continue
        amt = BANKROLL * k * KELLY_FRACTION
        amt = max(MIN_BET_AMOUNT, min(MAX_TICKET, amt))
        amt = int(round(amt / 100) * 100)
        if amt < MIN_BET_AMOUNT:
            continue
        candidates.append({
            'combo': combo, 'amount': amt, 'odds': raw_odds,
            'probability': prob, 'ev': ev,
        })
    # entropy filter (top1 軸)
    # max_entropy 適用は probs_1st 単位だが、ここは sanrentan 全体 → スキップ (元ロジックに依存)
    candidates.sort(key=lambda x: -x['ev'])
    candidates = candidates[:MAX_RECOMMENDED_BETS]
    total = sum(c['amount'] for c in candidates)
    if total > MAX_TOTAL and candidates:
        ratio = MAX_TOTAL / total
        for c in candidates:
            c['amount'] = int(round(c['amount'] * ratio / 100) * 100)
        candidates = [c for c in candidates if c['amount'] >= MIN_BET_AMOUNT]
    return candidates


def axis_dist(picks_list):
    cnt = Counter()
    for picks in picks_list:
        for c in picks:
            cnt[c['combo'].split('-')[0]] += 1
    total = sum(cnt.values())
    return {b: c/total for b, c in cnt.items()} if total else {}


def simulate(records, mode, v11_cfg, specs):
    spec_76, spec_82, spec_76_2nd, spec_76_3rd, pool_models = specs
    fe, scaler = FeatureEngineer(), pickle.load(open(SCALER_PATH, 'rb'))
    all_picks = []
    hit_races = 0
    total_inv = 0
    total_ret = 0
    races_with_picks = 0

    for r in records:
        p = r['prediction']
        v = r['venue_id']
        rn = p['race_number']
        if v not in INCLUDE_VENUES or rn > MAX_RACE:
            continue
        if r['date'] < TARGET_START or r['date'] > TARGET_END:
            continue
        try:
            features = fe.transform(p['race_data'], p['boats'])
            features = scaler.transform(features.reshape(1, -1)).flatten()
            local_adv = np.array([
                ((b.get('local_win_rate_2') or 0) - (b.get('win_rate_2') or 0)) / 100.0
                for b in p['boats']
            ], dtype=np.float32)
            features_82 = np.concatenate([features, local_adv])
            x_76 = features.reshape(1, -1)
            x_82 = features_82.reshape(1, -1)
        except Exception:
            continue

        v10_probs_1st = np.array(p['probs_1st'])

        # 1着 prob: V11 specialist 群 (両モード共通)
        probs_1st = predict_v11_probs_1st(
            v, x_76, x_82, v10_probs_1st,
            v11_cfg['venue_strategies'], v11_cfg['venue_distances'],
            spec_76, spec_82, pool_models,
        )

        # 2着3着 prob と sanrentan 計算
        if mode == 'v11':
            # 現行: V10 baseline + QMC v3
            sanrentan = qmc_sanrentan_v3(
                list(probs_1st), boats_data=p['boats'],
                race_data=p['race_data'], race_number=rn,
                n_simulations=8192, seed=42,
            )
        elif mode == 'v115_v4':
            # V11.5 specialist 2着/3着 + QMC v4
            m2 = spec_76_2nd.get(v)
            m3 = spec_76_3rd.get(v)
            probs_2nd = m2.predict(x_76, num_iteration=m2.best_iteration)[0] if m2 else np.array(p['probs_2nd'])
            probs_3rd = m3.predict(x_76, num_iteration=m3.best_iteration)[0] if m3 else np.array(p['probs_3rd'])
            sanrentan = qmc_sanrentan_v4(
                list(probs_1st), list(probs_2nd), list(probs_3rd),
                boats_data=p['boats'],
                race_data=p['race_data'], race_number=rn,
                n_simulations=8192, seed=42,
            )
        else:
            raise ValueError(mode)

        picks = filter_and_kelly(sanrentan, p['actual'], p['payout'])
        if not picks:
            continue
        races_with_picks += 1
        all_picks.append(picks)
        race_inv = sum(c['amount'] for c in picks)
        total_inv += race_inv
        race_ret = 0
        for c in picks:
            if c['combo'] == p['actual'] and p['payout']:
                race_ret += c['amount'] * p['payout'] / 100
        if race_ret > 0:
            hit_races += 1
            total_ret += race_ret

    axis = axis_dist(all_picks)
    roi = (total_ret - total_inv) / total_inv * 100 if total_inv else 0
    return {
        'races_with_picks': races_with_picks,
        'hit_races': hit_races,
        'hit_rate': hit_races / races_with_picks if races_with_picks else 0,
        'total_inv': total_inv,
        'total_ret': total_ret,
        'pnl': total_ret - total_inv,
        'roi': roi,
        'axis_dist': axis,
        'total_picks': sum(len(p) for p in all_picks),
    }


def main():
    logger.info("V11 vs V11.5 (QMC v4) forward simulate on 2026-04")
    venue_preds = pickle.load(open(PRED_PATH, 'rb'))
    v11_cfg = load_v11_config()
    specs = load_specialists()
    logger.info(f"Loaded: 76dim={len(specs[0])}, 82dim={len(specs[1])}, 2着={len(specs[2])}, 3着={len(specs[3])}, pool={len(specs[4])}")

    # Records build (in_scope venues only for speed)
    records = []
    for vid, preds in venue_preds.items():
        if vid not in INCLUDE_VENUES:
            continue
        for rid, p in preds.items():
            try:
                d = date.fromisoformat(p['race_date'])
                if d < TARGET_START or d > TARGET_END:
                    continue
                records.append({'venue_id': vid, 'date': d, 'prediction': p})
            except Exception:
                continue
    logger.info(f"target records: {len(records)} (R1-R3 + include 13 venues, 2026-04)")

    # V11 (現行)
    logger.info("Simulating V11 (QMC v3, 1着 only specialist)...")
    r_v11 = simulate(records, 'v11', v11_cfg, specs)

    # V11.5 + v4
    logger.info("Simulating V11.5 (QMC v4, 全 position specialist)...")
    r_v115 = simulate(records, 'v115_v4', v11_cfg, specs)

    # Report
    lines = []
    lines.append("# V11 vs V11.5 (QMC v4) forward simulate on 2026-04\n\n")
    lines.append("**注意**: odds は payout から推定 (的中時のみ実 odds、他は fair odds × 0.75 近似)。\n")
    lines.append("実際の運用 odds データではないため ROI 数値は参考程度。\n")
    lines.append("**主目的: 軸分布 (1号艇軸比率) と picks 数の差を確認**\n\n")

    lines.append("## サマリ\n\n")
    lines.append("| 項目 | V11 (現行: QMC v3 + 1着 only) | V11.5 (新: QMC v4 + 全 position) | 差 |\n|---|---|---|---|\n")
    lines.append(f"| races with picks | {r_v11['races_with_picks']} | {r_v115['races_with_picks']} | {r_v115['races_with_picks']-r_v11['races_with_picks']:+d} |\n")
    lines.append(f"| total picks | {r_v11['total_picks']} | {r_v115['total_picks']} | {r_v115['total_picks']-r_v11['total_picks']:+d} |\n")
    lines.append(f"| hit races | {r_v11['hit_races']} | {r_v115['hit_races']} | {r_v115['hit_races']-r_v11['hit_races']:+d} |\n")
    lines.append(f"| hit_rate | {r_v11['hit_rate']*100:.2f}% | {r_v115['hit_rate']*100:.2f}% | {(r_v115['hit_rate']-r_v11['hit_rate'])*100:+.2f}pt |\n")
    lines.append(f"| 投資 (推定) | ¥{r_v11['total_inv']:,} | ¥{r_v115['total_inv']:,} | ¥{r_v115['total_inv']-r_v11['total_inv']:+,} |\n")
    lines.append(f"| 払戻 (推定) | ¥{int(r_v11['total_ret']):,} | ¥{int(r_v115['total_ret']):,} | ¥{int(r_v115['total_ret']-r_v11['total_ret']):+,} |\n")
    lines.append(f"| 損益 (推定) | ¥{int(r_v11['pnl']):,} | ¥{int(r_v115['pnl']):,} | ¥{int(r_v115['pnl']-r_v11['pnl']):+,} |\n")
    lines.append(f"| ROI (推定) | {r_v11['roi']:+.1f}% | {r_v115['roi']:+.1f}% | {r_v115['roi']-r_v11['roi']:+.1f}pt |\n")

    lines.append("\n## 軸分布 (1着軸の boat 別比率)\n\n")
    lines.append("| boat | V11 軸比率 | V11.5 軸比率 | 差 |\n|---|---|---|---|\n")
    all_boats = sorted(set(r_v11['axis_dist'].keys()) | set(r_v115['axis_dist'].keys()))
    for b in all_boats:
        v1 = r_v11['axis_dist'].get(b, 0) * 100
        v2 = r_v115['axis_dist'].get(b, 0) * 100
        lines.append(f"| {b}号艇 | {v1:.1f}% | {v2:.1f}% | {v2-v1:+.1f}pt |\n")

    lines.append(f"\n## 評価\n\n")
    v11_1axis = r_v11['axis_dist'].get('1', 0) * 100
    v115_1axis = r_v115['axis_dist'].get('1', 0) * 100
    lines.append(f"- V11 の 1号艇軸比率: **{v11_1axis:.1f}%** (本番 88.9% に近いか?)\n")
    lines.append(f"- V11.5 の 1号艇軸比率: **{v115_1axis:.1f}%** (実 1着率 59% 近辺か?)\n")
    lines.append(f"- 1号艇軸偏重の改善: **{v115_1axis - v11_1axis:+.1f}pt**\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(''.join(lines), encoding='utf-8')
    logger.info(f"Report saved: {REPORT_PATH}")
    logger.info(f"\nV11 1軸比率: {v11_1axis:.1f}%, V11.5 1軸比率: {v115_1axis:.1f}%")
    logger.info(f"V11 hit: {r_v11['hit_rate']*100:.2f}%, V11.5 hit: {r_v115['hit_rate']*100:.2f}%")


if __name__ == '__main__':
    main()
