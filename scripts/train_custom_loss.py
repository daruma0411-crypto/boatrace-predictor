"""LightGBM カスタム損失関数による利益最適化実験

目的:
  現行PyTorchモデル (Focal Loss) は的中率最適化 → ROI=-53.7%
  オッズ加重学習で「利益」を直接最適化する実験

3モデル比較:
  1. LightGBM標準 (multi_logloss) — ベースライン
  2. LightGBM Profit-Weighted — カスタム目的関数
  3. 現行PyTorch (既存モデル読み込み) — 参考値

学習後シミュレーション:
  Sim1: 確信度閾値スイープ — 閾値0.20〜0.80で最適ベット条件を探索
  Sim2: 日別累積収支 — 最適閾値で日別P&Lを追跡、最大ドローダウン算出

カスタムLoss:
  weight_i = log(1 + payout_i / 1000)
  grad_{i,k} = w_i * (p_{i,k} - 1{k=y_i})
  hess_{i,k} = max(w_i * p_{i,k} * (1 - p_{i,k}), 1e-6)
"""
import sys
import os
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features import FeatureEngineerLegacy as FeatureEngineer
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _get_feature_names_208():
    """FeatureEngineerLegacy (208次元) の特徴量名を自動生成"""
    wind_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'calm']
    names = [
        'G_venue_id', 'G_month', 'G_distance', 'G_wind_speed',
    ]
    names.extend([f'G_wind_{d}' for d in wind_dirs])
    names.extend(['G_temperature', 'G_wave_height', 'G_water_temperature'])

    player_classes = ['A1', 'A2', 'B1', 'B2']
    for b in range(1, 7):
        p = f'B{b}'
        names.extend([f'{p}_class_{c}' for c in player_classes])
        names.extend([
            f'{p}_win_rate_rel', f'{p}_win_rate_2', f'{p}_win_rate_3',
            f'{p}_local_win_rate', f'{p}_local_win_rate_2',
            f'{p}_avg_st', f'{p}_inner_st_diff',
            f'{p}_motor_win_rate_2', f'{p}_motor_win_rate_3',
            f'{p}_boat_win_rate_2', f'{p}_is_new_motor',
            f'{p}_weight_diff', f'{p}_exhibition_time_diff',
        ])
        names.extend([f'{p}_approach_{j}' for j in range(1, 7)])
        names.append(f'{p}_fallback_flag')
        names.extend([f'{p}_boat_num_{j}' for j in range(1, 7)])
        names.extend([f'{p}_tilt', f'{p}_parts_changed'])

    assert len(names) == 208, f"Expected 208, got {len(names)}"
    return names


# ============================================================
# データロード
# ============================================================

def load_training_data_with_payout(years=3):
    """過去N年分のレースデータをDB一括取得 → 特徴量 + payout + 日付

    Returns:
        X: 特徴量 (N, 43)
        y1: 1着ラベル (N,)
        payouts: 三連単配当 (N,)
        dates: レース日付 (N,) — datetime.date の配列
    """
    feature_engineer = FeatureEngineer()
    cutoff_date = datetime.now() - timedelta(days=365 * years)

    logger.info("データ一括取得中 (payout付き)...")

    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.payout_sanrentan,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.race_date >= %s AND r.status = 'finished'
              AND r.result_1st IS NOT NULL
              AND r.payout_sanrentan IS NOT NULL
              AND r.payout_sanrentan > 0
              AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date
        """, (cutoff_date.date(),))
        races = cur.fetchall()
        logger.info(f"レース取得 (payout有り): {len(races):,}件")

        if not races:
            logger.warning("payout_sanrentan のあるレースが0件です")
            return None, None, None, None

        race_ids = [r['id'] for r in races]

        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor,
                   tilt, parts_changed
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()
        logger.info(f"ボート取得: {len(all_boats):,}件")

    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    logger.info("特徴量生成中...")
    X_list = []
    y1_list = []
    payout_list = []
    date_list = []

    for race in races:
        boats = boats_by_race.get(race['id'], [])
        if len(boats) != 6:
            continue

        race_data = {
            'venue_id': race['venue_id'],
            'month': race['race_date'].month,
            'distance': 1800,
            'wind_speed': race.get('wind_speed') or 0,
            'wind_direction': race.get('wind_direction') or 'calm',
            'temperature': race.get('temperature') or 20,
            'wave_height': race.get('wave_height') or 0,
            'water_temperature': race.get('water_temperature') or 20,
        }

        try:
            features = feature_engineer.transform(race_data, boats)
            X_list.append(features)
            y1_list.append(race['result_1st'] - 1)
            payout_list.append(race['payout_sanrentan'])
            date_list.append(race['race_date'])
        except Exception:
            continue

    if not X_list:
        logger.warning("訓練データが0件です")
        return None, None, None, None

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int64)
    payouts = np.array(payout_list, dtype=np.float64)
    dates = np.array(date_list)

    logger.info(f"訓練データ: {len(X):,}件, 特徴量次元: {X.shape[1]}")
    logger.info(f"配当統計: 中央値={np.median(payouts):.0f}円, "
                f"平均={np.mean(payouts):.0f}円, "
                f"最大={np.max(payouts):.0f}円")

    counts = Counter(y1.tolist())
    dist = " ".join(f"{i+1}号艇:{counts.get(i,0)/len(y1)*100:.1f}%"
                    for i in range(6))
    logger.info(f"1着分布: {dist}")

    return X, y1, payouts, dates


# ============================================================
# カスタム目的関数 / 評価関数
# ============================================================

def profit_weighted_objective(y_true, y_pred_raw, weights, num_class=6):
    """Profit-Weighted Softmax Cross-Entropy のカスタム目的関数

    LightGBM multiclass: y_pred_raw は列優先 (Fortran order) で
    [s0_c0, s1_c0, ..., sN_c0, s0_c1, s1_c1, ..., sN_c1, ...]
    reshape/ravel も order='F' を指定する必要がある。
    """
    n = len(y_true)
    y_pred = y_pred_raw.reshape(n, num_class, order='F')

    y_pred_max = y_pred.max(axis=1, keepdims=True)
    exp_pred = np.exp(y_pred - y_pred_max)
    probs = exp_pred / exp_pred.sum(axis=1, keepdims=True)

    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y_true.astype(int)] = 1.0

    # 重みを正規化して平均1.0にする（スケール安定化）
    w_norm = weights / weights.mean()

    grad = w_norm[:, np.newaxis] * (probs - y_onehot)
    hess = w_norm[:, np.newaxis] * probs * (1.0 - probs)
    hess = np.maximum(hess, 1e-6)

    return grad.ravel(order='F'), hess.ravel(order='F')


def profit_weighted_eval(y_true, y_pred_raw, weights, num_class=6):
    """Profit-Weighted CE のカスタム評価関数"""
    n = len(y_true)
    y_pred = y_pred_raw.reshape(n, num_class, order='F')

    y_pred_max = y_pred.max(axis=1, keepdims=True)
    exp_pred = np.exp(y_pred - y_pred_max)
    probs = exp_pred / exp_pred.sum(axis=1, keepdims=True)

    correct_probs = probs[np.arange(n), y_true.astype(int)]
    loss = -weights * np.log(np.maximum(correct_probs, 1e-15))

    return 'profit_weighted_ce', np.mean(loss), False


# ============================================================
# 評価ユーティリティ
# ============================================================

def evaluate_model_predictions(y_true, probs, payouts, model_name):
    """予測結果を評価して辞書で返す"""
    preds = probs.argmax(axis=1)
    accuracy = (preds == y_true).mean() * 100

    pred_counts = Counter(preds.tolist())
    pred_dist = {i: pred_counts.get(i, 0) / len(preds) * 100 for i in range(6)}

    bet_amount = 100
    total_bet = len(y_true) * bet_amount

    correct_mask = (preds == y_true)
    total_payout_if_correct = payouts[correct_mask].sum()
    roi_1st_hit = (total_payout_if_correct / total_bet - 1) * 100 if total_bet > 0 else 0

    correct_probs = probs[np.arange(len(y_true)), y_true]
    expected_payout = (correct_probs * payouts).sum()
    roi_expected = (expected_payout / total_bet - 1) * 100 if total_bet > 0 else 0

    high_payout_mask = payouts >= 10000
    if high_payout_mask.sum() > 0:
        high_acc = (preds[high_payout_mask] == y_true[high_payout_mask]).mean() * 100
        high_count = int(high_payout_mask.sum())
    else:
        high_acc, high_count = 0.0, 0

    low_payout_mask = payouts < 3000
    if low_payout_mask.sum() > 0:
        low_acc = (preds[low_payout_mask] == y_true[low_payout_mask]).mean() * 100
        low_count = int(low_payout_mask.sum())
    else:
        low_acc, low_count = 0.0, 0

    return {
        'model': model_name,
        'accuracy': accuracy,
        'roi_1st_hit': roi_1st_hit,
        'roi_expected': roi_expected,
        'pred_dist': pred_dist,
        'high_payout_acc': high_acc,
        'high_payout_count': high_count,
        'low_payout_acc': low_acc,
        'low_payout_count': low_count,
    }


# ============================================================
# Sim1: 確信度閾値スイープ
# ============================================================

def simulate_confidence_sweep(models_probs, y_val, p_val):
    """各モデルの確信度閾値を0.20〜0.80でスイープし、最適ROI閾値を探索

    Args:
        models_probs: [(model_name, probs_array), ...]
        y_val: 正解ラベル
        p_val: 三連単配当

    Returns:
        best_thresholds: {model_name: (best_threshold, best_roi, bet_count, hit_rate)}
    """
    print("\n")
    print("=" * 90)
    print("Sim1: 確信度閾値スイープ (閾値別ROI)")
    print("  確信度が閾値以上のレースのみベット → ROIが最大になる閾値を探索")
    print("=" * 90)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    bet_unit = 100  # 1レース100円

    best_thresholds = {}

    for model_name, probs in models_probs:
        print(f"\n--- {model_name} ---")
        print(f"  {'閾値':>6}  {'ベット数':>8}  {'的中数':>6}  {'的中率':>7}  "
              f"{'投資額':>10}  {'回収額':>10}  {'ROI':>8}  {'平均配当(的中)':>14}")
        print(f"  {'-'*84}")

        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        best_roi = -999
        best_thresh = 0.0

        for thresh in thresholds:
            mask = confidences >= thresh
            n_bets = mask.sum()

            if n_bets == 0:
                print(f"  {thresh:>5.2f}  {'---':>8}  {'---':>6}  {'---':>7}  "
                      f"{'---':>10}  {'---':>10}  {'---':>8}  {'---':>14}")
                continue

            bet_preds = preds[mask]
            bet_y = y_val[mask]
            bet_payouts = p_val[mask]

            hits = (bet_preds == bet_y)
            n_hits = hits.sum()
            hit_rate = n_hits / n_bets * 100

            total_invested = n_bets * bet_unit
            total_return = bet_payouts[hits].sum()
            roi = (total_return / total_invested - 1) * 100

            avg_payout_hit = bet_payouts[hits].mean() if n_hits > 0 else 0

            marker = ""
            if roi > best_roi and n_bets >= 20:
                best_roi = roi
                best_thresh = thresh
                marker = " <-- best"

            print(f"  {thresh:>5.2f}  {n_bets:>8,}  {n_hits:>6,}  {hit_rate:>6.1f}%  "
                  f"{total_invested:>9,}円  {total_return:>9,.0f}円  {roi:>+7.1f}%  "
                  f"{avg_payout_hit:>13,.0f}円{marker}")

        if best_roi > -999:
            # 最適閾値でのベット数と的中率を再計算
            mask = confidences >= best_thresh
            hits = (preds[mask] == y_val[mask])
            best_thresholds[model_name] = (
                best_thresh,
                best_roi,
                int(mask.sum()),
                float(hits.mean() * 100),
            )

    # サマリー
    print(f"\n{'='*60}")
    print("Sim1 サマリー: 各モデルの最適閾値")
    print(f"{'='*60}")
    print(f"  {'モデル':<28}  {'閾値':>5}  {'ROI':>8}  {'ベット数':>8}  {'的中率':>7}")
    print(f"  {'-'*60}")
    for model_name, (thresh, roi, n_bets, hit_rate) in best_thresholds.items():
        print(f"  {model_name:<28}  {thresh:>5.2f}  {roi:>+7.1f}%  {n_bets:>7,}件  {hit_rate:>6.1f}%")
    print(f"{'='*60}")

    return best_thresholds


# ============================================================
# Sim2: 日別累積収支シミュレーション
# ============================================================

def simulate_daily_pnl(models_probs, y_val, p_val, dates_val, best_thresholds):
    """最適閾値を使って日別の累積収支を追跡

    Args:
        models_probs: [(model_name, probs_array), ...]
        y_val: 正解ラベル
        p_val: 三連単配当
        dates_val: レース日付
        best_thresholds: Sim1で得た {model_name: (thresh, roi, n_bets, hit_rate)}
    """
    print("\n")
    print("=" * 90)
    print("Sim2: 日別累積収支シミュレーション")
    print("  Sim1で見つけた最適閾値を使い、日別P&L・最大ドローダウン・連敗を追跡")
    print("=" * 90)

    bet_unit = 100
    initial_bankroll = 200_000  # 初期資金20万円

    for model_name, probs in models_probs:
        if model_name not in best_thresholds:
            continue

        thresh, _, _, _ = best_thresholds[model_name]

        print(f"\n{'─'*70}")
        print(f"  {model_name}  (閾値={thresh:.2f}, 初期資金={initial_bankroll:,}円)")
        print(f"{'─'*70}")

        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        mask = confidences >= thresh

        # 日別にグループ化
        unique_dates = sorted(set(dates_val))
        daily_stats = OrderedDict()

        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        max_drawdown_date = None
        current_losing_streak = 0
        max_losing_streak = 0
        total_bets = 0
        total_hits = 0
        total_invested = 0
        total_returned = 0

        for date in unique_dates:
            day_mask = (dates_val == date) & mask
            n_day_bets = day_mask.sum()

            if n_day_bets == 0:
                continue

            day_preds = preds[day_mask]
            day_y = y_val[day_mask]
            day_payouts = p_val[day_mask]

            day_hits = (day_preds == day_y)
            n_day_hits = day_hits.sum()
            day_invested = n_day_bets * bet_unit
            day_returned = day_payouts[day_hits].sum()
            day_pnl = day_returned - day_invested

            cumulative_pnl += day_pnl
            total_bets += n_day_bets
            total_hits += n_day_hits
            total_invested += day_invested
            total_returned += day_returned

            # ドローダウン追跡
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            drawdown = peak_pnl - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_date = date

            # 連敗追跡（日単位: その日1件も当たらなければ負け日）
            if n_day_hits == 0:
                current_losing_streak += 1
                max_losing_streak = max(max_losing_streak, current_losing_streak)
            else:
                current_losing_streak = 0

            daily_stats[date] = {
                'bets': n_day_bets,
                'hits': n_day_hits,
                'invested': day_invested,
                'returned': day_returned,
                'pnl': day_pnl,
                'cumulative': cumulative_pnl,
            }

        if not daily_stats:
            print("  ベット対象レースなし")
            continue

        # 日別テーブル（最初5日 + 最後5日を表示）
        all_days = list(daily_stats.items())
        show_days = []
        if len(all_days) <= 12:
            show_days = all_days
        else:
            show_days = all_days[:5] + [None] + all_days[-5:]

        print(f"\n  {'日付':>12}  {'ベット':>6}  {'的中':>4}  {'投資':>8}  {'回収':>10}  "
              f"{'日別P&L':>10}  {'累積P&L':>12}  {'残高':>12}")
        print(f"  {'-'*86}")

        for item in show_days:
            if item is None:
                print(f"  {'... (中略) ...':^86}")
                continue
            date, s = item
            balance = initial_bankroll + s['cumulative']
            print(f"  {str(date):>12}  {s['bets']:>5,}件  {s['hits']:>3,}件  "
                  f"{s['invested']:>7,}円  {s['returned']:>9,.0f}円  "
                  f"{s['pnl']:>+9,.0f}円  {s['cumulative']:>+11,.0f}円  "
                  f"{balance:>11,}円")

        # サマリー統計
        final_balance = initial_bankroll + cumulative_pnl
        overall_roi = (total_returned / total_invested - 1) * 100 if total_invested > 0 else 0
        hit_rate = total_hits / total_bets * 100 if total_bets > 0 else 0
        n_days = len(daily_stats)
        n_profit_days = sum(1 for s in daily_stats.values() if s['pnl'] > 0)
        n_loss_days = sum(1 for s in daily_stats.values() if s['pnl'] < 0)
        n_even_days = n_days - n_profit_days - n_loss_days

        # 月別ROI
        monthly_stats = defaultdict(lambda: {'invested': 0, 'returned': 0})
        for date, s in daily_stats.items():
            month_key = date.strftime('%Y-%m')
            monthly_stats[month_key]['invested'] += s['invested']
            monthly_stats[month_key]['returned'] += s['returned']

        print(f"\n  {'─'*50}")
        print(f"  総合サマリー:")
        print(f"    期間        : {list(daily_stats.keys())[0]} 〜 {list(daily_stats.keys())[-1]} ({n_days}日間)")
        print(f"    初期資金    : {initial_bankroll:>12,}円")
        print(f"    最終残高    : {final_balance:>12,}円  ({cumulative_pnl:>+,}円)")
        print(f"    総投資額    : {total_invested:>12,}円")
        print(f"    総回収額    : {total_returned:>12,.0f}円")
        print(f"    ROI         : {overall_roi:>+11.1f}%")
        print(f"    的中率      : {hit_rate:>11.1f}%  ({total_hits:,}/{total_bets:,})")
        print(f"    勝ち日/負け日: {n_profit_days}勝 {n_loss_days}敗 {n_even_days}分 "
              f"(勝率{n_profit_days/n_days*100:.0f}%)")
        print(f"    最大ドローダウン: {max_drawdown:>10,.0f}円  ({max_drawdown_date})")
        print(f"    最長連敗    : {max_losing_streak}日")

        # 月別ROI
        if len(monthly_stats) > 1:
            print(f"\n  月別ROI:")
            print(f"    {'月':>8}  {'投資額':>10}  {'回収額':>10}  {'ROI':>8}")
            print(f"    {'-'*40}")
            for month_key in sorted(monthly_stats.keys()):
                ms = monthly_stats[month_key]
                m_roi = (ms['returned'] / ms['invested'] - 1) * 100 if ms['invested'] > 0 else 0
                print(f"    {month_key:>8}  {ms['invested']:>9,}円  {ms['returned']:>9,.0f}円  {m_roi:>+7.1f}%")

        # 累積P&Lの簡易チャート（テキスト）
        if len(all_days) >= 5:
            print(f"\n  累積P&Lチャート:")
            cum_values = [s['cumulative'] for _, s in all_days]
            chart_min = min(min(cum_values), 0)
            chart_max = max(max(cum_values), 0)
            chart_range = chart_max - chart_min if chart_max != chart_min else 1
            chart_width = 50

            # 10等分のポイントを表示
            n_points = min(20, len(all_days))
            step = max(1, len(all_days) // n_points)
            sampled = all_days[::step]
            if all_days[-1] not in sampled:
                sampled.append(all_days[-1])

            zero_pos = int((0 - chart_min) / chart_range * chart_width)

            for date, s in sampled:
                val = s['cumulative']
                pos = int((val - chart_min) / chart_range * chart_width)
                bar = [' '] * (chart_width + 1)

                # ゼロライン
                if 0 <= zero_pos <= chart_width:
                    bar[zero_pos] = '|'

                if val >= 0:
                    for j in range(zero_pos, min(pos + 1, chart_width + 1)):
                        bar[j] = '#'
                else:
                    for j in range(max(pos, 0), min(zero_pos + 1, chart_width + 1)):
                        bar[j] = '-'

                bar_str = ''.join(bar)
                print(f"    {str(date):>10} [{bar_str}] {val:>+,}円")


# ============================================================
# 学習 + 比較 + シミュレーション
# ============================================================

def train_and_compare():
    """3モデル比較実験 → 2シミュレーション実行"""
    import lightgbm as lgb

    logger.info("=" * 60)
    logger.info("LightGBM カスタム損失関数 利益最適化実験")
    logger.info("=" * 60)

    # === データロード ===
    X, y1, payouts, dates = load_training_data_with_payout()
    if X is None:
        logger.error("訓練データがありません。終了します。")
        return

    # === Train/Val 分割 (時系列: 先頭80%=train, 後半20%=val) ===
    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y1[:split], y1[split:]
    p_train, p_val = payouts[:split], payouts[split:]
    dates_val = dates[split:]

    logger.info(f"Train: {len(X_train):,}件, Val: {len(X_val):,}件")
    logger.info(f"Val期間: {dates_val[0]} 〜 {dates_val[-1]}")
    logger.info(f"Val期間配当: 中央値={np.median(p_val):.0f}円, 平均={np.mean(p_val):.0f}円")

    weights_train = np.log1p(p_train / 1000.0)
    weights_val = np.log1p(p_val / 1000.0)

    feature_names = _get_feature_names_208()
    results = []
    models_probs = []  # シミュレーション用に保持

    # =========================================
    # モデル1: LightGBM 標準 (multi_logloss)
    # =========================================
    logger.info("\n" + "=" * 40)
    logger.info("モデル1: LightGBM 標準 (multi_logloss)")
    logger.info("=" * 40)

    dtrain_std = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval_std = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain_std)

    params_std = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
    }

    model_std = lgb.train(
        params_std,
        dtrain_std,
        num_boost_round=500,
        valid_sets=[dval_std],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )

    probs_std = model_std.predict(X_val)
    results.append(evaluate_model_predictions(y_val, probs_std, p_val, "LightGBM標準"))
    models_probs.append(("LightGBM標準", probs_std))

    importance = model_std.feature_importance(importance_type='gain')
    top_features = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]
    logger.info("Top10特徴量 (gain):")
    for fname, imp in top_features:
        logger.info(f"  {fname}: {imp:.1f}")

    # =========================================
    # モデル2: LightGBM Profit-Weighted
    # =========================================
    logger.info("\n" + "=" * 40)
    logger.info("モデル2: LightGBM Profit-Weighted CE")
    logger.info("=" * 40)

    def custom_obj(y_pred_raw, dtrain):
        y_true = dtrain.get_label()
        grad, hess = profit_weighted_objective(y_true, y_pred_raw, weights_train)
        return grad, hess

    def custom_eval(y_pred_raw, dval_dataset):
        y_true = dval_dataset.get_label()
        name, val, is_higher_better = profit_weighted_eval(
            y_true, y_pred_raw, weights_val
        )
        return name, val, is_higher_better

    dtrain_pw = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval_pw = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain_pw)

    params_pw = {
        'objective': custom_obj,
        'num_class': 6,
        'metric': 'None',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
    }

    model_pw = lgb.train(
        params_pw,
        dtrain_pw,
        num_boost_round=500,
        valid_sets=[dval_pw],
        feval=custom_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )

    raw_pw = model_pw.predict(X_val)
    raw_pw_shifted = raw_pw - raw_pw.max(axis=1, keepdims=True)
    exp_pw = np.exp(raw_pw_shifted)
    probs_pw = exp_pw / exp_pw.sum(axis=1, keepdims=True)
    results.append(evaluate_model_predictions(y_val, probs_pw, p_val, "LightGBM Profit-Weighted"))
    models_probs.append(("LightGBM Profit-Weighted", probs_pw))

    # =========================================
    # モデル3: 現行PyTorch (参考値)
    # =========================================
    logger.info("\n" + "=" * 40)
    logger.info("モデル3: 現行PyTorch (Focal Loss)")
    logger.info("=" * 40)

    try:
        import torch
        from src.models import load_model as load_pytorch_model

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'boatrace_model.pth'
        )
        if os.path.exists(model_path):
            pt_model = load_pytorch_model(model_path)
            pt_model.eval()

            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val)
                outputs = pt_model(X_val_tensor)
                probs_pt = torch.softmax(outputs[0], dim=1).numpy()

            results.append(evaluate_model_predictions(y_val, probs_pt, p_val, "PyTorch Focal Loss"))
            models_probs.append(("PyTorch Focal Loss", probs_pt))
        else:
            logger.warning(f"PyTorchモデルが見つかりません: {model_path}")
    except Exception as e:
        logger.warning(f"PyTorchモデル読み込みエラー: {e}")

    # =========================================
    # 比較テーブル
    # =========================================
    print("\n")
    print("=" * 80)
    print("比較結果")
    print("=" * 80)
    print(f"{'モデル':<28} {'的中率':>7} {'ROI(1着的中)':>12} {'ROI(期待値)':>11} "
          f"{'穴的中率':>8} {'本命的中率':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<28} {r['accuracy']:>6.1f}% "
              f"{r['roi_1st_hit']:>+10.1f}% {r['roi_expected']:>+9.1f}% "
              f"{r['high_payout_acc']:>7.1f}% {r['low_payout_acc']:>9.1f}%")

    print("-" * 80)
    print(f"  穴=配当>=10,000円 ({results[0]['high_payout_count']:,}件), "
          f"本命=配当<3,000円 ({results[0]['low_payout_count']:,}件)")

    print("\n号艇別予測分布 (%):")
    print(f"{'モデル':<28}", end="")
    for i in range(6):
        print(f" {i+1}号艇", end="")
    print()
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<28}", end="")
        for i in range(6):
            print(f" {r['pred_dist'].get(i, 0):5.1f}", end="")
        print()

    print("\n" + "=" * 80)
    print("分析:")
    for r in results:
        boat1_pct = r['pred_dist'].get(0, 0)
        if boat1_pct > 60:
            print(f"  ! {r['model']}: 1号艇予測が{boat1_pct:.0f}% (偏重)")
        elif boat1_pct > 40:
            print(f"  ? {r['model']}: 1号艇予測が{boat1_pct:.0f}% (やや偏重)")
        else:
            print(f"  o {r['model']}: 1号艇予測が{boat1_pct:.0f}% (分散OK)")

    pw_result = results[1] if len(results) > 1 else None
    std_result = results[0] if len(results) > 0 else None
    if pw_result and std_result:
        if pw_result['high_payout_acc'] > std_result['high_payout_acc']:
            diff = pw_result['high_payout_acc'] - std_result['high_payout_acc']
            print(f"  + Profit-Weighted: 穴レース的中率が標準より +{diff:.1f}pt 改善")
        else:
            diff = std_result['high_payout_acc'] - pw_result['high_payout_acc']
            print(f"  - Profit-Weighted: 穴レース的中率が標準より -{diff:.1f}pt 低下")
    print("=" * 80)

    # =========================================
    # シミュレーション実行
    # =========================================
    best_thresholds = simulate_confidence_sweep(models_probs, y_val, p_val)
    simulate_daily_pnl(models_probs, y_val, p_val, dates_val, best_thresholds)

    print("\n" + "=" * 80)
    print("全工程完了")
    print("=" * 80)


if __name__ == '__main__':
    train_and_compare()
