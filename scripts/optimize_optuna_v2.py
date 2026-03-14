"""Optuna 7次元パラメータ最適化 + 日別収支バックテスト

Phase 1: LightGBM モデル学習 (標準 + Profit-Weighted + PyTorch参考)
Phase 2: Optuna ベイズ最適化 (7パラメータ x 各モデル)
Phase 3: Best Params で日別累積収支バックテスト (Sim2)

探索パラメータ (7次元):
  max_boat1_prob : [0.30, 0.70]  1号艇勝率の逆張り上限
  min_entropy    : [1.0, 2.5]    波乱度合いの下限 (bits)
  min_ev         : [1.00, 1.30]  期待値下限
  min_odds       : [3.0, 20.0]   オッズ下限
  max_odds       : [30.0, 150.0] オッズ上限
  min_probability: [0.01, 0.05]  最低確信度
  kelly_fraction : [0.05, 0.30]  Kelly分率

使い方:
  pip install lightgbm optuna
  DATABASE_URL=xxx python scripts/optimize_optuna_v2.py
  DATABASE_URL=xxx python scripts/optimize_optuna_v2.py --n-trials 500
"""
import sys
import os
import argparse
import logging
import numpy as np
from collections import OrderedDict, defaultdict

# パス設定
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_custom_loss import (
    load_training_data_with_payout,
    profit_weighted_objective,
    profit_weighted_eval,
    _get_feature_names_208,
)
from src.features import FeatureEngineerLegacy as FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PARAM_DEFS = [
    # (name, label, lo, hi)
    ('max_boat1_prob',  '1号艇確率上限',   0.30,   0.70),
    ('min_entropy',     'エントロピー下限', 1.0,    2.5),
    ('min_ev',          '期待値下限',       1.00,   1.30),
    ('min_odds',        'オッズ下限',       3.0,   20.0),
    ('max_odds',        'オッズ上限',      30.0,  150.0),
    ('min_probability', '最低確信度',       0.01,   0.05),
    ('kelly_fraction',  'Kelly分率',        0.05,   0.30),
]


# ============================================================
# Phase 1: モデル学習
# ============================================================

def train_models(X_train, y_train, p_train, X_val, y_val, p_val, feature_names):
    """LightGBM 標準 + Profit-Weighted + PyTorch(任意) を学習

    Returns:
        list of (name, probs_on_val)
    """
    import lightgbm as lgb

    weights_train = np.log1p(p_train / 1000.0)
    weights_val = np.log1p(p_val / 1000.0)
    models = []

    # --- 標準 LightGBM ---
    logger.info("Phase1: LightGBM標準 (multi_logloss) 学習中...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    params_std = {
        'objective': 'multiclass', 'num_class': 6,
        'metric': 'multi_logloss',
        'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'verbose': -1, 'seed': 42,
    }
    model_std = lgb.train(
        params_std, dtrain, num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )
    probs_std = model_std.predict(X_val)
    models.append(("LightGBM標準", probs_std))
    logger.info(f"  完了 (best_iteration={model_std.best_iteration})")

    # 特徴量重要度
    importance = model_std.feature_importance(importance_type='gain')
    top_feat = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5]
    logger.info("  Top5特徴量: " + ", ".join(f"{n}:{v:.0f}" for n, v in top_feat))

    # --- Profit-Weighted LightGBM ---
    logger.info("Phase1: LightGBM Profit-Weighted 学習中...")
    dtrain_pw = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval_pw = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain_pw)

    def custom_obj(y_pred_raw, dt):
        return profit_weighted_objective(dt.get_label(), y_pred_raw, weights_train)

    def custom_eval(y_pred_raw, dv):
        return profit_weighted_eval(dv.get_label(), y_pred_raw, weights_val)

    params_pw = {
        'objective': custom_obj,
        'num_class': 6,
        'metric': 'None',
        'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'verbose': -1, 'seed': 42,
    }
    model_pw = lgb.train(
        params_pw, dtrain_pw, num_boost_round=500,
        valid_sets=[dval_pw], feval=custom_eval,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )
    raw_pw = model_pw.predict(X_val)
    shifted = raw_pw - raw_pw.max(axis=1, keepdims=True)
    exp_pw = np.exp(shifted)
    probs_pw = exp_pw / exp_pw.sum(axis=1, keepdims=True)
    models.append(("LightGBM Profit-Weighted", probs_pw))
    logger.info(f"  完了 (best_iteration={model_pw.best_iteration})")

    # --- PyTorch (任意) ---
    try:
        import torch
        from src.models import load_model as load_pytorch_model
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'boatrace_model.pth'
        )
        if os.path.exists(model_path):
            pt_model = load_pytorch_model(model_path)
            pt_model.eval()
            with torch.no_grad():
                probs_pt = torch.softmax(
                    pt_model(torch.FloatTensor(X_val))[0], dim=1
                ).numpy()
            models.append(("PyTorch Focal Loss", probs_pt))
            logger.info("Phase1: PyTorchモデル読み込み成功")
    except Exception as e:
        logger.warning(f"Phase1: PyTorchモデルスキップ: {e}")

    return models


# ============================================================
# Phase 2: Optuna 7次元ベイズ最適化
# ============================================================

def compute_race_signals(probs, payouts):
    """モデル予測からレースシグナルを事前計算 (Optuna高速化用)

    entropy は bits (log2) で計算。6クラス均等時 max ~ 2.58
    """
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), pred]
    boat1_prob = probs[:, 0]
    entropy = -np.sum(probs * np.log2(np.maximum(probs, 1e-12)), axis=1)
    actual_odds = payouts / 100.0
    ev = confidence * actual_odds
    return {
        'pred': pred,
        'confidence': confidence,
        'boat1_prob': boat1_prob,
        'entropy': entropy,
        'actual_odds': actual_odds,
        'ev': ev,
    }


def create_optuna_objective(signals, y_val, initial_bankroll=200_000):
    """Optuna目的関数をクロージャで生成

    戻り値: ROI (%)
    ベット50件未満のtrial → ROI=-100% ペナルティ
    """
    pred = signals['pred']
    confidence = signals['confidence']
    boat1_prob = signals['boat1_prob']
    entropy = signals['entropy']
    actual_odds = signals['actual_odds']
    ev = signals['ev']
    correct = (pred == y_val)

    def objective(trial):
        params = {}
        for name, _, lo, hi in PARAM_DEFS:
            params[name] = trial.suggest_float(name, lo, hi)

        # ベクトル化フィルタ
        mask = (
            (boat1_prob <= params['max_boat1_prob']) &
            (entropy >= params['min_entropy']) &
            (ev >= params['min_ev']) &
            (actual_odds >= params['min_odds']) &
            (actual_odds <= params['max_odds']) &
            (confidence >= params['min_probability'])
        )

        bet_indices = np.where(mask)[0]
        if len(bet_indices) < 50:
            return -100.0

        # Kelly ベースの逐次シミュレーション
        kf = params['kelly_fraction']
        bankroll = float(initial_bankroll)
        total_invested = 0
        total_returned = 0.0
        bankroll_cap = initial_bankroll * 100  # 100倍キャップ（overflow防止）

        for idx in bet_indices:
            if bankroll < 100:
                break
            if bankroll > bankroll_cap:
                bankroll = bankroll_cap

            odds_i = actual_odds[idx]
            conf_i = confidence[idx]
            if odds_i <= 1.0:
                continue

            # Fractional Kelly
            kelly_f = (conf_i * odds_i - 1.0) / (odds_i - 1.0)
            if kelly_f <= 0 or kelly_f > 1.0:
                continue

            bet = bankroll * kf * kelly_f
            bet = max(100.0, min(bet, bankroll * 0.05))  # 1ベット最大5%
            bet = int(min(bet, 1e9) / 100) * 100  # int overflow防止
            if bet < 100 or bet > bankroll:
                continue

            total_invested += bet
            if correct[idx]:
                payout = bet * odds_i
                total_returned += payout
                bankroll += payout - bet
            else:
                bankroll -= bet

        if total_invested == 0:
            return -100.0

        roi = (total_returned / total_invested - 1.0) * 100.0
        return min(roi, 10000.0)  # ROIキャップ（過学習防止）

    return objective


def run_optuna(signals, y_val, model_name, n_trials=200, initial_bankroll=200_000):
    """Optuna study を実行して best_params を返す"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Phase2: Optuna最適化 -{model_name} ({n_trials} trials)")

    objective = create_optuna_objective(signals, y_val, initial_bankroll)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_roi = study.best_value

    # 統計情報
    all_rois = [t.value for t in study.trials if t.value is not None]
    penalized = sum(1 for v in all_rois if v == -100.0)
    positive = sum(1 for v in all_rois if v > 0)

    logger.info(f"  Best ROI: {best_roi:+.1f}%")
    logger.info(f"  Trials: {n_trials} (positive ROI: {positive}, "
                f"penalized(<50bets): {penalized})")

    return best, best_roi, study


# ============================================================
# Phase 3: 日別累積収支バックテスト (Sim2)
# ============================================================

def simulate_daily_pnl(signals, y_val, p_val, dates_val, params, model_name,
                       initial_bankroll=200_000):
    """Best Params による日別P&Lシミュレーション

    Kelly分率ベースのベットサイジング、日別グルーピングで追跡。
    最大ドローダウン・最長連敗・月別ROI・累積P&Lチャートを出力。
    """
    pred = signals['pred']
    confidence = signals['confidence']
    actual_odds = signals['actual_odds']
    correct = (pred == y_val)

    kf = params['kelly_fraction']

    # 7パラメータフィルタ
    mask = (
        (signals['boat1_prob'] <= params['max_boat1_prob']) &
        (signals['entropy'] >= params['min_entropy']) &
        (signals['ev'] >= params['min_ev']) &
        (actual_odds >= params['min_odds']) &
        (actual_odds <= params['max_odds']) &
        (confidence >= params['min_probability'])
    )

    print(f"\n{'='*90}")
    print(f"  Sim2: 日別累積収支バックテスト -{model_name}")
    print(f"  初期資金={initial_bankroll:,}円  Kelly分率={kf:.3f}")
    print(f"{'='*90}")

    unique_dates = sorted(set(dates_val))
    daily_stats = OrderedDict()

    bankroll = float(initial_bankroll)
    peak_bankroll = bankroll
    max_drawdown = 0.0
    max_drawdown_date = None
    current_losing_streak = 0
    max_losing_streak = 0
    total_bets = 0
    total_hits = 0
    total_invested = 0
    total_returned = 0.0

    for date in unique_dates:
        day_idx = np.where((dates_val == date) & mask)[0]
        if len(day_idx) == 0:
            continue

        day_invested = 0
        day_returned = 0.0
        day_hits = 0
        day_bets = 0

        for idx in day_idx:
            if bankroll < 100:
                break
            if bankroll > initial_bankroll * 100:
                bankroll = float(initial_bankroll * 100)

            odds_i = actual_odds[idx]
            conf_i = confidence[idx]
            if odds_i <= 1.0:
                continue

            kelly_f = (conf_i * odds_i - 1.0) / (odds_i - 1.0)
            if kelly_f <= 0 or kelly_f > 1.0:
                continue

            bet = bankroll * kf * kelly_f
            bet = max(100.0, min(bet, bankroll * 0.05))
            bet = int(min(bet, 1e9) / 100) * 100
            if bet < 100 or bet > bankroll:
                continue

            day_bets += 1
            day_invested += bet

            if correct[idx]:
                payout_amt = bet * odds_i
                day_returned += payout_amt
                bankroll += payout_amt - bet
                day_hits += 1
            else:
                bankroll -= bet

        if day_bets == 0:
            continue

        day_pnl = day_returned - day_invested
        total_bets += day_bets
        total_hits += day_hits
        total_invested += day_invested
        total_returned += day_returned

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = peak_bankroll - bankroll
        if dd > max_drawdown:
            max_drawdown = dd
            max_drawdown_date = date

        if day_hits == 0:
            current_losing_streak += 1
            max_losing_streak = max(max_losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0

        daily_stats[date] = {
            'bets': day_bets,
            'hits': day_hits,
            'invested': day_invested,
            'returned': day_returned,
            'pnl': day_pnl,
            'bankroll': bankroll,
        }

    if not daily_stats:
        print("  ベット対象レースなし (フィルタが厳しすぎる可能性)")
        return

    # --- 日別テーブル ---
    all_days = list(daily_stats.items())
    show_days = all_days if len(all_days) <= 14 else all_days[:6] + [None] + all_days[-6:]

    print(f"\n  {'日付':>12}  {'ベット':>6}  {'的中':>4}  {'投資':>10}  {'回収':>12}  "
          f"{'日別P&L':>12}  {'残高':>14}")
    print(f"  {'-'*82}")

    for item in show_days:
        if item is None:
            print(f"  {'... (中略) ...':^82}")
            continue
        date, s = item
        print(f"  {str(date):>12}  {s['bets']:>5,}件  {s['hits']:>3,}件  "
              f"{s['invested']:>9,}円  {s['returned']:>11,.0f}円  "
              f"{s['pnl']:>+11,.0f}円  {s['bankroll']:>13,.0f}円")

    # --- 総合サマリー ---
    final_bankroll = bankroll
    overall_roi = (total_returned / total_invested - 1) * 100 if total_invested > 0 else 0
    hit_rate = total_hits / total_bets * 100 if total_bets > 0 else 0
    n_days = len(daily_stats)
    n_profit_days = sum(1 for s in daily_stats.values() if s['pnl'] > 0)
    n_loss_days = sum(1 for s in daily_stats.values() if s['pnl'] < 0)
    n_even_days = n_days - n_profit_days - n_loss_days
    pnl = final_bankroll - initial_bankroll

    monthly = defaultdict(lambda: {'invested': 0, 'returned': 0.0})
    for date, s in daily_stats.items():
        mk = date.strftime('%Y-%m')
        monthly[mk]['invested'] += s['invested']
        monthly[mk]['returned'] += s['returned']

    print(f"\n  {'-'*55}")
    print(f"  総合サマリー:")
    dates_list = list(daily_stats.keys())
    print(f"    期間          : {dates_list[0]} ~ {dates_list[-1]} ({n_days}日間)")
    print(f"    初期資金      : {initial_bankroll:>14,}円")
    print(f"    最終残高      : {final_bankroll:>14,.0f}円  ({pnl:>+,.0f}円)")
    print(f"    総投資額      : {total_invested:>14,}円")
    print(f"    総回収額      : {total_returned:>14,.0f}円")
    print(f"    ROI           : {overall_roi:>+13.1f}%")
    print(f"    総ベット数    : {total_bets:>14,}件")
    print(f"    的中率        : {hit_rate:>13.1f}%  ({total_hits:,}/{total_bets:,})")
    print(f"    勝ち日/負け日 : {n_profit_days}勝 {n_loss_days}敗 {n_even_days}分 "
          f"(日勝率{n_profit_days / n_days * 100:.0f}%)")
    print(f"    最大DD        : {max_drawdown:>12,.0f}円  ({max_drawdown_date})")
    print(f"    最長連敗      : {max_losing_streak}日")

    # --- 月別ROI ---
    if len(monthly) > 1:
        print(f"\n  月別ROI:")
        print(f"    {'月':>8}  {'投資額':>10}  {'回収額':>12}  {'ROI':>8}  {'損益':>12}")
        print(f"    {'-'*56}")
        for mk in sorted(monthly.keys()):
            ms = monthly[mk]
            m_roi = (ms['returned'] / ms['invested'] - 1) * 100 if ms['invested'] > 0 else 0
            m_pnl = ms['returned'] - ms['invested']
            print(f"    {mk:>8}  {ms['invested']:>9,}円  {ms['returned']:>11,.0f}円  "
                  f"{m_roi:>+7.1f}%  {m_pnl:>+11,.0f}円")

    # --- 累積P&Lチャート ---
    if len(all_days) >= 5:
        print(f"\n  累積P&Lチャート:")
        cum_vals = [s['bankroll'] - initial_bankroll for _, s in all_days]
        chart_min = min(min(cum_vals), 0)
        chart_max = max(max(cum_vals), 0)
        chart_range = chart_max - chart_min if chart_max != chart_min else 1
        chart_w = 50

        n_pts = min(20, len(all_days))
        step = max(1, len(all_days) // n_pts)
        sampled = all_days[::step]
        if all_days[-1] not in sampled:
            sampled.append(all_days[-1])

        zero_pos = int((0 - chart_min) / chart_range * chart_w)

        for date, s in sampled:
            val = s['bankroll'] - initial_bankroll
            pos = int((val - chart_min) / chart_range * chart_w)
            bar = [' '] * (chart_w + 1)

            if 0 <= zero_pos <= chart_w:
                bar[zero_pos] = '|'

            if val >= 0:
                for j in range(zero_pos, min(pos + 1, chart_w + 1)):
                    bar[j] = '#'
            else:
                for j in range(max(pos, 0), min(zero_pos + 1, chart_w + 1)):
                    bar[j] = '-'

            print(f"    {str(date):>10} [{''.join(bar)}] {val:>+,}円")


# ============================================================
# メインフロー
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Optuna 7次元パラメータ最適化')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Optunaの探索回数 (default: 200)')
    parser.add_argument('--bankroll', type=int, default=200_000,
                        help='初期資金 (default: 200000)')
    parser.add_argument('--years', type=int, default=3,
                        help='学習データ年数 (default: 3)')
    args = parser.parse_args()

    # 依存チェック
    try:
        import lightgbm  # noqa: F401
        import optuna     # noqa: F401
    except ImportError as e:
        print(f"必要パッケージが未インストール: {e}")
        print("  pip install lightgbm optuna")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  Optuna 7次元パラメータ最適化 + 日別収支バックテスト")
    print("=" * 70)

    # === Phase 1: データロード & モデル学習 ===
    logger.info("Phase1: データロード中...")
    X, y1, payouts, dates = load_training_data_with_payout(years=args.years)
    if X is None:
        logger.error("訓練データがありません。終了。")
        return

    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y1[:split], y1[split:]
    p_train, p_val = payouts[:split], payouts[split:]
    dates_val = dates[split:]

    logger.info(f"  Train: {len(X_train):,}件, Val: {len(X_val):,}件")
    logger.info(f"  Val期間: {dates_val[0]} ~ {dates_val[-1]}")

    feature_names = _get_feature_names_208()
    trained_models = train_models(
        X_train, y_train, p_train, X_val, y_val, p_val, feature_names
    )

    # === Phase 2: 各モデルで Optuna 最適化 ===
    all_results = []

    for model_name, probs in trained_models:
        signals = compute_race_signals(probs, p_val)
        best_params, best_roi, study = run_optuna(
            signals, y_val, model_name,
            n_trials=args.n_trials,
            initial_bankroll=args.bankroll,
        )
        all_results.append({
            'name': model_name,
            'probs': probs,
            'signals': signals,
            'best_params': best_params,
            'best_roi': best_roi,
            'study': study,
        })

    # === Best Params 一覧 ===
    print("\n")
    print("=" * 90)
    print("  Optuna 最適化結果: Best Params")
    print("=" * 90)

    # ヘッダー
    col_w = max(28, max(len(r['name']) for r in all_results) + 2)
    print(f"\n  {'パラメータ':<20}", end="")
    for r in all_results:
        print(f"  {r['name']:>{col_w}}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in all_results:
        print(f"  {'-'*col_w}", end="")
    print()

    for pname, plabel, _, _ in PARAM_DEFS:
        print(f"  {plabel:<20}", end="")
        for r in all_results:
            val = r['best_params'][pname]
            print(f"  {val:>{col_w}.4f}", end="")
        print()

    print(f"  {'='*20}", end="")
    for _ in all_results:
        print(f"  {'='*col_w}", end="")
    print()
    print(f"  {'Best ROI':<20}", end="")
    for r in all_results:
        print(f"  {r['best_roi']:>+{col_w - 1}.1f}%", end="")
    print()

    # ベット件数 (best params適用時)
    print(f"  {'ベット件数':<20}", end="")
    for r in all_results:
        sig = r['signals']
        bp = r['best_params']
        m = (
            (sig['boat1_prob'] <= bp['max_boat1_prob']) &
            (sig['entropy'] >= bp['min_entropy']) &
            (sig['ev'] >= bp['min_ev']) &
            (sig['actual_odds'] >= bp['min_odds']) &
            (sig['actual_odds'] <= bp['max_odds']) &
            (sig['confidence'] >= bp['min_probability'])
        )
        print(f"  {int(m.sum()):>{col_w},}件", end="")
    print()

    # Top5 trialのROI分布
    print(f"\n  Optuna Top5 trial ROI:")
    for r in all_results:
        sorted_trials = sorted(r['study'].trials, key=lambda t: t.value or -999, reverse=True)
        top5 = [t.value for t in sorted_trials[:5] if t.value is not None and t.value > -100]
        top5_str = ", ".join(f"{v:+.1f}%" for v in top5) if top5 else "(全てペナルティ)"
        print(f"    {r['name']}: {top5_str}")

    print("=" * 90)

    # === Phase 3: 各モデルで Sim2 バックテスト ===
    for r in all_results:
        simulate_daily_pnl(
            r['signals'], y_val, p_val, dates_val,
            r['best_params'], r['name'],
            initial_bankroll=args.bankroll,
        )

    # === 最終比較 ===
    print(f"\n{'='*70}")
    print("  最終比較")
    print(f"{'='*70}")
    print(f"  {'モデル':<28}  {'Optuna ROI':>10}  {'Kelly分率':>9}  {'ベット件数':>10}")
    print(f"  {'-'*62}")
    for r in all_results:
        bp = r['best_params']
        sig = r['signals']
        m = (
            (sig['boat1_prob'] <= bp['max_boat1_prob']) &
            (sig['entropy'] >= bp['min_entropy']) &
            (sig['ev'] >= bp['min_ev']) &
            (sig['actual_odds'] >= bp['min_odds']) &
            (sig['actual_odds'] <= bp['max_odds']) &
            (sig['confidence'] >= bp['min_probability'])
        )
        print(f"  {r['name']:<28}  {r['best_roi']:>+9.1f}%  {bp['kelly_fraction']:>9.3f}"
              f"  {int(m.sum()):>9,}件")
    print(f"{'='*70}")
    print("全工程完了")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
