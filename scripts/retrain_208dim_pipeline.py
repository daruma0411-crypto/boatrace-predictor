"""208次元 最強再学習パイプライン

気象データ収集100%完了後に実行する。
3ステップで特徴量選別→学習→Optuna最適化を一気通貫で実行。

Step A: 特徴量重要度分析 (Permutation Importance)
  - 208次元で LightGBM を学習
  - 各特徴量を1つずつシャッフルして精度低下を計測
  - importance > 閾値 の特徴量のみ残す → N次元 (目標: 60-100次元)

Step B: 選別済み特徴量で PyTorch 4モデル再学習
  - Step A で選別した特徴量マスクを使用
  - input_dim=N, hidden は自動調整
  - Focal Loss + Early Stopping

Step C: Optuna 7次元パラメータ最適化
  - 選別済みモデルで最適ベットパラメータを探索

使い方:
    DATABASE_URL=xxx python scripts/retrain_208dim_pipeline.py
    DATABASE_URL=xxx python scripts/retrain_208dim_pipeline.py --importance-threshold 0.001
    DATABASE_URL=xxx python scripts/retrain_208dim_pipeline.py --skip-importance  # 選別スキップ(全208次元)
"""
import sys
import os
import argparse
import logging
import json
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features import FeatureEngineerLegacy
from src.database import get_db_connection
from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss, save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WIND_DIRS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'calm']
PLAYER_CLASSES = ['A1', 'A2', 'B1', 'B2']

MODELS = [
    {'path': 'models/boatrace_model.pth', 'gamma': 2.0, 'label': '標準'},
    {'path': 'models/boatrace_model_s05.pth', 'gamma': 1.5, 'label': 'マイルド'},
    {'path': 'models/boatrace_model_s07.pth', 'gamma': 2.5, 'label': 'やや強め'},
    {'path': 'models/boatrace_model_s085.pth', 'gamma': 3.0, 'label': 'アグレッシブ'},
]


def get_feature_names_208():
    """208次元の特徴量名"""
    names = ['G_venue_id', 'G_month', 'G_distance', 'G_wind_speed']
    names.extend([f'G_wind_{d}' for d in WIND_DIRS])
    names.extend(['G_temperature', 'G_wave_height', 'G_water_temperature'])
    for b in range(1, 7):
        p = f'B{b}'
        names.extend([f'{p}_class_{c}' for c in PLAYER_CLASSES])
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
    return names


# ============================================================
# データロード
# ============================================================

def load_data_208(years=3):
    """208次元特徴量 + ラベル + 配当をDB一括取得

    気象データ必須フィルタ (wind_speed IS NOT NULL)
    """
    fe = FeatureEngineerLegacy()
    cutoff = (datetime.now() - timedelta(days=365 * years)).date()

    logger.info("データ一括取得中 (208次元, 気象必須)...")

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
              AND r.wind_speed IS NOT NULL
            ORDER BY r.race_date
        """, (cutoff,))
        races = cur.fetchall()
        logger.info(f"レース取得: {len(races):,}件")

        race_ids = [r['id'] for r in races]
        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   approach_course, is_new_motor, tilt, parts_changed
            FROM boats WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()

    boats_by_race = defaultdict(list)
    for b in all_boats:
        boats_by_race[b['race_id']].append(dict(b))

    X_list, y1_list, y2_list, y3_list, payout_list, date_list = [], [], [], [], [], []

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
            features = fe.transform(race_data, boats)
            X_list.append(features)
            y1_list.append(race['result_1st'] - 1)
            y2_list.append(race['result_2nd'] - 1)
            y3_list.append(race['result_3rd'] - 1)
            payout_list.append(race.get('payout_sanrentan') or 0)
            date_list.append(race['race_date'])
        except Exception:
            continue

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int64)
    y2 = np.array(y2_list, dtype=np.int64)
    y3 = np.array(y3_list, dtype=np.int64)
    payouts = np.array(payout_list, dtype=np.float64)
    dates = np.array(date_list)

    logger.info(f"訓練データ: {len(X):,}件, 次元: {X.shape[1]}")
    return X, y1, y2, y3, payouts, dates


# ============================================================
# Step A: 特徴量重要度分析
# ============================================================

def analyze_feature_importance(X_train, y_train, X_val, y_val, feature_names,
                               threshold=0.0005):
    """Permutation Importance による特徴量選別

    LightGBM で学習 → 各特徴量をシャッフルして精度低下を計測
    低分散 + importance < threshold の特徴量を除外

    Returns:
        selected_mask: bool array (208,)
        importance_report: list of (name, importance, variance, selected)
    """
    import lightgbm as lgb

    logger.info("Step A: 特徴量重要度分析 (Permutation Importance)")

    # まず低分散チェック
    variance = np.var(X_train, axis=0)
    zero_var_mask = variance < 1e-8
    logger.info(f"  分散ゼロ特徴量: {zero_var_mask.sum()}/{len(variance)}")

    # LightGBM 学習
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    params = {
        'objective': 'multiclass', 'num_class': 6,
        'metric': 'multi_logloss',
        'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'verbose': -1, 'seed': 42,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )

    # ベースライン精度
    base_probs = model.predict(X_val)
    base_preds = base_probs.argmax(axis=1)
    base_acc = (base_preds == y_val).mean()
    logger.info(f"  ベースライン精度: {base_acc*100:.2f}%")

    # Permutation Importance
    importances = np.zeros(X_val.shape[1])
    rng = np.random.RandomState(42)
    n_repeats = 5

    for i in range(X_val.shape[1]):
        if zero_var_mask[i]:
            importances[i] = 0.0
            continue

        acc_drops = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            perm_probs = model.predict(X_permuted)
            perm_preds = perm_probs.argmax(axis=1)
            perm_acc = (perm_preds == y_val).mean()
            acc_drops.append(base_acc - perm_acc)
        importances[i] = np.mean(acc_drops)

    # 選別
    selected_mask = (importances > threshold) & (~zero_var_mask)

    # LightGBM の gain importance も参考
    gain_importance = model.feature_importance(importance_type='gain')

    # レポート作成
    report = []
    for i in range(len(feature_names)):
        report.append({
            'name': feature_names[i],
            'perm_importance': importances[i],
            'gain_importance': gain_importance[i],
            'variance': variance[i],
            'selected': bool(selected_mask[i]),
        })

    report.sort(key=lambda x: x['perm_importance'], reverse=True)

    logger.info(f"\n  特徴量選別結果: {selected_mask.sum()}/{len(selected_mask)} 採用")
    logger.info(f"  閾値: importance > {threshold}")
    logger.info(f"\n  Top 20 特徴量:")
    logger.info(f"  {'名前':35s} {'Perm_Imp':>10} {'Gain':>10} {'Var':>10} {'採用':>4}")
    logger.info(f"  {'-'*75}")
    for r in report[:20]:
        mark = "✓" if r['selected'] else "×"
        logger.info(f"  {r['name']:35s} {r['perm_importance']:>10.5f} "
                     f"{r['gain_importance']:>10.1f} {r['variance']:>10.5f} {mark:>4}")

    logger.info(f"\n  除外された主要カテゴリ:")
    excluded = [r for r in report if not r['selected']]
    categories = defaultdict(int)
    for r in excluded:
        cat = r['name'].split('_')[0] if '_' in r['name'] else r['name']
        categories[cat] += 1
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {cat}: {cnt}個除外")

    return selected_mask, report


# ============================================================
# Step B: PyTorch 4モデル再学習 (選別済み次元)
# ============================================================

def compute_class_weights(labels, num_classes=6, smoothing=0.7):
    """出現頻度の逆数ベースのクラス重み"""
    counts = Counter(labels.tolist())
    total = len(labels)
    raw = np.array([total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
                   dtype=np.float32)
    smoothed = (1 - smoothing) * raw + smoothing * np.ones(num_classes)
    smoothed = smoothed / smoothed.mean()
    return torch.FloatTensor(smoothed)


def train_one_model(X_train, y1_train, y2_train, y3_train,
                    X_val, y1_val, y2_val, y3_val,
                    gamma, save_path, label='',
                    epochs=100, batch_size=256, lr=0.0005,
                    patience=15, dropout=0.15):
    """1モデルを訓練して保存"""
    from torch.utils.data import DataLoader, TensorDataset

    logger.info(f"  === {save_path} (gamma={gamma}, {label}) ===")

    cw_2nd = compute_class_weights(y2_train, smoothing=0.7)
    cw_3rd = compute_class_weights(y3_train, smoothing=0.7)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y1_train),
        torch.LongTensor(y2_train), torch.LongTensor(y3_train))
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y1_val),
        torch.LongTensor(y2_val), torch.LongTensor(y3_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    hidden_dims = [512, 256, 128] if input_dim > 50 else [256, 128, 64]
    logger.info(f"    input={input_dim}, hidden={hidden_dims}")

    model = BoatraceMultiTaskModel(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
    ).to(device)
    criterion = BoatraceMultiTaskLoss(
        class_weights_1st=None,
        class_weights_2nd=cw_2nd.to(device),
        class_weights_3rd=cw_3rd.to(device),
        gamma=gamma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by1, by2, by3 in train_loader:
            bx = bx.to(device)
            targets = (by1.to(device), by2.to(device), by3.to(device))
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by1, by2, by3 in val_loader:
                bx = bx.to(device)
                targets = (by1.to(device), by2.to(device), by3.to(device))
                outputs = model(bx)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                correct += (outputs[0].argmax(dim=1) == targets[0]).sum().item()
                total += targets[0].size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total * 100
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_model(model, path=save_path, metadata={
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_1st': val_acc,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'focal_gamma': gamma,
                'dropout': dropout,
                'version': 'v5_208dim_selected',
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    Early Stop epoch {epoch+1}, "
                            f"best_val={best_val_loss:.4f}, acc={best_val_acc:.1f}%")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}: train={train_loss:.4f} "
                        f"val={val_loss:.4f} acc={val_acc:.1f}%")

    logger.info(f"    完了: val_loss={best_val_loss:.4f}, acc={best_val_acc:.1f}%")
    return best_val_loss, best_val_acc


# ============================================================
# Step C: Optuna 最適化
# ============================================================

def run_optuna_optimization(X_val, y_val, payouts_val, dates_val, n_trials=200):
    """選別済み次元のモデルで Optuna 最適化"""
    # optimize_optuna_v2.py の関数を再利用
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from optimize_optuna_v2 import (
        compute_race_signals, run_optuna, simulate_daily_pnl,
        PARAM_DEFS,
    )
    from src.models import load_model

    logger.info("Step C: Optuna 7次元パラメータ最適化")

    # 208次元モデルをロードして推論
    model = load_model('models/boatrace_model.pth', torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.FloatTensor(X_val))[0], dim=1
        ).numpy()

    signals = compute_race_signals(probs, payouts_val)
    best_params, best_roi, study = run_optuna(
        signals, y_val, "PyTorch v5", n_trials=n_trials
    )

    # 結果表示
    print("\n" + "=" * 70)
    print("  Optuna 最適化結果 (208次元選別済みモデル)")
    print("=" * 70)
    for pname, plabel, _, _ in PARAM_DEFS:
        print(f"  {plabel:<20}: {best_params[pname]:.4f}")
    print(f"  {'Best ROI':<20}: {best_roi:+.1f}%")
    print("=" * 70)

    # Sim2 バックテスト
    simulate_daily_pnl(
        signals, y_val, payouts_val, dates_val,
        best_params, "PyTorch v5 (208dim selected)"
    )

    return best_params, best_roi


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='208次元 最強再学習パイプライン')
    parser.add_argument('--importance-threshold', type=float, default=0.0005,
                        help='特徴量選別の閾値 (default: 0.0005)')
    parser.add_argument('--skip-importance', action='store_true',
                        help='特徴量選別をスキップ (全208次元で学習)')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Optunaの探索回数')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  208次元 最強再学習パイプライン")
    print("  Step A: 特徴量選別 → Step B: 4モデル学習 → Step C: Optuna最適化")
    print("=" * 70)

    # --- DB 充足率チェック ---
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as t, COUNT(wind_speed) as w FROM races WHERE status='finished'")
        r = cur.fetchone()
        pct = r['w'] / r['t'] * 100
        logger.info(f"DB気象データ充足率: {r['w']:,}/{r['t']:,} ({pct:.1f}%)")
        if pct < 80:
            logger.warning(f"充足率が80%未満 ({pct:.1f}%)。収集完了を待つことを推奨。")
            resp = input("続行しますか? [y/N]: ")
            if resp.lower() != 'y':
                print("中断しました。")
                return

    # --- データロード ---
    X, y1, y2, y3, payouts, dates = load_data_208()
    if X is None or len(X) == 0:
        logger.error("データなし。終了。")
        return

    # 時系列分割 (80/20)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y1_train, y1_val = y1[:split], y1[split:]
    y2_train, y2_val = y2[:split], y2[split:]
    y3_train, y3_val = y3[:split], y3[split:]
    p_val = payouts[split:]
    dates_val = dates[split:]

    feature_names = get_feature_names_208()

    # --- Step A: 特徴量選別 ---
    if args.skip_importance:
        logger.info("Step A: スキップ (全208次元を使用)")
        selected_mask = np.ones(208, dtype=bool)
    else:
        selected_mask, report = analyze_feature_importance(
            X_train, y1_train, X_val, y1_val, feature_names,
            threshold=args.importance_threshold,
        )
        # レポート保存
        report_path = os.path.join(
            os.path.dirname(__file__), 'feature_importance_208_report.json'
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"  重要度レポート保存: {report_path}")

    # 選別適用
    X_train_sel = X_train[:, selected_mask]
    X_val_sel = X_val[:, selected_mask]
    selected_dim = selected_mask.sum()
    logger.info(f"\n  選別後次元: {selected_dim}")

    # 選別マスク保存 (predictor が使う)
    mask_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'feature_mask_208.npy'
    )
    np.save(mask_path, selected_mask)
    logger.info(f"  特徴量マスク保存: {mask_path}")

    # --- Step B: 4モデル再学習 ---
    logger.info(f"\nStep B: PyTorch 4モデル再学習 ({selected_dim}次元)")

    # 共通分割 (ランダム)
    indices = np.random.permutation(len(X_train_sel))
    t_split = int(len(indices) * 0.85)
    t_idx, v_idx = indices[:t_split], indices[t_split:]

    results = []
    for m in MODELS:
        vl, va = train_one_model(
            X_train_sel[t_idx], y1_train[t_idx], y2_train[t_idx], y3_train[t_idx],
            X_train_sel[v_idx], y1_train[v_idx], y2_train[v_idx], y3_train[v_idx],
            gamma=m['gamma'], save_path=m['path'], label=m['label'],
            epochs=args.epochs, patience=args.patience,
        )
        results.append({'path': m['path'], 'gamma': m['gamma'],
                        'label': m['label'], 'val_loss': vl, 'val_acc': va})

    print("\n" + "=" * 70)
    print("  Step B 結果: 4モデル")
    print("=" * 70)
    for r in results:
        print(f"  {r['path']}: gamma={r['gamma']} ({r['label']}), "
              f"val_loss={r['val_loss']:.4f}, acc={r['val_acc']:.1f}%")
    print("=" * 70)

    # --- Step C: Optuna ---
    best_params, best_roi = run_optuna_optimization(
        X_val_sel, y1_val, p_val, dates_val, n_trials=args.n_trials
    )

    # --- config 更新提案 ---
    print("\n" + "=" * 70)
    print("  betting_config.json 更新提案 (optuna戦略)")
    print("=" * 70)
    print(json.dumps({
        "kelly_fraction": round(best_params['kelly_fraction'], 3),
        "min_expected_value": round(best_params['min_ev'], 3),
        "min_odds": round(best_params['min_odds'], 1),
        "max_odds": round(best_params['max_odds'], 0),
        "max_boat1_prob": round(best_params['max_boat1_prob'], 3),
        "min_entropy": round(best_params['min_entropy'], 3),
        "min_probability": round(best_params['min_probability'], 3),
    }, indent=2, ensure_ascii=False))
    print("=" * 70)
    print("パイプライン完了。")


if __name__ == '__main__':
    main()
