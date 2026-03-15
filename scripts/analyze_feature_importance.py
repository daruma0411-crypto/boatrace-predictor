"""特徴量重要度分析スクリプト

PyTorchモデルに対して2つの手法で特徴量重要度を計測:
1. Permutation Importance: 各特徴量をシャッフルして精度劣化を測定（最も信頼性高い）
2. Gradient-based Importance: 入力に対する勾配の絶対値平均（高速）

出力: 上位30個・下位30個のランキング + 全208特徴量のCSV
"""
import sys
import os
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import BoatraceMultiTaskModel, load_model
from src.features import FeatureEngineerLegacy as FeatureEngineer, WIND_DIRECTIONS, PLAYER_CLASSES
from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- 特徴量名の定義 (208次元) ---
def build_feature_names():
    """208次元の特徴量名リストを構築"""
    names = []

    # グローバル16次元
    names.append("G_venue_id")
    names.append("G_month")
    names.append("G_distance")
    names.append("G_wind_speed")
    for wd in WIND_DIRECTIONS:
        names.append(f"G_wind_{wd}")
    names.append("G_temperature")
    names.append("G_wave_height")
    names.append("G_water_temperature")

    # 艇別32次元 × 6艇
    for boat in range(1, 7):
        prefix = f"B{boat}"
        for cls in PLAYER_CLASSES:
            names.append(f"{prefix}_class_{cls}")
        names.append(f"{prefix}_win_rate_rel")
        names.append(f"{prefix}_win_rate_2")
        names.append(f"{prefix}_win_rate_3")
        names.append(f"{prefix}_local_win_rate")
        names.append(f"{prefix}_local_win_rate_2")
        names.append(f"{prefix}_avg_st")
        names.append(f"{prefix}_inner_st_diff")
        names.append(f"{prefix}_motor_win_rate_2")
        names.append(f"{prefix}_motor_win_rate_3")
        names.append(f"{prefix}_boat_win_rate_2")
        names.append(f"{prefix}_is_new_motor")
        names.append(f"{prefix}_weight_diff")
        names.append(f"{prefix}_exhibition_time_diff")
        for c in range(1, 7):
            names.append(f"{prefix}_approach_{c}")
        names.append(f"{prefix}_fallback_flag")
        for c in range(1, 7):
            names.append(f"{prefix}_slot_{c}")
        names.append(f"{prefix}_tilt")
        names.append(f"{prefix}_parts_changed")

    assert len(names) == 208, f"Expected 208, got {len(names)}"
    return names


def load_eval_data(n_samples=3000):
    """評価用データをDBから取得"""
    feature_engineer = FeatureEngineer()
    cutoff = datetime.now() - timedelta(days=90)  # 直近3ヶ月

    logger.info("評価データ取得中...")
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.race_date >= %s AND r.status = 'finished'
              AND r.result_1st IS NOT NULL
            ORDER BY r.race_date DESC
            LIMIT %s
        """, (cutoff.date(), n_samples * 2))
        races = cur.fetchall()
        race_ids = [r['id'] for r in races]

        if not race_ids:
            raise ValueError("評価データなし")

        cur.execute("""
            SELECT race_id, boat_number, player_class,
                   win_rate, win_rate_2, win_rate_3,
                   local_win_rate, local_win_rate_2,
                   avg_st, motor_win_rate_2, motor_win_rate_3,
                   boat_win_rate_2, weight, exhibition_time,
                   is_new_motor, approach_course,
                   tilt, parts_changed
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        all_boats = cur.fetchall()

    # ボートデータをrace_id別に整理
    boats_by_race = {}
    for b in all_boats:
        rid = b['race_id']
        if rid not in boats_by_race:
            boats_by_race[rid] = []
        boats_by_race[rid].append({
            'boat_number': b['boat_number'], 'player_class': b['player_class'],
            'win_rate': b['win_rate'], 'win_rate_2': b['win_rate_2'],
            'win_rate_3': b['win_rate_3'],
            'local_win_rate': b['local_win_rate'],
            'local_win_rate_2': b['local_win_rate_2'],
            'avg_st': b['avg_st'], 'motor_win_rate_2': b['motor_win_rate_2'],
            'motor_win_rate_3': b['motor_win_rate_3'],
            'boat_win_rate_2': b['boat_win_rate_2'],
            'weight': b['weight'], 'exhibition_time': b['exhibition_time'],
            'is_new_motor': b['is_new_motor'], 'approach_course': b['approach_course'],
            'tilt': b['tilt'], 'parts_changed': b['parts_changed'],
        })

    X_list = []
    y1_list = []
    y2_list = []
    y3_list = []

    for race in races:
        rid = race['id']
        if rid not in boats_by_race:
            continue
        boats = boats_by_race[rid]
        if len(boats) < 6:
            continue

        race_data = {
            'venue_id': race['venue_id'],
            'month': race['race_date'].month if race['race_date'] else 6,
            'distance': 1800,
            'wind_speed': race['wind_speed'] or 0,
            'wind_direction': race['wind_direction'] or 'calm',
            'temperature': race['temperature'] or 20,
            'wave_height': race['wave_height'] or 0,
            'water_temperature': race['water_temperature'] or 20,
        }

        try:
            features = feature_engineer.transform(race_data, boats)
            X_list.append(features)
            y1_list.append(race['result_1st'] - 1)  # 1着 (0-indexed)
            y2_list.append(race['result_2nd'] - 1)  # 2着
            y3_list.append(race['result_3rd'] - 1)  # 3着
        except (ValueError, Exception):
            continue

        if len(X_list) >= n_samples:
            break

    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.int64)
    y2 = np.array(y2_list, dtype=np.int64)
    y3 = np.array(y3_list, dtype=np.int64)

    logger.info(f"評価データ: {X.shape[0]}件 x {X.shape[1]}次元")
    return X, y1, y2, y3


def compute_accuracy(model, X_tensor, y_tensor, device):
    """1着予測の正答率を計算"""
    model.eval()
    with torch.no_grad():
        out1, _, _ = model(X_tensor.to(device))
        preds = out1.argmax(dim=1).cpu().numpy()
    return (preds == y_tensor.numpy()).mean()


def compute_log_loss(model, X_tensor, y_tensor, device):
    """1着予測のlog lossを計算（低いほど良い）"""
    model.eval()
    with torch.no_grad():
        out1, _, _ = model(X_tensor.to(device))
        probs = torch.softmax(out1, dim=1).cpu().numpy()
    # クリップして数値安定化
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    n = len(y_tensor)
    ll = -np.mean([np.log(probs[i, y_tensor[i]]) for i in range(n)])
    return ll


def permutation_importance(model, X, y1, device, n_repeats=5, metric='log_loss'):
    """Permutation Importance: 各特徴量をシャッフルして精度劣化を測定

    metric: 'accuracy' or 'log_loss'
    """
    n_features = X.shape[1]
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y1)

    if metric == 'accuracy':
        baseline_score = compute_accuracy(model, X_tensor, y_tensor, device)
        logger.info(f"ベースライン精度: {baseline_score:.4f}")
    else:
        baseline_score = compute_log_loss(model, X_tensor, y_tensor, device)
        logger.info(f"ベースライン log_loss: {baseline_score:.4f}")

    importances = np.zeros(n_features)

    for feat_idx in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feat_idx])
            X_perm_tensor = torch.FloatTensor(X_permuted)

            if metric == 'accuracy':
                score = compute_accuracy(model, X_perm_tensor, y_tensor, device)
                scores.append(baseline_score - score)  # 正=重要
            else:
                score = compute_log_loss(model, X_perm_tensor, y_tensor, device)
                scores.append(score - baseline_score)  # 正=重要（lossが増加）

        importances[feat_idx] = np.mean(scores)

        if (feat_idx + 1) % 20 == 0:
            logger.info(f"  Permutation: {feat_idx + 1}/{n_features} 完了")

    return importances, baseline_score


def gradient_importance(model, X, device):
    """Gradient-based Importance: 入力に対する勾配の絶対値平均"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad_(True)

    out1, _, _ = model(X_tensor)
    # 各サンプルの最大確率クラスに対する勾配
    max_probs = out1.max(dim=1).values.sum()
    max_probs.backward()

    grads = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    return grads


def feature_variance_analysis(X, feature_names):
    """各特徴量の分散・ゼロ率を分析"""
    results = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        var = np.var(col)
        zero_rate = (col == 0).mean()
        mean = np.mean(col)
        std = np.std(col)
        results.append({
            'name': name,
            'mean': mean,
            'std': std,
            'variance': var,
            'zero_rate': zero_rate,
        })
    return results


def main():
    logger.info("=" * 60)
    logger.info("特徴量重要度分析 開始")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"デバイス: {device}")

    feature_names = build_feature_names()

    # モデル読み込み（メインモデル）
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'models', 'boatrace_model.pth'
    )
    logger.info(f"モデル読み込み: {model_path}")
    model = load_model(model_path, device)
    model.eval()

    # 評価データ
    X, y1, y2, y3 = load_eval_data(n_samples=3000)

    # ----- 1. Permutation Importance (log_loss) -----
    logger.info("\n--- Permutation Importance (log_loss) ---")
    perm_imp, baseline_ll = permutation_importance(
        model, X, y1, device, n_repeats=5, metric='log_loss'
    )

    # ----- 2. Gradient-based Importance -----
    logger.info("\n--- Gradient-based Importance ---")
    grad_imp = gradient_importance(model, X, device)

    # ----- 3. Feature Variance -----
    logger.info("\n--- Feature Variance Analysis ---")
    var_results = feature_variance_analysis(X, feature_names)

    # ----- 結果整理 -----
    # Permutation Importanceでソート
    perm_order = np.argsort(perm_imp)[::-1]
    grad_order = np.argsort(grad_imp)[::-1]

    # === レポート出力 ===
    print("\n" + "=" * 80)
    print("特徴量重要度分析レポート")
    print(f"評価データ: {X.shape[0]}件, ベースライン log_loss: {baseline_ll:.4f}")
    print("=" * 80)

    print("\n### Permutation Importance 上位30 (log_lossベース) ###")
    print(f"{'Rank':<5} {'Feature':<35} {'Importance':>12} {'Gradient':>12} {'Variance':>10} {'ZeroRate':>10}")
    print("-" * 90)
    for rank, idx in enumerate(perm_order[:30], 1):
        vr = var_results[idx]
        print(f"{rank:<5} {feature_names[idx]:<35} {perm_imp[idx]:>12.6f} {grad_imp[idx]:>12.6f} {vr['variance']:>10.6f} {vr['zero_rate']:>10.1%}")

    print(f"\n### Permutation Importance 下位30 (ノイズ候補) ###")
    print(f"{'Rank':<5} {'Feature':<35} {'Importance':>12} {'Gradient':>12} {'Variance':>10} {'ZeroRate':>10}")
    print("-" * 90)
    bottom_30 = perm_order[-30:][::-1]  # 最下位から
    for rank, idx in enumerate(bottom_30, 1):
        vr = var_results[idx]
        neg_marker = " ***" if perm_imp[idx] < 0 else ""
        print(f"{rank:<5} {feature_names[idx]:<35} {perm_imp[idx]:>12.6f} {grad_imp[idx]:>12.6f} {vr['variance']:>10.6f} {vr['zero_rate']:>10.1%}{neg_marker}")

    # 負の重要度（シャッフルした方が良い＝ノイズ）の件数
    negative_count = (perm_imp < 0).sum()
    near_zero = (np.abs(perm_imp) < 0.001).sum()
    print(f"\n### サマリー ###")
    print(f"  負の重要度（ノイズ）: {negative_count}個 / 208個")
    print(f"  ほぼゼロ (|imp| < 0.001): {near_zero}個 / 208個")
    print(f"  削除候補: {negative_count + near_zero}個")

    # カテゴリ別の重要度集計
    print(f"\n### カテゴリ別 平均重要度 ###")
    categories = {}
    for i, name in enumerate(feature_names):
        if name.startswith("G_"):
            cat = "Global"
        else:
            # B1_win_rate_rel → 特徴名
            parts = name.split("_", 1)
            cat = parts[1] if len(parts) > 1 else name
            # B1_ → boat-agnostic name
            cat = name[3:]  # Remove "B1_" etc.
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(perm_imp[i])

    # ユニーク特徴カテゴリ（6艇分を集約）
    boat_cats = {}
    for i, name in enumerate(feature_names):
        if name.startswith("G_"):
            key = name
        else:
            key = name[3:]  # B1_xxx → xxx
        if key not in boat_cats:
            boat_cats[key] = []
        boat_cats[key].append(perm_imp[i])

    cat_avg = {k: np.mean(v) for k, v in boat_cats.items()}
    for k, v in sorted(cat_avg.items(), key=lambda x: -x[1]):
        marker = " <-- NOISE" if v < 0 else ""
        print(f"  {k:<35} avg_imp={v:>10.6f}{marker}")

    # ----- CSV保存 -----
    csv_path = os.path.join(os.path.dirname(__file__), 'feature_importance_report.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("rank,feature,perm_importance,grad_importance,variance,zero_rate,mean,std\n")
        for rank, idx in enumerate(perm_order, 1):
            vr = var_results[idx]
            f.write(f"{rank},{feature_names[idx]},{perm_imp[idx]:.8f},{grad_imp[idx]:.8f},"
                    f"{vr['variance']:.8f},{vr['zero_rate']:.4f},{vr['mean']:.6f},{vr['std']:.6f}\n")
    logger.info(f"\nCSV保存: {csv_path}")

    print(f"\n### 推奨アクション ###")
    if negative_count > 30:
        print(f"  重度のノイズ汚染: {negative_count}個の特徴量がモデルを劣化させている可能性大")
        print(f"  → 負の重要度 + |imp| < 0.001 の特徴量を全て削除して再学習を推奨")
    elif negative_count > 10:
        print(f"  中程度のノイズ: {negative_count}個の特徴量が有害")
        print(f"  → 下位50個を段階的に削除して精度変化を検証")
    else:
        print(f"  ノイズは軽微: {negative_count}個のみ")
        print(f"  → 次元の呪いは特徴量以外の原因を疑う（モデル容量、学習データ量）")


if __name__ == '__main__':
    main()
