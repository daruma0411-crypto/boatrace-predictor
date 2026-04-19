"""Walk-Forward詳細分析 v2: 3フォールド × 6つの分析軸

Fold定義:
  Fold 1: Test = 2026-02-13 ~ 2026-03-14
  Fold 2: Test = 2026-01-29 ~ 2026-02-28
  Fold 3: Test = 2026-01-14 ~ 2026-02-13

各フォールドで「テスト開始日より前」の全データで訓練→テスト期間で評価。

分析項目:
  1. 予測分布 vs 実績分布
  2. 艇番別precision
  3. 会場タイプ別精度 (荒れ場/本命場/グレー場)
  4. 3戦略のROIシミュレーション (Top1 / Top3 / Kelly)
  5. キャリブレーション (確率デシル別ヒット率)
  6. 2-6号艇予測パフォーマンス

使い方:
  DATABASE_URL=xxx python scripts/walk_forward_detailed.py
"""
import sys
import os
import logging
import numpy as np
import torch
from collections import Counter, defaultdict
from datetime import date
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault(
    'DATABASE_URL',
    'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
)

from src.models import BoatraceMultiTaskModel, BoatraceMultiTaskLoss
from src.features import FeatureEngineer
from src.database import get_db_connection
from scripts.train_model import compute_class_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# 会場分類
# ============================================================
CHAOTIC_VENUES = {2, 3, 4, 5, 6, 14, 15, 20}    # 荒れ場
STABLE_VENUES = {8, 11, 18, 19, 21, 24}           # 本命場
# それ以外 = グレー場

# フォールド定義
FOLDS = [
    {'name': 'Fold 1', 'test_start': date(2026, 2, 13), 'test_end': date(2026, 3, 14)},
    {'name': 'Fold 2', 'test_start': date(2026, 1, 29), 'test_end': date(2026, 2, 28)},
    {'name': 'Fold 3', 'test_start': date(2026, 1, 14), 'test_end': date(2026, 2, 13)},
]


def load_all_data():
    """全データ(日付・venue_id・payout付き)をDB取得→特徴量変換"""
    fe = FeatureEngineer()

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.venue_id, r.race_date, r.race_number,
                   r.result_1st, r.result_2nd, r.result_3rd,
                   r.payout_sanrentan,
                   r.wind_speed, r.wind_direction, r.temperature,
                   r.wave_height, r.water_temperature
            FROM races r
            WHERE r.status = 'finished'
              AND r.result_1st IS NOT NULL
              AND r.wind_speed IS NOT NULL
              AND r.payout_sanrentan IS NOT NULL
            ORDER BY r.race_date, r.race_number
        """)
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

    X_list, y1_list, y2_list, y3_list = [], [], [], []
    dates_list, venues_list, payouts_list = [], [], []

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
            dates_list.append(race['race_date'])
            venues_list.append(race['venue_id'])
            payouts_list.append(race['payout_sanrentan'] or 0)
        except Exception:
            continue

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y1_list, dtype=np.int64),
        np.array(y2_list, dtype=np.int64),
        np.array(y3_list, dtype=np.int64),
        dates_list,
        np.array(venues_list, dtype=np.int32),
        np.array(payouts_list, dtype=np.float32),
    )


def train_model(X_train, y1_train, y2_train, y3_train,
                epochs=60, lr=0.0005, gamma=2.0):
    """モデル訓練 → (model, scaler) を返す"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    cw_1st = compute_class_weights(y1_train, smoothing=0.3)
    cw_2nd = compute_class_weights(y2_train, smoothing=0.7)
    cw_3rd = compute_class_weights(y3_train, smoothing=0.7)

    device = torch.device('cpu')
    input_dim = X_train.shape[1]
    hidden_dims = [512, 256, 128] if input_dim > 50 else [256, 128, 64]

    model = BoatraceMultiTaskModel(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.15
    ).to(device)
    criterion = BoatraceMultiTaskLoss(
        class_weights_1st=cw_1st, class_weights_2nd=cw_2nd,
        class_weights_3rd=cw_3rd, gamma=gamma, label_smoothing_1st=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.7
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_s),
        torch.LongTensor(y1_train),
        torch.LongTensor(y2_train),
        torch.LongTensor(y3_train),
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by1, by2, by3 in loader:
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, (by1, by2, by3))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    return model, scaler


def predict(model, scaler, X_test):
    """テストデータに対する予測確率を返す"""
    X_test_s = scaler.transform(X_test)
    x_t = torch.FloatTensor(X_test_s)
    with torch.no_grad():
        out_1st, out_2nd, out_3rd = model(x_t)
    probs_1st = torch.softmax(out_1st, dim=1).numpy()
    probs_2nd = torch.softmax(out_2nd, dim=1).numpy()
    probs_3rd = torch.softmax(out_3rd, dim=1).numpy()
    return probs_1st, probs_2nd, probs_3rd


def get_top_trifectas(probs_1st, probs_2nd, probs_3rd, idx, top_n=3):
    """1レースの上位N三連単を返す: [(p1,p2,p3,combined_prob), ...]"""
    results = []
    for p1 in range(6):
        prob1 = probs_1st[idx, p1]
        if prob1 < 0.01:
            continue
        for p2 in range(6):
            if p2 == p1:
                continue
            prob2 = probs_2nd[idx, p2]
            if prob2 < 0.01:
                continue
            for p3 in range(6):
                if p3 == p1 or p3 == p2:
                    continue
                prob3 = probs_3rd[idx, p3]
                combined = prob1 * prob2 * prob3
                results.append((p1, p2, p3, combined))
    results.sort(key=lambda x: x[3], reverse=True)
    return results[:top_n]


def analyze_fold(fold_info, X, y1, y2, y3, dates, venues, payouts):
    """1フォールドの詳細分析"""
    test_start = fold_info['test_start']
    test_end = fold_info['test_end']
    fold_name = fold_info['name']

    # マスク生成
    train_mask = np.array([d < test_start for d in dates])
    test_mask = np.array([test_start <= d <= test_end for d in dates])

    n_train = train_mask.sum()
    n_test = test_mask.sum()

    logger.info(f"\n{'='*60}")
    logger.info(f"{fold_name}: 訓練 ~{test_start} ({n_train:,}件), "
                f"テスト {test_start}~{test_end} ({n_test:,}件)")
    logger.info(f"{'='*60}")

    if n_train < 1000 or n_test < 100:
        logger.warning(f"データ不足: train={n_train}, test={n_test}")
        return None

    # 訓練
    model, scaler = train_model(
        X[train_mask], y1[train_mask], y2[train_mask], y3[train_mask],
        epochs=80
    )

    # 予測
    probs_1st, probs_2nd, probs_3rd = predict(model, scaler, X[test_mask])

    # テストデータ
    y1_test = y1[test_mask]
    y2_test = y2[test_mask]
    y3_test = y3[test_mask]
    venues_test = venues[test_mask]
    payouts_test = payouts[test_mask]

    pred_1st = probs_1st.argmax(axis=1)
    max_probs = probs_1st.max(axis=1)

    result = {
        'fold_name': fold_name,
        'n_train': int(n_train),
        'n_test': int(n_test),
        'test_period': f"{test_start} ~ {test_end}",
    }

    # ===== 1. 予測分布 vs 実績分布 =====
    pred_dist = Counter(pred_1st.tolist())
    actual_dist = Counter(y1_test.tolist())
    result['pred_dist'] = {i: pred_dist.get(i, 0) for i in range(6)}
    result['actual_dist'] = {i: actual_dist.get(i, 0) for i in range(6)}

    # ===== 2. 艇番別precision =====
    precision_by_boat = {}
    for b in range(6):
        mask_pred_b = (pred_1st == b)
        n_pred = mask_pred_b.sum()
        if n_pred == 0:
            precision_by_boat[b] = {'predicted': 0, 'correct': 0, 'precision': 0.0}
            continue
        correct = ((pred_1st == b) & (y1_test == b)).sum()
        precision_by_boat[b] = {
            'predicted': int(n_pred),
            'correct': int(correct),
            'precision': float(correct / n_pred * 100),
        }
    result['precision_by_boat'] = precision_by_boat

    # ===== 3. 会場タイプ別精度 =====
    venue_groups = {
        '荒れ場': CHAOTIC_VENUES,
        '本命場': STABLE_VENUES,
        'グレー場': None,  # その他
    }
    accuracy_by_venue = {}
    for group_name, venue_set in venue_groups.items():
        if venue_set is not None:
            mask_v = np.array([v in venue_set for v in venues_test])
        else:
            mask_v = np.array([
                v not in CHAOTIC_VENUES and v not in STABLE_VENUES
                for v in venues_test
            ])
        n_v = mask_v.sum()
        if n_v == 0:
            accuracy_by_venue[group_name] = {'n': 0, 'acc': 0.0}
            continue
        correct_v = (pred_1st[mask_v] == y1_test[mask_v]).sum()
        accuracy_by_venue[group_name] = {
            'n': int(n_v),
            'acc': float(correct_v / n_v * 100),
        }
    result['accuracy_by_venue'] = accuracy_by_venue

    # ===== 4. 3戦略ROIシミュレーション =====
    # 4a. Top1 (最有力三連単に100円)
    top1_bet = 0
    top1_payout = 0
    top1_hits = 0
    # 4b. Top3 (上位3つに各100円)
    top3_bet = 0
    top3_payout = 0
    top3_hits = 0
    # 4c. Kelly filtered
    kelly_bet = 0
    kelly_payout = 0
    kelly_hits = 0

    for idx in range(n_test):
        trifectas = get_top_trifectas(probs_1st, probs_2nd, probs_3rd, idx, top_n=3)

        actual_combo = (y1_test[idx], y2_test[idx], y3_test[idx])
        payout_val = payouts_test[idx]

        # 市場implied prob (payout = 100円返しベース)
        if payout_val > 0:
            market_prob = 100.0 / payout_val  # 三連単配当から逆算
        else:
            market_prob = 0.01

        # Top1
        if trifectas:
            top1_bet += 100
            t = trifectas[0]
            if (t[0], t[1], t[2]) == actual_combo:
                top1_payout += payout_val
                top1_hits += 1

        # Top3
        for rank, t in enumerate(trifectas):
            top3_bet += 100
            if (t[0], t[1], t[2]) == actual_combo:
                top3_payout += payout_val
                top3_hits += 1

        # Kelly filtered: Kelly > 0 = model_prob > market_prob
        if trifectas:
            t = trifectas[0]
            model_prob = t[3]
            # Kelly criterion: f = (p*b - 1) / (b - 1), where b = odds, p = model_prob
            if payout_val > 100:
                odds = payout_val / 100.0
                kelly_f = (model_prob * odds - 1) / (odds - 1)
                if kelly_f > 0:
                    kelly_bet += 100
                    if (t[0], t[1], t[2]) == actual_combo:
                        kelly_payout += payout_val
                        kelly_hits += 1

    result['roi_top1'] = {
        'bet': int(top1_bet), 'payout': int(top1_payout),
        'hits': top1_hits,
        'roi': float(top1_payout / top1_bet * 100) if top1_bet > 0 else 0.0,
    }
    result['roi_top3'] = {
        'bet': int(top3_bet), 'payout': int(top3_payout),
        'hits': top3_hits,
        'roi': float(top3_payout / top3_bet * 100) if top3_bet > 0 else 0.0,
    }
    result['roi_kelly'] = {
        'bet': int(kelly_bet), 'payout': int(kelly_payout),
        'hits': kelly_hits,
        'roi': float(kelly_payout / kelly_bet * 100) if kelly_bet > 0 else 0.0,
    }

    # ===== 5. キャリブレーション (確率デシル別ヒット率) =====
    calibration = {}
    for decile in range(10):
        lo = decile * 0.1
        hi = (decile + 1) * 0.1
        mask_d = (max_probs >= lo) & (max_probs < hi)
        n_d = mask_d.sum()
        if n_d == 0:
            calibration[f"{int(lo*100)}-{int(hi*100)}%"] = {
                'n': 0, 'avg_prob': 0.0, 'actual_hit': 0.0
            }
            continue
        hits_d = (pred_1st[mask_d] == y1_test[mask_d]).sum()
        avg_p = max_probs[mask_d].mean()
        calibration[f"{int(lo*100)}-{int(hi*100)}%"] = {
            'n': int(n_d),
            'avg_prob': float(avg_p * 100),
            'actual_hit': float(hits_d / n_d * 100),
        }
    result['calibration'] = calibration

    # ===== 6. 2-6号艇予測パフォーマンス =====
    # モデルが2-6号艇を1着と予測したケース
    non1_mask = pred_1st != 0  # pred != 1号艇(idx 0)
    n_non1 = non1_mask.sum()
    if n_non1 > 0:
        non1_correct = ((pred_1st[non1_mask]) == y1_test[non1_mask]).sum()
        non1_acc = float(non1_correct / n_non1 * 100)

        # 非1号艇予測のうち、三連単top1が当たった場合のpayout
        non1_bet = 0
        non1_payout = 0
        non1_hits = 0
        non1_payouts_when_hit = []

        for idx in range(n_test):
            if pred_1st[idx] == 0:  # 1号艇予測はスキップ
                continue
            trifectas = get_top_trifectas(probs_1st, probs_2nd, probs_3rd, idx, top_n=1)
            if not trifectas:
                continue
            t = trifectas[0]
            actual_combo = (y1_test[idx], y2_test[idx], y3_test[idx])
            non1_bet += 100
            if (t[0], t[1], t[2]) == actual_combo:
                non1_payout += payouts_test[idx]
                non1_hits += 1
                non1_payouts_when_hit.append(float(payouts_test[idx]))

        result['non1_performance'] = {
            'n_predicted': int(n_non1),
            'n_correct_1st': int(non1_correct),
            'accuracy_1st': non1_acc,
            'trifecta_bet': int(non1_bet),
            'trifecta_payout': int(non1_payout),
            'trifecta_hits': non1_hits,
            'trifecta_roi': float(non1_payout / non1_bet * 100) if non1_bet > 0 else 0.0,
            'avg_payout_when_hit': float(np.mean(non1_payouts_when_hit)) if non1_payouts_when_hit else 0.0,
        }
    else:
        result['non1_performance'] = {
            'n_predicted': 0, 'accuracy_1st': 0.0,
            'trifecta_roi': 0.0, 'avg_payout_when_hit': 0.0,
        }

    # 全体精度
    result['overall_acc_1st'] = float((pred_1st == y1_test).mean() * 100)

    return result


def print_results(results):
    """結果を整形テーブルで表示"""
    print("\n" + "=" * 80)
    print("  Walk-Forward 詳細分析レポート")
    print("=" * 80)

    for r in results:
        if r is None:
            continue

        fold = r['fold_name']
        period = r['test_period']
        print(f"\n{'─' * 80}")
        print(f"  {fold}  |  テスト期間: {period}")
        print(f"  訓練: {r['n_train']:,}件  |  テスト: {r['n_test']:,}件")
        print(f"  全体1着精度: {r['overall_acc_1st']:.1f}%")
        print(f"{'─' * 80}")

        # 1. 予測分布 vs 実績分布
        print(f"\n  [1] 予測分布 vs 実績分布")
        print(f"  {'艇番':>6}  {'予測':>8}  {'予測%':>8}  {'実績':>8}  {'実績%':>8}")
        print(f"  {'─' * 44}")
        for b in range(6):
            p_cnt = r['pred_dist'][b]
            a_cnt = r['actual_dist'][b]
            n = r['n_test']
            print(f"  {b+1}号艇  {p_cnt:>8}  {p_cnt/n*100:>7.1f}%  "
                  f"{a_cnt:>8}  {a_cnt/n*100:>7.1f}%")

        # 2. 艇番別precision
        print(f"\n  [2] 艇番別Precision (モデルがX号艇1着と予測→正解率)")
        print(f"  {'艇番':>6}  {'予測回数':>8}  {'正解':>6}  {'Precision':>10}")
        print(f"  {'─' * 36}")
        for b in range(6):
            p = r['precision_by_boat'][b]
            print(f"  {b+1}号艇  {p['predicted']:>8}  {p['correct']:>6}  "
                  f"{p['precision']:>9.1f}%")

        # 3. 会場タイプ別精度
        print(f"\n  [3] 会場タイプ別1着精度")
        print(f"  {'タイプ':>10}  {'レース数':>8}  {'精度':>8}")
        print(f"  {'─' * 30}")
        for group_name in ['荒れ場', '本命場', 'グレー場']:
            v = r['accuracy_by_venue'][group_name]
            print(f"  {group_name:>10}  {v['n']:>8}  {v['acc']:>7.1f}%")

        # 4. ROIシミュレーション
        print(f"\n  [4] ROIシミュレーション")
        print(f"  {'戦略':>10}  {'投資':>10}  {'回収':>10}  {'的中':>6}  {'ROI':>8}")
        print(f"  {'─' * 50}")
        for key, label in [('roi_top1', 'Top1'), ('roi_top3', 'Top3'),
                           ('roi_kelly', 'Kelly')]:
            s = r[key]
            print(f"  {label:>10}  {s['bet']:>9}円  {s['payout']:>9}円  "
                  f"{s['hits']:>6}  {s['roi']:>7.1f}%")

        # 5. キャリブレーション
        print(f"\n  [5] キャリブレーション (確率デシル別)")
        print(f"  {'デシル':>10}  {'件数':>6}  {'平均予測確率':>12}  {'実際ヒット率':>12}")
        print(f"  {'─' * 46}")
        for label, cal in r['calibration'].items():
            if cal['n'] > 0:
                print(f"  {label:>10}  {cal['n']:>6}  "
                      f"{cal['avg_prob']:>11.1f}%  {cal['actual_hit']:>11.1f}%")

        # 6. 2-6号艇パフォーマンス
        print(f"\n  [6] 2-6号艇予測パフォーマンス")
        np1 = r['non1_performance']
        print(f"  非1号艇予測回数: {np1.get('n_predicted', 0):,}")
        print(f"  1着精度 (非1号艇予測時): {np1.get('accuracy_1st', 0):.1f}%")
        print(f"  三連単投資: {np1.get('trifecta_bet', 0):,}円")
        print(f"  三連単回収: {np1.get('trifecta_payout', 0):,}円")
        print(f"  三連単的中: {np1.get('trifecta_hits', 0)}回")
        print(f"  三連単ROI: {np1.get('trifecta_roi', 0):.1f}%")
        print(f"  的中時平均配当: {np1.get('avg_payout_when_hit', 0):,.0f}円")

    # === 全フォールド集計 ===
    valid = [r for r in results if r is not None]
    if len(valid) > 1:
        print(f"\n{'=' * 80}")
        print("  全フォールド集計")
        print(f"{'=' * 80}")

        print(f"\n  {'フォールド':>12}  {'テスト期間':>26}  {'1着精度':>8}  "
              f"{'Top1 ROI':>10}  {'Top3 ROI':>10}  {'Kelly ROI':>10}")
        print(f"  {'─' * 82}")
        for r in valid:
            print(f"  {r['fold_name']:>12}  {r['test_period']:>26}  "
                  f"{r['overall_acc_1st']:>7.1f}%  "
                  f"{r['roi_top1']['roi']:>9.1f}%  "
                  f"{r['roi_top3']['roi']:>9.1f}%  "
                  f"{r['roi_kelly']['roi']:>9.1f}%")

        avg_acc = np.mean([r['overall_acc_1st'] for r in valid])
        avg_top1 = np.mean([r['roi_top1']['roi'] for r in valid])
        avg_top3 = np.mean([r['roi_top3']['roi'] for r in valid])
        avg_kelly = np.mean([r['roi_kelly']['roi'] for r in valid])
        print(f"  {'平均':>12}  {'':>26}  "
              f"{avg_acc:>7.1f}%  {avg_top1:>9.1f}%  "
              f"{avg_top3:>9.1f}%  {avg_kelly:>9.1f}%")

        # 全フォールド予測分布集計
        print(f"\n  全フォールド予測分布集計:")
        print(f"  {'艇番':>6}  {'予測%':>8}  {'実績%':>8}")
        print(f"  {'─' * 26}")
        total_n = sum(r['n_test'] for r in valid)
        for b in range(6):
            pred_total = sum(r['pred_dist'][b] for r in valid)
            actual_total = sum(r['actual_dist'][b] for r in valid)
            print(f"  {b+1}号艇  {pred_total/total_n*100:>7.1f}%  "
                  f"{actual_total/total_n*100:>7.1f}%")

        # 全フォールド会場タイプ別集計
        print(f"\n  全フォールド会場タイプ別精度:")
        print(f"  {'タイプ':>10}  {'レース数':>8}  {'精度':>8}")
        print(f"  {'─' * 30}")
        for group_name in ['荒れ場', '本命場', 'グレー場']:
            total_v = sum(r['accuracy_by_venue'][group_name]['n'] for r in valid)
            if total_v > 0:
                # weighted average
                weighted = sum(
                    r['accuracy_by_venue'][group_name]['n'] *
                    r['accuracy_by_venue'][group_name]['acc']
                    for r in valid
                ) / total_v
                print(f"  {group_name:>10}  {total_v:>8}  {weighted:>7.1f}%")

        # 全フォールド非1号艇集計
        print(f"\n  全フォールド2-6号艇パフォーマンス:")
        total_non1_pred = sum(r['non1_performance'].get('n_predicted', 0) for r in valid)
        total_non1_bet = sum(r['non1_performance'].get('trifecta_bet', 0) for r in valid)
        total_non1_pay = sum(r['non1_performance'].get('trifecta_payout', 0) for r in valid)
        total_non1_hits = sum(r['non1_performance'].get('trifecta_hits', 0) for r in valid)
        total_non1_correct = sum(r['non1_performance'].get('n_correct_1st', 0) for r in valid)
        print(f"  非1号艇予測回数合計: {total_non1_pred:,}")
        if total_non1_pred > 0:
            print(f"  1着精度: {total_non1_correct / total_non1_pred * 100:.1f}%")
        if total_non1_bet > 0:
            print(f"  三連単ROI: {total_non1_pay / total_non1_bet * 100:.1f}%")
        print(f"  三連単的中: {total_non1_hits}回")

    print(f"\n{'=' * 80}")
    print("  分析完了")
    print(f"{'=' * 80}\n")


def main():
    logger.info("=== Walk-Forward 詳細分析 ===")

    # データ読み込み
    X, y1, y2, y3, dates, venues, payouts = load_all_data()
    logger.info(f"データ: {len(X):,}件, 期間: {dates[0]} ~ {dates[-1]}")
    logger.info(f"特徴量次元: {X.shape[1]}")

    results = []
    for fold in FOLDS:
        result = analyze_fold(fold, X, y1, y2, y3, dates, venues, payouts)
        results.append(result)

    print_results(results)


if __name__ == '__main__':
    main()
