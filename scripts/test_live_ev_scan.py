"""Phase 2 ライブEVスキャン — フルパイプライン・ペーパートレード・デモ

データフロー:
  boatrace.jp → scraper → feature_engineer → PyTorch model
                                              ↓
  boatrace.jp/odds3t → RealtimeOddsProvider → EV = prob × odds
                                              ↓
                                        Kelly基準 → 推奨買い目
"""
import sys
import os
import time as _time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import torch
from datetime import date, datetime
from itertools import permutations

from src.scraper import _get_session, scrape_racelist, scrape_beforeinfo, scrape_race_deadlines
from src.odds_estimator import RealtimeOddsProvider
from src.features import FeatureEngineer
from src.models import load_model, BoatraceMultiTaskModel
from utils.timezone import now_jst

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── 場名マスタ ───
VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

# ─── ベッティング定数 ───
BANKROLL = 100_000  # 仮想資金 10万円
KELLY_FRACTION = 0.5  # ハーフケリー
MIN_BET = 100
MAX_BET = 5000
MAX_BET_RATIO = 0.05


def find_active_race(session, today):
    """締切時刻ベースで最適なレースを選択

    全開催場の締切時刻を取得し、現在時刻から15-20分以内のレースを優先選択。
    該当なしの場合は、最も直近の将来レースを選択してEVスキャン実行。
    """
    current = now_jst()
    print(f"本日の開催場を検索中... (現在 {current.strftime('%H:%M')})")

    # 全開催場の締切時刻を収集
    all_races = []  # (venue_id, race_number, deadline_dt, venue_name)

    for venue_id in range(1, 25):
        test = scrape_racelist(session, today, venue_id, 1)
        if not test:
            _time.sleep(0.3)
            continue

        venue_name = VENUE_NAMES.get(venue_id, f'場{venue_id}')
        print(f"  {venue_name}: 開催中")

        # 締切時刻を一括取得
        deadlines = scrape_race_deadlines(session, today, venue_id)
        _time.sleep(0.5)

        if not deadlines:
            print(f"    締切時刻取得失敗 → スキップ")
            continue

        for rno, dl_str in deadlines.items():
            try:
                h, m = dl_str.split(':')
                deadline_dt = current.replace(
                    hour=int(h), minute=int(m), second=0, microsecond=0,
                )
                all_races.append((venue_id, rno, deadline_dt, venue_name))
            except (ValueError, TypeError):
                pass

        print(f"    締切: R1={deadlines.get(1, '?')} 〜 R12={deadlines.get(12, '?')}")
        _time.sleep(0.3)

    if not all_races:
        print("開催中の場が見つかりませんでした。")
        return None, None, None

    # 締切まで15-20分以内のレースを抽出
    target_races = []
    for venue_id, rno, deadline_dt, venue_name in all_races:
        minutes_left = (deadline_dt - current).total_seconds() / 60
        if 15 <= minutes_left <= 20:
            target_races.append((venue_id, rno, deadline_dt, venue_name, minutes_left))

    # 15-20分以内のレースがある場合
    if target_races:
        # 最も締切が近い（minutes_left が小さい）ものを選択
        target_races.sort(key=lambda x: x[4])
        venue_id, rno, deadline_dt, venue_name, minutes_left = target_races[0]
        print(f"\n  ★ 締切直前レース発見: {venue_name} R{rno}")
        print(f"    締切: {deadline_dt.strftime('%H:%M')} (残り{minutes_left:.0f}分)")

        # オッズ取得
        odds_provider = RealtimeOddsProvider()
        odds = odds_provider.fetch_odds(today, venue_id, rno)
        if odds and len(odds) >= 60:
            print(f"    オッズ取得成功 ({len(odds)}通り)")
            return venue_id, rno, odds
        else:
            print(f"    オッズ取得失敗 → 他の候補を探索")

    # 15-20分以内がない or オッズ取得失敗 → 最も直近の将来レースを選択
    future_races = []
    for venue_id, rno, deadline_dt, venue_name in all_races:
        minutes_left = (deadline_dt - current).total_seconds() / 60
        if minutes_left > 0:
            future_races.append((venue_id, rno, deadline_dt, venue_name, minutes_left))

    if future_races:
        future_races.sort(key=lambda x: x[4])
        venue_id, rno, deadline_dt, venue_name, minutes_left = future_races[0]
        print(f"\n  → 最も直近のレース: {venue_name} R{rno}")
        print(f"    締切: {deadline_dt.strftime('%H:%M')} (残り{minutes_left:.0f}分)")

        if minutes_left > 20:
            print(f"    ※ 締切まで{minutes_left:.0f}分あります（理想は15-20分前）")
            print(f"    参考としてEVスキャンを実行します")

        odds_provider = RealtimeOddsProvider()
        odds = odds_provider.fetch_odds(today, venue_id, rno)
        if odds and len(odds) >= 60:
            print(f"    オッズ取得成功 ({len(odds)}通り)")
            return venue_id, rno, odds

    # 全レース終了 → 最終レースのオッズで参考スキャン
    past_races = sorted(all_races, key=lambda x: x[2], reverse=True)
    for venue_id, rno, deadline_dt, venue_name in past_races:
        odds_provider = RealtimeOddsProvider()
        odds = odds_provider.fetch_odds(today, venue_id, rno)
        if odds and len(odds) >= 60:
            minutes_ago = (current - deadline_dt).total_seconds() / 60
            print(f"\n  → {venue_name} R{rno} (締切{minutes_ago:.0f}分前 — 参考スキャン)")
            return venue_id, rno, odds

    print("オッズ取得可能なレースが見つかりませんでした。")
    return None, None, None


def build_boats_data(boats_raw):
    """scraper出力 → predictor/feature_engineer 用フォーマット変換"""
    boats_data = []
    for b in boats_raw:
        boats_data.append({
            'boat_number': b['boat_number'],
            'player_class': b.get('player_class'),
            'win_rate': b.get('win_rate'),
            'win_rate_2': b.get('win_rate_2'),
            'win_rate_3': b.get('win_rate_3'),
            'local_win_rate': b.get('local_win_rate'),
            'local_win_rate_2': b.get('local_win_rate_2'),
            'avg_st': b.get('avg_st'),
            'motor_win_rate_2': b.get('motor_win_rate_2'),
            'motor_win_rate_3': b.get('motor_win_rate_3'),
            'boat_win_rate_2': b.get('boat_win_rate_2'),
            'weight': b.get('weight'),
            'exhibition_time': None,
            'tilt': None,
            'parts_changed': False,
            'approach_course': b['boat_number'],  # 枠なり仮定
            'is_new_motor': False,
            'fallback_flag': True,
        })
    return boats_data


def calculate_sanrentan_probs(probs_1st, probs_2nd, probs_3rd):
    """条件付き確率で3連単120通りの確率を計算"""
    sanrentan = {}
    for combo in permutations(range(6), 3):
        i, j, k = combo
        p1 = probs_1st[i]
        if p1 <= 0:
            continue

        remaining_2nd = [probs_2nd[x] for x in range(6) if x != i]
        sum_r2 = sum(remaining_2nd)
        if sum_r2 <= 0:
            continue
        p2 = probs_2nd[j] / sum_r2

        remaining_3rd = [probs_3rd[x] for x in range(6) if x != i and x != j]
        sum_r3 = sum(remaining_3rd)
        if sum_r3 <= 0:
            continue
        p3 = probs_3rd[k] / sum_r3

        prob = p1 * p2 * p3
        if prob > 0:
            sanrentan[f"{i+1}-{j+1}-{k+1}"] = prob

    return sanrentan


def kelly_bet(prob, odds, bankroll):
    """ハーフケリー基準で推奨額を計算"""
    b = odds - 1.0
    if b <= 0 or prob <= 0:
        return 0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    if kelly <= 0:
        return 0
    amount = bankroll * kelly * KELLY_FRACTION
    max_bet = min(MAX_BET, bankroll * MAX_BET_RATIO)
    amount = max(MIN_BET, min(max_bet, amount))
    return int(round(amount / 100) * 100)


def main():
    today = now_jst().date()
    now = now_jst()
    session = _get_session()

    print("=" * 70)
    print(f"  Phase 2 ライブEVスキャン — ペーパートレード・デモ")
    print(f"  日付: {today}  時刻: {now.strftime('%H:%M:%S')}")
    print(f"  仮想資金: ¥{BANKROLL:,}")
    print("=" * 70)

    # ── Step 1: ターゲットレース選択 ──
    print("\n[Step 1] ターゲットレース選択")
    venue_id, race_number, realtime_odds = find_active_race(session, today)

    if venue_id is None:
        print("本日の開催レースが見つかりませんでした。")
        print("(全レース終了 or オッズ未発売)")
        sys.exit(0)

    venue_name = VENUE_NAMES.get(venue_id, f'場{venue_id}')
    print(f"\n  ターゲット: {venue_name} R{race_number}")

    # ── Step 2: 出走表 + 直前情報取得 → 特徴量生成 → AIモデル推論 ──
    print("\n[Step 2] 出走表 + 直前情報取得 → AIモデル推論")
    boats_raw = scrape_racelist(session, today, venue_id, race_number)
    if not boats_raw or len(boats_raw) != 6:
        print("  出走表取得失敗")
        sys.exit(1)

    # 直前情報取得（天候 + 展示タイム + チルト + 部品交換）
    beforeinfo = scrape_beforeinfo(session, today, venue_id, race_number)

    boats_data = build_boats_data(boats_raw)
    race_data = {
        'venue_id': venue_id,
        'month': today.month,
        'distance': 1800,
        'wind_speed': 0,
        'wind_direction': 'calm',
        'temperature': 20,
        'wave_height': 0,
        'water_temperature': 20,
    }

    # 直前情報をマージ
    if beforeinfo:
        weather = beforeinfo.get('weather', {})
        if weather.get('wind_speed') is not None:
            race_data['wind_speed'] = weather['wind_speed']
        if weather.get('wind_direction'):
            race_data['wind_direction'] = weather['wind_direction']
        if weather.get('temperature') is not None:
            race_data['temperature'] = weather['temperature']
        race_data['wave_height'] = weather.get('wave_height', 0) or 0
        race_data['water_temperature'] = weather.get('water_temperature', 20) or 20

        # 展示タイム・チルト・部品交換・進入コース・体重をboats_dataにマージ
        bi_boats = beforeinfo.get('boats', [])
        for bi_boat in bi_boats:
            for bd in boats_data:
                if bd['boat_number'] == bi_boat['boat_number']:
                    bd['exhibition_time'] = bi_boat.get('exhibition_time')
                    bd['tilt'] = bi_boat.get('tilt')
                    bd['parts_changed'] = bi_boat.get('parts_changed', False)
                    bd['approach_course'] = bi_boat.get(
                        'approach_course', bd['boat_number']
                    )
                    bd['fallback_flag'] = False
                    if bi_boat.get('weight') is not None:
                        bd['weight'] = bi_boat['weight']
                    break

        # 天候情報表示
        print(f"\n  天候: {weather.get('wind_direction', '?')}の風 "
              f"{weather.get('wind_speed', '?')}m  "
              f"波高{weather.get('wave_height', '?')}cm  "
              f"気温{weather.get('temperature', '?')}℃  "
              f"水温{weather.get('water_temperature', '?')}℃")
    else:
        print("\n  直前情報取得失敗 → デフォルト値使用")

    # 選手情報表示（展示タイム・チルト追加）
    print(f"\n  {'枠':>2}  {'選手名':<8} {'級':>2} {'勝率':>5} {'2連率':>6} "
          f"{'ST':>5} {'展示':>5} {'ﾁﾙﾄ':>5} {'部品':>4}")
    print("  " + "-" * 62)
    for bd in boats_data:
        name = (bd.get('player_name') or '???')[:8]
        # player_name is from boats_raw, need to look it up
        raw = next((b for b in boats_raw if b['boat_number'] == bd['boat_number']), {})
        name = (raw.get('player_name') or '???')[:8]
        cls = raw.get('player_class') or '??'
        wr = raw.get('win_rate') or 0.0
        w2 = raw.get('win_rate_2') or 0.0
        st = raw.get('avg_st') or 0.0
        ex = bd.get('exhibition_time')
        ex_str = f"{ex:>5.2f}" if ex else "  N/A"
        tilt = bd.get('tilt')
        tilt_str = f"{tilt:>+5.1f}" if tilt is not None else "  N/A"
        parts = "YES" if bd.get('parts_changed') else "-"
        print(f"  {bd['boat_number']:>2}  {name:<8} {cls:>2} {wr:>5.2f} "
              f"{w2:>5.1f}% {st:>5.2f} {ex_str} {tilt_str} {parts:>4}")

    # モデル推論
    fe = FeatureEngineer()
    try:
        model = load_model('models/boatrace_model.pth', torch.device('cpu'))
        # 次元チェック: 旧モデル(194次元) vs 新特徴量(208次元)
        if model.input_dim != fe.TOTAL_DIM:
            print(f"  モデル次元不一致 ({model.input_dim} != {fe.TOTAL_DIM}) "
                  f"→ 新ダミーモデル使用")
            model = BoatraceMultiTaskModel(input_dim=fe.TOTAL_DIM)
            model.eval()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  モデル読み込み不可({type(e).__name__}) → ダミーモデル使用")
        model = BoatraceMultiTaskModel(input_dim=fe.TOTAL_DIM)
        model.eval()

    features = fe.transform(race_data, boats_data)
    x = torch.FloatTensor(features).unsqueeze(0)
    with torch.no_grad():
        out_1st, out_2nd, out_3rd = model(x)

    probs_1st = torch.softmax(out_1st, dim=1).squeeze().numpy()
    probs_2nd = torch.softmax(out_2nd, dim=1).squeeze().numpy()
    probs_3rd = torch.softmax(out_3rd, dim=1).squeeze().numpy()

    print(f"\n  AIモデル 1着確率:")
    for i in range(6):
        bar = "█" * int(probs_1st[i] * 50)
        print(f"    {i+1}号艇: {probs_1st[i]*100:5.1f}%  {bar}")

    # ── Step 3: リアルタイムオッズ取得 ──
    print(f"\n[Step 3] リアルタイムオッズ（{len(realtime_odds)}通り取得済み）")
    sorted_market = sorted(realtime_odds.items(), key=lambda x: x[1])
    print(f"  人気上位3: ", end="")
    for c, o in sorted_market[:3]:
        print(f"{c}({o:.1f}倍) ", end="")
    print(f"\n  穴上位3:   ", end="")
    for c, o in sorted_market[-3:]:
        print(f"{c}({o:.1f}倍) ", end="")
    print()

    # ── Step 4: EV計算（全120通り） ──
    print(f"\n[Step 4] EV = 予測確率 × リアルオッズ（全120通り）")
    sanrentan_probs = calculate_sanrentan_probs(
        probs_1st.tolist(), probs_2nd.tolist(), probs_3rd.tolist()
    )

    ev_list = []
    for combo, prob in sanrentan_probs.items():
        odds = realtime_odds.get(combo, 0.0)
        if odds > 0 and prob > 0:
            ev = prob * odds
            ev_list.append({
                'combo': combo,
                'prob': prob,
                'odds': odds,
                'ev': ev,
            })

    ev_list.sort(key=lambda x: x['ev'], reverse=True)

    # EV分布サマリー
    evs = [e['ev'] for e in ev_list]
    print(f"  計算済み: {len(ev_list)}通り")
    print(f"  EV分布: min={min(evs):.3f}  median={np.median(evs):.3f}  "
          f"max={max(evs):.3f}")
    print(f"  EV > 1.0: {sum(1 for e in evs if e > 1.0)}通り")
    print(f"  EV > 1.2: {sum(1 for e in evs if e > 1.2)}通り")

    # ── Step 5: +EV ベット抽出 + ケリー基準 ──
    print(f"\n[Step 5] +EV 買い目 & ケリー基準（ハーフケリー）")
    print("=" * 70)

    positive_ev = [e for e in ev_list if e['ev'] > 1.0]

    if not positive_ev:
        print("\n  *** +EV買い目なし — このレースは見送り ***")
        print("\n  市場オッズがモデル確率に対して十分に高い組み合わせがありません。")
        print("  → 市場が正しく価格付けしている or モデルが自信を持てないレースです。")
        print("\n  上位5通りの参考データ:")
        print(f"  {'買い目':>8} {'確率':>7} {'オッズ':>7} {'EV':>6}")
        print("  " + "-" * 34)
        for e in ev_list[:5]:
            print(f"  {e['combo']:>8} {e['prob']*100:>6.2f}% {e['odds']:>6.1f}x "
                  f"{e['ev']:>5.3f}")
    else:
        print(f"\n  +EV買い目: {len(positive_ev)}通り")
        print()
        print(f"  {'#':>2} {'買い目':>8} {'予測確率':>8} {'市場ｵｯｽﾞ':>8} "
              f"{'EV':>6} {'ｹﾘｰf':>6} {'推奨額':>7} {'期待利益':>8}")
        print("  " + "-" * 68)

        total_bet = 0
        total_expected_profit = 0

        for idx, e in enumerate(positive_ev, 1):
            bet = kelly_bet(e['prob'], e['odds'], BANKROLL)
            kelly_f = 0
            b = e['odds'] - 1.0
            if b > 0:
                q = 1.0 - e['prob']
                kelly_f = (b * e['prob'] - q) / b * KELLY_FRACTION

            expected_profit = bet * (e['ev'] - 1.0)
            total_bet += bet
            total_expected_profit += expected_profit

            print(f"  {idx:>2} {e['combo']:>8} {e['prob']*100:>7.2f}% "
                  f"{e['odds']:>7.1f}x {e['ev']:>5.2f} "
                  f"{kelly_f:>5.1f}% ¥{bet:>6,} "
                  f"¥{expected_profit:>+7,.0f}")

        print("  " + "-" * 68)
        print(f"  {'合計':>11} {' '*24} "
              f"¥{total_bet:>6,} ¥{total_expected_profit:>+7,.0f}")
        print(f"\n  投資率: {total_bet/BANKROLL*100:.1f}% of ¥{BANKROLL:,}")
        avg_ev = np.mean([e['ev'] for e in positive_ev])
        print(f"  平均EV: {avg_ev:.3f}")
        print(f"  期待ROI: {total_expected_profit/total_bet*100:+.1f}%" if total_bet > 0 else "")

    # ── EV分布ヒストグラム（テキスト） ──
    print(f"\n{'─' * 70}")
    print(f"  EV分布ヒストグラム（全{len(ev_list)}通り）")
    print(f"{'─' * 70}")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8),
            (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 99)]
    for lo, hi in bins:
        count = sum(1 for e in evs if lo <= e < hi)
        label = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f"{lo:.1f}+"
        bar = "▓" * count
        marker = " ← +EV" if lo >= 1.0 and count > 0 else ""
        print(f"  {label:>8}: {count:>3} {bar}{marker}")

    print(f"\n{'=' * 70}")
    print(f"  デモ完了 — これはペーパートレードです（実際の購入は行いません）")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
