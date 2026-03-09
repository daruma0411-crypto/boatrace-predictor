"""3連単オッズスクレイパー テスト

1. _decode_odds_position() の120通り一意性検証（オフライン）
2. ライブ接続テスト（本日の開催レースでオッズ取得確認）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import _decode_odds_position, scrape_odds_3t, _get_session
from datetime import date


def test_decode_uniqueness():
    """120通りのデコード結果が全て一意な3連単組み合わせであることを検証"""
    print("=" * 60)
    print("TEST: _decode_odds_position() 120通り一意性検証")
    print("=" * 60)

    combos = set()
    errors = []

    for pos in range(120):
        result = _decode_odds_position(pos)
        if result is None:
            errors.append(f"position {pos}: None returned")
            continue

        first, second, third = result

        # 艇番号の範囲チェック (1-6)
        for boat, label in [(first, '1着'), (second, '2着'), (third, '3着')]:
            if boat < 1 or boat > 6:
                errors.append(f"position {pos}: {label}={boat} 範囲外")

        # 同一艇チェック
        if len({first, second, third}) != 3:
            errors.append(f"position {pos}: 重複 ({first},{second},{third})")

        combo_key = f"{first}-{second}-{third}"
        if combo_key in combos:
            errors.append(f"position {pos}: 重複組み合わせ {combo_key}")
        combos.add(combo_key)

    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  {e}")
        return False

    # 120通り全てが生成されたか
    if len(combos) != 120:
        print(f"FAIL: {len(combos)}通り (120通り必要)")
        return False

    # 全6P3 = 120 通りが網羅されているか
    from itertools import permutations
    expected = set()
    for perm in permutations(range(1, 7), 3):
        expected.add(f"{perm[0]}-{perm[1]}-{perm[2]}")

    missing = expected - combos
    extra = combos - expected

    if missing:
        print(f"FAIL: 不足 {len(missing)}通り: {list(missing)[:5]}")
        return False
    if extra:
        print(f"FAIL: 余分 {len(extra)}通り: {list(extra)[:5]}")
        return False

    print(f"PASS: 120通り全て一意、6P3完全網羅")

    # サンプル出力（最初と最後の5通り）
    print("\n最初の5通り:")
    for pos in range(5):
        r = _decode_odds_position(pos)
        print(f"  position {pos:3d} → {r[0]}-{r[1]}-{r[2]}")
    print("最後の5通り:")
    for pos in range(115, 120):
        r = _decode_odds_position(pos)
        print(f"  position {pos:3d} → {r[0]}-{r[1]}-{r[2]}")

    return True


def test_live_scrape():
    """本日の開催レースでライブスクレイピングテスト"""
    print("\n" + "=" * 60)
    print("TEST: ライブスクレイピング（本日の開催レース）")
    print("=" * 60)

    today = date.today()
    session = _get_session()

    # 主要場をテスト (桐生=1, 戸田=2, 江戸川=3, 平和島=4, 多摩川=5, 浜名湖=6 ...)
    test_venues = list(range(1, 25))
    found = False

    for venue_id in test_venues:
        odds = scrape_odds_3t(session, today, venue_id, 1, max_retries=1)
        if odds:
            print(f"\n場{venue_id} R1: {len(odds)}通り取得成功")
            found = True

            # 上位5通り（低オッズ＝人気順）
            sorted_odds = sorted(odds.items(), key=lambda x: x[1])
            print("\n人気上位5通り:")
            for combo, val in sorted_odds[:5]:
                print(f"  {combo}: {val:.1f}倍")

            # 下位5通り（高オッズ＝穴順）
            print("穴上位5通り:")
            for combo, val in sorted_odds[-5:]:
                print(f"  {combo}: {val:.1f}倍")

            break

    if not found:
        print("本日の開催レースが見つかりませんでした（レース未開催 or 全場終了）")
        return False

    return True


if __name__ == '__main__':
    ok1 = test_decode_uniqueness()
    ok2 = test_live_scrape()

    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"  デコーダー一意性: {'PASS' if ok1 else 'FAIL'}")
    print(f"  ライブスクレイピング: {'PASS' if ok2 else 'FAIL/SKIP'}")

    sys.exit(0 if ok1 else 1)
