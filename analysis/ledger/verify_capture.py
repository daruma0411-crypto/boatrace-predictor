import sys
from db import connect


def main():
    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT to_regclass('public.race_odds_board') AS t")
    if cur.fetchone()["t"] is None:
        print("NG: race_odds_board テーブルが存在しない（未デプロイ/未発火）"); return 1

    cur.execute("""SELECT count(*) n, min(n_combos) mn, max(n_combos) mx,
                          min(captured_at)::date d0, max(captured_at)::date d1
                   FROM race_odds_board""")
    s = cur.fetchone()
    print(f"行数={s['n']} n_combos={s['mn']}〜{s['mx']} 期間={s['d0']}〜{s['d1']}")
    if s["n"] == 0:
        print("NG: 行が0（キャプチャ未発火＝サイレント失敗の疑い）"); return 1

    # 採用bet のオッズ と 盤 の一致クロスチェック（直近で bet がある盤レースを1件）
    cur.execute("""
        SELECT b.race_id, b.combination, b.odds bet_odds, (ob.odds_3t ->> b.combination)::float board_odds
        FROM bets b JOIN race_odds_board ob ON ob.race_id = b.race_id
        WHERE ob.odds_3t ? b.combination
        ORDER BY b.id DESC LIMIT 5
    """)
    rows = cur.fetchall()
    if not rows:
        print("注意: 盤とbetが揃うレースがまだ無い（蓄積待ち）。テーブル/行は健全。")
        return 0
    ok = True
    for r in rows:
        match = r["board_odds"] is not None and abs(r["bet_odds"] - r["board_odds"]) < 0.05 * max(1.0, r["bet_odds"])
        print(f"  race={r['race_id']} {r['combination']} bet={r['bet_odds']} board={r['board_odds']} {'OK' if match else 'MISMATCH'}")
        ok = ok and match
    print("PASS: 盤と採用betのオッズが一致" if ok else "NG: オッズ不一致あり")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
