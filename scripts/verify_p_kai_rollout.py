"""P改ロールアウトのデプロイ後検証 (read-only)。
実行: venv/Scripts/python.exe scripts/verify_p_kai_rollout.py ["YYYY-MM-DD HH:MM:SS"(デプロイ時刻)]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "ledger"))
from db import connect   # read-only 接続

DEFAULT_DEPLOY_AT = "2026-07-31 12:00:00"

def main():
    deploy_at = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEPLOY_AT
    conn = connect(); cur = conn.cursor()
    print(f"デプロイ時刻: {deploy_at} 以降を検証")
    # (a) v11_var13 の新規が止まっている
    cur.execute("SELECT count(*) n FROM predictions WHERE strategy_type='v11_var13' AND created_at > %s", (deploy_at,))
    n = cur.fetchone()["n"]
    print(("OK" if n == 0 else "NG"), f"(a) v11_var13 新規prediction={n} (0であること)")
    # (b) 新戦略が発火
    for s in ("mc_venue_all", "p_kai"):
        cur.execute("SELECT count(*) n, max(created_at) mx FROM predictions WHERE strategy_type=%s AND created_at > %s", (s, deploy_at))
        r = cur.fetchone()
        print(("OK" if r["n"] > 0 else "待ち(蓄積待ちなら数レース後に再確認)"), f"(b) {s} predictions={r['n']} 最新={r['mx']}")
    # (c) p_kai ⊂ P (同レース同comboで金額は等倍or半額[銀行丸め])
    cur.execute("""SELECT pk.race_id, pk.combination, pk.amount ka, p.amount pa
                   FROM bets pk JOIN bets p ON p.race_id=pk.race_id AND p.combination=pk.combination
                        AND p.strategy_type='mc_venue_focus'
                   WHERE pk.strategy_type='p_kai' AND pk.created_at > %s LIMIT 20""", (deploy_at,))
    rows = cur.fetchall()
    def half_round(pa):
        return round(float(pa) * 0.5 / 100) * 100
    bad = [r for r in rows if not (abs(float(r["ka"]) - float(r["pa"])) < 1
                                   or abs(float(r["ka"]) - half_round(r["pa"])) < 1)]
    print(("OK" if rows and not bad else ("待ち" if not rows else "NG")),
          f"(c) p_kai⊂P 金額整合 checked={len(rows)} bad={len(bad)}")
    # (d) race_softness が増えている
    cur.execute("SELECT to_regclass('public.race_softness') AS t")
    if cur.fetchone()["t"] is None:
        print("NG (d) race_softness テーブル未作成(未デプロイ/未発火/バックフィル未実行)")
    else:
        cur.execute("SELECT count(*) n, max(race_date) mx FROM race_softness")
        r = cur.fetchone()
        print(("OK" if r["n"] > 0 else "NG"), f"(d) race_softness rows={r['n']} 最新={r['mx']}")
    # (e) 既存Pが引き続き発火
    cur.execute("SELECT count(*) n FROM predictions WHERE strategy_type='mc_venue_focus' AND created_at > %s", (deploy_at,))
    print("INFO (e) P predictions after deploy =", cur.fetchone()["n"])
    print("---- 判定: NG が無く (b)(c) が OK になれば合格 (「待ち」は数レース処理後に再実行) ----")

if __name__ == "__main__":
    main()
