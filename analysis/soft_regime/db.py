import os
import psycopg2

DSN = os.environ.get(
    "BR_DATABASE_URL",
    "postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable",
)

# --- a-priori カレンダー定数（レース前に既知） ---
GW_START, GW_END = "2026-04-29", "2026-05-07"
SPLIT_DATE = "2026-05-16"        # test = この日以降（構築時に覗かない）
ARE_TRAIN_START = "2026-02-01"   # 波乱ターゲットのtrain開始
ROI_TRAIN_START = "2026-04-08"   # 戦略ベット開始
DATA_END = "2026-12-31"

VENUE_GOOD = (6, 13, 4)
VENUE_BAD = (23, 12)
STRATS = ("mc_venue_focus", "mc2_venue_focus")  # P, P2


def connect():
    conn = psycopg2.connect(DSN, connect_timeout=25)
    conn.set_session(readonly=True, autocommit=True)  # 書込みをサーバ側で禁止
    return conn


def fetch(conn, sql, params=None):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    cur.close()
    return cols, rows


def gw_flag(d):
    return GW_START <= str(d) <= GW_END


def classify_period(d):
    s = str(d)
    if s <= "2026-04-28":
        return "1_preGW"
    if s <= "2026-05-07":
        return "2_GW"
    if s <= "2026-06-13":
        return "3_dead"
    return "4_breakout"


def strategy_roi(conn, strat, date_from, date_to):
    sql = """
        select count(*) n,
               sum(b.amount) inv,
               sum(coalesce(b.return_amount, b.payout, 0) - b.amount) profit,
               sum(coalesce(b.return_amount, b.payout, 0)) ret
        from bets b join races r on b.race_id = r.id
        where b.strategy_type = %s and r.race_date between %s and %s
          and r.result_1st is not null
    """
    _, rows = fetch(conn, sql, (strat, date_from, date_to))
    n, inv, profit, ret = rows[0]
    inv = float(inv or 0); profit = float(profit or 0); ret = float(ret or 0)
    roi = (ret / inv * 100.0) if inv else 0.0
    return {"n": int(n or 0), "inv": inv, "profit": profit, "roi": round(roi, 1)}
