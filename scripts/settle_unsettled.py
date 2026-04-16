"""未確定ベットを過去日付含めて一括精算する"""
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault(
    'DATABASE_URL',
    'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db'
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

from src.result_collector import ResultCollector
from src.database import get_db_connection

def main():
    # 未確定ベットの日付一覧を取得
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r.race_date
            FROM bets b
            JOIN races r ON b.race_id = r.id
            WHERE b.result IS NULL
            ORDER BY r.race_date
        """)
        dates = [row['race_date'] for row in cur.fetchall()]

    if not dates:
        print("未確定ベットなし")
        return

    print(f"未確定日付: {[str(d) for d in dates]}")

    collector = ResultCollector()

    total = 0
    for race_date in dates:
        # _get_unsettled_bets は today 引数を取るので、日付ごとに呼ぶ
        unsettled = collector._get_unsettled_bets(race_date)
        if not unsettled:
            continue

        race_keys = set()
        for bet in unsettled:
            race_keys.add((bet['venue_id'], bet['race_number']))

        print(f"\n--- {race_date} ({len(race_keys)}レース, {len(unsettled)}件) ---")

        from src.scraper import _get_session, scrape_result, scrape_odds_3t, scrape_odds_2t

        for venue_id, race_number in sorted(race_keys):
            try:
                result = scrape_result(collector.session, race_date, venue_id, race_number)
            except Exception as e:
                print(f"  結果取得失敗: 場{venue_id} R{race_number}: {e}")
                continue

            if not result:
                print(f"  結果データなし: 場{venue_id} R{race_number}")
                continue

            winning = f"{result['result_1st']}-{result['result_2nd']}-{result['result_3rd']}"
            payoff_3t = result['payout_sanrentan'] or 0

            winning_2t = f"{result['result_1st']}-{result['result_2nd']}"
            payoff_2t = result.get('payout_nirentan', 0) or 0

            closing_3t = {}
            closing_2t = {}
            try:
                closing_3t = scrape_odds_3t(collector.session, race_date, venue_id, race_number) or {}
                closing_2t = scrape_odds_2t(collector.session, race_date, venue_id, race_number) or {}
            except Exception:
                pass

            collector._save_race_result(race_date, venue_id, race_number, result, payoff_3t)

            count = collector._settle_race_bets(
                race_date, venue_id, race_number,
                winning, payoff_3t,
                winning_combo_2t=winning_2t,
                payoff_2t=payoff_2t,
                closing_odds_3t=closing_3t,
                closing_odds_2t=closing_2t,
            )
            total += count
            print(f"  場{venue_id} R{race_number}: {winning} (払戻{payoff_3t}) → {count}件精算")

    print(f"\n合計 {total}件 精算完了")


if __name__ == '__main__':
    main()
