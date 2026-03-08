"""レース結果収集 & ベット答え合わせ（scraper.py ベース）

レース終了後に実際の着順・払戻金をスクレイピングし、
bets テーブルの result / payout を UPDATE する。
スケジューラの23時以降ループから呼ばれる。
"""
import logging
from src.scraper import _get_session, scrape_result
from src.database import get_db_connection
from utils.timezone import now_jst

logger = logging.getLogger(__name__)


class ResultCollector:
    """レース結果を収集し、ベットの勝敗・払戻を確定する"""

    def __init__(self):
        self.session = _get_session()

    def settle_today(self):
        """本日の未確定ベットを全て精算する"""
        today = now_jst().date()
        unsettled = self._get_unsettled_bets(today)
        if not unsettled:
            logger.info("未確定ベットなし")
            return 0

        # (venue_id, race_number) の一意セットを取得
        race_keys = set()
        for bet in unsettled:
            race_keys.add((bet['venue_id'], bet['race_number']))

        settled_count = 0
        for venue_id, race_number in race_keys:
            try:
                result = scrape_result(self.session, today, venue_id, race_number)
            except Exception as e:
                logger.warning(
                    f"結果取得失敗: 場{venue_id} R{race_number}: {e}"
                )
                continue

            if not result:
                logger.warning(
                    f"結果データなし: 場{venue_id} R{race_number}"
                )
                continue

            # scrape_result は {result_1st, result_2nd, result_3rd, payout_sanrentan}
            winning_combo = (
                f"{result['result_1st']}-{result['result_2nd']}-{result['result_3rd']}"
            )
            payoff_per_100 = result['payout_sanrentan'] or 0

            # 着順を races テーブルにも書き込む
            self._save_race_result(
                today, venue_id, race_number, result, payoff_per_100
            )

            # 該当レースのベットを精算
            count = self._settle_race_bets(
                today, venue_id, race_number,
                winning_combo, payoff_per_100,
            )
            settled_count += count
            logger.info(
                f"精算完了: 場{venue_id} R{race_number} "
                f"(確定: {winning_combo}, 払戻: ¥{payoff_per_100:,}/100円) "
                f"{count}件更新"
            )

        logger.info(f"本日の精算合計: {settled_count}件")
        return settled_count

    def _get_unsettled_bets(self, race_date):
        """未精算のベットを取得"""
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT b.id, b.combination, b.amount, b.strategy_type,
                       r.venue_id, r.race_number
                FROM bets b
                JOIN races r ON b.race_id = r.id
                WHERE r.race_date = %s AND b.result IS NULL
            """, (race_date,))
            return cur.fetchall()

    def _settle_race_bets(self, race_date, venue_id, race_number,
                          winning_combo, payoff_per_100):
        """1レース分のベットを精算"""
        with get_db_connection() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT b.id, b.combination, b.amount
                FROM bets b
                JOIN races r ON b.race_id = r.id
                WHERE r.race_date = %s
                  AND r.venue_id = %s
                  AND r.race_number = %s
                  AND b.result IS NULL
            """, (race_date, venue_id, race_number))
            bets = cur.fetchall()

            count = 0
            for bet in bets:
                bet_combo = self._normalize_combo(bet['combination'])
                if bet_combo == winning_combo:
                    payout = int(bet['amount'] / 100 * payoff_per_100)
                    cur.execute("""
                        UPDATE bets SET result = 'win', payout = %s
                        WHERE id = %s
                    """, (payout, bet['id']))
                else:
                    cur.execute("""
                        UPDATE bets SET result = 'lose', payout = 0
                        WHERE id = %s
                    """, (bet['id'],))
                count += 1

            cur.execute("""
                UPDATE races SET status = 'settled'
                WHERE race_date = %s AND venue_id = %s AND race_number = %s
            """, (race_date, venue_id, race_number))

            return count

    def _save_race_result(self, race_date, venue_id, race_number,
                          result_data, payoff_per_100):
        """着順・払戻金を races テーブルに保存"""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE races
                    SET result_1st = %s, result_2nd = %s, result_3rd = %s,
                        payout_sanrentan = %s, status = 'finished'
                    WHERE race_date = %s AND venue_id = %s AND race_number = %s
                """, (
                    result_data['result_1st'],
                    result_data['result_2nd'],
                    result_data['result_3rd'],
                    payoff_per_100,
                    race_date, venue_id, race_number,
                ))
                logger.info(
                    f"着順保存: 場{venue_id} R{race_number} "
                    f"→ {result_data['result_1st']}-{result_data['result_2nd']}"
                    f"-{result_data['result_3rd']}"
                )
        except Exception as e:
            logger.warning(f"着順保存失敗: 場{venue_id} R{race_number}: {e}")

    def _normalize_combo(self, combo):
        """組み合わせ文字列を "1-2-3" 形式に正規化"""
        if not combo:
            return ''
        s = str(combo).strip()
        s = s.replace('=', '-')
        s = s.replace('−', '-').replace('ー', '-')
        if ' ' in s and '-' not in s:
            s = '-'.join(s.split())
        if len(s) == 3 and s.isdigit():
            s = f"{s[0]}-{s[1]}-{s[2]}"
        return s
