"""テレボート自動購入ボット（ローカルPC常駐）

60秒間隔でDBをポーリングし、未購入ベットをテレボートで購入する。
本番運用では100円/ベット固定（Kelly金額は無視）。

使い方:
    python scripts/teleboat_purchaser.py [--dry-run] [--strategy mc_quarter_kelly]

環境変数（.envに設定、gitignore対象）:
    TELEBOAT_MEMBER_ID  加入者番号
    TELEBOAT_PIN        暗証番号
    TELEBOAT_AUTH       認証番号
    DATABASE_URL        PostgreSQL接続文字列
    LINE_ACCESS_TOKEN   LINE通知用トークン
    LINE_USER_ID        LINE通知先ユーザーID
"""
import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.database import get_db_connection
from src.teleboat import TelebotPurchaser
from src.notifier import send_line_purchase_notification

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('teleboat_purchaser')

# ポーリング間隔（秒）
# 2026-04-24: 45→20秒に短縮。DEADLINE_WINDOW_MIN=3 の狭い窓内で retry を収めるため。
POLL_INTERVAL = 20

# 終了時刻（JST 23:00）
END_HOUR = 23

# 締切ウィンドウ（分）: 締切この時間以内のbetを処理対象に。
# 設計意図: scheduler が LEAD_TIME 1.5〜3分前に予測＆bet生成するため、
# この窓も 3分にして「予測時と実売時の EV 乖離」を最小化する（オッズは
# 締切直前に激変するため）。
# 2026-04-23 に10分に広げたが、retry時にオッズが大きくズレるため3分に戻した。
DEADLINE_WINDOW_MIN = 3

# 同一betの最大再試行回数（purchase_logにfailedが入ってても締切前ならこの回数までリトライ）
MAX_RETRY_PER_BET = 3

# 本番ベット金額: Kelly計算の金額をそのまま使用（100円単位に丸め）
USE_KELLY_AMOUNT = True


def get_pending_bets(strategy_types):
    """未購入または未成功の締切近傍ベットを取得（複数戦略対応）

    取得条件:
      - 当該戦略の本日作成bet
      - 締切が今から {DEADLINE_WINDOW_MIN} 分以内で未来
      - purchase_log に成功記録が無い
      - purchase_log の失敗回数が {MAX_RETRY_PER_BET} 未満

    Returns:
        list[dict]: bets + races情報 + 過去試行回数
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT b.id, b.race_id, b.combination, b.amount, b.strategy_type,
                   b.odds, b.expected_value,
                   r.venue_id, r.race_number, r.deadline_time,
                   COALESCE(pl_stats.fail_count, 0) as fail_count,
                   COALESCE(pl_stats.has_success, false) as has_success
            FROM bets b
            JOIN races r ON b.race_id = r.id
            LEFT JOIN (
                SELECT bet_id,
                       COUNT(*) FILTER (WHERE status != 'success') as fail_count,
                       BOOL_OR(status = 'success') as has_success
                FROM purchase_log
                GROUP BY bet_id
            ) pl_stats ON pl_stats.bet_id = b.id
            WHERE b.strategy_type = ANY(%s)
              AND b.created_at >= CURRENT_DATE
              AND r.deadline_time > NOW()
              AND r.deadline_time < NOW() + INTERVAL '{DEADLINE_WINDOW_MIN} minutes'
              AND COALESCE(pl_stats.has_success, false) = false
              AND COALESCE(pl_stats.fail_count, 0) < {MAX_RETRY_PER_BET}
            ORDER BY r.deadline_time ASC, b.strategy_type ASC
        """, (strategy_types,))
        return cur.fetchall()


def record_purchase(bet_id, race_id, strategy_type, combination, amount,
                    status, error_message=None, screenshot_path=None):
    """購入結果をpurchase_logに記録"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO purchase_log
            (bet_id, race_id, strategy_type, combination, amount,
             status, error_message, screenshot_path, purchased_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'success' THEN NOW() ELSE NULL END)
        """, (bet_id, race_id, strategy_type, combination, amount,
              status, error_message, screenshot_path, status))


def get_today_stats(strategy_types):
    """当日の購入統計（複数戦略対応）"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'success') as success_count,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_count,
                COALESCE(SUM(amount) FILTER (WHERE status = 'success'), 0) as total_amount
            FROM purchase_log
            WHERE strategy_type = ANY(%s)
              AND created_at >= CURRENT_DATE
        """, (strategy_types,))
        row = cur.fetchone()
        return {
            'success_count': row['success_count'],
            'failed_count': row['failed_count'],
            'total_amount': row['total_amount'],
        }


async def _login_with_retry(purchaser, max_attempts=5):
    """ログインを指数バックオフでリトライ。

    Returns:
        bool: 最終的にログイン成功したか
    """
    for attempt in range(1, max_attempts + 1):
        try:
            ok = await purchaser.login()
            if ok:
                return True
        except Exception as e:
            logger.warning(f"ログイン例外 ({attempt}/{max_attempts}): {e}")
        wait = min(30 * attempt, 180)
        logger.warning(f"ログイン失敗 ({attempt}/{max_attempts})、{wait}秒後にリトライ...")
        try:
            await purchaser.close()
        except Exception:
            pass
        await asyncio.sleep(wait)
        try:
            await purchaser.start()
        except Exception as e:
            logger.error(f"ブラウザ再起動失敗: {e}")
    return False


async def _process_one_bet(purchaser, bet):
    """1件のベットを購入しDB/LINEに記録。

    Returns:
        bool: 購入試行が完了したか（成功・失敗に関わらずrecord済み）
    """
    bet_id = bet['id']
    race_id = bet['race_id']
    combination = bet['combination']
    kelly_amount = int(bet['amount'])
    strategy = bet['strategy_type']
    venue_id = bet['venue_id']
    race_number = bet['race_number']
    fail_count = bet.get('fail_count', 0)

    purchase_amount = max(100, (kelly_amount // 100) * 100)

    retry_tag = f" (再試行{fail_count + 1}回目)" if fail_count > 0 else ""
    logger.info(f"  購入実行: bet_id={bet_id}{retry_tag} "
                f"場{venue_id} {race_number}R {combination} "
                f"¥{purchase_amount:,} (Kelly: ¥{kelly_amount:,})")

    try:
        result = await purchaser.purchase(
            venue_id=venue_id,
            race_number=race_number,
            combination=combination,
            amount=purchase_amount,
        )
    except Exception as e:
        logger.error(f"  購入中に例外: {e}")
        result = {"success": False, "message": f"exception: {e}", "screenshot": ""}

    status = 'success' if result['success'] else 'failed'
    try:
        record_purchase(
            bet_id=bet_id,
            race_id=race_id,
            strategy_type=strategy,
            combination=combination,
            amount=purchase_amount,
            status=status,
            error_message=result.get('message', '') if not result['success'] else None,
            screenshot_path=result.get('screenshot', ''),
        )
    except Exception as e:
        logger.error(f"  purchase_log書き込み失敗: {e}")

    try:
        send_line_purchase_notification(
            venue_id=venue_id,
            race_number=race_number,
            combination=combination,
            amount=purchase_amount,
            success=result['success'],
            message=result.get('message', ''),
        )
    except Exception as e:
        logger.warning(f"  LINE通知失敗: {e}")

    logger.info(f"  結果: {status} - {result.get('message', '')}")
    return True


async def _run_session(strategy_types, dry_run, member_id, pin, auth):
    """1セッション: ログイン → 終了時刻までポーリング → close。

    例外が発生した場合は呼び出し元(main_loop)でキャッチして再起動する。
    """
    purchaser = TelebotPurchaser(member_id, pin, auth, dry_run=dry_run)
    try:
        await purchaser.start()

        if not await _login_with_retry(purchaser, max_attempts=5):
            logger.error("テレボートログイン5回失敗。セッション終了します。")
            return

        logger.info("テレボートログイン成功。ポーリング開始...")

        balance = await purchaser.get_balance()
        if balance is not None:
            logger.info(f"テレボート残高: ¥{balance:,}")

        consecutive_errors = 0

        while True:
            now = datetime.now()

            if now.hour >= END_HOUR:
                stats = get_today_stats(strategy_types)
                logger.info(f"=== {END_HOUR}:00 終了 ===")
                logger.info(f"  成功: {stats['success_count']}件 / "
                            f"失敗: {stats['failed_count']}件 / "
                            f"合計金額: ¥{stats['total_amount']:,}")
                return

            try:
                pending_bets = get_pending_bets(strategy_types)
            except Exception as e:
                logger.error(f"get_pending_bets失敗: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    logger.error("DB接続エラー5回連続 → セッション再起動")
                    return
                await asyncio.sleep(POLL_INTERVAL)
                continue
            consecutive_errors = 0

            if pending_bets:
                try:
                    current_balance = await purchaser.get_balance()
                except Exception as e:
                    logger.warning(f"残高取得失敗: {e} → セッション再起動")
                    return

                if current_balance is not None and current_balance <= 0:
                    logger.warning(f"残高不足 (¥0) — 入金待ち。{len(pending_bets)}件スキップ")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                bal_str = f"¥{current_balance:,}" if current_balance is not None else "取得失敗(処理続行)"
                logger.info(f"未購入/未成功ベット: {len(pending_bets)}件 (残高: {bal_str})")

                for bet in pending_bets:
                    await _process_one_bet(purchaser, bet)
                    await asyncio.sleep(3)

            await asyncio.sleep(POLL_INTERVAL)

    finally:
        try:
            await purchaser.close()
        except Exception:
            pass
        logger.info("セッションclose完了")


async def main_loop(strategy_types, dry_run):
    """外側ループ: セッションがクラッシュしても END_HOUR まで再起動し続ける。"""

    member_id = os.environ.get("TELEBOAT_MEMBER_ID")
    pin = os.environ.get("TELEBOAT_PIN")
    auth = os.environ.get("TELEBOAT_AUTH")

    if not all([member_id, pin, auth]):
        logger.error("TELEBOAT_MEMBER_ID, TELEBOAT_PIN, TELEBOAT_AUTH を .env に設定してください")
        sys.exit(1)

    logger.info("=== テレボート自動購入ボット起動 ===")
    logger.info(f"  戦略: {','.join(strategy_types)} ({len(strategy_types)}個)")
    logger.info(f"  ベット金額: Kelly計算額（100円単位丸め）")
    logger.info(f"  DRY_RUN: {dry_run}")
    logger.info(f"  ポーリング間隔: {POLL_INTERVAL}秒 / 締切ウィンドウ: {DEADLINE_WINDOW_MIN}分 / "
                f"再試行上限: {MAX_RETRY_PER_BET}回")
    logger.info(f"  終了時刻: {END_HOUR}:00")

    attempt = 0
    while True:
        attempt += 1
        now = datetime.now()
        if now.hour >= END_HOUR:
            logger.info(f"=== {END_HOUR}:00 到達 メインループ終了 ===")
            return

        logger.info(f"=== セッション attempt={attempt} 開始 ===")
        try:
            await _run_session(strategy_types, dry_run, member_id, pin, auth)
            # 正常returnした場合もEND_HOUR判定して再起動
            logger.info(f"セッション attempt={attempt} 正常終了")
        except KeyboardInterrupt:
            logger.info("Ctrl+C で中断")
            return
        except Exception as e:
            logger.error(f"セッション attempt={attempt} で例外: {e}", exc_info=True)

        # セッション間の待機（再ログインラッシュを防ぐ）
        now = datetime.now()
        if now.hour >= END_HOUR:
            return
        wait = min(30 * attempt, 300)
        logger.info(f"次のセッションまで {wait}秒 待機...")
        await asyncio.sleep(wait)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='テレボート自動購入ボット')
    parser.add_argument('--dry-run', action='store_true',
                        help='購入確定せず確認画面まで（テスト用）')
    parser.add_argument('--strategy', default=None,
                        help='購入対象戦略、カンマ区切りで複数可 (例: mc_early_race,mc_venue_focus)')
    args = parser.parse_args()

    # 戦略の決定（引数 > 環境変数 > デフォルト）。カンマ区切りを list 化。
    strategy_env = args.strategy or os.environ.get("TELEBOAT_STRATEGY", "mc_early_race")
    strategy_types = [s.strip() for s in strategy_env.split(',') if s.strip()]

    # DRY_RUNの決定（引数 > 環境変数）
    dry_run = args.dry_run or os.environ.get("TELEBOAT_DRY_RUN", "false").lower() == "true"

    asyncio.run(main_loop(strategy_types, dry_run))
