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
POLL_INTERVAL = 60

# 終了時刻（JST 23:00）
END_HOUR = 23

# 本番ベット金額: Kelly計算の金額をそのまま使用（100円単位に丸め）
USE_KELLY_AMOUNT = True


def get_pending_bets(strategy_types):
    """未購入かつ締切3分以内のベットを取得（複数戦略対応）

    Args:
        strategy_types: list[str] 購入対象の戦略名リスト

    Returns:
        list[dict]: bets + races情報
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT b.id, b.race_id, b.combination, b.amount, b.strategy_type,
                   b.odds, b.expected_value,
                   r.venue_id, r.race_number, r.deadline_time
            FROM bets b
            JOIN races r ON b.race_id = r.id
            LEFT JOIN purchase_log pl ON pl.bet_id = b.id
            WHERE pl.id IS NULL
              AND b.strategy_type = ANY(%s)
              AND b.created_at >= CURRENT_DATE
              AND r.deadline_time > NOW()
              AND r.deadline_time < NOW() + INTERVAL '3 minutes'
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


async def main_loop(strategy_types, dry_run):
    """メインポーリングループ（複数戦略対応）"""

    # 環境変数チェック
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
    logger.info(f"  ポーリング間隔: {POLL_INTERVAL}秒")
    logger.info(f"  終了時刻: {END_HOUR}:00")

    # テレボートログイン（リトライ付き）
    purchaser = TelebotPurchaser(member_id, pin, auth, dry_run=dry_run)
    await purchaser.start()

    login_ok = False
    for login_attempt in range(3):
        login_ok = await purchaser.login()
        if login_ok:
            break
        logger.warning(f"ログイン失敗 ({login_attempt + 1}/3)、60秒後にリトライ...")
        await purchaser.close()
        await asyncio.sleep(60)
        await purchaser.start()

    if not login_ok:
        logger.error("テレボートログイン3回失敗。終了します。")
        await purchaser.close()
        sys.exit(1)

    logger.info("テレボートログイン成功。ポーリング開始...")

    # 残高確認
    balance = await purchaser.get_balance()
    if balance is not None:
        logger.info(f"テレボート残高: ¥{balance:,}")

    try:
        while True:
            now = datetime.now()

            # 終了時刻チェック
            if now.hour >= END_HOUR:
                stats = get_today_stats(strategy_types)
                logger.info(f"=== {END_HOUR}:00 終了 ===")
                logger.info(f"  成功: {stats['success_count']}件")
                logger.info(f"  失敗: {stats['failed_count']}件")
                logger.info(f"  合計金額: ¥{stats['total_amount']:,}")
                break

            # 未購入ベット取得
            pending_bets = get_pending_bets(strategy_types)

            if pending_bets:
                # 残高チェック（入金忘れ対策: 残高不足なら購入スキップ）
                current_balance = await purchaser.get_balance()
                if current_balance is not None and current_balance <= 0:
                    logger.warning(f"残高不足 (¥0) — 入金待ち。{len(pending_bets)}件スキップ")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                logger.info(f"未購入ベット: {len(pending_bets)}件 (残高: ¥{current_balance:,})")

                for bet in pending_bets:
                    # RealDictCursor: dict形式で返る
                    bet_id = bet['id']
                    race_id = bet['race_id']
                    combination = bet['combination']
                    kelly_amount = int(bet['amount'])
                    strategy = bet['strategy_type']
                    venue_id = bet['venue_id']
                    race_number = bet['race_number']

                    # Kelly金額を100円単位に丸め（最低100円）
                    purchase_amount = max(100, (kelly_amount // 100) * 100)

                    logger.info(f"  購入実行: bet_id={bet_id} "
                                f"場{venue_id} {race_number}R {combination} "
                                f"¥{purchase_amount:,} (Kelly: ¥{kelly_amount:,})")

                    # テレボートで購入
                    result = await purchaser.purchase(
                        venue_id=venue_id,
                        race_number=race_number,
                        combination=combination,
                        amount=purchase_amount,
                    )

                    # 結果記録
                    status = 'success' if result['success'] else 'failed'
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

                    # LINE通知
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

                    logger.info(f"  結果: {status} - {result['message']}")

                    # 連続購入時の間隔
                    await asyncio.sleep(3)

            # ポーリング間隔
            await asyncio.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Ctrl+C で中断")
    finally:
        await purchaser.close()
        logger.info("テレボートブラウザ終了")


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
