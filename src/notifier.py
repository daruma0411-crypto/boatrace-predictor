"""LINE通知モジュール - ベット結果をLINEに送信"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

# 会場コード→名前マッピング
VENUE_NAMES = {
    '01': '桐生', '02': '戸田', '03': '江戸川', '04': '平和島',
    '05': '多摩川', '06': '浜名湖', '07': '蒲郡', '08': '常滑',
    '09': '津', '10': '三国', '11': 'びわこ', '12': '住之江',
    '13': '尼崎', '14': '鳴門', '15': '丸亀', '16': '児島',
    '17': '宮島', '18': '徳山', '19': '下関', '20': '若松',
    '21': '芦屋', '22': '福岡', '23': '唐津', '24': '大村',
}


def _get_venue_name(venue_id):
    """会場IDから名前を取得"""
    vid = str(venue_id).zfill(2)
    return VENUE_NAMES.get(vid, f'場{venue_id}')


def send_line_bet_notification(venue_id, race_number, all_bets):
    """
    買い目が決定した際にLINEに通知を送る

    Args:
        venue_id: 会場ID
        race_number: レース番号
        all_bets: dict {strategy_type: [bet_dict, ...]}
    """
    token = os.environ.get("LINE_ACCESS_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")

    if not token or not user_id:
        logger.debug("LINE設定がないため通知をスキップ")
        return

    venue_name = _get_venue_name(venue_id)

    # メッセージ組み立て
    msg = f"🚤 【自動ベット】{venue_name} {race_number}R\n\n"
    total_amount = 0
    total_bets = 0

    for strategy_type, bets in all_bets.items():
        if not bets:
            continue
        msg += f"📊 {strategy_type}\n"
        for b in bets:
            combo = b.get('combination', '不明')
            amt = b.get('amount', 0)
            odds = b.get('odds', 0.0)
            ev = b.get('expected_value', 0.0)
            msg += f"  🎯 {combo} ¥{amt:,} (odds {odds:.1f} EV {ev:.2f})\n"
            total_amount += amt
            total_bets += 1
        msg += "\n"

    msg += f"💰 合計: {total_bets}点 ¥{total_amount:,}"

    # LINE Push API
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": msg}],
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"✅ LINE通知送信: {venue_name} {race_number}R ({total_bets}点)")
    except Exception as e:
        logger.warning(f"❌ LINE通知エラー: {e}")


def send_line_purchase_notification(venue_id, race_number, combination, amount,
                                    success, message=""):
    """テレボート購入結果をLINE通知

    Args:
        venue_id: 会場ID
        race_number: レース番号
        combination: 買い目 ("1-2-3")
        amount: 金額
        success: 購入成功か
        message: 追加メッセージ
    """
    token = os.environ.get("LINE_ACCESS_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")

    if not token or not user_id:
        logger.debug("LINE設定がないため通知をスキップ")
        return

    venue_name = _get_venue_name(venue_id)

    if success:
        msg = f"✅ 購入完了: {venue_name} {race_number}R 3連単 {combination} ¥{amount:,}"
    else:
        msg = f"❌ 購入失敗: {venue_name} {race_number}R 3連単 {combination} ({message})"

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": msg}],
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"✅ LINE購入通知送信: {venue_name} {race_number}R")
    except Exception as e:
        logger.warning(f"❌ LINE購入通知エラー: {e}")


def send_line_daily_summary(summary_text):
    """日次サマリーをLINEに送信"""
    token = os.environ.get("LINE_ACCESS_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")

    if not token or not user_id:
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": summary_text}],
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("✅ LINE日次サマリー送信完了")
    except Exception as e:
        logger.warning(f"❌ LINE日次サマリー送信エラー: {e}")
