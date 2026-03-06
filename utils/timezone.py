"""タイムゾーンユーティリティ"""
import os
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

JST = pytz.timezone('Asia/Tokyo')


def _check_tz():
    tz = os.environ.get('TZ')
    if tz and tz != 'Asia/Tokyo':
        logger.warning(f"TZ環境変数が 'Asia/Tokyo' ではありません: {tz}")


_check_tz()


def now_jst():
    """現在の日本時間を返す"""
    return datetime.now(JST)


def to_jst(dt):
    """datetimeをJSTに変換"""
    if dt.tzinfo is None:
        return JST.localize(dt)
    return dt.astimezone(JST)


def format_jst(dt, fmt='%Y-%m-%d %H:%M:%S'):
    """JSTでフォーマット"""
    jst_dt = to_jst(dt)
    return jst_dt.strftime(fmt)
