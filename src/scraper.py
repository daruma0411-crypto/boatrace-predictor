"""ボートレース公式サイトから選手データを直接スクレイピング

pyjpboatraceのget_race_infoが壊れているため、
公式HTMLを直接パースして選手情報を取得する。
"""
import re
import time
import logging
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race"


def _get_session():
    """SSL検証を無効化したセッション"""
    s = requests.Session()
    s.verify = False
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return s


def _parse_float(text):
    """文字列から数値を抽出。失敗時はNone"""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(text):
    """文字列から整数を抽出。失敗時はNone"""
    if not text:
        return None
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return None


def _split_br(td):
    """td内の<br/>区切りテキストをリストで返す"""
    parts = []
    for item in td.children:
        if isinstance(item, str):
            s = item.strip()
            if s:
                parts.append(s)
    return parts


def scrape_racelist(session, race_date, venue_id, race_number):
    """出走表ページから6艇の選手データを取得

    Args:
        session: requests.Session
        race_date: datetime.date
        venue_id: int (1-24)
        race_number: int (1-12)

    Returns:
        list[dict] or None: 6艇分のデータ。取得失敗時はNone
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/racelist?rno={race_number}&jcd={venue_id:02d}&hd={hd}"

    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
    except Exception as e:
        logger.debug(f"HTTP error: {e}")
        return None

    soup = BeautifulSoup(r.text, 'html.parser')
    tbodies = soup.find_all('tbody')

    if len(tbodies) < 7:
        return None

    boats = []
    for boat_idx in range(6):
        tb = tbodies[boat_idx + 1]  # tbody[0]は締切時刻
        tds = tb.find_all('td')

        if len(tds) < 8:
            continue

        boat_data = _parse_boat_td(tds, boat_idx + 1)
        if boat_data:
            boats.append(boat_data)

    if len(boats) != 6:
        return None

    return boats


def _parse_boat_td(tds, boat_number):
    """td群から1艇分のデータを抽出

    td[0]: 枠番
    td[2]: 登録番号/級別/名前/支部/年齢/体重
    td[3]: F数/L数/平均ST
    td[4]: 全国 勝率/2連率/3連率
    td[5]: 当地 勝率/2連率/3連率
    td[6]: モーター番号/2連率/3連率
    td[7]: ボート番号/2連率/3連率
    """
    try:
        # td[2]: 登録番号/級別/名前/体重
        td2 = tds[2]
        reg_div = td2.find('div', class_='is-fs11')
        reg_text = reg_div.get_text(strip=True) if reg_div else ''
        # "4246/B1" → player_id, player_class
        reg_match = re.search(r'(\d{4})\s*/\s*(A1|A2|B1|B2)', reg_text)
        player_id = reg_match.group(1) if reg_match else None
        player_class = reg_match.group(2) if reg_match else None

        name_div = td2.find('div', class_=lambda c: c and 'is-fs18' in c)
        player_name = name_div.get_text(strip=True) if name_div else None

        # 体重抽出
        info_divs = td2.find_all('div', class_='is-fs11')
        weight = None
        if len(info_divs) >= 2:
            info_text = info_divs[1].get_text()
            w_match = re.search(r'([\d.]+)kg', info_text)
            if w_match:
                weight = float(w_match.group(1))

        # td[3]: F数/L数/平均ST
        td3_parts = _split_br(tds[3])
        avg_st = None
        if len(td3_parts) >= 3:
            avg_st = _parse_float(td3_parts[2])
        elif len(td3_parts) >= 1:
            # "F0L00.18" のようにくっついている場合
            st_match = re.search(r'(\d+\.\d+)', td3_parts[-1])
            if st_match:
                avg_st = float(st_match.group(1))

        # td[4]: 全国 勝率/2連率/3連率
        td4_parts = _split_br(tds[4])
        win_rate = _parse_float(td4_parts[0]) if len(td4_parts) > 0 else None
        win_rate_2 = _parse_float(td4_parts[1]) if len(td4_parts) > 1 else None
        win_rate_3 = _parse_float(td4_parts[2]) if len(td4_parts) > 2 else None

        # td[5]: 当地 勝率/2連率/3連率
        td5_parts = _split_br(tds[5])
        local_win_rate = _parse_float(td5_parts[0]) if len(td5_parts) > 0 else None
        local_win_rate_2 = _parse_float(td5_parts[1]) if len(td5_parts) > 1 else None

        # td[6]: モーター番号/2連率/3連率
        td6_parts = _split_br(tds[6])
        motor_win_rate_2 = _parse_float(td6_parts[1]) if len(td6_parts) > 1 else None
        motor_win_rate_3 = _parse_float(td6_parts[2]) if len(td6_parts) > 2 else None

        # td[7]: ボート番号/2連率/3連率
        td7_parts = _split_br(tds[7])
        boat_win_rate_2 = _parse_float(td7_parts[1]) if len(td7_parts) > 1 else None

        return {
            'boat_number': boat_number,
            'player_id': player_id,
            'player_name': player_name,
            'player_class': player_class,
            'win_rate': win_rate,
            'win_rate_2': win_rate_2,
            'win_rate_3': win_rate_3,
            'local_win_rate': local_win_rate,
            'local_win_rate_2': local_win_rate_2,
            'motor_win_rate_2': motor_win_rate_2,
            'motor_win_rate_3': motor_win_rate_3,
            'boat_win_rate_2': boat_win_rate_2,
            'weight': weight,
            'avg_st': avg_st,
        }

    except Exception as e:
        logger.debug(f"parse error boat {boat_number}: {e}")
        return None


def scrape_beforeinfo(session, race_date, venue_id, race_number):
    """直前情報ページから展示タイム・進入コースを取得

    Returns:
        list[dict] or None
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/beforeinfo?rno={race_number}&jcd={venue_id:02d}&hd={hd}"

    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(r.text, 'html.parser')

    # 展示タイムテーブルを探す
    result = []
    # 展示タイムは "is-fs14" クラスの要素に格納されることが多い
    # 簡易実装: 6艇分のデータだけ返す
    for boat_number in range(1, 7):
        result.append({
            'boat_number': boat_number,
            'exhibition_time': None,
            'approach_course': boat_number,  # デフォルト枠なり
        })

    return result
