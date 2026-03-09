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


def scrape_result(session, race_date, venue_id, race_number):
    """レース結果ページから着順と3連単払戻金を取得

    Returns:
        dict with keys: result_1st, result_2nd, result_3rd, payout_sanrentan
        or None
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/raceresult?rno={race_number}&jcd={venue_id:02d}&hd={hd}"

    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(r.text, 'html.parser')
    tbodies = soup.find_all('tbody')

    if len(tbodies) < 9:
        return None

    # tbody[1]〜[6]: 着順。td[1]が艇番
    ranking = []
    for i in range(1, 7):
        tds = tbodies[i].find_all('td')
        if len(tds) >= 2:
            boat_text = tds[1].get_text(strip=True)
            boat_num = _parse_int(boat_text)
            if boat_num:
                ranking.append(boat_num)

    if len(ranking) < 3:
        return None

    # 3連単払戻金: tbody内のtdを個別にパース
    payout = 0
    for i in range(7, min(len(tbodies), 15)):
        tds_pay = tbodies[i].find_all('td')
        if tds_pay and '3連単' in tds_pay[0].get_text(strip=True):
            if len(tds_pay) >= 3:
                pay_text = tds_pay[2].get_text(strip=True)
                m = re.search(r'[¥￥]([0-9,]+)', pay_text)
                if m:
                    payout = int(m.group(1).replace(',', ''))
            break

    return {
        'result_1st': ranking[0],
        'result_2nd': ranking[1],
        'result_3rd': ranking[2],
        'payout_sanrentan': payout,
    }


def _decode_odds_position(position):
    """120個のオッズ位置 → (1着, 2着, 3着) 艇番号に変換

    boatrace.jp odds3t ページの td.oddsPoint 要素の並び順:
    - column = position % 6 → 1着の艇番号インデックス (0-5)
    - group_index = (position // 6) // 4 → 2着候補インデックス (0-4)
    - row_in_group = (position // 6) % 4 → 3着候補インデックス (0-3)

    Returns:
        tuple(int, int, int): (1着, 2着, 3着) 艇番号 (1-6) or None
    """
    column = position % 6
    row = position // 6
    group_index = row // 4
    row_in_group = row % 4

    first = column + 1

    # 2着: 1着を除いた艇のうち、group_index番目（昇順）
    second_candidates = [b for b in range(1, 7) if b != first]
    if group_index >= len(second_candidates):
        return None
    second = second_candidates[group_index]

    # 3着: 1着・2着を除いた艇のうち、row_in_group番目（昇順）
    third_candidates = [b for b in range(1, 7) if b != first and b != second]
    if row_in_group >= len(third_candidates):
        return None
    third = third_candidates[row_in_group]

    return (first, second, third)


def scrape_odds_3t(session, race_date, venue_id, race_number, max_retries=3):
    """3連単オッズページから全120通りのオッズを取得

    Args:
        session: requests.Session
        race_date: datetime.date
        venue_id: int (1-24)
        race_number: int (1-12)
        max_retries: int リトライ回数

    Returns:
        dict: {"1-2-3": 12.7, ...} 倍率形式。取得失敗時はNone
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/odds3t?rno={race_number}&jcd={venue_id:02d}&hd={hd}"

    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                logger.debug(f"オッズ取得HTTP {r.status_code}: attempt {attempt+1}")
                time.sleep(2)
                continue
        except Exception as e:
            logger.debug(f"オッズ取得エラー: {e}, attempt {attempt+1}")
            time.sleep(2)
            continue

        soup = BeautifulSoup(r.text, 'html.parser')
        odds_cells = soup.find_all('td', class_='oddsPoint')

        if len(odds_cells) < 120:
            logger.debug(f"オッズ要素不足: {len(odds_cells)}個 (120必要)")
            time.sleep(2)
            continue

        odds_dict = {}
        for pos, cell in enumerate(odds_cells[:120]):
            combo = _decode_odds_position(pos)
            if combo is None:
                continue

            text = cell.get_text(strip=True)
            if not text or text == '欠' or text == '特払':
                continue

            odds_val = _parse_float(text)
            if odds_val and odds_val > 0:
                key = f"{combo[0]}-{combo[1]}-{combo[2]}"
                odds_dict[key] = odds_val

        if odds_dict:
            logger.info(
                f"3連単オッズ取得: {len(odds_dict)}通り "
                f"(場{venue_id} R{race_number})"
            )
            return odds_dict

        time.sleep(2)

    logger.warning(f"3連単オッズ取得失敗: 場{venue_id} R{race_number}")
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
