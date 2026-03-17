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
    """td内のテキストをタグ(spanやbrなど)を無視してリストで返す"""
    if not td:
        return []
    return list(td.stripped_strings)


def scrape_race_deadlines(session, race_date, venue_id):
    """raceindexページから12R分の締切時刻を一括取得

    Args:
        session: requests.Session
        race_date: datetime.date
        venue_id: int (1-24)

    Returns:
        dict: {1: "15:20", 2: "15:45", ..., 12: "20:38"}
              パース失敗時は空dict
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/raceindex?jcd={venue_id:02d}&hd={hd}"

    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            logger.debug(f"raceindex HTTP {r.status_code}: 場{venue_id}")
            return {}
    except Exception as e:
        logger.debug(f"raceindex取得エラー: 場{venue_id}: {e}")
        return {}

    soup = BeautifulSoup(r.text, 'html.parser')

    # HH:MM 形式の td を全て取得
    time_cells = soup.find_all('td', string=re.compile(r'^\s*\d{2}:\d{2}\s*$'))

    if len(time_cells) < 12:
        logger.debug(
            f"締切時刻の要素不足: 場{venue_id} ({len(time_cells)}個, 12個必要)"
        )
        return {}

    deadlines = {}
    for i in range(12):
        time_text = time_cells[i].get_text(strip=True)
        deadlines[i + 1] = time_text

    logger.info(f"締切時刻取得: 場{venue_id} 12R分 ({deadlines.get(1)}〜{deadlines.get(12)})")
    return deadlines


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

    # 2連単払戻金
    payout_nirentan = 0
    for i in range(7, min(len(tbodies), 15)):
        tds_pay = tbodies[i].find_all('td')
        if tds_pay and '2連単' in tds_pay[0].get_text(strip=True):
            if len(tds_pay) >= 3:
                pay_text = tds_pay[2].get_text(strip=True)
                m = re.search(r'[¥￥]([0-9,]+)', pay_text)
                if m:
                    payout_nirentan = int(m.group(1).replace(',', ''))
            break

    return {
        'result_1st': ranking[0],
        'result_2nd': ranking[1],
        'result_3rd': ranking[2],
        'payout_sanrentan': payout,
        'payout_nirentan': payout_nirentan,
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


def _decode_odds_position_2t(position):
    """30個のオッズ位置 → (1着, 2着) 艇番号に変換

    boatrace.jp odds2t ページの td.oddsPoint 要素の並び順:
    - column = position % 6 → 1着の艇番号インデックス (0-5)
    - row = position // 6 → 2着候補インデックス (0-4, 1着を除く5艇)

    Returns:
        tuple(int, int): (1着, 2着) 艇番号 (1-6) or None
    """
    column = position % 6
    row = position // 6

    first = column + 1

    # 2着: 1着を除いた艇のうち、row番目（昇順）
    second_candidates = [b for b in range(1, 7) if b != first]
    if row >= len(second_candidates):
        return None
    second = second_candidates[row]

    return (first, second)


def scrape_odds_2t(session, race_date, venue_id, race_number, max_retries=3):
    """2連単オッズページから全30通りのオッズを取得

    Args:
        session: requests.Session
        race_date: datetime.date
        venue_id: int (1-24)
        race_number: int (1-12)
        max_retries: int リトライ回数

    Returns:
        dict: {"1-2": 4.5, ...} 倍率形式。取得失敗時はNone
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/odds2t?rno={race_number}&jcd={venue_id:02d}&hd={hd}"

    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                logger.debug(f"2連単オッズ取得HTTP {r.status_code}: attempt {attempt+1}")
                time.sleep(2)
                continue
        except Exception as e:
            logger.debug(f"2連単オッズ取得エラー: {e}, attempt {attempt+1}")
            time.sleep(2)
            continue

        soup = BeautifulSoup(r.text, 'html.parser')
        odds_cells = soup.find_all('td', class_='oddsPoint')

        if len(odds_cells) < 30:
            logger.debug(f"2連単オッズ要素不足: {len(odds_cells)}個 (30必要)")
            time.sleep(2)
            continue

        odds_dict = {}
        for pos, cell in enumerate(odds_cells[:30]):
            combo = _decode_odds_position_2t(pos)
            if combo is None:
                continue

            text = cell.get_text(strip=True)
            if not text or text == '欠' or text == '特払':
                continue

            odds_val = _parse_float(text)
            if odds_val and odds_val > 0:
                key = f"{combo[0]}-{combo[1]}"
                odds_dict[key] = odds_val

        if odds_dict:
            logger.info(
                f"2連単オッズ取得: {len(odds_dict)}通り "
                f"(場{venue_id} R{race_number})"
            )
            return odds_dict

        time.sleep(2)

    logger.warning(f"2連単オッズ取得失敗: 場{venue_id} R{race_number}")
    return None


# 風向CSSクラス番号 → 8方位マッピング（is-wind1〜16 → N/NE/E/SE/S/SW/W/NW, 17=calm）
_WIND_CLASS_MAP = {
    1: 'N', 2: 'N', 3: 'NE', 4: 'NE',
    5: 'E', 6: 'E', 7: 'SE', 8: 'SE',
    9: 'S', 10: 'S', 11: 'SW', 12: 'SW',
    13: 'W', 14: 'W', 15: 'NW', 16: 'NW',
    17: 'calm',
}


def _parse_weather(soup):
    """weather1 セクションから天候データを抽出

    Returns:
        dict: {temperature, wind_speed, wind_direction, wave_height, water_temperature}
    """
    weather = {
        'temperature': None,
        'wind_speed': None,
        'wind_direction': 'calm',
        'wave_height': None,
        'water_temperature': None,
    }

    weather_div = soup.find('div', class_='weather1')
    if not weather_div:
        return weather

    # 風速: "風速" ラベルの隣テキスト "2m"
    wind_label = weather_div.find(string=re.compile(r'風速'))
    if wind_label:
        sib = wind_label.find_next(string=re.compile(r'\d+'))
        if sib:
            m = re.search(r'(\d+)', sib)
            if m:
                weather['wind_speed'] = int(m.group(1))

    # 波高: "波高" ラベルの隣テキスト "1cm"
    wave_label = weather_div.find(string=re.compile(r'波高'))
    if wave_label:
        sib = wave_label.find_next(string=re.compile(r'\d+'))
        if sib:
            m = re.search(r'(\d+)', sib)
            if m:
                weather['wave_height'] = int(m.group(1))

    # 気温: "気温" ラベルの隣テキスト "10.0℃"
    temp_label = weather_div.find(string=re.compile(r'気温'))
    if temp_label:
        sib = temp_label.find_next(string=re.compile(r'[\d.]+'))
        if sib:
            m = re.search(r'([\d.]+)', sib)
            if m:
                weather['temperature'] = float(m.group(1))

    # 水温: "水温" ラベルの隣テキスト "11.0℃"
    water_label = weather_div.find(string=re.compile(r'水温'))
    if water_label:
        sib = water_label.find_next(string=re.compile(r'[\d.]+'))
        if sib:
            m = re.search(r'([\d.]+)', sib)
            if m:
                weather['water_temperature'] = float(m.group(1))

    # 風向: CSSクラス is-windXX → 8方位
    wind_dir_p = weather_div.find(
        'p', class_=lambda c: c and any('is-wind' in x for x in (c if isinstance(c, list) else [c]))
    )
    if wind_dir_p:
        classes = wind_dir_p.get('class', [])
        for cls in classes:
            m = re.match(r'is-wind(\d+)', cls)
            if m:
                wind_num = int(m.group(1))
                weather['wind_direction'] = _WIND_CLASS_MAP.get(wind_num, 'calm')
                break

    return weather


def _parse_boat_beforeinfo(tbody):
    """tbody (1艇分) から直前情報を抽出

    td[0]: 枠番
    td[3]: 体重 "53.7kg"
    td[4]: 展示タイム "6.75"
    td[5]: チルト "-0.5"
    td[7]: 部品交換 (空 or テキスト)
    """
    tds = tbody.find_all('td')
    if len(tds) < 8:
        return None

    boat_number = _parse_int(tds[0].get_text(strip=True))
    if not boat_number:
        return None

    # 体重
    weight_text = tds[3].get_text(strip=True)
    weight = None
    w_match = re.search(r'([\d.]+)', weight_text)
    if w_match:
        weight = float(w_match.group(1))

    # 展示タイム
    exhibition_time = _parse_float(tds[4].get_text(strip=True))

    # チルト
    tilt = _parse_float(tds[5].get_text(strip=True))

    # 部品交換
    parts_text = tds[7].get_text(strip=True) if len(tds) > 7 else ''
    parts_changed = bool(parts_text)

    return {
        'boat_number': boat_number,
        'weight': weight,
        'exhibition_time': exhibition_time,
        'tilt': tilt,
        'parts_changed': parts_changed,
        'parts_detail': parts_text if parts_changed else None,
    }


def _parse_start_exhibition(soup):
    """スタート展示セクションから進入コースを抽出

    tbody の最後のブロック (6セル) に "X.YY" 形式のデータが格納。
    整数部分が艇番号、小数部分×100で進入コースを推定。

    Returns:
        dict: {boat_number: approach_course, ...}
    """
    tbodies = soup.find_all('tbody')
    approach = {}

    # 最後の tbody (6セル) がスタート展示
    for tb in reversed(tbodies):
        tds = tb.find_all('td')
        if len(tds) == 6:
            for course, td in enumerate(tds, 1):
                text = td.get_text(strip=True)
                # "1.18" → 整数部=艇番号
                m = re.match(r'^(\d)', text)
                if m:
                    boat_num = int(m.group(1))
                    approach[boat_num] = course
            if approach:
                break

    return approach


def scrape_beforeinfo(session, race_date, venue_id, race_number):
    """直前情報ページから天候・展示タイム・チルト・部品交換・進入コースを取得

    Returns:
        dict: {
            'weather': {temperature, wind_speed, wind_direction, wave_height, water_temperature},
            'boats': [{boat_number, weight, exhibition_time, tilt, parts_changed,
                       parts_detail, approach_course}, ...]
        }
        or None
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

    # 天候データ
    weather = _parse_weather(soup)

    # 6艇分の直前データ
    tbodies = soup.find_all('tbody')
    boats = []
    for i in range(1, 7):
        if i < len(tbodies):
            boat_data = _parse_boat_beforeinfo(tbodies[i])
            if boat_data:
                boats.append(boat_data)

    if len(boats) != 6:
        logger.debug(
            f"直前情報パース不完全: 場{venue_id} R{race_number} "
            f"({len(boats)}/6艇)"
        )

    # 進入コース
    approach = _parse_start_exhibition(soup)
    for boat in boats:
        boat['approach_course'] = approach.get(
            boat['boat_number'], boat['boat_number']
        )

    logger.info(
        f"直前情報取得: 場{venue_id} R{race_number} "
        f"(風{weather['wind_direction']}{weather.get('wind_speed', '?')}m "
        f"波{weather.get('wave_height', '?')}cm "
        f"展示{[b.get('exhibition_time') for b in boats]})"
    )

    return {'weather': weather, 'boats': boats}


def scrape_race_result(session, race_date, venue_id, race_number):
    """レース結果ページから3連単の着順と払戻金を抽出する

    戻り値: {"trifecta": "1-5-2", "payout": 5830} （未確定や失敗時はNone）
    """
    hd = race_date.strftime('%Y%m%d')
    url = f"{BASE_URL}/raceresult?rno={race_number}&jcd={venue_id:02d}&hd={hd}"

    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, 'html.parser')

        # ページ内の全行(tr)をループして「3連単」の行を探す
        rows = soup.find_all('tr')
        for tr in rows:
            tds = tr.find_all('td')
            if not tds:
                continue
            if '3連単' not in tds[0].get_text(strip=True):
                continue

            # 1. 払戻金の抽出 (td[2]: "¥15,260" -> 15260)
            if len(tds) < 3:
                continue
            pay_text = tds[2].get_text(strip=True)
            payout_match = re.search(r'([0-9,]+)', pay_text.replace('¥', ''))
            if not payout_match:
                continue
            payout = int(payout_match.group(1).replace(',', ''))

            # 2. 着順の抽出 (td[1]: spanのis-typeXクラスから)
            result_numbers = []

            # パターンA: is-typeX クラスから艇番号を抽出
            for span in tds[1].find_all(
                'span', class_=re.compile(r'is-type[1-6]')
            ):
                classes = " ".join(span.get('class', []))
                m = re.search(r'is-type([1-6])', classes)
                if m:
                    result_numbers.append(m.group(1))

            # パターンB: テキストから直接拾う (フォールバック)
            if not result_numbers:
                td1_text = tds[1].get_text(strip=True)
                # "2-5-1" のようなハイフン区切り
                m = re.findall(r'[1-6]', td1_text)
                if len(m) >= 3:
                    result_numbers = m[:3]

            # 3つの数字が取れていれば成功
            if len(result_numbers) >= 3:
                trifecta = (
                    f"{result_numbers[0]}-"
                    f"{result_numbers[1]}-"
                    f"{result_numbers[2]}"
                )
                return {
                    "trifecta": trifecta,
                    "payout": payout,
                }

        return None

    except Exception as e:
        logger.warning(f"結果取得エラー 場{venue_id} R{race_number}: {e}")
        return None
