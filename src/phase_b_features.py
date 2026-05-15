"""Phase B 新特徴量の純関数 (B1/B2/B4)

- B1: classify_race_category(subtitle) + detect_planned_race(title)
- B2: compute_skill_gap(boats)
- B4: parse_day_label(label)

B3 (a_class_consumed) は SQL window aggregation で計算するため
analysis/21_build_phase_b_features.py に直書き (本ファイル対象外)。
"""
import re


_CATEGORY_RULES = [
    ("優勝戦", "final"),
    ("準優", "semifinal"),
    ("予選", "qualifier"),
    ("一般", "general"),
]


def classify_race_category(subtitle):
    """subtitle から race_category を分類

    Args:
        subtitle: str | None

    Returns:
        str: "qualifier" / "semifinal" / "final" / "general" / "other"
    """
    if not subtitle:
        return "other"
    for keyword, label in _CATEGORY_RULES:
        if keyword in subtitle:
            return label
    return "other"


_PLANNED_KEYWORDS = [
    "サンライズ",
    "ゴールデン",
    "Ｖプレミア",
    "Vプレミア",
    "V プレミア",
    "プレミアム",
    "GW特選",
    "GW 特選",
    "ゴールデンウィーク特選",
    "ＧＷ特選",
    "新春特選",
    "夏季特選",
    "特別選抜",
]


def detect_planned_race(title):
    """title が企画レース語を含むか

    Args:
        title: str | None

    Returns:
        bool
    """
    if not title:
        return False
    return any(kw in title for kw in _PLANNED_KEYWORDS)


_DAY_PATTERN = re.compile(r'^([０-９0-9]+)日目$')
_ZEN_TO_HAN = str.maketrans('０１２３４５６７８９', '0123456789')


def parse_day_label(label):
    """day_label を 1..N の integer に変換

    Args:
        label: str | None (例: "初日", "２日目", "５日目", "最終日", "優勝戦")

    Returns:
        int | None: 整数。最終日/優勝戦/不正値/None は None (節長依存)
    """
    if not label:
        return None
    if label == "初日":
        return 1
    m = _DAY_PATTERN.match(label)
    if m:
        num_str = m.group(1).translate(_ZEN_TO_HAN)
        return int(num_str)
    return None


def compute_skill_gap(boats):
    """1号艇 win_rate_2 − 平均(2..6号艇 win_rate_2)

    Args:
        boats: list[dict]、各要素は {'boat_number': int, 'win_rate_2': float|None}

    Returns:
        float | None: 6艇揃わない / win_rate_2 に NULL 含む / 1号艇欠落 時は None
    """
    if not boats or len(boats) != 6:
        return None
    by_num = {b['boat_number']: b.get('win_rate_2') for b in boats}
    if any(by_num.get(i) is None for i in range(1, 7)):
        return None
    boat1 = by_num[1]
    others_mean = sum(by_num[i] for i in range(2, 7)) / 5
    return float(boat1 - others_mean)
